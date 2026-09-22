"""Runner local delgado: consume el gateway por HTTP como los workers GPU.

Recibe el mensaje de la cola dev via webhook, trae los blobs por las
rutas dev, corre el pipeline existente contra el sidecar ML y reporta
progreso + bundle por HTTP. Solo stdlib, sin framework nuevo.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

from backend.domain import (
    BaseUrl,
    CompareResult,
    DomainError,
    Err,
    ImageBytes,
    InvalidR2Key,
    JobId,
    MlFailed,
    MlTransport,
    NotFound,
    Ok,
    Progress,
    QueueJob,
    R2Key,
    Stage,
    ZipBundle,
    domain_to_message,
    parse_base_url,
    parse_image_bytes,
    parse_progress,
    parse_queue_envelope,
    parse_r2_key,
)
from backend.gnm import build_result_zip, uv_to_png
from backend.pipeline_local import (
    MlSidecarClient,
    ProgressSink,
    default_config,
    run_pair,
)
from backend.shell_secrets import CfToken, SecretStr

logger = logging.getLogger("vultus-runner")

TERMINAL_STATUSES = ("done", "failed", "expired")

UNKNOWN_JOB_ID = "unknown"

# parse_r2_key vive en el dominio como dueno unico; el alias fija el borde
# de cola del runner sobre ese parser sin duplicar la regla.
_R2_KEY_PARSER = parse_r2_key


@dataclass(frozen=True, slots=True)
class DeadLetter:
    """Sobre shell para mensajes malos: bytes crudos mas error tipado."""

    raw: bytes
    reason: DomainError


_DEAD_LETTERS: list[DeadLetter] = []

_DEAD_LETTER_MAX = 100

# Lock para append/del + _METRICS: _Handler es ThreadingHTTPServer, sin
# esto dos POST /hooks/queue concurrentes pueden corromper la lista.
_DEAD_LETTER_LOCK = threading.Lock()

_METRICS: dict[str, int] = {"queue_skip": 0}


def _raw_to_bytes(raw: object) -> bytes:
    if isinstance(raw, (bytes, bytearray)):
        data = bytes(raw)
    else:
        try:
            data = json.dumps(raw, default=str).encode("utf-8")
        except (TypeError, ValueError):
            data = repr(raw).encode("utf-8")
    # Cap dead-letter payload: evita que un body gigante crezca sin cota
    # en memoria (100 cartas x N bytes).
    if len(data) > 4096:
        return data[:4096]
    return data


def _dead_letter(raw: object, reason: DomainError) -> DeadLetter:
    letter = DeadLetter(raw=_raw_to_bytes(raw), reason=reason)
    with _DEAD_LETTER_LOCK:
        _DEAD_LETTERS.append(letter)
        if len(_DEAD_LETTERS) > _DEAD_LETTER_MAX:
            del _DEAD_LETTERS[0 : len(_DEAD_LETTERS) - _DEAD_LETTER_MAX]
        _METRICS["queue_skip"] = _METRICS.get("queue_skip", 0) + 1
    return letter


def _load_cf_token() -> CfToken | None:
    raw = os.environ.get("CF_TOKEN", "").strip()
    if not raw:
        return None
    return CfToken(_inner=SecretStr(_value=raw))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _gateway_base() -> BaseUrl:
    raw = _env("GATEWAY_URL", "http://localhost:8000")
    parsed = parse_base_url(raw)
    if isinstance(parsed, Ok):
        return parsed.value
    logger.warning("invalid GATEWAY_URL, using default")
    fallback = parse_base_url("http://localhost:8000")
    assert isinstance(fallback, Ok)
    return fallback.value


def _http(method: str, url: str, body: bytes | None = None, content_type: str = "application/json") -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        # Red caida / gateway apagado / timeout: no tumbar el thread del
        # runner; el caller lo trata como status no-200 (599 = sin respuesta).
        logger.warning("http %s failed url=%s err=%s", method, url, exc)
        return 599, b""


def _gateway_status(job_id: JobId) -> str | None:
    base = _gateway_base()
    url = base.join("/v1/jobs/") + job_id.as_str()
    status, raw = _http("GET", url)
    if status != 200:
        return None
    try:
        return str(json.loads(raw.decode()).get("status"))
    except ValueError:
        return None


def _parse_queue_message(raw: object) -> Ok[QueueJob] | Err[DomainError]:
    """Borde del runner: delega al unico parse_queue_envelope del dominio.

    Solo canonico; legacy a/b es Err desde Wave 5.
    """
    return parse_queue_envelope(raw)


def _fetch_blob(job_id: JobId, key: R2Key, slot: Literal["a", "b"]) -> Ok[ImageBytes] | Err[DomainError]:
    # Contrato gateway vs R2Key: el gateway sirve /dev/blobs/{job}/{slot},
    # pero la R2Key probada gana sobre el slot cuando trae sufijo /a o /b.
    # Sufijo presente que discrepa del slot es Err sin fetch; sin sufijo
    # se usa el slot explicito bajo el contrato dev documentado.
    key_str = key.as_str()
    slot_value: str = slot
    if slot_value not in ("a", "b"):
        logger.warning("r2key invalid slot job=%s slot=%s key=%s", job_id.as_str(), slot_value, key_str)
        return Err(InvalidR2Key())
    safe_slot: Literal["a", "b"]
    if key_str.endswith("/a"):
        if slot != "a":
            logger.warning("r2key slot mismatch job=%s slot=%s key=%s", job_id.as_str(), slot, key_str)
            return Err(InvalidR2Key())
        safe_slot = "a"
    elif key_str.endswith("/b"):
        if slot != "b":
            logger.warning("r2key slot mismatch job=%s slot=%s key=%s", job_id.as_str(), slot, key_str)
            return Err(InvalidR2Key())
        safe_slot = "b"
    else:
        safe_slot = slot
    logger.debug("fetch blob job=%s slot=%s key=%s", job_id.as_str(), safe_slot, key_str)
    base = _gateway_base()
    url = base.join("/dev/blobs/") + f"{job_id.as_str()}/{safe_slot}"
    status, raw = _http("GET", url)
    if status == 404:
        return Err(NotFound(job_id=job_id.as_str()))
    if status != 200:
        return Err(MlFailed(detail=MlTransport(details=f"blob fetch {safe_slot} status={status}")))
    parsed = parse_image_bytes(raw)
    if isinstance(parsed, Err):
        return parsed
    return parsed


class HttpProgressSink:
    """Sink sobre HTTP: mismo patron que los workers GPU en prod.

    Analogo a OrderRepository en good-python: puerto estrecho que el
    pipeline consume sin conocer el transporte; tests inyectan el sink
    en memoria, prod inyecta este sink HTTP.
    """

    def __init__(self, job_id: JobId) -> None:
        self._job_id = job_id

    def _post_progress(self, payload: dict[str, object]) -> Ok[None] | Err[DomainError]:
        base = _gateway_base()
        url = base.join("/v1/jobs/") + f"{self._job_id.as_str()}/progress"
        status, _ = _http(
            "POST",
            url,
            json.dumps(payload).encode(),
        )
        if status == 404:
            return Err(NotFound(job_id=self._job_id.as_str()))
        if status < 200 or status >= 300:
            return Err(MlFailed(detail=MlTransport(details=f"progress post status={status}")))
        return Ok(None)

    def report(self, progress: Progress, stage: Stage) -> Ok[None] | Err[DomainError]:
        return self._post_progress({"progress": progress.value(), "stage": stage.as_str()})

    def complete(self, result: CompareResult) -> Ok[None] | Err[DomainError]:
        uv_a_png = uv_to_png(result.uv_a)
        uv_b_png = uv_to_png(result.uv_b)
        heatmap_png = uv_to_png(result.heatmap)
        bundle = ZipBundle(
            uv_a_png=uv_a_png,
            uv_b_png=uv_b_png,
            heatmap_png=heatmap_png,
            mesh_a_glb=result.mesh_a.as_bytes(),
            mesh_b_glb=result.mesh_b.as_bytes(),
            pbr_a=uv_a_png,
            pbr_b=uv_b_png,
        )
        blob = build_result_zip(bundle)
        base = _gateway_base()
        url = base.join("/dev/results/") + self._job_id.as_str()
        status, _ = _http(
            "PUT",
            url,
            blob,
            "application/zip",
        )
        if status < 200 or status >= 300:
            return Err(MlFailed(detail=MlTransport(details=f"result put status={status}")))
        done = parse_progress(1.0)
        assert isinstance(done, Ok)
        return self._post_progress({"progress": done.value.value(), "stage": Stage.DONE.as_str(), "status": "done"})

    def fail(self) -> Ok[None] | Err[DomainError]:
        result = self._post_progress({"status": "failed"})
        if isinstance(result, Err):
            return result
        return Ok(None)


def _sidecar_client() -> MlSidecarClient | None:
    raw = _env("ML_SIDECAR_URL", "http://ml-sidecar:8081")
    parsed = parse_base_url(raw)
    if isinstance(parsed, Err):
        logger.warning("invalid ML_SIDECAR_URL, job skipped")
        return None
    return MlSidecarClient(base_url=parsed.value)


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job_id: str
    action: str


def process_message(raw: object) -> JobOutcome:
    """Un job por llamada: valida, omite terminales, corre pipeline, reporta."""
    start = time.monotonic()
    parsed = _parse_queue_message(raw)
    if isinstance(parsed, Err):
        _dead_letter(raw, parsed.error)
        logger.warning("bad queue message skipped err=%s", domain_to_message(parsed.error))
        return JobOutcome(job_id=UNKNOWN_JOB_ID, action="skipped-bad-message")
    job = parsed.value
    job_id = job.job_id
    status = _gateway_status(job_id)
    if status in TERMINAL_STATUSES:
        logger.info("job already terminal skipped job=%s status=%s", job_id.as_str(), status)
        return JobOutcome(job_id=job_id.as_str(), action="skipped-terminal")
    ml = _sidecar_client()
    if ml is None:
        return JobOutcome(job_id=job_id.as_str(), action="skipped-no-sidecar")
    blob_a = _fetch_blob(job_id, job.r2_a, "a")
    if isinstance(blob_a, Err):
        HttpProgressSink(job_id).fail()
        return JobOutcome(job_id=job_id.as_str(), action="failed-blob")
    blob_b = _fetch_blob(job_id, job.r2_b, "b")
    if isinstance(blob_b, Err):
        HttpProgressSink(job_id).fail()
        return JobOutcome(job_id=job_id.as_str(), action="failed-blob")
    sink: ProgressSink = HttpProgressSink(job_id)
    result = run_pair(sink, ml, job_id, blob_a.value, blob_b.value, default_config())
    elapsed_ms = int((time.monotonic() - start) * 1000)
    if isinstance(result, Err):
        logger.warning("pipeline failed job=%s duration_ms=%d", job_id.as_str(), elapsed_ms)
        return JobOutcome(job_id=job_id.as_str(), action="failed-pipeline")
    logger.info("pipeline done job=%s duration_ms=%d", job_id.as_str(), elapsed_ms)
    return JobOutcome(job_id=job_id.as_str(), action="done")


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/hooks/queue":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode())
        except ValueError:
            payload = None
        outcome = process_message(payload)
        body = json.dumps({"job_id": outcome.job_id, "action": outcome.action}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        logger.info("runner http %s", args[1] if len(args) > 1 else "")


def serve_forever(host: str = "0.0.0.0", port: int = 8001) -> None:
    server = ThreadingHTTPServer((host, port), _Handler)
    logger.info("runner serving host=%s port=%d gateway=%s", host, port, _gateway_base().as_str())
    server.serve_forever()


def main(argv: list[str]) -> int:
    """Modo one-shot para debug: `local_runner.py --once message.json`."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    if len(argv) == 3 and argv[1] == "--once":
        with open(argv[2], encoding="utf-8") as f:
            payload = json.load(f)
        outcome = process_message(payload)
        print(json.dumps({"job_id": outcome.job_id, "action": outcome.action}))
        return 0 if outcome.action in ("done", "skipped-terminal") else 1
    host = _env("RUNNER_HOST", "0.0.0.0")
    port = int(_env("RUNNER_PORT", "8001"))
    serve_forever(host, port)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
