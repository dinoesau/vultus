"""Runner local delgado: consume el gateway por HTTP como los workers GPU.

Recibe el mensaje de la cola dev via webhook, trae los blobs por las
rutas dev, corre el pipeline existente contra el sidecar ML y reporta
progreso + bundle por HTTP. Solo stdlib, sin framework nuevo.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend.domain import (
    CompareResult,
    DomainError,
    Err,
    ImageBytes,
    JobId,
    Ok,
    Progress,
    Stage,
    domain_to_message,
    parse_base_url,
    parse_image_bytes,
    parse_job_id,
    parse_progress,
)
from backend.gnm import build_result_zip, uv_to_png
from backend.pipeline_local import (
    MlSidecarClient,
    ProgressSink,
    default_config,
    run_pair,
)

logger = logging.getLogger("vultus-runner")

TERMINAL_STATUSES = ("done", "failed", "expired")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _gateway_base() -> str:
    return _env("GATEWAY_URL", "http://localhost:8000").rstrip("/")


def _http(method: str, url: str, body: bytes | None = None, content_type: str = "application/json") -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _gateway_status(job_id: JobId) -> str | None:
    status, raw = _http("GET", f"{_gateway_base()}/v1/jobs/{job_id.as_str()}")
    if status != 200:
        return None
    try:
        return str(json.loads(raw.decode()).get("status"))
    except ValueError:
        return None


def _parse_queue_message(raw: object) -> Ok[tuple[JobId, str, str]] | Err[str]:
    """Borde del runner: valida job_id y punteros sin `..` una sola vez."""
    if not isinstance(raw, dict):
        return Err("queue body not object")
    job_raw = raw.get("job_id")
    keys = raw.get("r2_keys")
    if not isinstance(keys, dict):
        return Err("queue body missing r2_keys")
    job_result = parse_job_id(job_raw)
    if isinstance(job_result, Err):
        return Err(domain_to_message(job_result.error))
    for name in ("image_a", "image_b"):
        key = keys.get(name)
        if not isinstance(key, str) or not key.strip() or ".." in key or len(key.strip()) > 1024:
            return Err(f"invalid r2 key for {name}")
    assert isinstance(keys.get("image_a"), str) and isinstance(keys.get("image_b"), str)
    return Ok((job_result.value, str(keys["image_a"]).strip(), str(keys["image_b"]).strip()))


def _fetch_blob(job_id: JobId, key: str) -> Ok[ImageBytes] | Err[DomainError]:
    slot = "a" if key.rstrip().endswith("/a") else "b"
    status, raw = _http("GET", f"{_gateway_base()}/dev/blobs/{job_id.as_str()}/{slot}")
    if status != 200:
        from backend.domain import MlFailed, MlTransport

        return Err(MlFailed(detail=MlTransport(details=f"blob fetch {slot} status={status}")))
    parsed = parse_image_bytes(raw)
    if isinstance(parsed, Err):
        return parsed
    return parsed


class HttpProgressSink:
    """Sink sobre HTTP: mismo patron que los workers GPU en prod."""

    def __init__(self, job_id: JobId) -> None:
        self._job_id = job_id

    def _post_progress(self, payload: dict[str, object]) -> Ok[None] | Err[DomainError]:
        from backend.domain import NotFound

        status, _ = _http(
            "POST",
            f"{_gateway_base()}/v1/jobs/{self._job_id.as_str()}/progress",
            json.dumps(payload).encode(),
        )
        if status == 404:
            return Err(NotFound(job_id=self._job_id.as_str()))
        if status < 200 or status >= 300:
            from backend.domain import MlFailed, MlTransport

            return Err(MlFailed(detail=MlTransport(details=f"progress post status={status}")))
        return Ok(None)

    def report(self, progress: Progress, stage: Stage) -> Ok[None] | Err[DomainError]:
        return self._post_progress({"progress": progress.value(), "stage": stage.as_str()})

    def complete(self, result: CompareResult) -> Ok[None] | Err[DomainError]:
        blob = build_result_zip(
            uv_to_png(result.uv_a.as_bytes()),
            uv_to_png(result.uv_b.as_bytes()),
            uv_to_png(result.heatmap.as_bytes()),
            result.mesh_a.as_bytes(),
            result.mesh_b.as_bytes(),
        )
        status, _ = _http(
            "PUT",
            f"{_gateway_base()}/dev/results/{self._job_id.as_str()}",
            blob,
            "application/zip",
        )
        if status < 200 or status >= 300:
            from backend.domain import MlFailed, MlTransport

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
        logger.warning("bad queue message skipped err=%s", parsed.error)
        return JobOutcome(job_id="unknown", action="skipped-bad-message")
    job_id, key_a, key_b = parsed.value
    status = _gateway_status(job_id)
    if status in TERMINAL_STATUSES:
        logger.info("job already terminal skipped job=%s status=%s", job_id.as_str(), status)
        return JobOutcome(job_id=job_id.as_str(), action="skipped-terminal")
    ml = _sidecar_client()
    if ml is None:
        return JobOutcome(job_id=job_id.as_str(), action="skipped-no-sidecar")
    blob_a = _fetch_blob(job_id, key_a)
    if isinstance(blob_a, Err):
        HttpProgressSink(job_id).fail()
        return JobOutcome(job_id=job_id.as_str(), action="failed-blob")
    blob_b = _fetch_blob(job_id, key_b)
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
    logger.info("runner serving host=%s port=%d gateway=%s", host, port, _gateway_base())
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
