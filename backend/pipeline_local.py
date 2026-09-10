"""Orquestador local: paralelismo por cara, timeouts fijos y deadline total = TTL.

Fit y textura via sidecar HTTP (`/ml/fit`, `/ml/texture`); assemble, heatmap
y GLB son funciones puras CPU sin I/O via `gnm_fit`/`gnm_texture`/`gnm_assemble`.
Solo bordes externos: cola, tiempo, filesystem efimero, sidecar.
Logs solo con job id y duraciones, nunca bytes.
"""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from backend.domain import (
    BaseUrl,
    CompareResult,
    CompleteUv,
    DomainError,
    Err,
    FitFailed,
    FitResult,
    ImageBytes,
    JobId,
    Landmarks,
    MlBadStatus,
    MlDecode,
    MlEmpty,
    MlFailed,
    MlTransport,
    NotFound,
    Ok,
    Progress,
    Stage,
    encode_fit_request,
    parse_camera_params,
    parse_complete_uv,
    parse_gnm_coeffs,
    parse_landmarks,
    parse_progress,
)
from backend.gnm import compute_heatmap
from backend.gnm_assemble import build_personalized_glb

logger = logging.getLogger("vultus-pipeline")


def job_dir(job_id: JobId) -> Path:
    """Directorio efimero del job en tmpfs. Ciclo de vida del pipeline, no de la cola."""
    return Path(tempfile.gettempdir()) / f"vultus-{job_id.as_str()}"


def cleanup_job_dir(job_id: JobId) -> None:
    directory = job_dir(job_id)
    try:
        shutil.rmtree(directory, ignore_errors=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("tmp cleanup failed job=%s path=%s err=%s", job_id.as_str(), str(directory), exc)


class ProgressSink(Protocol):
    """Seam estrecho de progreso: el pipeline reporta, completa o falla.

    Los tests inyectan el sink en memoria; el runner local y los workers
    GPU lo implementan sobre HTTP. Misma forma en ambos entornos.
    """

    def report(self, progress: Progress, stage: Stage) -> Ok[None] | Err[DomainError]:
        ...

    def complete(self, result: CompareResult) -> Ok[None] | Err[DomainError]:
        ...

    def fail(self) -> Ok[None] | Err[DomainError]:
        ...


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Presupuesto por etapa dentro del TTL total (S10).

    Base medida: fit doble p95 local <1s (test_fit_p95_local_double_inside_ttl_gate).
    T4 real sin medir (fitter pendiente Step 4): el total se queda en 60s y
    los stages heredan el presupuesto de la via anterior (5+10+30<=60).
    """

    landmarks_timeout_secs: float = 5.0
    fit_timeout_secs: float = 10.0
    texture_timeout_secs: float = 30.0
    total_timeout_secs: float = 60.0


def default_config() -> PipelineConfig:
    return PipelineConfig()


# --- Contrato wire /ml/fit y /ml/texture (S8 lo espeja en modal_app) ---
#
# Fit request: encode_fit_request (u32 BE len + landmarks_json + image).
# Fit response: 253 f32 LE + 12 f32 LE = 1060 bytes.
# Texture request: u32 BE len(fit_request) + fit_request + fit_result(1060).
# Texture response: UV_LEN bytes crudos -> parse_complete_uv.

FIT_RESULT_LEN = 253 * 4 + 12 * 4


def encode_fit_result(fit: FitResult) -> bytes:
    import struct as _struct

    return _struct.pack("<253f", *fit.coeffs.as_tuple()) + _struct.pack("<12f", *fit.camera.as_tuple())


def parse_fit_result(raw: object) -> Ok[FitResult] | Err[DomainError]:
    import struct as _struct

    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return Err(FitFailed(detail=MlDecode(details="fit result <1060 bytes")))
    data = bytes(raw)
    if len(data) != FIT_RESULT_LEN:
        return Err(FitFailed(detail=MlDecode(details=f"fit result len {len(data)} != {FIT_RESULT_LEN}")))
    try:
        coeff_values = list(_struct.unpack_from("<253f", data, 0))
        camera_values = list(_struct.unpack_from("<12f", data, 253 * 4))
    except _struct.error as exc:
        return Err(FitFailed(detail=MlDecode(details=f"fit result unpack: {exc}")))
    coeffs = parse_gnm_coeffs(coeff_values)
    if isinstance(coeffs, Err):
        return coeffs
    camera = parse_camera_params(camera_values)
    if isinstance(camera, Err):
        return camera
    return Ok(FitResult(coeffs=coeffs.value, camera=camera.value))


def encode_texture_request(image: ImageBytes, fit: FitResult, landmarks: Landmarks) -> bytes:
    fit_req = encode_fit_request(image, landmarks)
    return len(fit_req).to_bytes(4, "big") + fit_req + encode_fit_result(fit)


def decode_texture_request(raw: object) -> Ok[tuple[ImageBytes, FitResult, Landmarks]] | Err[DomainError]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return Err(FitFailed(detail=MlDecode(details="texture payload <4 bytes")))
    data = bytes(raw)
    if len(data) < 4:
        return Err(FitFailed(detail=MlDecode(details="texture payload <4 bytes")))
    size = int.from_bytes(data[0:4], "big")
    if len(data) < 4 + size + FIT_RESULT_LEN:
        return Err(FitFailed(detail=MlDecode(details="texture payload truncated")))
    from backend.domain import decode_fit_request

    fit_req = decode_fit_request(data[4 : 4 + size])
    if isinstance(fit_req, Err):
        return fit_req
    fit_res = parse_fit_result(data[4 + size : 4 + size + FIT_RESULT_LEN])
    if isinstance(fit_res, Err):
        return fit_res
    landmarks, image = fit_req.value
    return Ok((image, fit_res.value, landmarks))


class MlSidecarClient:
    def __init__(self, base_url: BaseUrl, timeout_secs: float = 35.0) -> None:
        self._base = base_url
        self._timeout = timeout_secs

    def base_url(self) -> BaseUrl:
        return self._base

    def _post(self, path: str, job_id: JobId, payload: bytes) -> Ok[bytes] | Err[DomainError]:
        url = self._base.join(path)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    url,
                    content=payload,
                    headers={"X-Job-Id": job_id.as_str(), "Content-Type": "application/octet-stream"},
                )
        except httpx.HTTPError as exc:
            return Err(MlFailed(detail=MlTransport(details=str(exc))))
        if resp.status_code < 200 or resp.status_code >= 300:
            return Err(MlFailed(detail=MlBadStatus(status=resp.status_code)))
        if len(resp.content) == 0:
            return Err(MlFailed(detail=MlEmpty()))
        return Ok(bytes(resp.content))

    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        result = self._post("/ml/landmarks", job_id, image.as_bytes())
        if isinstance(result, Err):
            return result
        parsed = parse_landmarks(result.value)
        if isinstance(parsed, Err):
            return parsed
        return parsed

    def fit(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
        payload = encode_fit_request(image, landmarks)
        result = self._post("/ml/fit", job_id, payload)
        if isinstance(result, Err):
            return result
        return parse_fit_result(result.value)

    def texture(
        self, job_id: JobId, image: ImageBytes, fit: FitResult, landmarks: Landmarks
    ) -> Ok[CompleteUv] | Err[DomainError]:
        payload = encode_texture_request(image, fit, landmarks)
        result = self._post("/ml/texture", job_id, payload)
        if isinstance(result, Err):
            return result
        parsed = parse_complete_uv(result.value)
        if isinstance(parsed, Err):
            return parsed
        return parsed


def run_pair(
    sink: ProgressSink,
    ml: MlSidecarClient,
    job_id: JobId,
    image_a: ImageBytes,
    image_b: ImageBytes,
    cfg: PipelineConfig | None = None,
) -> Ok[CompareResult] | Err[DomainError]:
    config = cfg if cfg is not None else default_config()
    start = time.monotonic()
    deadline = start + config.total_timeout_secs
    try:
        job_dir(job_id).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    def _expired() -> bool:
        return time.monotonic() > deadline

    def _fail(message: str) -> Err[DomainError]:
        sink.fail()
        try:
            cleanup_job_dir(job_id)
        except OSError:
            pass
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.warning("pipeline failed job=%s duration_ms=%d err=%s", job_id.as_str(), duration_ms, message)
        return Err(MlFailed(detail=MlTransport(details=message)))

    p40 = parse_progress(0.40)
    assert isinstance(p40, Ok)
    progress_result = sink.report(p40.value, Stage.FIT)
    if isinstance(progress_result, Err):
        cleanup_job_dir(job_id)
        return Err(progress_result.error)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_lm:
            fut_lm_a = pool_lm.submit(ml.landmarks, job_id, image_a)
            fut_lm_b = pool_lm.submit(ml.landmarks, job_id, image_b)
            try:
                ra_lm = fut_lm_a.result(timeout=config.landmarks_timeout_secs + 25.0)
                rb_lm = fut_lm_b.result(timeout=config.landmarks_timeout_secs + 25.0)
            except concurrent.futures.TimeoutError:
                return _fail("landmarks timeout")
            if isinstance(ra_lm, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return ra_lm
            if isinstance(rb_lm, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return rb_lm
            landmarks_a = ra_lm.value
            landmarks_b = rb_lm.value
    except Exception as exc:  # noqa: BLE001
        return _fail(f"landmarks failed: {exc}")

    if _expired():
        return _fail("pipeline total timeout")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_fit:
            fut_fit_a = pool_fit.submit(ml.fit, job_id, image_a, landmarks_a)
            fut_fit_b = pool_fit.submit(ml.fit, job_id, image_b, landmarks_b)
            try:
                ra_fit = fut_fit_a.result(timeout=config.fit_timeout_secs + 50.0)
                rb_fit = fut_fit_b.result(timeout=config.fit_timeout_secs + 50.0)
            except concurrent.futures.TimeoutError:
                return _fail("fit timeout")
            if isinstance(ra_fit, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return ra_fit
            if isinstance(rb_fit, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return rb_fit
            fit_a = ra_fit.value
            fit_b = rb_fit.value
    except Exception as exc:  # noqa: BLE001
        return _fail(f"fit failed: {exc}")

    if _expired():
        return _fail("pipeline total timeout")
    p75 = parse_progress(0.75)
    assert isinstance(p75, Ok)
    if isinstance(sink.report(p75.value, Stage.TEXTURE), Err):
        cleanup_job_dir(job_id)
        return Err(NotFound(job_id=job_id.as_str()))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_tx:
            fut_tx_a = pool_tx.submit(ml.texture, job_id, image_a, fit_a, landmarks_a)
            fut_tx_b = pool_tx.submit(ml.texture, job_id, image_b, fit_b, landmarks_b)
            try:
                ra_tx = fut_tx_a.result(timeout=config.texture_timeout_secs + 30.0)
                rb_tx = fut_tx_b.result(timeout=config.texture_timeout_secs + 30.0)
            except concurrent.futures.TimeoutError:
                return _fail("texture timeout")
            if isinstance(ra_tx, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return ra_tx
            if isinstance(rb_tx, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return rb_tx
            uv_a = ra_tx.value
            uv_b = rb_tx.value
    except Exception as exc:  # noqa: BLE001
        return _fail(f"texture failed: {exc}")

    p95 = parse_progress(0.95)
    assert isinstance(p95, Ok)
    if isinstance(sink.report(p95.value, Stage.ASSEMBLE), Err):
        cleanup_job_dir(job_id)
        return Err(NotFound(job_id=job_id.as_str()))

    assemble_start = time.monotonic()
    mesh_a_result = build_personalized_glb(fit_a, uv_a)
    if isinstance(mesh_a_result, Err):
        return _fail(f"assemble mesh_a failed: {mesh_a_result.error}")
    mesh_b_result = build_personalized_glb(fit_b, uv_b)
    if isinstance(mesh_b_result, Err):
        return _fail(f"assemble mesh_b failed: {mesh_b_result.error}")
    mesh_a = mesh_a_result.value
    mesh_b = mesh_b_result.value
    heatmap = compute_heatmap(uv_a, uv_b)
    assemble_ms = int((time.monotonic() - assemble_start) * 1000)
    logger.info(
        "assemble gnm done job=%s assemble_ms=%d mesh_a_len=%d mesh_b_len=%d",
        job_id.as_str(),
        assemble_ms,
        len(mesh_a.as_bytes()),
        len(mesh_b.as_bytes()),
    )
    output = CompareResult(uv_a=uv_a, uv_b=uv_b, heatmap=heatmap, mesh_a=mesh_a, mesh_b=mesh_b)
    completed = sink.complete(output)
    if isinstance(completed, Err):
        cleanup_job_dir(job_id)
        return Err(completed.error)
    cleanup_job_dir(job_id)
    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info("pipeline done job=%s duration_ms=%d", job_id.as_str(), duration_ms)
    return Ok(output)
