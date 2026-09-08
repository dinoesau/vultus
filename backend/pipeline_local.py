"""Orquestador local: paralelismo por cara, timeouts fijos y deadline total = TTL.

Bake, heatmap, GLB y zip son funciones puras CPU sin I/O via `gnm`.
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
    FlawUv,
    ImageBytes,
    JobId,
    Landmarks,
    MlBadStatus,
    MlEmpty,
    MlFailed,
    MlTransport,
    NotFound,
    Ok,
    Progress,
    Stage,
    encode_flame_payload,
    parse_complete_uv,
    parse_flaw_uv,
    parse_landmarks,
    parse_progress,
)
from backend.gnm import bake_bfm_to_gnm, build_gnm_glb, compute_heatmap

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
    landmarks_timeout_secs: float = 5.0
    flame_timeout_secs: float = 10.0
    freeuv_timeout_secs: float = 30.0
    total_timeout_secs: float = 60.0


def default_config() -> PipelineConfig:
    return PipelineConfig()


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

    def flame(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FlawUv] | Err[DomainError]:
        payload = encode_flame_payload(landmarks, image)
        result = self._post("/ml/flame", job_id, payload)
        if isinstance(result, Err):
            return result
        parsed = parse_flaw_uv(result.value)
        if isinstance(parsed, Err):
            return parsed
        return parsed

    def freeuv(self, job_id: JobId, flaw: FlawUv) -> Ok[CompleteUv] | Err[DomainError]:
        result = self._post("/ml/freeuv", job_id, flaw.as_bytes())
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

    p15 = parse_progress(0.15)
    assert isinstance(p15, Ok)
    progress_result = sink.report(p15.value, Stage.LANDMARKS)
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
    p40 = parse_progress(0.40)
    assert isinstance(p40, Ok)
    if isinstance(sink.report(p40.value, Stage.FLAME), Err):
        cleanup_job_dir(job_id)
        return Err(NotFound(job_id=job_id.as_str()))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_flame:
            fut_flame_a = pool_flame.submit(ml.flame, job_id, image_a, landmarks_a)
            fut_flame_b = pool_flame.submit(ml.flame, job_id, image_b, landmarks_b)
            try:
                ra_flame = fut_flame_a.result(timeout=config.flame_timeout_secs + 50.0)
                rb_flame = fut_flame_b.result(timeout=config.flame_timeout_secs + 50.0)
            except concurrent.futures.TimeoutError:
                return _fail("flame timeout")
            if isinstance(ra_flame, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return ra_flame
            if isinstance(rb_flame, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return rb_flame
            flaw_a = ra_flame.value
            flaw_b = rb_flame.value
    except Exception as exc:  # noqa: BLE001
        return _fail(f"flame failed: {exc}")

    if _expired():
        return _fail("pipeline total timeout")
    p75 = parse_progress(0.75)
    assert isinstance(p75, Ok)
    if isinstance(sink.report(p75.value, Stage.FREEUV), Err):
        cleanup_job_dir(job_id)
        return Err(NotFound(job_id=job_id.as_str()))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_uv:
            fut_uv_a = pool_uv.submit(ml.freeuv, job_id, flaw_a)
            fut_uv_b = pool_uv.submit(ml.freeuv, job_id, flaw_b)
            try:
                ra_uv = fut_uv_a.result(timeout=config.freeuv_timeout_secs + 30.0)
                rb_uv = fut_uv_b.result(timeout=config.freeuv_timeout_secs + 30.0)
            except concurrent.futures.TimeoutError:
                return _fail("freeuv timeout")
            if isinstance(ra_uv, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return ra_uv
            if isinstance(rb_uv, Err):
                sink.fail()
                cleanup_job_dir(job_id)
                return rb_uv
            uv_a = ra_uv.value
            uv_b = rb_uv.value
    except Exception as exc:  # noqa: BLE001
        return _fail(f"freeuv failed: {exc}")

    p95 = parse_progress(0.95)
    assert isinstance(p95, Ok)
    if isinstance(sink.report(p95.value, Stage.BAKE), Err):
        cleanup_job_dir(job_id)
        return Err(NotFound(job_id=job_id.as_str()))

    bake_start = time.monotonic()
    baked_a = bake_bfm_to_gnm(uv_a)
    baked_b = bake_bfm_to_gnm(uv_b)
    mesh_a = build_gnm_glb(baked_a)
    mesh_b = build_gnm_glb(baked_b)
    heatmap = compute_heatmap(uv_a, uv_b)
    bake_ms = int((time.monotonic() - bake_start) * 1000)
    logger.info(
        "bake gnm done job=%s bake_ms=%d mesh_a_len=%d mesh_b_len=%d",
        job_id.as_str(),
        bake_ms,
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
