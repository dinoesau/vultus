"""API local Python: duena de API local, cola en memoria y orquestador local.

Shell delgado: parsea una vez en el borde, llama al core, mapea errores a HTTP.
Sin revalidar en caliente lo ya probado. Logs solo con job id y duraciones.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.domain import (
    BaseUrl,
    Err,
    JobId,
    JobStatus,
    Ok,
    Stage,
    default_ttl,
    domain_to_message,
    parse_base_url,
    parse_image_bytes,
    parse_job_id,
    parse_progress,
    parse_stage,
    parse_ttl_secs,
)
from backend.domain import EnqueueCommand as DomainEnqueueCommand
from backend.gnm import build_result_zip, uv_to_png
from backend.pipeline_local import MlSidecarClient, QueueLike, default_config, run_pair
from backend.store import MemoryQueue, R2PointerQueue

logger = logging.getLogger("vultus-api")


def _parse_ttl_env() -> object:
    from backend.domain import TtlSecs

    raw = os.environ.get("R2_TTL_SECONDS", "").strip() or os.environ.get("RESULT_TTL_SECONDS", "").strip()
    if not raw:
        return default_ttl()
    try:
        number = int(raw.strip())
    except ValueError:
        return default_ttl()
    result = parse_ttl_secs(number)
    if isinstance(result, Err):
        return default_ttl()
    ttl: TtlSecs = result.value
    return ttl


def _queue_from_env() -> QueueLike:
    ttl = _parse_ttl_env()
    assert not isinstance(ttl, Err)
    driver = os.environ.get("QUEUE_DRIVER", "memory").strip().lower()
    if driver in ("r2pointer", "r2_pointer", "cloudflare", "r2"):
        return R2PointerQueue(ttl=ttl)  # type: ignore[arg-type]
    return MemoryQueue(ttl=ttl)  # type: ignore[arg-type]


def _sidecar_from_env() -> MlSidecarClient | None:
    raw = os.environ.get("ML_SIDECAR_URL", "http://ml-sidecar:8081").strip()
    if not raw:
        return None
    parsed = parse_base_url(raw)
    if isinstance(parsed, Err):
        logger.warning("invalid ML_SIDECAR_URL, pipeline disabled")
        return None
    return MlSidecarClient(base_url=parsed.value)


def create_app(queue: QueueLike | None = None, sidecar: MlSidecarClient | None = None) -> FastAPI:
    owned_queue: QueueLike = queue if queue is not None else _queue_from_env()
    owned_sidecar = sidecar if sidecar is not None else (_sidecar_from_env() if queue is None else None)

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        ttl_secs = owned_queue.ttl().value()
        interval = max(1, (ttl_secs + 1) // 2)

        async def _reaper() -> None:
            while True:
                await asyncio.sleep(float(interval))
                try:
                    purged = owned_queue.purge_expired()
                    if purged:
                        logger.info("ttl reaper purged jobs purged=%d", purged)
                except OSError:
                    continue

        task = asyncio.create_task(_reaper())
        yield
        task.cancel()

    app = FastAPI(title="vultus-api-local", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.state.queue = owned_queue
    app.state.sidecar = owned_sidecar

    @app.get("/health")
    async def health() -> JSONResponse:
        probe_result = owned_queue.status(JobId(_value="00000000-0000-4000-8000-000000000000"))
        queue_ok = isinstance(probe_result, (Ok, Err))
        ttl_secs = owned_queue.ttl().value()
        sidecar_state = "disabled" if owned_sidecar is None else "ok"
        body = {"status": "ok", "queue": "ok" if queue_ok else "error", "ttl_secs": ttl_secs, "sidecar": sidecar_state}
        return JSONResponse(status_code=200, content=body)

    @app.post("/v1/compare")
    async def compare(request: Request) -> Response:
        ctype = request.headers.get("content-type", "")
        if "multipart/form-data" not in ctype:
            return JSONResponse(status_code=400, content={"detail": "missing multipart"})
        try:
            form = await request.form()
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": f"invalid multipart: {exc}"})
        raw_a = form.get("image_a")
        raw_b = form.get("image_b")
        if raw_a is None or raw_b is None:
            return JSONResponse(status_code=400, content={"detail": "missing image_a or image_b"})
        if isinstance(raw_a, str) or isinstance(raw_b, str):
            return JSONResponse(status_code=400, content={"detail": "missing image_a or image_b"})
        try:
            data_a = await raw_a.read()
            data_b = await raw_b.read()
        except (AttributeError, OSError, ValueError):
            return JSONResponse(status_code=400, content={"detail": "missing image_a or image_b"})
        if not isinstance(data_a, (bytes, bytearray)) or not isinstance(data_b, (bytes, bytearray)):
            return JSONResponse(status_code=400, content={"detail": "missing image_a or image_b"})
        parsed_a = parse_image_bytes(data_a)
        if isinstance(parsed_a, Err):
            return JSONResponse(status_code=400, content={"detail": domain_to_message(parsed_a.error)})
        parsed_b = parse_image_bytes(data_b)
        if isinstance(parsed_b, Err):
            return JSONResponse(status_code=400, content={"detail": domain_to_message(parsed_b.error)})
        cmd = DomainEnqueueCommand(image_a=parsed_a.value, image_b=parsed_b.value)
        enqueued = owned_queue.enqueue(cmd)
        job_id = enqueued.job_id
        logger.info("job enqueued job=%s", job_id.as_str())
        if owned_sidecar is not None:
            ml = owned_sidecar
            img_a = parsed_a.value
            img_b = parsed_b.value
            cfg = default_config()

            async def _background() -> None:
                start = asyncio.get_event_loop().time()
                try:
                    result = await asyncio.to_thread(run_pair, owned_queue, ml, job_id, img_a, img_b, cfg)
                    elapsed_ms = int((asyncio.get_event_loop().time() - start) * 1000)
                    if isinstance(result, Err):
                        logger.warning("pipeline failed job=%s duration_ms=%d", job_id.as_str(), elapsed_ms)
                    else:
                        logger.info("pipeline done job=%s duration_ms=%d", job_id.as_str(), elapsed_ms)
                except OSError as exc:
                    logger.warning("pipeline failed job=%s err=%s", job_id.as_str(), exc)

            asyncio.create_task(_background())
        return JSONResponse(status_code=202, content={"job_id": job_id.as_str(), "status": "queued"})

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str) -> Response:
        parsed = parse_job_id(job_id)
        if isinstance(parsed, Err):
            return JSONResponse(status_code=400, content={"detail": domain_to_message(parsed.error)})
        status_result = owned_queue.status(parsed.value)
        if isinstance(status_result, Err):
            return JSONResponse(status_code=404, content={"detail": "not found"})
        return JSONResponse(
            status_code=200, content={"job_id": parsed.value.as_str(), "status": status_result.value.as_str()}
        )

    @app.get("/v1/jobs/{job_id}/result")
    async def job_result(job_id: str) -> Response:
        parsed = parse_job_id(job_id)
        if isinstance(parsed, Err):
            return JSONResponse(status_code=400, content={"detail": domain_to_message(parsed.error)})
        jid = parsed.value
        status_result = owned_queue.status(jid)
        if isinstance(status_result, Err):
            return JSONResponse(status_code=404, content={"detail": "not found"})
        if status_result.value != JobStatus.DONE:
            return JSONResponse(status_code=409, content={"detail": "not done"})
        fetched = owned_queue.fetch_result(jid)
        if isinstance(fetched, Err):
            return JSONResponse(status_code=404, content={"detail": "not found"})
        result = fetched.value
        a_png = uv_to_png(result.uv_a.as_bytes())
        b_png = uv_to_png(result.uv_b.as_bytes())
        h_png = uv_to_png(result.heatmap.as_bytes())
        blob = build_result_zip(a_png, b_png, h_png, result.mesh_a.as_bytes(), result.mesh_b.as_bytes())
        logger.info("result zip served job=%s zip_len=%d", jid.as_str(), len(blob))
        return Response(
            content=blob,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="result-{jid.as_str()}.zip"'},
        )

    @app.post("/v1/jobs/{job_id}/progress")
    async def job_progress(job_id: str, request: Request) -> Response:
        parsed = parse_job_id(job_id)
        if isinstance(parsed, Err):
            return JSONResponse(status_code=400, content={"detail": domain_to_message(parsed.error)})
        jid = parsed.value
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "invalid json"})
        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"detail": "invalid json"})
        status_raw = payload.get("status")
        if status_raw == "failed":
            failed = owned_queue.fail_job(jid)
            if isinstance(failed, Err):
                return JSONResponse(status_code=404, content={"detail": "not found"})
            return JSONResponse(status_code=200, content={"ok": True})
        if status_raw == "done":
            return JSONResponse(status_code=400, content={"detail": "invalid status"})
        if status_raw is not None and status_raw != "processing":
            return JSONResponse(status_code=400, content={"detail": "invalid status"})
        progress_raw = payload.get("progress")
        stage_raw = payload.get("stage")
        if progress_raw is not None:
            progress_parsed = parse_progress(progress_raw)
            if isinstance(progress_parsed, Err):
                return JSONResponse(status_code=400, content={"detail": "invalid progress"})
        if stage_raw is not None:
            stage_parsed = parse_stage(stage_raw)
            if isinstance(stage_parsed, Err):
                return JSONResponse(status_code=400, content={"detail": "invalid stage"})
        if progress_raw is None or stage_raw is None:
            current = owned_queue.progress(jid)
            if isinstance(current, Err):
                return JSONResponse(status_code=404, content={"detail": "not found"})
            return JSONResponse(status_code=200, content={"ok": True})
        progress_parsed = parse_progress(progress_raw)
        stage_parsed = parse_stage(stage_raw)
        assert isinstance(progress_parsed, Ok) and isinstance(stage_parsed, Ok)
        stage_value: Stage = stage_parsed.value
        updated = owned_queue.set_progress(jid, progress_parsed.value, stage_value)
        if isinstance(updated, Err):
            return JSONResponse(status_code=404, content={"detail": "not found"})
        return JSONResponse(status_code=200, content={"ok": True})

    @app.websocket("/v1/jobs/{job_id}/events")
    async def job_events(websocket: WebSocket, job_id: str) -> None:
        parsed = parse_job_id(job_id)
        if isinstance(parsed, Err):
            await websocket.close(code=4400)
            return
        jid = parsed.value
        status_result = owned_queue.status(jid)
        if isinstance(status_result, Err):
            await websocket.close(code=4404)
            return
        await websocket.accept()
        try:
            for _ in range(120):
                status_now = owned_queue.status(jid)
                if isinstance(status_now, Err):
                    break
                progress_now = owned_queue.progress(jid)
                if isinstance(progress_now, Ok):
                    progress_value, stage_value = progress_now.value
                    payload = {
                        "job_id": jid.as_str(),
                        "status": status_now.value.as_str(),
                        "progress": progress_value.value(),
                        "stage": stage_value.as_str(),
                    }
                else:
                    payload = {
                        "job_id": jid.as_str(),
                        "status": status_now.value.as_str(),
                        "progress": 0.0,
                        "stage": "queued",
                    }
                try:
                    await websocket.send_json(payload)
                except (RuntimeError, WebSocketDisconnect):
                    break
                if status_now.value in (JobStatus.DONE, JobStatus.FAILED, JobStatus.EXPIRED):
                    break
                await asyncio.sleep(0.5)
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app


app = create_app()


def _base_url_from_env() -> BaseUrl | None:
    raw = os.environ.get("ML_SIDECAR_URL", "").strip()
    if not raw:
        return None
    parsed = parse_base_url(raw)
    if isinstance(parsed, Err):
        return None
    return parsed.value
