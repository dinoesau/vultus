"""
Modal GPU workers para Vultus (híbrido Rust + Python).

Arquitectura híbrida (ADR-005):
- Rust (Axum) es dueño de Seam 1 API + Seam 2 queue + Worker 4 CPU
  (GNM bake, heatmap, report). Ver backend/crates/.
- Python aquí es solo sidecar ML GPU: MediaPipe / FLAME / FreeUV.
  Rust nunca importa torch/diffusers/mediapipe; los consume vía HTTP:
  `MlSidecarClient { landmarks, flame, freeuv }` -> `POST /ml/*`.

Prod: Cloudflare Queues (HTTP Pull Consumer) -> Modal -> R2
Local sin Modal: `python -m modal_app --serve :8081` expone el mismo
contrato /ml/* y el binario Rust lo consume vía ML_SIDECAR_URL.

Starter plan: $30/mes free (~50h T4 = ~9.300 compares). Cold start 1-2s.
Deploy: modal deploy backend/modal_app.py
Logs:   modal app logs vultus-workers
"""

import asyncio
import hashlib
import json
import logging
import math
import os

logger = logging.getLogger("vultus-ml-sidecar")

# Canónicas: espejo de backend/crates/core/src/job.rs (LANDMARKS_LEN, UV_LEN).
# No duplicar literales 478 / 786432 en el código: usar estas consts.
LANDMARKS_LEN = 478
UV_WIDTH = 512
UV_HEIGHT = 512
UV_CHANNELS = 3
UV_LEN = UV_WIDTH * UV_HEIGHT * UV_CHANNELS  # 786432

# Env con defaults seguros, nada hardcodeado.
ML_PORT = int(os.environ.get("ML_PORT", "8081"))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")

try:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    logger.info("weights dir ready path=%s", WEIGHTS_DIR)
except OSError as e:
    logger.warning("weights dir not writable path=%s err=%s", WEIGHTS_DIR, e)

# Transform leve determinista para /ml/freeuv (eco +1 por byte, a nivel C).
_FREEUV_TABLE = bytes((i + 1) % 256 for i in range(256))

# Paso pesado sidecar: paridad local con Modal concurrency_limit=1.
# En prod la serialización la impone Modal por réplica; aquí el semáforo
# local evita OOM concurrente en Docker CPU. No usar semáforo en Rust.
_FREEUV_SEMAPHORE = asyncio.Semaphore(1)


def _is_jpeg(b: bytes) -> bool:
    return len(b) >= 3 and b[0] == 0xFF and b[1] == 0xD8 and b[2] == 0xFF


def _is_png(b: bytes) -> bool:
    return len(b) >= 8 and b[0:8] == b"\x89PNG\r\n\x1a\n"


def _deterministic_landmarks(image: bytes) -> bytes:
    """Doble determinista: grilla derivada de sha256(image), 478 puntos finitos."""
    seed = hashlib.sha256(image).digest()
    pts = []
    for i in range(LANDMARKS_LEN):
        d = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        x = int.from_bytes(d[0:4], "big") / 4294967295.0
        y = int.from_bytes(d[4:8], "big") / 4294967295.0
        z = int.from_bytes(d[8:12], "big") / 4294967295.0
        pts.append([x, y, z])
    return json.dumps(pts).encode("utf-8")


def _check_landmarks_json(raw: bytes) -> None:
    """Valida JSON [[x,y,z],...] con LANDMARKS_LEN puntos finitos. Lanza ValueError(detail)."""
    try:
        pts = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"invalid landmarks json: {e}") from e
    if not isinstance(pts, list) or len(pts) != LANDMARKS_LEN:
        n = len(pts) if isinstance(pts, list) else -1
        raise ValueError(f"expected {LANDMARKS_LEN} points, got {n}")
    for p in pts:
        if not isinstance(p, list) or len(p) != 3:
            raise ValueError("invalid landmark point, expected [x,y,z]")
        for v in p:
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError("non-finite landmark")


def _split_flame_payload(payload: bytes) -> tuple:
    """Paridad con Rust FlamePayload::decode: u32 BE len + landmarks_json + image_bytes."""
    if len(payload) < 4:
        raise ValueError("flame payload <4 bytes")
    n = int.from_bytes(payload[0:4], "big")
    if len(payload) < 4 + n:
        raise ValueError("flame payload truncated")
    return payload[4 : 4 + n], payload[4 + n :]


def _check_image_bytes(img: bytes) -> None:
    if not img:
        raise ValueError("empty image in flame payload")
    if not (_is_jpeg(img) or _is_png(img)):
        raise ValueError("unsupported image format, expected JPEG or PNG")


def _deterministic_uv(seed: bytes) -> bytes:
    """Doble determinista: expansión de sha256(seed) hasta UV_LEN bytes raw."""
    digest = hashlib.sha256(seed).digest()
    reps = UV_LEN // len(digest) + 1
    return (digest * reps)[:UV_LEN]


def _inpaint_uv(flaw: bytes) -> bytes:
    """Eco con transform leve: simula inpaint manteniendo UV_LEN."""
    return flaw.translate(_FREEUV_TABLE)


try:
    import modal

    HAVE_MODAL = True
except ImportError:  # local Docker sin modal: solo corre sidecar FastAPI
    modal = None  # type: ignore
    HAVE_MODAL = False

if HAVE_MODAL:
    app = modal.App("vultus-workers")

    # Imagen base GPU con torch + diffusers + mediapipe
    # Reusa Dockerfile.gpu local para paridad
    image = modal.Image.from_dockerfile("backend/Dockerfile.gpu").pip_install(
        "boto3", "httpx"
    )  # R2 + Queues HTTP Pull

    # Volume para cachear pesos FreeUV / FLAME / GNM (evita re-descarga en cold start)
    weights = modal.Volume.from_name("vultus-weights", create_if_missing=True)
else:
    app = None  # type: ignore
    image = None  # type: ignore
    weights = None  # type: ignore

# Secrets: Cloudflare R2 + Queues creds
# modal secret create vultus-cloudflare CLOUDFLARE_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...


def freeuv_worker(job_id: str, r2_keys: dict):
    """
    Worker 3 - FreeUV SD1.5 inpainting.
    En prod consume desde Cloudflare Queues via HTTP Pull Consumer (ver pull_consumer.py).
    Lee flaw-uv + image desde R2, escribe complete-uv a R2.
    En local este mismo código corre en Docker sin Modal.
    """
    # TODO: import models.freeuv, descargar pesos a /weights si no existen
    # r2 = boto3.client("s3", endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com")
    # flaw_uv = r2.get_object(Bucket="vultus-jobs", Key=r2_keys["flaw_uv"])["Body"].read()
    # complete_uv = models.freeuv.inpaint(flaw_uv)
    # r2.put_object(Bucket="vultus-jobs", Key=f"{job_id}/uv.png", Body=complete_uv)
    pass


if HAVE_MODAL:
    freeuv_worker = app.function(
        image=image,
        gpu="T4",
        cpu=2,
        memory=16384,
        volumes={"/weights": weights},
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        concurrency_limit=1,  # FreeUV OOM si >1 por GPU
        timeout=60,
        min_containers=0,
        max_containers=10,  # Starter free: 10 GPU concurrency
    )(freeuv_worker)


def flame_worker(job_id: str, r2_keys: dict):
    """Worker 2 - FLAME Fitting 3DDFA_V3/DECA."""
    pass


if HAVE_MODAL:
    flame_worker = app.function(
        image=image,
        gpu="T4",
        cpu=4,
        memory=32768,
        volumes={"/weights": weights},
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        concurrency_limit=1,
        timeout=60,
    )(flame_worker)


def mediapipe_worker(job_id: str, r2_keys: dict):
    """Worker 1 - MediaPipe 478 landmarks CPU (puede correr también en Cloudflare Workers si se portara a WASM)."""
    pass


if HAVE_MODAL:
    mediapipe_worker = app.function(
        image=image,
        cpu=2,
        memory=4096,
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        concurrency_limit=4,
        timeout=30,
    )(mediapipe_worker)


def gnm_bake_worker(job_id: str, r2_keys: dict):
    """Worker 4 - DEPRECATED en híbrido: vive en Rust `vultus-workers-cpu`.

    Se mantiene el stub para no romper deploys antiguos.
    No añadir lógica aquí: `compute_heatmap` + `bake_bfm_to_gnm` en Rust.
    """
    raise NotImplementedError("moved to Rust vultus-workers-cpu")


if HAVE_MODAL:
    gnm_bake_worker = app.function(
        image=image,
        cpu=2,
        memory=4096,
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        timeout=30,
    )(gnm_bake_worker)


def queue_pull_consumer():
    """
    HTTP Pull Consumer para Cloudflare Queues.
    Polls Queues REST API, despacha a workers Modal via .spawn().
    Alternativa a Queues consumer binding (que solo funciona dentro de Workers).
    Ver https://developers.cloudflare.com/queues/configuration/pull-consumers/
    """
    # TODO: implementar con httpx + Cloudflare Queues REST API
    # messages = httpx.get(f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/queues/{QUEUE_ID}/messages/pull")
    # for msg in messages: freeuv_worker.spawn(msg["job_id"], msg["r2_keys"])
    pass


if HAVE_MODAL:
    queue_pull_consumer = app.function(
        image=image,
        cpu=1,
        memory=1024,
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        schedule=modal.Period(seconds=5),
    )(queue_pull_consumer)


# --- Sidecar HTTP consumido por Rust (MlSidecarClient) ---
# Mismo contrato en Modal (@app.function con web_endpoint) y en local
# (`python modal_app.py --serve`). Rust envía bytes, recibe bytes.
# Nunca se expone fuera del VPC/prod interno; sin auth externa.

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse

    sidecar = FastAPI(title="vultus-ml-sidecar")

    @sidecar.post("/ml/landmarks")
    async def http_landmarks(request: Request):
        body = await request.body()
        if not body:
            return JSONResponse(status_code=400, content={"detail": "empty body"})
        if not (_is_jpeg(body) or _is_png(body)):
            return JSONResponse(
                status_code=400,
                content={"detail": "unsupported image format, expected JPEG or PNG"},
            )
        try:
            out = _deterministic_landmarks(body)
        except Exception as e:
            logger.exception("landmarks failed len=%d", len(body))
            return JSONResponse(status_code=500, content={"detail": str(e)})
        logger.info("landmarks ok image_len=%d points=%d", len(body), LANDMARKS_LEN)
        return Response(content=out, media_type="application/octet-stream")

    @sidecar.post("/ml/flame")
    async def http_flame(request: Request):
        payload = await request.body()
        try:
            lm_raw, img_raw = _split_flame_payload(payload)
            _check_landmarks_json(lm_raw)
            _check_image_bytes(img_raw)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        try:
            out = _deterministic_uv(payload)
        except Exception as e:
            logger.exception("flame failed payload_len=%d", len(payload))
            return JSONResponse(status_code=500, content={"detail": str(e)})
        assert len(out) == UV_LEN
        logger.info("flame ok payload_len=%d uv_len=%d", len(payload), len(out))
        return Response(content=out, media_type="application/octet-stream")

    @sidecar.post("/ml/freeuv")
    async def http_freeuv(request: Request):
        body = await request.body()
        if len(body) != UV_LEN:
            return JSONResponse(
                status_code=400,
                content={"detail": f"expected {UV_LEN} uv bytes, got {len(body)}"},
            )
        try:
            async with _FREEUV_SEMAPHORE:
                out = _inpaint_uv(body)
        except Exception as e:
            logger.exception("freeuv failed in_len=%d", len(body))
            return JSONResponse(status_code=500, content={"detail": str(e)})
        assert len(out) == UV_LEN
        logger.info("freeuv ok in_len=%d out_len=%d", len(body), len(out))
        return Response(content=out, media_type="application/octet-stream")

    if HAVE_MODAL:

        @app.function(image=image, cpu=1, memory=1024)
        @modal.fastapi_endpoint(method="POST")
        async def ml_endpoint(request: Request):
            path = request.url.path
            body = await request.body()
            job_id = request.headers.get("X-Job-Id", "unknown")
            if path.endswith("/landmarks"):
                if not body:
                    return JSONResponse(status_code=400, content={"detail": "empty body"})
                if not (_is_jpeg(body) or _is_png(body)):
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "unsupported image format, expected JPEG or PNG"},
                    )
                try:
                    out = _deterministic_landmarks(body)
                except Exception as e:
                    logger.exception("landmarks failed job=%s len=%d", job_id, len(body))
                    return JSONResponse(status_code=500, content={"detail": str(e)})
                logger.info("landmarks ok job=%s image_len=%d", job_id, len(body))
                return Response(content=out, media_type="application/octet-stream")
            if path.endswith("/flame"):
                try:
                    lm_raw, img_raw = _split_flame_payload(body)
                    _check_landmarks_json(lm_raw)
                    _check_image_bytes(img_raw)
                except ValueError as e:
                    return JSONResponse(status_code=400, content={"detail": str(e)})
                try:
                    out = _deterministic_uv(body)
                except Exception as e:
                    logger.exception("flame failed job=%s len=%d", job_id, len(body))
                    return JSONResponse(status_code=500, content={"detail": str(e)})
                logger.info("flame ok job=%s uv_len=%d", job_id, len(out))
                return Response(content=out, media_type="application/octet-stream")
            if path.endswith("/freeuv"):
                if len(body) != UV_LEN:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": f"expected {UV_LEN} uv bytes, got {len(body)}"},
                    )
                try:
                    # En prod Modal ya serializa con concurrency_limit=1;
                    # el semáforo local mantiene paridad si el endpoint corre fuera.
                    # En prod delega a freeuv_worker.spawn(job_id, ...) y lee R2.
                    async with _FREEUV_SEMAPHORE:
                        out = _inpaint_uv(body)
                except Exception as e:
                    logger.exception("freeuv failed job=%s len=%d", job_id, len(body))
                    return JSONResponse(status_code=500, content={"detail": str(e)})
                logger.info("freeuv ok job=%s uv_len=%d", job_id, len(out))
                return Response(content=out, media_type="application/octet-stream")
            return JSONResponse(status_code=404, content={"detail": "unknown ml route"})
except ImportError:  # Modal image sin fastapi en tests unitarios
    sidecar = None  # type: ignore


def _serve_arg_port(argv) -> int:
    """Parsea `--serve [:PORT|PORT]` con default ML_PORT. Sin fugas: solo el puerto."""
    for i, a in enumerate(argv):
        if a == "--serve" and i + 1 < len(argv):
            raw = argv[i + 1].lstrip(":")
            try:
                return int(raw)
            except ValueError:
                pass
    return ML_PORT


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        import uvicorn

        port = _serve_arg_port(sys.argv)
        logger.info("serving sidecar port=%d weights=%s", port, WEIGHTS_DIR)
        uvicorn.run(sidecar, host="0.0.0.0", port=port)
