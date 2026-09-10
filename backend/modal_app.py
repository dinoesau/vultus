"""
Modal GPU workers para Vultus (Python + TypeScript).

Arquitectura (sin Rust):
- Python FastAPI es dueno de API local + cola en memoria + orquestador local
  + Worker CPU (GNM assemble, heatmap, report) via `backend/gnm_assemble.py`.
- Python aqui es sidecar ML GPU: MediaPipe / fit GNM / textura GNM.
  La API nunca importa torch/diffusers/mediapipe; los consume vía HTTP:
  `MlSidecarClient { landmarks, fit, texture }` -> `POST /ml/*`.

Cadena GNM (plan-gnm-fit-texture-pbr):
- landmarks: MediaPipe Tasks `face_landmarker.task`, 478 puntos.
- fit: fitter GNM directo -> 253 coefs + camara 3x4 (1060 bytes).
- texture: proyeccion foto + warp TPS + inpaint solo ocluidas -> albedo 512.
- Sin CUDA ni pesos, los dobles deterministas siguen respondiendo el
  mismo contrato para regresión rápida local (CPU).

Prod: Cloudflare Queues (HTTP Pull Consumer) -> Modal -> R2
Local sin Modal: `python -m modal_app --serve :8081` expone el mismo
contrato /ml/* y la API Python lo consume vía ML_SIDECAR_URL.

Starter plan: $30/mes free (~50h T4 = ~9.300 compares). Cold start 1-2s.
Deploy: modal deploy backend/modal_app.py
Logs:   modal app logs vultus-workers
"""

import asyncio
import hashlib
import io
import json
import logging
import math
import os
import sys
import threading
import time

logger = logging.getLogger("vultus-ml-sidecar")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

# Canónicas: LANDMARKS_LEN 478, UV 512x512x3, UV_LEN 786432 (ver edge/contract.ts).
# No duplicar literales 478 / 786432 en el código: usar estas consts.
LANDMARKS_LEN = 478
UV_WIDTH = 512
UV_HEIGHT = 512
UV_CHANNELS = 3
UV_LEN = UV_WIDTH * UV_HEIGHT * UV_CHANNELS  # 786432
# Env con defaults seguros, nada hardcodeado.
ML_PORT = int(os.environ.get("ML_PORT", "8081"))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
# Ruta de assets GNM: template + islas en Volume.
GNM_ASSETS_DIR = os.environ.get("GNM_ASSETS_DIR", os.path.join(WEIGHTS_DIR, "gnm"))
# VULTUS_REAL_ML: 1 fuerza real, 0 fuerza dobles, auto decide por pesos+deps.
REAL_MODE = os.environ.get("VULTUS_REAL_ML", "auto").lower()

# Nombres del bundle (contrato con edge/contract.ts ZIP_MANIFEST).
ZIP_UV_A = "uv_a.png"
ZIP_UV_B = "uv_b.png"
ZIP_HEATMAP = "heatmap.png"
ZIP_MESH_A = "mesh_a.glb"
ZIP_MESH_B = "mesh_b.glb"
GLB_MAGIC = b"glTF"

try:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    logger.info("weights dir ready path=%s", WEIGHTS_DIR)
except OSError as e:
    logger.warning("weights dir not writable path=%s err=%s", WEIGHTS_DIR, e)

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


# --- Inferencia real: MediaPipe tras el mismo contrato ---
# Todo import pesado es lazy dentro de los singletons: sin CUDA ni pesos el
# módulo importa igual y sirve dobles. En prod el fallo es ruidoso (500 con
# causa) en vez de devolver un doble silencioso: Error Hiding prohibido.


def _gnm_npz_present() -> bool:
    """True si el npz GNM existe en alguna ruta candidata de `gnm_head`.

    S7 prod-block: la misma lista de candidatos que `load_gnm_head`
    (GNM_NPZ_PATH directo, GNM_ASSETS_DIR, WEIGHTS_DIR, raiz del repo).
    Import lazy para no acoplar el import del modulo a numpy.
    """
    try:
        from backend.gnm_head import _candidate_npz_paths as _cands
    except ImportError:  # pragma: no cover - paridad ruta plana en imagen
        import gnm_head as _gh  # type: ignore[import-not-found]

        _cands = _gh._candidate_npz_paths
    try:
        return any(os.path.isfile(p) for p in _cands())
    except Exception:
        return False


def _weights_present() -> bool:
    # S7: real exige MediaPipe task Y npz GNM. Sin npz no hay fit real
    # (ver `gnm_fit._real_fit_available`) y auto debe seguir en dobles.
    need = [
        os.path.join(WEIGHTS_DIR, "mediapipe", "face_landmarker.task"),
    ]
    if not all(os.path.exists(p) for p in need):
        return False
    return _gnm_npz_present()


def _use_real() -> bool:
    if REAL_MODE == "1":
        return True
    if REAL_MODE == "0":
        return False
    return _weights_present()


def _pil_from_image_bytes(raw: bytes):
    from PIL import Image

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"cannot decode image bytes: {e}") from e


_LM_LOCK = threading.Lock()
_LM = None


def _landmarker():
    """Singleton MediaPipe FaceLandmarker (478 puntos). Lanza RuntimeError con causa."""
    global _LM
    if _LM is not None:
        return _LM
    with _LM_LOCK:
        if _LM is not None:
            return _LM
        task = os.path.join(WEIGHTS_DIR, "mediapipe", "face_landmarker.task")
        if not os.path.exists(task):
            raise RuntimeError(f"mediapipe task missing: {task}")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise RuntimeError(f"mediapipe package missing: {e}") from e
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _LM = mp_vision.FaceLandmarker.create_from_options(opts)
        return _LM


def _real_landmarks(image: bytes) -> bytes:
    """Landmarks reales 478 [[x,y,z]...] finitos. ValueError = 400, otro = 500."""
    import mediapipe as mp
    import numpy as np

    img = _pil_from_image_bytes(image)
    arr = np.asarray(img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = _landmarker().detect(mp_image)
    if not res.face_landmarks:
        raise ValueError("no face detected")
    face = res.face_landmarks[0]
    if len(face) != LANDMARKS_LEN:
        raise ValueError(f"expected {LANDMARKS_LEN} points, got {len(face)}")
    pts = [[float(p.x), float(p.y), float(p.z)] for p in face]
    for p in pts:
        for v in p:
            if not math.isfinite(v):
                raise ValueError("non-finite landmark")
    return json.dumps(pts).encode("utf-8")


def _impl_landmarks(body: bytes) -> bytes:
    if not body:
        raise ValueError("empty body")
    if not (_is_jpeg(body) or _is_png(body)):
        raise ValueError("unsupported image format, expected JPEG or PNG")
    if _use_real():
        return _real_landmarks(body)
    return _deterministic_landmarks(body)


def _impl_fit(payload: bytes) -> bytes:
    """Delega al modulo `gnm_fit`: fit-request -> 1060 bytes (253f + 12f LE).

    S7 prod-block: con `_use_real()` el doble esta prohibido. Si el fit
    real no esta disponible (npz ausente) se falla ruidoso (RuntimeError
    -> 500 con causa) antes de delegar, nunca doble silencioso.
    """
    try:
        from backend.domain import Err as _Err
        from backend.domain import FitFailed as _FitFailed
        from backend.domain import domain_to_message as _msg
        from backend.gnm_fit import fit_gnm_from_request
        from backend.pipeline_local import encode_fit_result
    except ImportError:  # pragma: no cover - paridad ruta plana en imagen
        from domain import Err as _Err  # type: ignore[no-redef]
        from domain import FitFailed as _FitFailed  # type: ignore[no-redef]
        from domain import domain_to_message as _msg  # type: ignore[no-redef]
        from gnm_fit import fit_gnm_from_request  # type: ignore[no-redef]
        from pipeline_local import encode_fit_result  # type: ignore[no-redef]
    if _use_real() and not _gnm_npz_present():
        # Sin npz no hay fit real (`_real_fit_available` seria False y el
        # seam caeria al doble): fallar ruidoso aqui, nunca doble
        # silencioso. Casos resto (npz corrupto, landmarks ausentes) ya
        # llegan como Err(FitFailed) -> RuntimeError abajo.
        raise RuntimeError("real fit required but GNM head weights missing (npz ausente)")

    result = fit_gnm_from_request(payload)
    if isinstance(result, _Err):
        if isinstance(result.error, _FitFailed):
            raise RuntimeError(_msg(result.error))
        raise ValueError(_msg(result.error))
    return encode_fit_result(result.value)


def _impl_texture(payload: bytes) -> bytes:
    """Delega a `gnm_texture.build_albedo`: texture-request -> UV_LEN bytes."""
    try:
        from backend.domain import Err as _Err2
        from backend.domain import FitFailed as _FitFailed2
        from backend.domain import MlFailed as _MlFailed2
        from backend.domain import domain_to_message as _msgT
        from backend.gnm_texture import build_albedo
        from backend.pipeline_local import decode_texture_request
    except ImportError:  # pragma: no cover - paridad ruta plana en imagen
        from domain import Err as _Err2  # type: ignore[no-redef]
        from domain import FitFailed as _FitFailed2  # type: ignore[no-redef]
        from domain import MlFailed as _MlFailed2  # type: ignore[no-redef]
        from domain import domain_to_message as _msgT  # type: ignore[no-redef]
        from gnm_texture import build_albedo  # type: ignore[no-redef]
        from pipeline_local import decode_texture_request  # type: ignore[no-redef]

    decoded = decode_texture_request(payload)
    if isinstance(decoded, _Err2):
        raise ValueError(_msgT(decoded.error))
    image, fit, landmarks = decoded.value
    result = build_albedo(image, fit, landmarks)
    if isinstance(result, _Err2):
        if isinstance(result.error, (_FitFailed2, _MlFailed2)):
            raise RuntimeError(_msgT(result.error))
        raise ValueError(_msgT(result.error))
    out = result.value.as_bytes()
    assert len(out) == UV_LEN
    return out


try:
    import modal

    HAVE_MODAL = True
except ImportError:  # local Docker sin modal: solo corre sidecar FastAPI
    modal = None  # type: ignore
    HAVE_MODAL = False

if HAVE_MODAL:
    app = modal.App("vultus-workers")

    # Imagen por receta (no from_dockerfile): cada paso se cachea por hash y
    # el codigo viaja en el paquete del deploy, asi los deploys de solo-codigo
    # no reconstruyen nada (<60s). Solo cambian la imagen los cambios a esta
    # receta o a requirements.txt. Paridad con Dockerfile.gpu (uso local):
    # misma base devel, mismos paquetes, mismo orden torch primero.
    # Base devel (no runtime): el rasterizador CUDA se compila desde source;
    # sin nvcc quedaria solo-CPU.
    # Deploys estrictamente secuenciales: dos builds concurrentes no comparten
    # cache y ambos pagan el build completo.
    image = (
        modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.10")
        .apt_install("build-essential", "python3-dev", "ninja-build", "curl", "libgl1", "libglib2.0-0", "git")
        .pip_install(
            "torch==2.13.0",
            "torchvision==0.28.0",
            index_url="https://download.pytorch.org/whl/cu126",
        )
        .pip_install_from_requirements("backend/requirements.txt")
        .run_commands(
            "pip install --no-cache-dir fvcore iopath",
            # Sin `|| echo`: pytorch3d es requerido en prod (rasterizador DECA).
            # --no-build-isolation: su setup.py importa torch y el env aislado
            # PEP 517 no lo trae (ahi moria con ModuleNotFoundError: torch).
            # CXX=g++: torch elige clang++ por defecto y no existe en la imagen.
            # FORCE_CUDA=1: el builder no tiene GPU y setup.py decidiria solo-CPU
            # aunque haya nvcc; T4 es sm_75, una sola arch para compilar rapido.
            "FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.5 CXX=g++ CC=gcc pip install --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git",
        )
        # Codigo compartido en la imagen: Modal solo monta `modal_app.py`;
        # sin esto el consumer muere con ModuleNotFoundError al importar
        # `backend.domain` / `backend.gnm_assemble` en heatmap/GLB/zip.
        # Solo .py (los .bin viven en el Volume). Al final para no
        # invalidar las capas pesadas de torch.
        .add_local_python_source("backend")
    )

    # Volume para cachear pesos MediaPipe / GNM (evita re-descarga en cold start)
    weights = modal.Volume.from_name("vultus-weights", create_if_missing=True)
else:
    app = None  # type: ignore
    image = None  # type: ignore
    weights = None  # type: ignore

# Secrets: Cloudflare R2 + Queues creds
# modal secret create vultus-cloudflare CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... CLOUDFLARE_QUEUE_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=vultus-jobs VULTUS_API_URL=https://api.vultus.esau.com.mx
# Timeouts espejo de PipelineConfig S10 (5+10+30 dentro de TTL 60).
# Base: fit doble p95 local <1s; T4 real pendiente Step 4, el total no se mueve.
# S7: el fit real itera 3 outer fijos (<10s, ver test_fit_p95_inside_fit_timeout);
# una optimizacion lenta no estira timeouts: fit_gnm devuelve Err(FitFailed) y
# _impl_fit lo propaga como 500 ruidoso antes del TTL 60 de fit_worker.
LANDMARKS_TIMEOUT_SECS = 5
FIT_TIMEOUT_SECS = 10
TEXTURE_TIMEOUT_SECS = 30
TOTAL_TIMEOUT_SECS = 60
# Visibilidad del mensaje en cola: cubre la cadena fria (~90-120s) para que
# un job frio no se reentregue y se procese duplicado. Espejo del consumer
# HTTP pull de la cola (visibility_timeout_ms 180000, batch 2, retries 2).
QUEUE_VISIBILITY_TIMEOUT_SECS = 180
# Progreso canonico espejo de pipeline run_pair (etapas GNM).
PROGRESS_FIT = 0.40
PROGRESS_TEXTURE = 0.75
PROGRESS_ASSEMBLE = 0.95
PROGRESS_DONE = 1.0


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _r2_client():
    import boto3

    account = _env("CLOUDFLARE_ACCOUNT_ID")
    key_id = _env("R2_ACCESS_KEY_ID")
    secret = _env("R2_SECRET_ACCESS_KEY")
    if not account or not key_id or not secret:
        raise RuntimeError("r2 creds missing: CLOUDFLARE_ACCOUNT_ID/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )


def _r2_bucket() -> str:
    return _env("R2_BUCKET", "vultus-jobs") or "vultus-jobs"


def _api_base() -> str:
    return _env("VULTUS_API_URL", "https://api.vultus.esau.com.mx").rstrip("/") or "https://api.vultus.esau.com.mx"


def _report_progress(job_id: str, progress: float, stage: str) -> None:
    """Best-effort: actualiza DO via gateway. Nunca tumba el job por fallo de progreso."""
    import urllib.request

    url = f"{_api_base()}/v1/jobs/{job_id}/progress"
    body = json.dumps({"progress": progress, "stage": stage}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (compatible; VultusModal/1.0)"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:
        logger.warning("progress update failed job=%s stage=%s err=%s", job_id, stage, e)


def _report_failed(job_id: str) -> None:
    """Marca DO como failed tras error/timeout. Best-effort con log, nunca lanza."""
    import urllib.request

    url = f"{_api_base()}/v1/jobs/{job_id}/progress"
    body = json.dumps({"status": "failed"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (compatible; VultusModal/1.0)"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:
        logger.warning("failed report failed job=%s err=%s", job_id, e)


class _ExpiredAbort(Exception):
    """El DO ya no quiere el resultado (expired/failed): abortar sin ruido."""


def _job_alive(job_id: str) -> bool:
    """True si el DO aun quiere el resultado (queued/processing). Tras expired
    o failed el trabajo GPU es inutil: abortar libera el slot para jobs vivos.
    Ante error de transporte se asume vivo (un blip no debe matar un job)."""
    import urllib.request

    url = f"{_api_base()}/v1/jobs/{job_id}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; VultusModal/1.0)"}, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status != 200:
                return True
            status = json.loads(r.read().decode()).get("status")
            return status in ("queued", "processing")
    except Exception as e:
        logger.warning("alive check failed job=%s err=%s (se asume vivo)", job_id, e)
        return True


def _fetch_r2_bytes(bucket: str, key: str) -> bytes:
    r2 = _r2_client()
    obj = r2.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    if not data:
        raise ValueError(f"empty r2 object {key}")
    return data


def _heatmap_abs_diff(uv_a: bytes, uv_b: bytes) -> bytes:
    if len(uv_a) != UV_LEN or len(uv_b) != UV_LEN:
        raise ValueError(f"heatmap needs {UV_LEN} bytes per uv")
    try:
        from backend.domain import Ok as _Ok
        from backend.domain import parse_complete_uv as _parse_uv
        from backend.gnm import compute_heatmap as _shared_heatmap
    except ImportError:
        from domain import Ok as _Ok  # type: ignore[no-redef]
        from domain import parse_complete_uv as _parse_uv  # type: ignore[no-redef]
        from gnm import compute_heatmap as _shared_heatmap  # type: ignore[no-redef]
    ra = _parse_uv(bytes(uv_a))
    rb = _parse_uv(bytes(uv_b))
    assert isinstance(ra, _Ok) and isinstance(rb, _Ok)
    return bytes(_shared_heatmap(ra.value, rb.value).as_bytes())





def mediapipe_infer(job_id: str, image: bytes) -> bytes:
    """Nucleo landmarks real. Falla ruidoso sin pesos/CUDA-paquetizados, nunca doble silencioso."""
    t0 = time.perf_counter()
    out = _real_landmarks(image)
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("mediapipe ok job=%s out_len=%d duration_ms=%d", job_id, len(out), dt)
    return out


def fit_infer(job_id: str, payload: bytes) -> bytes:
    """Nucleo fit GNM: fit-request -> 1060 bytes (253 coefs + 12 camara)."""
    t0 = time.perf_counter()
    out = _impl_fit(payload)
    dt = int((time.perf_counter() - t0) * 1000)
    try:
        from backend.gnm_fit import _LAST_FIT_STATS as _fit_stats
    except ImportError:  # pragma: no cover - paridad ruta plana en imagen
        from gnm_fit import _LAST_FIT_STATS as _fit_stats  # type: ignore[no-redef]
    iterations = int(_fit_stats.get("iterations", 0))
    loss = float(_fit_stats.get("loss", float("nan")))
    logger.info(
        "fit ok job=%s out_len=%d duration_ms=%d iterations=%d loss=%.6g",
        job_id,
        len(out),
        dt,
        iterations,
        loss,
    )
    return out


def texture_infer(job_id: str, payload: bytes) -> bytes:
    """Nucleo textura GNM: texture-request -> albedo UV_LEN."""
    t0 = time.perf_counter()
    out = _impl_texture(payload)
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("texture ok job=%s out_len=%d duration_ms=%d", job_id, len(out), dt)
    return out


def fit_worker(job_id: str, r2_key: str, landmarks_json: bytes):
    """Worker fit - GNM fitting directo (GPU, 1 input por GPU). Lee imagen de R2."""
    t0 = time.perf_counter()
    bucket = _r2_bucket()
    image = _fetch_r2_bytes(bucket, r2_key)
    n = len(landmarks_json)
    payload = n.to_bytes(4, "big") + landmarks_json + image
    out = fit_infer(job_id, payload)
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("fit_worker ok job=%s out_len=%d duration_ms=%d", job_id, len(out), dt)
    return out


if HAVE_MODAL:
    fit_worker = app.function(
        image=image,
        gpu="T4",
        cpu=4,
        memory=32768,
        volumes={"/weights": weights},
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        max_containers=2,  # A/B en paralelo en 2 GPUs; 1 input por GPU
        timeout=60,
    )(fit_worker)


def texture_worker(job_id: str, payload: bytes):
    """
    Worker textura - proyeccion + warp + inpaint solo ocluidas (GPU).
    Entrada: texture-request. Salida: albedo UV_LEN bytes.
    Pool de 2 contenedores para paralelizar cara A/B del mismo job.
    """
    return texture_infer(job_id, payload)


if HAVE_MODAL:
    texture_worker = app.function(
        image=image,
        gpu="T4",
        cpu=2,
        memory=16384,
        volumes={"/weights": weights},
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        max_containers=2,  # A/B en paralelo en 2 GPUs; 1 input por GPU (anti-OOM)
        timeout=60,
        min_containers=0,
    )(texture_worker)


def mediapipe_worker(job_id: str, r2_key: str):
    """Worker 1 - MediaPipe 478 landmarks CPU. Lee imagen de R2, retorna JSON 478 finitos."""
    t0 = time.perf_counter()
    bucket = _r2_bucket()
    image = _fetch_r2_bytes(bucket, r2_key)
    out = mediapipe_infer(job_id, image)
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("mediapipe_worker ok job=%s out_len=%d duration_ms=%d", job_id, len(out), dt)
    return out


if HAVE_MODAL:
    mediapipe_worker = app.function(
        image=image,
        cpu=2,
        memory=4096,
        volumes={"/weights": weights},
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        max_containers=4,
        timeout=30,
    )(mediapipe_worker)


def _cf_pull_messages(batch_size: int = 1):
    """Pull de Cloudflare Queues via REST. Retorna lista de dicts con id/lease_id/body."""
    import httpx

    account = _env("CLOUDFLARE_ACCOUNT_ID")
    token = _env("CLOUDFLARE_API_TOKEN") or _env("CLOUDFLARE_API_KEY")
    queue_id = _env("CLOUDFLARE_QUEUE_ID") or _env("QUEUE_ID") or "vultus-jobs"
    if not account or not token:
        logger.info("queues creds missing, skip pull")
        return []
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/queues/{queue_id}/messages/pull"
    try:
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"visibility_timeout_ms": QUEUE_VISIBILITY_TIMEOUT_SECS * 1000, "batch_size": batch_size},
            timeout=10.0,
        )
    except Exception as e:
        logger.warning("queues pull transport failed err=%s", e)
        return []
    if r.status_code != 200:
        # Diagnostico sin exponer el secreto: longitudes y queue_id no sensible.
        logger.warning(
            "queues pull status=%d body=%.200s (token_len=%d queue_id=%s)",
            r.status_code,
            r.text,
            len(token),
            _env("CLOUDFLARE_QUEUE_ID") or _env("QUEUE_ID") or "vultus-jobs",
        )
        return []
    try:
        data = r.json()
    except Exception as e:
        logger.warning("queues pull bad json err=%s", e)
        return []
    msgs = ((data.get("result") or {}).get("messages")) or data.get("messages") or []
    return msgs if isinstance(msgs, list) else []


def _cf_ack_messages(acks: list) -> None:
    import httpx

    if not acks:
        return
    account = _env("CLOUDFLARE_ACCOUNT_ID")
    token = _env("CLOUDFLARE_API_TOKEN") or _env("CLOUDFLARE_API_KEY")
    queue_id = _env("CLOUDFLARE_QUEUE_ID") or _env("QUEUE_ID") or "vultus-jobs"
    if not account or not token:
        return
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/queues/{queue_id}/messages/ack"
    try:
        httpx.post(url, headers={"Authorization": f"Bearer {token}"}, json={"acks": acks}, timeout=10.0)
    except Exception as e:
        logger.warning("queues ack failed err=%s", e)


def _parse_queue_body(msg: dict) -> tuple:
    """Extrae (job_id, r2_a, r2_b) del body. La cola solo lleva IDs+punteros, nunca bytes."""
    body = msg.get("body") or msg.get("message") or {}
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception as e:
            raise ValueError(f"queue body not json: {e}") from e
    if not isinstance(body, dict):
        raise ValueError("queue body not object")
    job_id = str(body.get("job_id") or body.get("jobId") or "")
    r2_keys = body.get("r2_keys") or body.get("r2Keys") or {}
    r2_a = str(r2_keys.get("image_a") or r2_keys.get("a") or "")
    r2_b = str(r2_keys.get("image_b") or r2_keys.get("b") or "")
    if not job_id or not r2_a or not r2_b:
        raise ValueError("queue body missing job_id/r2_keys")
    if ".." in r2_a or ".." in r2_b:
        raise ValueError("invalid r2 key")
    return job_id, r2_a, r2_b


def _run_job_from_r2(job_id: str, r2_a: str, r2_b: str) -> None:
    """Orquestador produccion: fetch R2, cadenas A/B en paralelo, join, zip a R2, progreso vivo."""
    import concurrent.futures

    t0 = time.perf_counter()
    bucket = _r2_bucket()
    logger.info("job start job=%s a=%s b=%s", job_id, r2_a, r2_b)
    if not _job_alive(job_id):
        logger.info("job ya terminal antes de empezar job=%s, se omite", job_id)
        return
    _report_progress(job_id, PROGRESS_FIT, "fit")
    r2 = _r2_client()
    img_a = r2.get_object(Bucket=bucket, Key=r2_a)["Body"].read()
    img_b = r2.get_object(Bucket=bucket, Key=r2_b)["Body"].read()
    if not img_a or not img_b:
        raise ValueError("empty image from r2")

    def _is_modal_function(fn) -> bool:
        return HAVE_MODAL and hasattr(fn, "remote")

    # Landmarks A/B en paralelo. En Modal via workers remotos (CPU pool x4);
    # en local/Docker via inferencia directa (mismo nucleo real).
    if _is_modal_function(mediapipe_worker):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(mediapipe_worker.remote, job_id, r2_a)
            fut_b = ex.submit(mediapipe_worker.remote, job_id, r2_b)
            lm_a = fut_a.result(timeout=LANDMARKS_TIMEOUT_SECS + 25)
            lm_b = fut_b.result(timeout=LANDMARKS_TIMEOUT_SECS + 25)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(mediapipe_infer, job_id, img_a)
            fut_b = ex.submit(mediapipe_infer, job_id, img_b)
            lm_a = fut_a.result(timeout=LANDMARKS_TIMEOUT_SECS + 25)
            lm_b = fut_b.result(timeout=LANDMARKS_TIMEOUT_SECS + 25)
    _check_landmarks_json(lm_a)
    _check_landmarks_json(lm_b)
    if not _job_alive(job_id):
        raise _ExpiredAbort(f"job expiro durante landmarks job={job_id}")
    _report_progress(job_id, PROGRESS_FIT, "fit")

    def _fit_payload(image: bytes, landmarks_json: bytes) -> bytes:
        n = len(landmarks_json)
        return n.to_bytes(4, "big") + landmarks_json + image

    if _is_modal_function(fit_worker):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(fit_worker.remote, job_id, r2_a, lm_a)
            fut_b = ex.submit(fit_worker.remote, job_id, r2_b, lm_b)
            fit_a = fut_a.result(timeout=FIT_TIMEOUT_SECS + 50)
            fit_b = fut_b.result(timeout=FIT_TIMEOUT_SECS + 50)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(fit_infer, job_id, _fit_payload(img_a, lm_a))
            fut_b = ex.submit(fit_infer, job_id, _fit_payload(img_b, lm_b))
            fit_a = fut_a.result(timeout=FIT_TIMEOUT_SECS + 50)
            fit_b = fut_b.result(timeout=FIT_TIMEOUT_SECS + 50)
    try:
        from backend.pipeline_local import FIT_RESULT_LEN
    except ImportError:
        FIT_RESULT_LEN = 1060
    if len(fit_a) != FIT_RESULT_LEN or len(fit_b) != FIT_RESULT_LEN:
        raise ValueError("fit result bad length")
    if not _job_alive(job_id):
        raise _ExpiredAbort(f"job expiro durante fit job={job_id}")
    _report_progress(job_id, PROGRESS_TEXTURE, "texture")

    pay_a = _fit_payload(img_a, lm_a)
    pay_b = _fit_payload(img_b, lm_b)
    tex_a = len(pay_a).to_bytes(4, "big") + pay_a + bytes(fit_a)
    tex_b = len(pay_b).to_bytes(4, "big") + pay_b + bytes(fit_b)
    if _is_modal_function(texture_worker):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(texture_worker.remote, job_id, tex_a)
            fut_b = ex.submit(texture_worker.remote, job_id, tex_b)
            uv_a = fut_a.result(timeout=TEXTURE_TIMEOUT_SECS + 30)
            uv_b = fut_b.result(timeout=TEXTURE_TIMEOUT_SECS + 30)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(texture_infer, job_id, tex_a)
            fut_b = ex.submit(texture_infer, job_id, tex_b)
            uv_a = fut_a.result(timeout=TEXTURE_TIMEOUT_SECS + 30)
            uv_b = fut_b.result(timeout=TEXTURE_TIMEOUT_SECS + 30)
    if len(uv_a) != UV_LEN or len(uv_b) != UV_LEN:
        raise ValueError("albedo bad length")
    _report_progress(job_id, PROGRESS_ASSEMBLE, "assemble")

    t_assemble = time.perf_counter()
    try:
        from backend.domain import Ok as _OkA
        from backend.domain import parse_complete_uv as _parse_uv_a
        from backend.gnm_assemble import build_full_zip as _full_zip
        from backend.gnm_assemble import build_personalized_glb as _glb
        from backend.gnm_assemble import pbr_from_albedo as _pbr
        from backend.pipeline_local import parse_fit_result as _parse_fit
    except ImportError:
        from domain import Ok as _OkA  # type: ignore[no-redef]
        from domain import parse_complete_uv as _parse_uv_a  # type: ignore[no-redef]
        from gnm_assemble import build_full_zip as _full_zip  # type: ignore[no-redef]
        from gnm_assemble import (
            build_personalized_glb as _glb,  # type: ignore[no-redef]
        )
        from gnm_assemble import pbr_from_albedo as _pbr  # type: ignore[no-redef]
        from pipeline_local import (
            parse_fit_result as _parse_fit,  # type: ignore[no-redef]
        )
    ra_fit = _parse_fit(bytes(fit_a))
    rb_fit = _parse_fit(bytes(fit_b))
    ra_uv = _parse_uv_a(bytes(uv_a))
    rb_uv = _parse_uv_a(bytes(uv_b))
    assert isinstance(ra_fit, _OkA) and isinstance(rb_fit, _OkA)
    assert isinstance(ra_uv, _OkA) and isinstance(rb_uv, _OkA)
    heat = _heatmap_abs_diff(bytes(uv_a), bytes(uv_b))
    r_mesh_a = _glb(ra_fit.value, ra_uv.value)
    r_mesh_b = _glb(rb_fit.value, rb_uv.value)
    r_pbr_a = _pbr(ra_uv.value)
    r_pbr_b = _pbr(rb_uv.value)
    assert isinstance(r_mesh_a, _OkA) and isinstance(r_mesh_b, _OkA)
    assert isinstance(r_pbr_a, _OkA) and isinstance(r_pbr_b, _OkA)
    mesh_a = bytes(r_mesh_a.value.as_bytes())
    mesh_b = bytes(r_mesh_b.value.as_bytes())
    assemble_ms = int((time.perf_counter() - t_assemble) * 1000)
    logger.info(
        "assemble gnm done job=%s assemble_ms=%d mesh_a_len=%d mesh_b_len=%d",
        job_id,
        assemble_ms,
        len(mesh_a),
        len(mesh_b),
    )
    # PNG + zip en memoria, sin disco. Nombres del manifiesto versionado (7 archivos).
    import io as _io

    from PIL import Image

    def _to_png(raw: bytes) -> bytes:
        img = Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(raw))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    zip_bytes = _full_zip(
        _to_png(uv_a),
        _to_png(uv_b),
        _to_png(heat),
        mesh_a,
        mesh_b,
        _to_png(r_pbr_a.value),
        _to_png(r_pbr_b.value),
    )
    r2.put_object(Bucket=bucket, Key=f"jobs/{job_id}/result.zip", Body=zip_bytes, ContentType="application/zip")
    _report_progress(job_id, PROGRESS_DONE, "done")
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("job done job=%s zip_len=%d duration_ms=%d", job_id, len(zip_bytes), dt)


def queue_pull_consumer():
    """
    HTTP Pull Consumer para Cloudflare Queues (orquestador produccion).
    Polls Queues REST API cada 5s, despacha cadenas por cara en paralelo,
    join ambas ramas, escribe result.zip a R2 y actualiza progreso vivo.
    Ver https://developers.cloudflare.com/queues/configuration/pull-consumers/
    """
    t0 = time.perf_counter()
    try:
        msgs = _cf_pull_messages(batch_size=1)
    except Exception as e:
        logger.warning("pull failed err=%s", e)
        return
    if not msgs:
        return
    for msg in msgs:
        msg_id = str(msg.get("id") or msg.get("message_id") or "")
        lease = msg.get("lease_id") or msg.get("leaseId")
        try:
            job_id, r2_a, r2_b = _parse_queue_body(msg)
        except Exception as e:
            logger.warning("bad queue message skipped err=%s", e)
            if msg_id:
                _cf_ack_messages([{"id": msg_id, **({"lease_id": lease} if lease else {})}])
            continue
        try:
            # Deadline total = TTL: la primera llamada paga cold+carga, el resto warm.
            deadline = TOTAL_TIMEOUT_SECS
            import concurrent.futures as _cf

            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run_job_from_r2, job_id, r2_a, r2_b)
                fut.result(timeout=deadline)
        except _ExpiredAbort as e:
            logger.info("job abortado por expiracion job=%s (%s)", job_id if "job_id" in locals() else "unknown", e)
        except Exception:
            failed_id = job_id if "job_id" in locals() else "unknown"
            logger.exception("job failed job=%s", failed_id)
            # _report_failed es best-effort con log interno, nunca lanza.
            if failed_id != "unknown":
                _report_failed(failed_id)
        finally:
            if msg_id:
                ack = {"id": msg_id}
                if lease:
                    ack["lease_id"] = lease
                _cf_ack_messages([ack])
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("pull tick done msgs=%d duration_ms=%d", len(msgs), dt)


if HAVE_MODAL:
    queue_pull_consumer = app.function(
        image=image,
        cpu=1,
        memory=1024,
        volumes={"/weights": weights},
        secrets=[
            modal.Secret.from_name("vultus-cloudflare"),
            modal.Secret.from_name("vultus-queues-token"),
        ],
        # Cableado explicito: `gnm` resuelve assets en `GNM_ASSETS_DIR`.
        # Sin esto cae al `assets/` del repo, que no existe en la imagen
        # (solo viajan .py) y el bake muere con `gnm asset missing` en prod.
        env={"GNM_ASSETS_DIR": "/weights/gnm"},
        schedule=modal.Period(seconds=5),
    )(queue_pull_consumer)


# --- Sidecar HTTP consumido por el pipeline (MlSidecarClient) ---
# Mismo contrato en Modal (@app.function con web_endpoint) y en local
# (`python modal_app.py --serve`). El pipeline envía bytes, recibe bytes.
# Nunca se expone fuera del VPC/prod interno; sin auth externa.
# Endpoints delgados: delegan a _impl_* (dobles o inferencia real según
# pesos+env). El ML pesado corre en hilo para no bloquear el loop.

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse

    sidecar = FastAPI(title="vultus-ml-sidecar")

    async def _run_impl(job_id: str, label: str, fn, *args):
        t0 = time.perf_counter()
        try:
            out = await asyncio.to_thread(fn, *args)
        except ValueError as e:
            logger.info("%s 400 job=%s detail=%s", label, job_id, e)
            return JSONResponse(status_code=400, content={"detail": str(e)})
        except Exception as e:
            logger.exception("%s failed job=%s", label, job_id)
            return JSONResponse(status_code=500, content={"detail": str(e)})
        dt = int((time.perf_counter() - t0) * 1000)
        logger.info("%s ok job=%s out_len=%d duration_ms=%d", label, job_id, len(out), dt)
        return Response(content=out, media_type="application/octet-stream")

    @sidecar.post("/ml/landmarks")
    async def http_landmarks(request: Request):
        body = await request.body()
        job_id = request.headers.get("X-Job-Id", "unknown")
        return await _run_impl(job_id, "landmarks", _impl_landmarks, body)

    @sidecar.post("/ml/fit")
    async def http_fit(request: Request):
        payload = await request.body()
        job_id = request.headers.get("X-Job-Id", "unknown")
        return await _run_impl(job_id, "fit", _impl_fit, payload)

    @sidecar.post("/ml/texture")
    async def http_texture(request: Request):
        payload = await request.body()
        job_id = request.headers.get("X-Job-Id", "unknown")
        return await _run_impl(job_id, "texture", _impl_texture, payload)

except ImportError:  # Entorno sin fastapi: solo dobles vía _impl_* (tests unitarios)
    sidecar = None  # type: ignore


if HAVE_MODAL:

    @app.cls(
        image=image,
        gpu="T4",
        cpu=4,
        memory=16384,
        volumes={"/weights": weights},
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        # Sin @modal.concurrent: 1 input por container = una inferencia
        # pesada por GPU, sin OOM. max_containers=10 (Starter).
        # buffer_containers=1 absorbe la rafaga A/B del pipeline en paralelo.
        max_containers=10,
        buffer_containers=1,
        timeout=600,  # red amplia: la primera llamada paga cold + carga de pesos
        env={"VULTUS_REAL_ML": "1"},
    )
    class MlSidecar:
        @modal.enter()
        def warm(self):
            # Precalienta MediaPipe al arrancar el container para que
            # el primer request ya este en warm (el timeout de 5s en
            # landmarks no perdona la carga lazy de TFLite).
            # El fitter GNM real se cablea en Step 4 (dobles hasta entonces).
            _landmarker()
            logger.info("sidecar warm: mediapipe cargado")


        @modal.asgi_app()
        def app(self):
            # Misma tabla de rutas que el serve local: un solo contrato.
            # Fuera del try de fastapi: el registro no depende del env local
            # del CLI (la imagen remota sí trae fastapi vía requirements).
            return sidecar


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
        logger.info("serving sidecar port=%d weights=%s real=%s", port, WEIGHTS_DIR, _use_real())
        uvicorn.run(sidecar, host="0.0.0.0", port=port)
