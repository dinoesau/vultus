"""
Modal GPU workers para Vultus (híbrido Rust + Python).

Arquitectura híbrida (ADR-005):
- Rust (Axum) es dueño de Seam 1 API + Seam 2 queue + Worker 4 CPU
  (GNM bake, heatmap, report). Ver backend/crates/.
- Python aquí es solo sidecar ML GPU: MediaPipe / FLAME / FreeUV.
  Rust nunca importa torch/diffusers/mediapipe; los consume vía HTTP:
  `MlSidecarClient { landmarks, flame, freeuv }` -> `POST /ml/*`.

Cadena real (deploy-real-models):
- landmarks: MediaPipe Tasks `face_landmarker.task`, 478 puntos.
- flame: DECA encode/decode + `flame2023_Open.pkl` -> `uv_texture_gt`
  256 (oclusiones visibles) redimensionada a 512 en este sidecar.
- freeuv: SD v1-5 (`from_pretrained` + `subfolder="unet"`,
  `safety_checker=None`) + `flaw_tolerant_facial_detail_extractor.bin`
  + `uv_structure_aligner.bin` -> `complete-uv` 512 fotorrealista.
- Sin CUDA ni pesos, los dobles deterministas siguen respondiendo el
  mismo contrato para regresión rápida local (CPU).

Prod: Cloudflare Queues (HTTP Pull Consumer) -> Modal -> R2
Local sin Modal: `python -m modal_app --serve :8081` expone el mismo
contrato /ml/* y el binario Rust lo consume vía ML_SIDECAR_URL.

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

# Canónicas: espejo de backend/crates/core/src/job.rs (LANDMARKS_LEN, UV_LEN).
# No duplicar literales 478 / 786432 en el código: usar estas consts.
LANDMARKS_LEN = 478
UV_WIDTH = 512
UV_HEIGHT = 512
UV_CHANNELS = 3
UV_LEN = UV_WIDTH * UV_HEIGHT * UV_CHANNELS  # 786432
# FreeUV trabaja internamente a 256 (data-process config UV_SIZE);
# el redimensionado a 512 vive en este sidecar antes de responder.
FLAME_UV_SIZE = 256

# Env con defaults seguros, nada hardcodeado.
ML_PORT = int(os.environ.get("ML_PORT", "8081"))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
# VULTUS_REAL_ML: 1 fuerza real, 0 fuerza dobles, auto decide por pesos+deps.
REAL_MODE = os.environ.get("VULTUS_REAL_ML", "auto").lower()
DECA_CODE_DIR = os.environ.get("DECA_CODE_DIR", os.path.join(WEIGHTS_DIR, "deca-code"))
FREEUV_CODE_DIR = os.environ.get("FREEUV_CODE_DIR", os.path.join(WEIGHTS_DIR, "freeuv-code"))

try:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    logger.info("weights dir ready path=%s", WEIGHTS_DIR)
except OSError as e:
    logger.warning("weights dir not writable path=%s err=%s", WEIGHTS_DIR, e)

# Transform leve determinista para /ml/freeuv (eco +1 por byte, a nivel C).
_FREEUV_TABLE = bytes((i + 1) % 256 for i in range(256))

# Paso pesado sidecar: paridad local con max_containers=1.
# En prod la serialización la impone Modal por réplica (web endpoint sin
# @modal.concurrent = 1 input por container); aquí el semáforo
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


# --- Inferencia real: MediaPipe + DECA + FreeUV tras el mismo contrato ---
# Todo import pesado es lazy dentro de los singletons: sin CUDA ni pesos el
# módulo importa igual y sirve dobles. En prod el fallo es ruidoso (500 con
# causa) en vez de devolver un doble silencioso: Error Hiding prohibido.


def _weights_present() -> bool:
    need = [
        os.path.join(WEIGHTS_DIR, "mediapipe", "face_landmarker.task"),
        os.path.join(WEIGHTS_DIR, "flame", "flame2023_Open.pkl"),
        os.path.join(WEIGHTS_DIR, "freeuv-checkpoints", "flaw_tolerant_facial_detail_extractor.bin"),
        os.path.join(WEIGHTS_DIR, "freeuv-checkpoints", "uv_structure_aligner.bin"),
        os.path.join(WEIGHTS_DIR, "sdv1-5", "model_index.json"),
    ]
    return all(os.path.exists(p) for p in need)


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
            from mediapipe.tasks.python import base_options as mp_base
            from mediapipe.tasks.python.vision import face_landmarker as mp_lm
        except ImportError as e:
            raise RuntimeError(f"mediapipe package missing: {e}") from e
        opts = mp_lm.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=task),
            running_mode=mp_lm.RunningMode.IMAGE,
            num_faces=1,
        )
        _LM = mp_lm.FaceLandmarker.create_from_options(opts)
        return _LM


def _real_landmarks(image: bytes) -> bytes:
    """Landmarks reales 478 [[x,y,z]...] finitos. ValueError = 400, otro = 500."""
    import numpy as np
    from mediapipe.tasks.python.vision import face_landmarker as mp_lm

    img = _pil_from_image_bytes(image)
    arr = np.asarray(img)
    mp_image = mp_lm.Image(image_format=mp_lm.ImageFormat.SRGB, data=arr)
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


_DECA_LOCK = threading.Lock()
_DECA = None


def _numpy_compat_shim():
    """DECA es era numpy<1.24: restaura alias eliminados si faltan. Solo shim, sin lógica."""
    import numpy as np

    for old, new in (("float", "float64"), ("int", "int64"), ("bool", "bool_")):
        if not hasattr(np, old) and hasattr(np, new):
            setattr(np, old, getattr(np, new))


def _deca_model():
    """Singleton DECA en CUDA con FLAME 2023 Open. Lanza RuntimeError con causa."""
    global _DECA
    if _DECA is not None:
        return _DECA
    with _DECA_LOCK:
        if _DECA is not None:
            return _DECA
        _numpy_compat_shim()
        if DECA_CODE_DIR not in sys.path:
            sys.path.insert(0, DECA_CODE_DIR)
        if FREEUV_CODE_DIR not in sys.path:
            sys.path.insert(0, FREEUV_CODE_DIR)
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(f"torch missing: {e}") from e
        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable, real flame needs GPU")
        try:
            from decalib.deca import DECA
            from decalib.utils import config as deca_config
        except ImportError as e:
            raise RuntimeError(f"deca code missing in {DECA_CODE_DIR}: {e}") from e
        flame_pkl = os.path.join(WEIGHTS_DIR, "flame", "flame2023_Open.pkl")
        if not os.path.exists(flame_pkl):
            raise RuntimeError(f"flame model missing: {flame_pkl}")
        cfg = deca_config.cfg.clone()
        cfg.deca_dir = DECA_CODE_DIR
        cfg.model.flame_model_path = flame_pkl
        cfg.model.use_tex = True
        cfg.model.extract_tex = True
        try:
            _DECA = DECA(config=cfg, device="cuda")
            _DECA.eval()
        except Exception as e:
            raise RuntimeError(f"deca init failed: {e}") from e
        return _DECA


def _real_flaw_uv(payload: bytes) -> bytes:
    """flaw-uv real 512 RGB crudo con oclusiones. ValueError = 400, otro = 500."""
    import numpy as np
    import torch
    from PIL import Image

    lm_raw, img_raw = _split_flame_payload(payload)
    _check_landmarks_json(lm_raw)
    _check_image_bytes(img_raw)
    pil = _pil_from_image_bytes(img_raw).resize((224, 224), Image.BILINEAR)
    arr = (np.asarray(pil).astype(np.float32) / 255.0 - 0.5) / 0.5
    ten = torch.from_numpy(arr.transpose(2, 0, 1)[None]).float().cuda()
    deca = _deca_model()
    with torch.no_grad():
        codedict = deca.encode(ten)
        codedict["images"] = ten
        opdict, _vis = deca.decode(codedict)
        if "uv_texture_gt" not in opdict:
            raise RuntimeError(f"deca decode sin uv_texture_gt, claves={sorted(opdict.keys())}")
        uv = opdict["uv_texture_gt"][0].clamp(0, 1)
    out = Image.fromarray((uv.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    out = out.resize((UV_WIDTH, UV_HEIGHT), Image.BILINEAR).convert("RGB")
    raw = out.tobytes()
    assert len(raw) == UV_LEN, f"flaw-uv len {len(raw)} != {UV_LEN}"
    return raw


_PIPE_LOCK = threading.Lock()
_PIPE = None


def _freeuv_pipe():
    """Singleton SD v1-5 + ControlNet + detail_encoder en CUDA. Lanza RuntimeError."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    with _PIPE_LOCK:
        if _PIPE is not None:
            return _PIPE
        if FREEUV_CODE_DIR not in sys.path:
            sys.path.insert(0, FREEUV_CODE_DIR)
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(f"torch missing: {e}") from e
        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable, real freeuv needs GPU")
        sdv = os.path.join(WEIGHTS_DIR, "sdv1-5")
        enc = os.path.join(WEIGHTS_DIR, "image_encoder_l")
        det = os.path.join(WEIGHTS_DIR, "freeuv-checkpoints", "flaw_tolerant_facial_detail_extractor.bin")
        ali = os.path.join(WEIGHTS_DIR, "freeuv-checkpoints", "uv_structure_aligner.bin")
        for p in (sdv, enc, det, ali):
            if not os.path.exists(p):
                raise RuntimeError(f"freeuv weight missing: {p}")
        try:
            from diffusers import DDIMScheduler, UNet2DConditionModel as UNet, ControlNetModel
            from pipeline_sd15 import StableDiffusionControlNetPipeline
            from detail_encoder.encoder_freeuv import detail_encoder
        except ImportError as e:
            raise RuntimeError(f"freeuv code missing in {FREEUV_CODE_DIR}: {e}") from e
        try:
            unet = UNet.from_pretrained(sdv, subfolder="unet").to("cuda")
            aligner = ControlNetModel.from_unet(unet)
            encoder = detail_encoder(unet, enc + "/", "cuda", dtype=torch.float32)
            aligner.load_state_dict(torch.load(ali, map_location="cpu"), strict=False)
            encoder.load_state_dict(torch.load(det, map_location="cpu"), strict=False)
            aligner.to("cuda")
            encoder.to("cuda")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                sdv, safety_checker=None, unet=unet, controlnet=aligner, torch_dtype=torch.float32
            ).to("cuda")
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        except Exception as e:
            raise RuntimeError(f"freeuv init failed: {e}") from e
        _PIPE = (pipe, encoder)
        return _PIPE


def _real_complete_uv(flaw: bytes) -> bytes:
    """complete-uv real 512 RGB. ValueError = 400 (longitud), otro = 500."""
    import io as _io

    import torch
    from PIL import Image

    if len(flaw) != UV_LEN:
        raise ValueError(f"expected {UV_LEN} uv bytes, got {len(flaw)}")
    pipe, encoder = _freeuv_pipe()
    flaw_img = Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), flaw)
    uv_template = os.path.join(WEIGHTS_DIR, "freeuv-resources", "uv.jpg")
    if os.path.exists(uv_template):
        uv_img = Image.open(uv_template).convert("RGB").resize((UV_WIDTH, UV_HEIGHT), Image.BILINEAR)
    else:
        uv_img = flaw_img
    with torch.no_grad():
        out = encoder.generate(
            uv_structure_image=uv_img, flaw_uv_image=flaw_img, pipe=pipe, guidance_scale=1.4
        )
    if not isinstance(out, Image.Image):
        out = Image.open(_io.BytesIO(bytes(out)))
    out = out.resize((UV_WIDTH, UV_HEIGHT), Image.BILINEAR).convert("RGB")
    raw = out.tobytes()
    assert len(raw) == UV_LEN, f"complete-uv len {len(raw)} != {UV_LEN}"
    return raw


def _impl_landmarks(body: bytes) -> bytes:
    if not body:
        raise ValueError("empty body")
    if not (_is_jpeg(body) or _is_png(body)):
        raise ValueError("unsupported image format, expected JPEG or PNG")
    if _use_real():
        return _real_landmarks(body)
    return _deterministic_landmarks(body)


def _impl_flame(payload: bytes) -> bytes:
    if _use_real():
        return _real_flaw_uv(payload)
    lm_raw, img_raw = _split_flame_payload(payload)
    _check_landmarks_json(lm_raw)
    _check_image_bytes(img_raw)
    out = _deterministic_uv(payload)
    assert len(out) == UV_LEN
    return out


def _impl_freeuv(body: bytes) -> bytes:
    if _use_real():
        return _real_complete_uv(body)
    if len(body) != UV_LEN:
        raise ValueError(f"expected {UV_LEN} uv bytes, got {len(body)}")
    out = _inpaint_uv(body)
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
        max_containers=1,  # FreeUV OOM si >1 por GPU (pool de 1 container)
        timeout=60,
        min_containers=0,
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
        max_containers=1,
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
        max_containers=4,
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

    @sidecar.post("/ml/flame")
    async def http_flame(request: Request):
        payload = await request.body()
        job_id = request.headers.get("X-Job-Id", "unknown")
        return await _run_impl(job_id, "flame", _impl_flame, payload)

    @sidecar.post("/ml/freeuv")
    async def http_freeuv(request: Request):
        body = await request.body()
        job_id = request.headers.get("X-Job-Id", "unknown")
        if not _use_real() and len(body) != UV_LEN:
            return JSONResponse(
                status_code=400,
                content={"detail": f"expected {UV_LEN} uv bytes, got {len(body)}"},
            )
        if _use_real():
            return await _run_impl(job_id, "freeuv", _impl_freeuv, body)
        try:
            async with _FREEUV_SEMAPHORE:
                out = await asyncio.to_thread(_impl_freeuv, body)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        except Exception as e:
            logger.exception("freeuv failed job=%s", job_id)
            return JSONResponse(status_code=500, content={"detail": str(e)})
        assert len(out) == UV_LEN
        logger.info("freeuv ok job=%s in_len=%d out_len=%d", job_id, len(body), len(out))
        return Response(content=out, media_type="application/octet-stream")

    if HAVE_MODAL:

        @app.function(
            image=image,
            gpu="T4",
            cpu=4,
            memory=16384,
            volumes={"/weights": weights},
            secrets=[modal.Secret.from_name("vultus-cloudflare")],
            # Sin @modal.concurrent: 1 input por container = una inferencia
            # pesada por GPU, sin OOM. max_containers=10 (Starter).
            max_containers=10,
            timeout=600,  # red amplia: la primera llamada paga cold + carga de pesos
            env={"VULTUS_REAL_ML": "1"},
        )
        @modal.fastapi_endpoint(method="POST")
        async def ml_endpoint(request: Request):
            path = request.url.path
            body = await request.body()
            job_id = request.headers.get("X-Job-Id", "unknown")
            if path.endswith("/landmarks"):
                return await _run_impl(job_id, "landmarks", _impl_landmarks, body)
            if path.endswith("/flame"):
                return await _run_impl(job_id, "flame", _impl_flame, body)
            if path.endswith("/freeuv"):
                # En prod Modal ya serializa (1 input por container);
                # el semáforo local mantiene paridad si el endpoint corre fuera.
                # En prod delega a freeuv_worker.spawn(job_id, ...) y lee R2.
                async with _FREEUV_SEMAPHORE:
                    return await _run_impl(job_id, "freeuv", _impl_freeuv, body)
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
        logger.info("serving sidecar port=%d weights=%s real=%s", port, WEIGHTS_DIR, _use_real())
        uvicorn.run(sidecar, host="0.0.0.0", port=port)
