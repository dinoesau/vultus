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
    import numpy as np

    import mediapipe as mp

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
        # Sin modelo de textura (FLAME_albedo_from_BFM ausente del bundle):
        # flaw-uv = remuestreo de la foto al UV 256 via geometria DECA+FLAME,
        # con oclusiones visibles. El fitting sigue siendo DECA + Open.
        cfg.model.use_tex = False
        cfg.model.extract_tex = False
        cfg.pretrained_modelpath = os.path.join(WEIGHTS_DIR, "deca", "deca_model.tar")
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
            # torch>=2.6 usa weights_only=True por defecto: estos .bin son
            # checkpoints propios (no solo tensores), forzar False como en 2.4.
            aligner.load_state_dict(torch.load(ali, map_location="cpu", weights_only=False), strict=False)
            encoder.load_state_dict(torch.load(det, map_location="cpu", weights_only=False), strict=False)
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
        # 12 pasos: ~16s en T4 warm frente a ~26s con 20. El SLO warm (<20s
        # p95 par) y el timeout Rust de 30s por cara lo exigen; la revision
        # a ojo del golden congela la calidad a este valor.
        out = encoder.generate(
            uv_structure_image=uv_img,
            flaw_uv_image=flaw_img,
            pipe=pipe,
            guidance_scale=1.4,
            num_inference_steps=12,
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

    # Imagen por receta (no from_dockerfile): cada paso se cachea por hash y
    # el codigo viaja en el paquete del deploy, asi los deploys de solo-codigo
    # no reconstruyen nada (<60s). Solo cambian la imagen los cambios a esta
    # receta o a requirements.txt. Paridad con Dockerfile.gpu (uso local):
    # misma base devel, mismos paquetes, mismo orden torch primero.
    # Base devel (no runtime): pytorch3d se compila desde source y sin nvcc
    # queda solo-CPU -> `_C.rasterize_meshes` falla sin GPU en /ml/flame.
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
    )

    # Volume para cachear pesos FreeUV / FLAME / GNM (evita re-descarga en cold start)
    weights = modal.Volume.from_name("vultus-weights", create_if_missing=True)
else:
    app = None  # type: ignore
    image = None  # type: ignore
    weights = None  # type: ignore

# Secrets: Cloudflare R2 + Queues creds
# modal secret create vultus-cloudflare CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... CLOUDFLARE_QUEUE_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=vultus-jobs VULTUS_API_URL=https://api.vultus.esau.com.mx
# Timeouts espejo de PipelineConfig Rust (5+10+30+60=TTL). Sin magic numbers sueltos.
LANDMARKS_TIMEOUT_SECS = 5
FLAME_TIMEOUT_SECS = 10
FREEUV_TIMEOUT_SECS = 30
TOTAL_TIMEOUT_SECS = 60
# Progreso canonico espejo de pipeline.rs run_pair_inner.
PROGRESS_LANDMARKS = 0.15
PROGRESS_FLAME = 0.40
PROGRESS_FREEUV = 0.75
PROGRESS_BAKE = 0.95
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
    return bytes(x - y if x >= y else y - x for x, y in zip(uv_a, uv_b))


def _png_from_uv_raw(raw: bytes):
    from PIL import Image

    if len(raw) != UV_LEN:
        raise ValueError(f"expected {UV_LEN} uv bytes, got {len(raw)}")
    return Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), raw)


def _build_result_zip(uv_a_png: bytes, uv_b_png: bytes, heat_png: bytes) -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("uv_a.png", uv_a_png)
        z.writestr("uv_b.png", uv_b_png)
        z.writestr("heatmap.png", heat_png)
    return buf.getvalue()


def mediapipe_infer(job_id: str, image: bytes) -> bytes:
    """Nucleo landmarks real. Falla ruidoso sin pesos/CUDA-paquetizados, nunca doble silencioso."""
    t0 = time.perf_counter()
    out = _real_landmarks(image)
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("mediapipe ok job=%s out_len=%d duration_ms=%d", job_id, len(out), dt)
    return out


def flame_infer(job_id: str, image: bytes, landmarks_json: bytes) -> bytes:
    """Nucleo fitting real: payload u32 BE + landmarks + imagen -> flaw-uv 512."""
    t0 = time.perf_counter()
    n = len(landmarks_json)
    payload = n.to_bytes(4, "big") + landmarks_json + image
    out = _real_flaw_uv(payload)
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("flame ok job=%s out_len=%d duration_ms=%d", job_id, len(out), dt)
    return out


def freeuv_infer(job_id: str, flaw_uv: bytes) -> bytes:
    """Nucleo inpainting real: flaw-uv -> complete-uv 512."""
    t0 = time.perf_counter()
    out = _real_complete_uv(flaw_uv)
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("freeuv ok job=%s out_len=%d duration_ms=%d", job_id, len(out), dt)
    return out


def freeuv_worker(job_id: str, flaw_uv: bytes):
    """
    Worker 3 - FreeUV SD1.5 inpainting (GPU, estrictamente 1 input por GPU).
    Entrada: flaw-uv 786432 bytes. Salida: complete-uv 786432 bytes.
    Pool de 2 contenedores para paralelizar cara A/B del mismo job;
    cada container procesa 1 input (sin @modal.concurrent = sin OOM).
    Llama inferencia real, nunca doble silencioso.
    """
    return freeuv_infer(job_id, flaw_uv)


if HAVE_MODAL:
    freeuv_worker = app.function(
        image=image,
        gpu="T4",
        cpu=2,
        memory=16384,
        volumes={"/weights": weights},
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        max_containers=2,  # A/B en paralelo en 2 GPUs; 1 input por GPU (anti-OOM)
        timeout=60,
        min_containers=0,
    )(freeuv_worker)


def flame_worker(job_id: str, r2_key: str, landmarks_json: bytes):
    """Worker 2 - FLAME fitting DECA+Open (GPU, 1 input por GPU). Lee imagen de R2."""
    t0 = time.perf_counter()
    bucket = _r2_bucket()
    image = _fetch_r2_bytes(bucket, r2_key)
    out = flame_infer(job_id, image, landmarks_json)
    dt = int((time.perf_counter() - t0) * 1000)
    logger.info("flame_worker ok job=%s out_len=%d duration_ms=%d", job_id, len(out), dt)
    return out


if HAVE_MODAL:
    flame_worker = app.function(
        image=image,
        gpu="T4",
        cpu=4,
        memory=32768,
        volumes={"/weights": weights},
        secrets=[modal.Secret.from_name("vultus-cloudflare")],
        max_containers=2,  # A/B en paralelo en 2 GPUs; 1 input por GPU
        timeout=60,
    )(flame_worker)


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
            json={"visibility_timeout_ms": TOTAL_TIMEOUT_SECS * 1000, "batch_size": batch_size},
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
    _report_progress(job_id, PROGRESS_LANDMARKS, "landmarks")
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
    _report_progress(job_id, PROGRESS_FLAME, "flame")

    if _is_modal_function(flame_worker):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(flame_worker.remote, job_id, r2_a, lm_a)
            fut_b = ex.submit(flame_worker.remote, job_id, r2_b, lm_b)
            flaw_a = fut_a.result(timeout=FLAME_TIMEOUT_SECS + 50)
            flaw_b = fut_b.result(timeout=FLAME_TIMEOUT_SECS + 50)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(flame_infer, job_id, img_a, lm_a)
            fut_b = ex.submit(flame_infer, job_id, img_b, lm_b)
            flaw_a = fut_a.result(timeout=FLAME_TIMEOUT_SECS + 50)
            flaw_b = fut_b.result(timeout=FLAME_TIMEOUT_SECS + 50)
    if len(flaw_a) != UV_LEN or len(flaw_b) != UV_LEN:
        raise ValueError("flaw-uv bad length")
    _report_progress(job_id, PROGRESS_FREEUV, "freeuv")

    if _is_modal_function(freeuv_worker):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(freeuv_worker.remote, job_id, flaw_a)
            fut_b = ex.submit(freeuv_worker.remote, job_id, flaw_b)
            uv_a = fut_a.result(timeout=FREEUV_TIMEOUT_SECS + 30)
            uv_b = fut_b.result(timeout=FREEUV_TIMEOUT_SECS + 30)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(freeuv_infer, job_id, flaw_a)
            fut_b = ex.submit(freeuv_infer, job_id, flaw_b)
            uv_a = fut_a.result(timeout=FREEUV_TIMEOUT_SECS + 30)
            uv_b = fut_b.result(timeout=FREEUV_TIMEOUT_SECS + 30)
    if len(uv_a) != UV_LEN or len(uv_b) != UV_LEN:
        raise ValueError("complete-uv bad length")
    _report_progress(job_id, PROGRESS_BAKE, "bake")

    heat = _heatmap_abs_diff(bytes(uv_a), bytes(uv_b))
    # PNG + zip en memoria, sin disco. Nombres exactos del contrato.
    import io as _io

    from PIL import Image

    def _to_png(raw: bytes) -> bytes:
        img = Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(raw))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    zip_bytes = _build_result_zip(_to_png(uv_a), _to_png(uv_b), _to_png(heat))
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
        secrets=[
            modal.Secret.from_name("vultus-cloudflare"),
            modal.Secret.from_name("vultus-queues-token"),
        ],
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
        # buffer_containers=1 absorbe la rafaga A/B de Rust en paralelo.
        max_containers=10,
        buffer_containers=1,
        timeout=600,  # red amplia: la primera llamada paga cold + carga de pesos
        env={"VULTUS_REAL_ML": "1"},
    )
    class MlSidecar:
        @modal.enter()
        def warm(self):
            # Precalienta los tres modelos al arrancar el container para que
            # el primer request ya este en warm (el timeout Rust de 5s en
            # landmarks no perdona la carga lazy de TFLite/CUDA).
            # Sin fastapi aqui: importa pesado lazy igual que en remoto.
            _landmarker()
            _deca_model()
            _freeuv_pipe()
            logger.info("sidecar warm: mediapipe+deca+freeuv cargados")

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
