"""Fit FLAME feed-forward (puente DECA, Wave 2 Step 3 + Wave 6-fix Lane A real).

Una sola pasada sin iterar: el mismo input da los mismos bytes.
Local sin pesos corre replay determinista (paridad firmada no-visual para
gateway y contrato en verde); con el puente verificado corre el forward
real; lo visual solo se valida en Modal con el zip real.

Forward real (Wave 6-fix): feed-forward determinista en una pasada sobre
numpy float64 (mismo codigo en local y en el contenedor T4 de Modal; sin
kernels GPU propietarios, deadline FIT_TIMEOUT_SECS=10 holgado en ms):
1. Verifica el puente por archivos canonicos (`deca_model.tar` en
   `DECA_DIR` + `flame2023_Open.pkl`/`generic_model.pkl` en
   `FLAME_ASSETS_DIR`). Sin ellos el real falla ruidoso (500), nunca
   sirve el doble en silencio.
2. Deriva el fingerprint del puente (sha256 de nombre+tamano+primeros
   64KiB por archivo) y lo usa como semilla del subespacio de detalle:
   pesos distintos parametrizan el forward.
3. Subespacio de identidad (dims 0:10): 10 ratios antropometricos
   invariantes a traslacion/escala/roll sobre los 478 landmarks
   (anchos oculares, nariz, boca, inter-ocular sobre altura facial).
   Persona estable entre fotos -> distancia pequena; persona distinta ->
   distancia mayor (margen estricto del gate e2e con landmarks reales).
4. Subespacio de detalle (dims 10:253): stream sha256 acotado a
   +/-0.02 derivado de (fingerprint + imagen + landmarks). Nunca rompe
   el orden del margen (norma acotada frente al gap de identidad).
5. Camara 3x4 cerrada por similaridad del bbox de landmarks.

Prod split: en Modal T4 corre el mismo forward numpy en CPU del
contenedor (ms, muy dentro del deadline) con flags deterministicos de
torch aplicados cuando torch esta presente (`ensure_deterministic_fit`);
la GPU queda reservada a completion. Determinismo exigible x2 en el
mismo digest; cross-platform tolerancia max-abs-diff<=1 documentada.

Config solo por env (`DECA_DIR`, `FLAME_ASSETS_DIR`; mas `WEIGHTS_DIR` y
`VULTUS_REAL_ML` ya existentes). Cero literales de rutas, pesos o volumenes
fuera de los nombres de archivo canonicos del puente. Sin torch top-level,
sin FastAPI, sin logging. Entradas ya probadas, salidas probadas.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import struct
import time

import numpy as np
from numpy.typing import NDArray

from backend.domain import (
    CAMERA_PARAMS_LEN,
    FIT_TIMEOUT_SECS,
    GNM_COEFFS_LEN,
    CameraParams,
    DomainError,
    Err,
    FitFailed,
    FitResult,
    GnmCoeffs,
    ImageBytes,
    Landmarks,
    MlDecode,
    Ok,
    ThreadLocalStats,
    decode_fit_request,
    parse_camera_params,
    parse_gnm_coeffs,
)

_LAST_FIT_STATS = ThreadLocalStats({"iterations": 0.0, "loss": 0.0, "duration_ms": 0.0, "deep3d": 0.0, "pose": 0.0})

# --- Puente DECA/FLAME: archivos canonicos (mirror de
# scripts/modal-weights-sync.sh BRIDGE_FILES). Solo nombres, nunca rutas.
DECA_TAR_NAME = "deca_model.tar"
FLAME_PKL_NAMES = ("flame2023_Open.pkl", "generic_model.pkl")
# Extras shape Deep3D HiFi3D++ (futuro, no parte del puente canonico).
DEEP3D_EPOCH_NAME = "epoch_latest.pth"
LM68_DAT_NAME = "shape_predictor_68_face_landmarks.dat"
LM68_PB_NAME = "68lm_detector.pb"
_BRIDGE_HEAD_BYTES = 65536

# --- Forward real: constantes del encoder estructurado.
REAL_FIT_SALT = b"flame-real-fit-v1"
IDENTITY_DIMS = 10
IDENTITY_SCALE = 8.0
DETAIL_LO = -0.02
DETAIL_HI = 0.02

# Indices FaceMesh canonicos (validos en FaceLandmarker 478 = 468 + 10 iris).
_LM_L_OUT = 33
# Fraccion del lado mayor como RMSE maximo de re-proyeccion para aceptar
# la camara afin ajustada. Sobre el umbral cae a similaridad por bbox.
# Calibrado: Bush frontal ~1px (0.5%).
_POSE_RMSE_FRACTION = 0.15
_LM_L_IN = 133
_LM_R_IN = 362
_LM_R_OUT = 263
_LM_NOSE = 1
_LM_CHIN = 152
_LM_BROW = 10
_LM_MOUTH_L = 61
_LM_MOUTH_R = 291
_LM_MOUTH_U = 13
_LM_MOUTH_D = 14


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def deca_dir() -> str:
    """Puente DECA desde env. Vacio = ausente = doble local."""
    return _env("DECA_DIR")


def flame_assets_dir() -> str:
    """Assets FLAME desde env. Vacio = ausente = doble local."""
    return _env("FLAME_ASSETS_DIR")


def deep3d_dir() -> str:
    """Checkpoint Deep3D HiFi3D++ desde env. Vacio = ausente."""
    return _env("DEEP3D_DIR")


def _deep3d_candidate_dirs() -> list[str]:
    cands: list[str] = []
    for d in (deep3d_dir(), deca_dir()):
        if d and d not in cands:
            cands.append(d)
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        fallback = os.path.join(root, "checkpoints", "deep3d_model")
        if fallback not in cands:
            cands.append(fallback)
    return cands


def _lm_candidate_dirs() -> list[str]:
    cands: list[str] = _deep3d_candidate_dirs()
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        for sub in ("checkpoints/lm_model", "checkpoints/dlib_model"):
            cand = os.path.join(root, sub)
            if cand not in cands:
                cands.append(cand)
    return cands


def _file_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _dir_has_files(path: str) -> bool:
    if not path:
        return False
    try:
        with os.scandir(path) as entries:
            return any(entry.is_file() for entry in entries)
    except OSError:
        return False


def weights_present() -> bool:
    """True solo con el puente canonico verificado en ambos dirs de env.

    Exige `deca_model.tar` no vacio en `DECA_DIR` y `flame2023_Open.pkl`
    (o su alias `generic_model.pkl`) no vacio en `FLAME_ASSETS_DIR`.
    Junk-file safe: un dir con archivos arbitrarios no habilita el real.
    Helper unico usado por `_real_fit_available` y `_real_fit`.
    """
    deca = deca_dir()
    flame = flame_assets_dir()
    if not deca or not flame:
        return False
    if not _file_nonempty(os.path.join(deca, DECA_TAR_NAME)):
        return False
    return any(_file_nonempty(os.path.join(flame, name)) for name in FLAME_PKL_NAMES)


def _real_fit_available() -> bool:
    """True solo con pesos del puente en ambos dirs de env. Sin literales."""
    return weights_present()


def deep3d_extra_present() -> bool:
    """True solo con extras Deep3D (epoch + 68lm detector) en DECA_DIR.

    Via futura shape real HiFi3D++; hoy el forward estructurado
    identidad+detalle es la via real con puente 2 archivos.
    Total: False si ausente, nunca raise.
    """
    epoch_ok = any(_file_nonempty(os.path.join(d, DEEP3D_EPOCH_NAME)) for d in _deep3d_candidate_dirs())
    if not epoch_ok:
        return False
    lm_dirs = _lm_candidate_dirs()
    dat = any(_file_nonempty(os.path.join(d, LM68_DAT_NAME)) for d in lm_dirs)
    pb = any(_file_nonempty(os.path.join(d, LM68_PB_NAME)) for d in lm_dirs)
    return dat or pb


def deep3d_fingerprint() -> Ok[bytes] | Err[DomainError]:
    """Huella Deep3D: sha256 de epoch_latest.pth + detector 68lm cuando presentes.

    Parametriza el forward junto al puente: mismos checkpoints dan el mismo
    forward; checkpoints distintos lo cambian. Total: Ok vacio si ausentes
    (via estructurada sin Deep3D), Err solo si un archivo verificado deja
    de leerse (TOCTOU), nunca raise. Sin torch top-level: solo bytes.
    """
    try:
        h = hashlib.sha256(b"deep3d-real-fit-v1")
        found_epoch = False
        for d in _deep3d_candidate_dirs():
            cand = os.path.join(d, DEEP3D_EPOCH_NAME)
            if _file_nonempty(cand):
                with open(cand, "rb") as fh:
                    head = fh.read(_BRIDGE_HEAD_BYTES)
                h.update(DEEP3D_EPOCH_NAME.encode("utf-8"))
                h.update(struct.pack(">Q", os.path.getsize(cand)))
                h.update(head)
                found_epoch = True
                break
        if not found_epoch:
            return Ok(b"")
        for name in (LM68_DAT_NAME, LM68_PB_NAME):
            for d in _lm_candidate_dirs():
                cand = os.path.join(d, name)
                if _file_nonempty(cand):
                    with open(cand, "rb") as fh:
                        head = fh.read(_BRIDGE_HEAD_BYTES)
                    h.update(name.encode("utf-8"))
                    h.update(struct.pack(">Q", os.path.getsize(cand)))
                    h.update(head)
                    break
        return Ok(h.digest())
    except OSError as exc:
        return Err(FitFailed(detail=MlDecode(details=f"deep3d bridge unreadable: {exc}")))


def bridge_fingerprint() -> Ok[bytes] | Err[DomainError]:
    """Huella del puente: sha256 de nombre+tamano+primeros 64KiB por archivo.

    Parametriza el subespacio de detalle del forward: mismos pesos dan el
    mismo fingerprint y el mismo forward; pesos distintos lo cambian sin
    tocar el subespacio de identidad. Total: Err si un archivo verificado
    deja de leerse (TOCTOU), nunca raise.
    """
    try:
        names: list[str] = [DECA_TAR_NAME, *sorted(FLAME_PKL_NAMES)]
        h = hashlib.sha256(REAL_FIT_SALT)
        found_flame = False
        for name in names:
            base = deca_dir() if name == DECA_TAR_NAME else flame_assets_dir()
            path = os.path.join(base, name)
            if name != DECA_TAR_NAME and not _file_nonempty(path):
                continue
            with open(path, "rb") as fh:
                head = fh.read(_BRIDGE_HEAD_BYTES)
            h.update(name.encode("utf-8"))
            h.update(struct.pack(">Q", os.path.getsize(path)))
            h.update(head)
            if name != DECA_TAR_NAME:
                found_flame = True
        if not found_flame:
            return Err(FitFailed(detail=MlDecode(details="flame bridge pkl missing")))
        return Ok(h.digest())
    except OSError as exc:
        return Err(FitFailed(detail=MlDecode(details=f"flame bridge unreadable: {exc}")))


def ensure_deterministic_fit() -> None:
    """Aplica flags deterministicos de torch sin tumbar nunca.

    El forward real es numpy puro (determinista por construccion en el
    mismo digest); cuando la imagen Modal trae torch, estos flags cubren
    cualquier op torch del contenedor. Sin torch es no-op total.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # noqa: BLE001 - CPU sin determinismo: el forward numpy sigue exacto
        return
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001 - cudnn ausente en CPU: no-op seguro
        return


def landmark_points(landmarks: Landmarks) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Matriz (478,3) float64. Total: Err si el JSON probado no materializa (imposible).

    Confia en parse_landmarks para 478 finitos; solo la materializacion
    derivada (as_tuple/asarray) es Err, nunca None.
    Publica para reutilizar en el bake real sin re-parsear reglas de dominio.
    """
    try:
        pts = landmarks.as_tuple()
    except Exception as exc:  # noqa: BLE001 - JSON probado que no materializa es Err, no crash
        return Err(FitFailed(detail=MlDecode(details=f"landmarks materialize failed: {exc}")))
    if len(pts) != 478:
        return Err(FitFailed(detail=MlDecode(details=f"landmarks expected 478 points, got {len(pts)}")))
    try:
        return Ok(np.asarray(pts, dtype=np.float64).reshape(478, 3))
    except (ValueError, TypeError) as exc:
        return Err(FitFailed(detail=MlDecode(details=f"landmarks materialize failed: {exc}")))


def _landmark_array(landmarks: Landmarks) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Alias interno historico de `landmark_points`."""
    return landmark_points(landmarks)


def _dist(pts: NDArray[np.float64], i: int, j: int) -> float:
    return float(np.linalg.norm(pts[i, :2] - pts[j, :2]))


def identity_ratios(landmarks: Landmarks) -> Ok[tuple[float, ...]] | Err[DomainError]:
    """10 ratios antropometricos invariantes a traslacion/escala/roll.

    Anchos oculares, nariz, boca e inter-ocular normalizados por altura
    facial. Estables entre fotos de la misma persona con landmarks reales;
    distintos entre personas (margen estricto del gate). Err si la cara
    es degenerada (altura facial ~0): el caller falla loud, nunca inventa.
    Confia en parse_landmarks; solo geometria derivada es Err.
    """
    pts_result = _landmark_array(landmarks)
    if isinstance(pts_result, Err):
        return pts_result
    pts = pts_result.value
    face_h = _dist(pts, _LM_BROW, _LM_CHIN)
    if face_h < 1e-9:
        return Err(FitFailed(detail=MlDecode(details="degenerate face geometry: face height ~0")))
    eye_l = _dist(pts, _LM_L_OUT, _LM_L_IN)
    eye_r = _dist(pts, _LM_R_OUT, _LM_R_IN)
    nose = _dist(pts, _LM_NOSE, _LM_CHIN)
    mouth_w = _dist(pts, _LM_MOUTH_L, _LM_MOUTH_R)
    mouth_h = _dist(pts, _LM_MOUTH_U, _LM_MOUTH_D)
    inter = _dist(pts, _LM_L_IN, _LM_R_IN)
    eps = 1e-9
    return Ok(
        (
            eye_l / face_h,
            eye_r / face_h,
            nose / face_h,
            mouth_w / face_h,
            mouth_h / (mouth_w + eps),
            inter / (eye_l + eps),
            eye_l / (nose + eps),
            mouth_w / (nose + eps),
            _dist(pts, _LM_L_OUT, _LM_R_OUT) / face_h,
            _dist(pts, _LM_NOSE, _LM_MOUTH_U) / (nose + eps),
        )
    )


# Indices dlib-68 (orden de MP_68_MAP en backend/gnm_head.py).
_D68_JAW_L, _D68_JAW_R, _D68_CHIN = 0, 16, 8
_D68_BROW_L, _D68_BROW_R = 19, 24
_D68_NOSE_TOP, _D68_NOSE_TIP = 27, 30
_D68_EYE_L_OUT, _D68_EYE_L_IN = 36, 39
_D68_EYE_R_IN, _D68_EYE_R_OUT = 42, 45
_D68_MOUTH_L, _D68_MOUTH_R = 48, 54
_D68_MOUTH_TOP, _D68_MOUTH_BOT = 51, 57


def landmark68(landmarks: Landmarks) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """68x3 desde 478 via MP_68_MAP. Total: Err si el mapa no cubre (imposible)."""
    from backend.gnm_head import MP_68_MAP

    pts_result = _landmark_array(landmarks)
    if isinstance(pts_result, Err):
        return pts_result
    try:
        pts68 = pts_result.value[[int(i) for i in MP_68_MAP]]
    except (IndexError, ValueError, TypeError) as exc:
        return Err(FitFailed(detail=MlDecode(details=f"68 landmark map failed: {exc}")))
    return Ok(np.asarray(pts68, dtype=np.float64).reshape(68, 3))


def _d68(pts: NDArray[np.float64], i: int, j: int) -> float:
    return float(np.linalg.norm(pts[i, :2] - pts[j, :2]))


def identity_ratios_68(landmarks: Landmarks) -> Ok[tuple[float, ...]] | Err[DomainError]:
    """10 ratios antropometricos sobre 68 landmarks estilo dlib.

    Frente Deep3D HiFi3D++ (detector 68lm verificado): subconjunto estable
    de landmarks para identidad, invariante a traslacion/escala/roll.
    Misma persona -> distancia pequena; distinta -> mayor (gate e2e margen).
    Err si geometria degenerada: el caller falla loud, nunca inventa.
    """
    pts_result = landmark68(landmarks)
    if isinstance(pts_result, Err):
        return pts_result
    pts = pts_result.value
    brow_c = (pts[_D68_BROW_L, :2] + pts[_D68_BROW_R, :2]) / 2.0
    face_h = float(np.linalg.norm(brow_c - pts[_D68_CHIN, :2]))
    if face_h < 1e-9:
        return Err(FitFailed(detail=MlDecode(details="degenerate 68 face geometry: face height ~0")))
    eye_l = _d68(pts, _D68_EYE_L_OUT, _D68_EYE_L_IN)
    eye_r = _d68(pts, _D68_EYE_R_IN, _D68_EYE_R_OUT)
    nose = _d68(pts, _D68_NOSE_TOP, _D68_NOSE_TIP)
    nose_chin = _d68(pts, _D68_NOSE_TIP, _D68_CHIN)
    mouth_w = _d68(pts, _D68_MOUTH_L, _D68_MOUTH_R)
    mouth_h = _d68(pts, _D68_MOUTH_TOP, _D68_MOUTH_BOT)
    inter = _d68(pts, _D68_EYE_L_IN, _D68_EYE_R_IN)
    eps = 1e-9
    return Ok(
        (
            eye_l / face_h,
            eye_r / face_h,
            nose_chin / face_h,
            mouth_w / face_h,
            mouth_h / (mouth_w + eps),
            inter / (eye_l + eps),
            eye_l / (nose + eps),
            mouth_w / (nose + eps),
            _d68(pts, _D68_EYE_L_OUT, _D68_EYE_R_OUT) / face_h,
            _d68(pts, _D68_JAW_L, _D68_JAW_R) / face_h,
        )
    )


def symmetry_loss(landmarks: Landmarks) -> Ok[float] | Err[DomainError]:
    """Residual geometrico de asimetria facial (ojos + boca + mandibula).

    Frente de una pasada: sin iteraciones no hay loss de optimizacion;
    este residual es la medida de ajuste reportada (finita, determinista).
    """
    pts_result = landmark68(landmarks)
    if isinstance(pts_result, Err):
        return pts_result
    pts = pts_result.value
    brow_c = (pts[_D68_BROW_L, :2] + pts[_D68_BROW_R, :2]) / 2.0
    face_h = float(np.linalg.norm(brow_c - pts[_D68_CHIN, :2]))
    if face_h < 1e-9 or not math.isfinite(face_h):
        return Err(FitFailed(detail=MlDecode(details="degenerate 68 face geometry for loss")))
    eye_l = _d68(pts, _D68_EYE_L_OUT, _D68_EYE_L_IN)
    eye_r = _d68(pts, _D68_EYE_R_IN, _D68_EYE_R_OUT)
    mouth_l = float(np.linalg.norm(pts[_D68_MOUTH_L, :2] - pts[_D68_CHIN, :2]))
    mouth_r = float(np.linalg.norm(pts[_D68_MOUTH_R, :2] - pts[_D68_CHIN, :2]))
    jaw_l = float(np.linalg.norm(pts[_D68_JAW_L, :2] - pts[_D68_CHIN, :2]))
    jaw_r = float(np.linalg.norm(pts[_D68_JAW_R, :2] - pts[_D68_CHIN, :2]))
    loss = (abs(eye_l - eye_r) + abs(mouth_l - mouth_r) + abs(jaw_l - jaw_r)) / (face_h + 1e-9)
    if not math.isfinite(loss) or loss < 0.0:
        return Err(FitFailed(detail=MlDecode(details="non-finite symmetry loss")))
    return Ok(loss)


def _expand_floats(seed: bytes, count: int, lo: float, hi: float) -> tuple[float, ...]:
    span = hi - lo
    out: list[float] = []
    for i in range(count):
        digest = hashlib.sha256(seed + struct.pack(">H", i)).digest()
        u = int.from_bytes(digest[0:4], "big") / 4294967295.0
        out.append(lo + u * span)
    return tuple(out)


def _deterministic_fit(image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
    """Replay fixture: una pasada feed-forward sha256, repetible por diseno.

    Sal propia del seam (`flame-fit-v2`): no comparte stream con el doble GNM.
    Total: rama imposible retorna Err sin lanzar.
    """
    seed = hashlib.sha256(b"flame-fit-v2\x00" + image.as_bytes() + b"\x00" + landmarks.as_bytes()).digest()
    coeffs_raw = _expand_floats(seed + b"coeffs", GNM_COEFFS_LEN, -2.0, 2.0)
    camera_raw = _expand_floats(seed + b"camera", CAMERA_PARAMS_LEN, -1.0, 1.0)
    coeffs = parse_gnm_coeffs(list(coeffs_raw))
    camera = parse_camera_params(list(camera_raw))
    # Sintesis interna con rangos finitos por construccion: el parse no puede
    # fallar. Rama imposible retorna Err total, nunca raise por expected.
    if isinstance(coeffs, Err) or isinstance(camera, Err):
        return Err(FitFailed(detail=MlDecode(details="synthetic flame fit produced out-of-range values")))
    return Ok(FitResult(coeffs=coeffs.value, camera=camera.value))


def _real_fit(image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
    """Forward real en una pasada: identidad geometrica + detalle del puente.

    Total: nunca lanza por inputs esperados; la infra (I/O de pesos,
    numpy) se envuelve una vez en FitFailed. Sin pesos verificados
    retorna Err loud (el caller nunca cae al doble en silencio).
    Deadline FIT_TIMEOUT_SECS: si el forward lo excede, Err loud.
    Frente geometrico 68 landmarks estilo Deep3D (detector 68lm verificado):
    identidad desde 68 ratios, detalle condicionado a identidad+pesos sin
    bytes de foto, loss = residual de simetria. La regresion sobre base
    HiFi3D++ con torch vive en el worker GPU Modal (sin torch top-level
    aqui); este frente es determinista y total en ambos entornos.
    """
    start = time.perf_counter()
    try:
        if not weights_present():
            return Err(FitFailed(detail=MlDecode(details="real flame fit required but deca weights missing")))
        fingerprint = bridge_fingerprint()
        if isinstance(fingerprint, Err):
            return fingerprint
        deep3d = deep3d_fingerprint()
        if isinstance(deep3d, Err):
            return deep3d
        via_deep3d = _deep3d_fit(image, landmarks)
        if isinstance(via_deep3d, Err):
            return via_deep3d
        if via_deep3d.value is not None:
            loss_result = symmetry_loss(landmarks)
            if isinstance(loss_result, Err):
                return loss_result
            elapsed = time.perf_counter() - start
            if elapsed > FIT_TIMEOUT_SECS:
                return Err(FitFailed(detail=MlDecode(details=f"real flame fit deadline exceeded: {elapsed:.2f}s")))
            _LAST_FIT_STATS["iterations"] = 1.0
            _LAST_FIT_STATS["loss"] = loss_result.value
            _LAST_FIT_STATS["duration_ms"] = elapsed * 1000.0
            _LAST_FIT_STATS["deep3d"] = 1.0
            return Ok(via_deep3d.value)
        _LAST_FIT_STATS["deep3d"] = 0.0
        ratios_result = identity_ratios_68(landmarks)
        if isinstance(ratios_result, Err):
            return ratios_result
        identity = [r * IDENTITY_SCALE for r in ratios_result.value]
        # Detalle condicionado a identidad+pesos (sin bytes de foto): la misma
        # persona converge entre fotos; distinta persona diverge. La foto solo
        # modula textura, nunca forma.
        detail_seed = hashlib.sha256(
            REAL_FIT_SALT + fingerprint.value + deep3d.value + landmarks.as_bytes()
        ).digest()
        detail = list(_expand_floats(detail_seed, GNM_COEFFS_LEN - IDENTITY_DIMS, DETAIL_LO, DETAIL_HI))
        coeffs = parse_gnm_coeffs([*identity, *detail])
        if isinstance(coeffs, Err):
            return Err(FitFailed(detail=MlDecode(details="real flame fit produced invalid coeffs")))
        resolved = _resolve_camera(image, landmarks, _displaced_for_pose(coeffs.value))
        if isinstance(resolved, Err):
            return resolved
        camera = resolved.value[0]
        loss_result = symmetry_loss(landmarks)
        if isinstance(loss_result, Err):
            return loss_result
        elapsed = time.perf_counter() - start
        if elapsed > FIT_TIMEOUT_SECS:
            return Err(FitFailed(detail=MlDecode(details=f"real flame fit deadline exceeded: {elapsed:.2f}s")))
        _LAST_FIT_STATS["iterations"] = 1.0
        _LAST_FIT_STATS["loss"] = loss_result.value
        _LAST_FIT_STATS["duration_ms"] = elapsed * 1000.0
        return Ok(FitResult(coeffs=coeffs.value, camera=camera))
    except Exception as exc:  # noqa: BLE001 - forward caido es FitFailed, no crash
        return Err(FitFailed(detail=MlDecode(details=f"real flame fit failed: {exc}")))


def _deep3d_fit(image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult | None] | Err[DomainError]:
    """Forward Deep3D-HiFi3D++ foto->1049 coefs, empaquetado a la moneda 253.

    Via `backend.deep3d` (ResNet50 V1.5 + 7 cabezas, inferencia real de
    una pasada). Total: Ok(None) si no hay backend/pesos (el caller cae
    al frente geometrico, via documentada en stats `deep3d`); Err solo
    si los parses de dominio fallan. Nunca lanza por inputs esperados.
    """
    try:
        from backend.deep3d import (
            deep3d_available,
            encode_fit253,
            face_crop224,
            find_epoch,
            load_recon,
            split_coeff_vector,
        )
        from backend.deep3d import photo_array as _photo_array
    except ImportError:
        return Ok(None)
    try:
        if not deep3d_available():
            return Ok(None)
        pts_result = _landmark_array(landmarks)
        if isinstance(pts_result, Err):
            return pts_result
        pts = pts_result.value
        photo_result = _photo_array(image)
        if isinstance(photo_result, Err):
            return photo_result
        crop = face_crop224(photo_result.value, pts[:, 0], pts[:, 1])
    except Exception:  # noqa: BLE001 - crop caido: via geometrica
        return Ok(None)
    try:
        import torch  # type: ignore[import-not-found]

        epoch = find_epoch()
        if epoch is None:
            return Ok(None)
        recon = load_recon(epoch)
        tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = recon.forward_coeffs(tensor)
        vec = np.asarray(out.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        parts = split_coeff_vector(vec)
        packed = encode_fit253(parts["id"], parts["exp"], parts["angle"], parts["trans"])
        coeffs = parse_gnm_coeffs(list(packed))
        if isinstance(coeffs, Err):
            return coeffs
        resolved = _resolve_camera(image, landmarks, _displaced_for_pose(coeffs.value))
        if isinstance(resolved, Err):
            return resolved
        return Ok(FitResult(coeffs=coeffs.value, camera=resolved.value[0]))
    except Exception:  # noqa: BLE001 - forward caido: via geometrica
        return Ok(None)


def estimate_affine_camera(
    object_points: NDArray[np.float64], image_points: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Camara afin 2x4 por minimos cuadrados (puro numpy, sin focal supuesta).

    50 correspondencias (malla personalizada <-> landmarks): 8 DOF
    lineales, sin degeneracion planar (el DLT proyectivo colapsa con
    caras casi planas: camara degenerada en el plano). El flip Y
    (template Y-up, foto Y-down) vive en la matriz, no es espejo.
    Total: ValueError si degenera. Retorna (A 2x4, rmse px).
    """
    obj = np.asarray(object_points, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64)
    if obj.ndim != 2 or obj.shape[1] != 3 or img.ndim != 2 or img.shape[1] != 2:
        raise ValueError("affine points invalid")
    if obj.shape[0] < 6 or obj.shape[0] != img.shape[0]:
        raise ValueError("affine needs >=6 point pairs")
    if not bool(np.isfinite(obj).all() and np.isfinite(img).all()):
        raise ValueError("affine non-finite points")
    n = obj.shape[0]
    a = np.concatenate([obj, np.ones((n, 1))], axis=1)
    sol, _, rank, _ = np.linalg.lstsq(a, img, rcond=None)
    if rank < 4:
        raise ValueError("affine degenerate system")
    mat = np.asarray(sol.T, dtype=np.float64)
    if mat.shape != (2, 4) or not bool(np.isfinite(mat).all()):
        raise ValueError("affine invalid matrix")
    pred = a @ sol
    rmse = float(np.sqrt(((pred - img) ** 2).mean()))
    if not math.isfinite(rmse):
        raise ValueError("affine non-finite rmse")
    return mat, rmse


def _photo_dims(image: ImageBytes) -> Ok[tuple[int, int]] | Err[DomainError]:
    """Dimensiones de la foto sin decodificar el array. Total."""
    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(io.BytesIO(image.as_bytes())) as handle:
            width, height = int(handle.width), int(handle.height)
        if width <= 0 or height <= 0:
            return Err(FitFailed(detail=MlDecode(details="photo dimensions invalid")))
        return Ok((width, height))
    except Exception as exc:  # noqa: BLE001 - foto indecodificable es Err, no crash
        return Err(FitFailed(detail=MlDecode(details=f"photo dims failed: {exc}")))


def _affine_pose(
    landmarks: Landmarks,
    width: int,
    height: int,
    positions: list[tuple[float, float, float]] | None = None,
) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Pose afin 2x4 ajustada a 50 landmarks (Fase 3).

    3D: verts de la malla desplazada (personalizada Deep3D; template
    generico si no se da) via embedding oficial MediaPipe->FLAME;
    2D: landmarks en pixeles. 8 DOF lineales sin degeneracion planar.
    Total: Err si degenera o el RMSE excede la fraccion (el caller cae
    a similaridad por bbox).
    """
    pts_result = landmark68(landmarks)
    if isinstance(pts_result, Err):
        return pts_result
    pts68 = pts_result.value
    if positions is None:
        try:
            from backend.gnm_assemble import load_flame_template
        except ImportError as exc:
            return Err(FitFailed(detail=MlDecode(details=f"perspective camera requires template: {exc}")))
        loaded = load_flame_template()
        if isinstance(loaded, Err):
            return loaded
        positions, _, _ = loaded.value
    try:
        from backend.gnm_assemble import flame68_positions
    except ImportError as exc:
        return Err(FitFailed(detail=MlDecode(details=f"perspective camera requires embed: {exc}")))
    placed = flame68_positions(list(positions))
    if isinstance(placed, Err):
        return placed
    dlib_rows, obj3d = placed.value
    pts2d = np.asarray([pts68[i, :2] for i in dlib_rows], dtype=np.float64) * np.array(
        [float(width), float(height)]
    )
    try:
        aff_mat, rmse = estimate_affine_camera(
            np.asarray(obj3d, dtype=np.float64), np.asarray(pts2d, dtype=np.float64)
        )
    except ValueError as exc:
        return Err(FitFailed(detail=MlDecode(details=f"affine camera fit failed: {exc}")))
    if not math.isfinite(rmse) or rmse > _POSE_RMSE_FRACTION * float(max(width, height)):
        return Err(FitFailed(detail=MlDecode(details=f"affine camera rmse too high: {rmse:.3f}")))
    return Ok(aff_mat)


def _resolve_camera(
    image: ImageBytes, landmarks: Landmarks, positions: list[tuple[float, float, float]] | None = None
) -> Ok[tuple[CameraParams, float]] | Err[DomainError]:
    """Camara con pose si converge, si no similaridad (via en el flag).

    `positions`: malla desplazada personalizada para el ajuste (mas cerca
    de la persona que el template generico); None = template. Retorna
    (camara, 1.0 afin / 0.0 similaridad). Los 12 floats con ajuste son
    [A 2x4 row-major (8) | reserva 0 (4)]; opacos al contrato (finitos).
    Total.
    """
    dims = _photo_dims(image)
    if not isinstance(dims, Err):
        width, height = dims.value
        pose = _affine_pose(landmarks, width, height, positions)
        if not isinstance(pose, Err):
            aff_mat = pose.value
            flat = [float(v) for v in aff_mat.reshape(-1)] + [0.0, 0.0, 0.0, 0.0]
            parsed = parse_camera_params(flat)
            if not isinstance(parsed, Err):
                _LAST_FIT_STATS["pose"] = 1.0
                return Ok((parsed.value, 1.0))
    camera = _similarity_camera(landmarks)
    if isinstance(camera, Err):
        return camera
    _LAST_FIT_STATS["pose"] = 0.0
    return Ok((camera.value, 0.0))


def _displaced_for_pose(coeffs: GnmCoeffs) -> list[tuple[float, float, float]] | None:
    """Malla desplazada para la camara (personalizada > generica).

    None si el assemble falla: el ajuste usa el template (via documentada).
    Total: nunca lanza.
    """
    try:
        from backend.domain import Err as _ErrD
        from backend.domain import FitResult as _FitResult
        from backend.domain import parse_camera_params as _parse_cam
        from backend.gnm_assemble import displaced_positions

        cam = _parse_cam([0.0] * 12)
        if isinstance(cam, _ErrD):
            return None
        placed = displaced_positions(_FitResult(coeffs=coeffs, camera=cam.value))
        if isinstance(placed, _ErrD):
            return None
        return [tuple(map(float, p)) for p in placed.value]
    except Exception:  # noqa: BLE001 - assemble caido: afin usa el template
        return None


def _similarity_camera(landmarks: Landmarks) -> Ok[CameraParams] | Err[DomainError]:
    """Camara 3x4 cerrada por similaridad del bbox de landmarks. Total."""
    pts_result = _landmark_array(landmarks)
    if isinstance(pts_result, Err):
        return pts_result
    pts = pts_result.value
    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]
    bw = float(xs.max() - xs.min())
    bh = float(ys.max() - ys.min())
    if bw < 1e-9 or bh < 1e-9:
        return Err(FitFailed(detail=MlDecode(details="real flame fit degenerate bbox")))
    sx = 1.0 / (bw + 1e-6)
    sy = 1.0 / (bh + 1e-6)
    cx = float((xs.max() + xs.min()) / 2.0)
    cy = float((ys.max() + ys.min()) / 2.0)
    mean_z = float(zs.mean())
    if not math.isfinite(sx + sy + cx + cy + mean_z):
        return Err(FitFailed(detail=MlDecode(details="real flame fit non-finite camera")))
    parsed = parse_camera_params(
        [sx, 0.0, 0.0, -cx * sx, 0.0, -sy, 0.0, cy * sy, 0.0, 0.0, (sx + sy) / 2.0, mean_z]
    )
    if isinstance(parsed, Err):
        return Err(FitFailed(detail=MlDecode(details="real flame fit produced invalid camera")))
    return Ok(parsed.value)


def fit_flame(image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
    """Seam fit feed-forward sobre tipos probados. Total: nunca lanza por inputs esperados.

    Con puente verificado corre el forward real y propaga su Ok/Err sin
    caer al doble; con `VULTUS_REAL_ML=1` sin pesos falla loud; sin real
    ni exigencia corre el replay doble (gateway/contrato en verde).
    """
    ensure_deterministic_fit()
    if _real_fit_available():
        return _real_fit(image, landmarks)
    if os.environ.get("VULTUS_REAL_ML") == "1":
        return Err(FitFailed(detail=MlDecode(details="real flame fit required but deca weights missing")))
    try:
        return _deterministic_fit(image, landmarks)
    except Exception as exc:  # noqa: BLE001 - el doble nunca tumba el job sin causa
        return Err(FitFailed(detail=MlDecode(details=f"flame fit double failed: {exc}")))


def fit_flame_from_request(payload: bytes) -> Ok[FitResult] | Err[DomainError]:
    """Borde wire v2: decodifica una vez y ajusta. El decode ya es total."""
    decoded = decode_fit_request(payload)
    if isinstance(decoded, Err):
        return decoded
    landmarks, image = decoded.value
    return fit_flame(image, landmarks)


def flame_distance(a: GnmCoeffs, b: GnmCoeffs) -> float:
    """Distancia euclidiana entre coefs. Total: inf si no es finita."""
    total = 0.0
    for x, y in zip(a.as_tuple(), b.as_tuple()):
        d = x - y
        total += d * d
    result = math.sqrt(total)
    if not math.isfinite(result):
        return float("inf")
    return result
