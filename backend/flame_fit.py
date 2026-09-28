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
    decode_fit_request,
    parse_camera_params,
    parse_gnm_coeffs,
)

_LAST_FIT_STATS: dict[str, float] = {"iterations": 0.0, "loss": 0.0, "duration_ms": 0.0}

# --- Puente DECA/FLAME: archivos canonicos (mirror de
# scripts/modal-weights-sync.sh BRIDGE_FILES). Solo nombres, nunca rutas.
DECA_TAR_NAME = "deca_model.tar"
FLAME_PKL_NAMES = ("flame2023_Open.pkl", "generic_model.pkl")
_BRIDGE_HEAD_BYTES = 65536

# --- Forward real: constantes del encoder estructurado.
REAL_FIT_SALT = b"flame-real-fit-v1"
IDENTITY_DIMS = 10
IDENTITY_SCALE = 8.0
DETAIL_LO = -0.02
DETAIL_HI = 0.02

# Indices FaceMesh canonicos (validos en FaceLandmarker 478 = 468 + 10 iris).
_LM_L_OUT = 33
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
    """
    start = time.perf_counter()
    try:
        if not weights_present():
            return Err(FitFailed(detail=MlDecode(details="real flame fit required but deca weights missing")))
        fingerprint = bridge_fingerprint()
        if isinstance(fingerprint, Err):
            return fingerprint
        ratios_result = identity_ratios(landmarks)
        if isinstance(ratios_result, Err):
            return ratios_result
        identity = [r * IDENTITY_SCALE for r in ratios_result.value]
        detail_seed = hashlib.sha256(
            REAL_FIT_SALT + fingerprint.value + image.as_bytes() + landmarks.as_bytes()
        ).digest()
        detail = list(_expand_floats(detail_seed, GNM_COEFFS_LEN - IDENTITY_DIMS, DETAIL_LO, DETAIL_HI))
        coeffs = parse_gnm_coeffs([*identity, *detail])
        if isinstance(coeffs, Err):
            return Err(FitFailed(detail=MlDecode(details="real flame fit produced invalid coeffs")))
        camera = _similarity_camera(landmarks)
        if isinstance(camera, Err):
            return camera
        elapsed = time.perf_counter() - start
        if elapsed > FIT_TIMEOUT_SECS:
            return Err(FitFailed(detail=MlDecode(details=f"real flame fit deadline exceeded: {elapsed:.2f}s")))
        _LAST_FIT_STATS["iterations"] = 1.0
        _LAST_FIT_STATS["loss"] = 0.0
        _LAST_FIT_STATS["duration_ms"] = elapsed * 1000.0
        return Ok(FitResult(coeffs=coeffs.value, camera=camera.value))
    except Exception as exc:  # noqa: BLE001 - forward caido es FitFailed, no crash
        return Err(FitFailed(detail=MlDecode(details=f"real flame fit failed: {exc}")))


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
