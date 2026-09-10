"""Fitter GNM directo: 253 coefs + camara 3x4 tras un solo seam.

Doble determinista sha256 cuando el fitter publico no esta disponible.
Sin torch, sin FastAPI, sin logging. Entradas ya probadas, salidas probadas.
"""

from __future__ import annotations

import hashlib
import math
import struct

from backend.domain import (
    CAMERA_PARAMS_LEN,
    GNM_COEFFS_LEN,
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


def _expand_floats(seed: bytes, count: int, lo: float, hi: float) -> tuple[float, ...]:
    span = hi - lo
    out: list[float] = []
    for i in range(count):
        digest = hashlib.sha256(seed + struct.pack(">H", i)).digest()
        u = int.from_bytes(digest[0:4], "big") / 4294967295.0
        out.append(lo + u * span)
    return tuple(out)


def _deterministic_fit(image: ImageBytes, landmarks: Landmarks) -> FitResult:
    seed = hashlib.sha256(image.as_bytes() + b"\x00" + landmarks.as_bytes()).digest()
    coeffs_raw = _expand_floats(seed + b"coeffs", GNM_COEFFS_LEN, -2.0, 2.0)
    camera_raw = _expand_floats(seed + b"camera", CAMERA_PARAMS_LEN, -1.0, 1.0)
    coeffs = parse_gnm_coeffs(list(coeffs_raw))
    camera = parse_camera_params(list(camera_raw))
    assert isinstance(coeffs, Ok)
    assert isinstance(camera, Ok)
    return FitResult(coeffs=coeffs.value, camera=camera.value)


def _real_fit_available() -> bool:
    # El fitter publico se asume disponible en la imagen GPU (ver plan S4).
    # Licencia y VRAM pendientes de verificacion en Step 4 antes de cablear
    # la inferencia real: aqui solo el doble para no bloquear Waves 2-3.
    return False


def fit_gnm(image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
    if _real_fit_available():
        return Err(FitFailed(detail=MlDecode(details="real fitter not wired yet")))
    try:
        return Ok(_deterministic_fit(image, landmarks))
    except Exception as exc:  # noqa: BLE001 - el doble nunca debe tumbar el job sin causa
        return Err(FitFailed(detail=MlDecode(details=f"fit double failed: {exc}")))


def fit_gnm_from_request(payload: bytes) -> Ok[FitResult] | Err[DomainError]:
    decoded = decode_fit_request(payload)
    if isinstance(decoded, Err):
        return decoded
    landmarks, image = decoded.value
    return fit_gnm(image, landmarks)


def coeff_distance(a: GnmCoeffs, b: GnmCoeffs) -> float:
    total = 0.0
    for x, y in zip(a.as_tuple(), b.as_tuple()):
        d = x - y
        total += d * d
    result = math.sqrt(total)
    if not math.isfinite(result):
        return float("inf")
    return result
