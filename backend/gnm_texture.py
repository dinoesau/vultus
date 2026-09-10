"""Textura GNM: proyeccion foto + warp TPS + inpaint solo de ocluidas.

Doble determinista sha256 cuando SD no esta disponible.
Sin torch, sin FastAPI, sin logging. Entradas ya probadas, salidas probadas.
"""

from __future__ import annotations

import hashlib

from backend.domain import (
    UV_LEN,
    CompleteUv,
    DomainError,
    Err,
    FitResult,
    ImageBytes,
    Landmarks,
    MlDecode,
    MlFailed,
    Ok,
    parse_complete_uv,
)

_MASK_PERIOD = 8


def _seed(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
        h.update(b"\x00")
    return h.digest()


def _expand(seed: bytes, count: int) -> bytes:
    reps = count // len(seed) + 1
    return (seed * reps)[:count]


def project_texture(image: ImageBytes, fit: FitResult) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        img = image.as_bytes()
        coeff_bytes = b"".join(
            float(v).hex().encode("utf-8") for v in fit.coeffs.as_tuple()[:16]
        )
        seed = _seed(img, coeff_bytes)
        stream = _expand(seed, UV_LEN)
        out = bytes((img[i % len(img)] + stream[i]) % 256 for i in range(UV_LEN))
        parsed = parse_complete_uv(out)
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001 - el doble nunca tumba sin causa
        return Err(MlFailed(detail=MlDecode(details=f"project failed: {exc}")))


def warp_with_landmarks(albedo: CompleteUv, landmarks: Landmarks) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        raw = albedo.as_bytes()
        seed = _seed(landmarks.as_bytes(), b"warp")
        shift = seed[0] % 7 + 1
        out = raw[-shift:] + raw[:-shift]
        assert len(out) == UV_LEN
        parsed = parse_complete_uv(out)
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"warp failed: {exc}")))


def occlusion_mask(landmarks: Landmarks) -> bytes:
    seed = _seed(landmarks.as_bytes(), b"occlusion")
    return bytes(seed[i % len(seed)] % _MASK_PERIOD == 0 for i in range(UV_LEN))


def inpaint_occluded(albedo: CompleteUv, landmarks: Landmarks) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        raw = bytearray(albedo.as_bytes())
        mask = occlusion_mask(landmarks)
        for i in range(UV_LEN):
            if mask[i]:
                left = raw[i - 1] if i > 0 else raw[i + 1]
                right = raw[i + 1] if i + 1 < UV_LEN else raw[i - 1]
                raw[i] = (left + right) // 2
        parsed = parse_complete_uv(bytes(raw))
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"inpaint failed: {exc}")))


def build_albedo(
    image: ImageBytes, fit: FitResult, landmarks: Landmarks
) -> Ok[CompleteUv] | Err[DomainError]:
    projected = project_texture(image, fit)
    if isinstance(projected, Err):
        return projected
    warped = warp_with_landmarks(projected.value, landmarks)
    if isinstance(warped, Err):
        return warped
    return inpaint_occluded(warped.value, landmarks)
