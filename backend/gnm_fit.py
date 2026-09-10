"""Fitter GNM directo: 253 coefs + camara 3x4 tras un solo seam.

Fit real (ridge sobre la cabeza GNM de `backend/gnm_head.py`) cuando los
pesos estan disponibles; doble determinista sha256 como fallback local.
Sin torch, sin FastAPI, sin logging. Entradas ya probadas, salidas probadas.
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
from backend.gnm_head import (
    LANDMARKS68,
    load_gnm_head,
    mediapipe478_to_gnm68_targets,
)

_RIDGE_LAMBDA = 1e-3
_OUTER_ITERS = 3
_COEF_CLIP = 3.0

_LAST_FIT_STATS: dict[str, float] = {"iterations": 0.0, "loss": 0.0, "duration_ms": 0.0}

_LM_X0: NDArray[np.float64] | None = None
_LM_BASIS: NDArray[np.float64] | None = None


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
    try:
        load_gnm_head()
    except Exception:  # noqa: BLE001 - sin pesos no hay fit real, el caller decide
        return False
    return True


def _landmark_model() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Media-cara 68x3 (X0) y base 253x68x3 ya pesada por el mapa disperso."""
    global _LM_X0, _LM_BASIS
    cached_x0 = _LM_X0
    cached_basis = _LM_BASIS
    if cached_x0 is not None and cached_basis is not None:
        return (cached_x0, cached_basis)
    head = load_gnm_head()
    idx: NDArray[np.int64] = np.asarray(head.landmark_indices, dtype=np.int64)
    weights: NDArray[np.float64] = np.asarray(head.landmark_weights, dtype=np.float64)
    template: NDArray[np.float64] = np.asarray(head.template_positions, dtype=np.float64)
    x0: NDArray[np.float64] = (template[idx] * weights[:, :, None]).sum(axis=1)
    gathered: NDArray[np.float32] = np.asarray(head.identity_basis, dtype=np.float32)[:, idx]
    basis: NDArray[np.float64] = (gathered.astype(np.float64) * weights[None, :, :, None]).sum(axis=2)
    _LM_X0 = x0
    _LM_BASIS = basis
    return (x0, basis)


def _estimate_camera(
    pred: NDArray[np.float64], targets: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Weak-perspective cerrada: x = s*X + tx, y = s*Y + ty por lstsq."""
    design: NDArray[np.float64] = np.zeros((2 * LANDMARKS68, 3), dtype=np.float64)
    rhs: NDArray[np.float64] = np.zeros((2 * LANDMARKS68,), dtype=np.float64)
    design[0::2, 0] = pred[:, 0]
    design[0::2, 1] = 1.0
    design[1::2, 0] = pred[:, 1]
    design[1::2, 2] = 1.0
    rhs[0::2] = targets[:, 0]
    rhs[1::2] = targets[:, 1]
    raw = np.linalg.lstsq(design, rhs, rcond=None)[0]
    sol: NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
    s = float(sol[0])
    tx = float(sol[1])
    ty = float(sol[2])
    if not (math.isfinite(s) and math.isfinite(tx) and math.isfinite(ty)):
        raise ValueError("non-finite camera estimate")
    return (s, tx, ty)


def _solve_coefs(
    x0: NDArray[np.float64],
    basis: NDArray[np.float64],
    targets: NDArray[np.float64],
    cam: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Ridge cerrada sobre 253 coefs con camara fija, recorte a [-3, 3]."""
    s, tx, ty = cam
    dim = GNM_COEFFS_LEN
    count = 2 * LANDMARKS68
    mat: NDArray[np.float64] = np.zeros((count, dim), dtype=np.float64)
    mat[0::2, :] = s * basis[:, :, 0].T
    mat[1::2, :] = s * basis[:, :, 1].T
    rhs: NDArray[np.float64] = np.zeros((count,), dtype=np.float64)
    rhs[0::2] = targets[:, 0] - (s * x0[:, 0] + tx)
    rhs[1::2] = targets[:, 1] - (s * x0[:, 1] + ty)
    aug: NDArray[np.float64] = np.vstack([mat, math.sqrt(_RIDGE_LAMBDA) * np.eye(dim, dtype=np.float64)])
    aug_rhs: NDArray[np.float64] = np.concatenate([rhs, np.zeros((dim,), dtype=np.float64)])
    raw = np.linalg.lstsq(aug, aug_rhs, rcond=None)[0]
    vec: NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
    coefs: NDArray[np.float64] = np.clip(vec, -_COEF_CLIP, _COEF_CLIP)
    if not bool(np.all(np.isfinite(coefs))):
        raise ValueError("non-finite coef solve")
    return coefs


def _real_fit(image: ImageBytes, landmarks: Landmarks) -> FitResult:
    del image  # el fit geometrico solo usa landmarks; la imagen viaja al seam texture
    start = time.perf_counter()
    targets: NDArray[np.float64] = np.asarray(
        mediapipe478_to_gnm68_targets(landmarks.as_bytes()), dtype=np.float64
    )
    if targets.shape != (LANDMARKS68, 2):
        raise ValueError(f"targets shape {targets.shape} != {(LANDMARKS68, 2)}")
    if not bool(np.all(np.isfinite(targets))):
        raise ValueError("non-finite fit targets")
    x0, basis = _landmark_model()
    coefs: NDArray[np.float64] = np.zeros((GNM_COEFFS_LEN,), dtype=np.float64)
    cam: tuple[float, float, float] = (1.0, 0.0, 0.0)
    for _ in range(_OUTER_ITERS):
        pred: NDArray[np.float64] = x0 + np.einsum("d,dmc->mc", coefs, basis)
        cam = _estimate_camera(pred, targets)
        coefs = _solve_coefs(x0, basis, targets, cam)
    s, tx, ty = cam
    final_mesh: NDArray[np.float64] = x0 + np.einsum("d,dmc->mc", coefs, basis)
    proj: NDArray[np.float64] = np.stack(
        [s * final_mesh[:, 0] + tx, s * final_mesh[:, 1] + ty], axis=1
    )
    loss = float(((proj - targets) ** 2).mean())
    if not math.isfinite(loss):
        raise ValueError("non-finite fit loss")
    duration_ms = (time.perf_counter() - start) * 1000.0
    _LAST_FIT_STATS["iterations"] = float(_OUTER_ITERS)
    _LAST_FIT_STATS["loss"] = loss
    _LAST_FIT_STATS["duration_ms"] = duration_ms
    coeffs_parsed = parse_gnm_coeffs([float(v) for v in coefs])
    camera_parsed = parse_camera_params(
        [s, 0.0, 0.0, tx, 0.0, s, 0.0, ty, 0.0, 0.0, 0.0, 0.0]
    )
    assert isinstance(coeffs_parsed, Ok)
    assert isinstance(camera_parsed, Ok)
    typed_coeffs: GnmCoeffs = coeffs_parsed.value
    return FitResult(coeffs=typed_coeffs, camera=camera_parsed.value)


def fit_gnm(image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
    if _real_fit_available():
        try:
            return Ok(_real_fit(image, landmarks))
        except Exception as exc:  # noqa: BLE001 - optimizacion caida es FitFailed, no crash
            return Err(FitFailed(detail=MlDecode(details=f"real fit failed: {exc}")))
    if os.environ.get("VULTUS_REAL_ML") == "1":
        return Err(FitFailed(detail=MlDecode(details="real fit required but head weights missing")))
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
