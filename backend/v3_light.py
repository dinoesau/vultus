"""Luz v3 separada (DPR Hourglass real + relighting con esferas).

- Checkpoint real `trained_model_03.t7` (via `backend.dpr`: `find_t7`,
  `dpr_available`, `estimate_sh`, `sh_basis`). Aqui se re-exporta la base
  SH9 en numpy para el renderer sin torch top-level.
- Albedo = foto/shading SH9 con cota `[0.5, 2.0]` (tono preservado: misma
  escala a los 3 canales).
- Renderer de relighting (neutral + 3 luces con esferas como la figura):
  `relight(albedo, normals, sh)` = `albedo * shading(sh)` con esferas
  de sonda (las esferas las pinta el visor; aqui el shading esferico por
  normal es el que las haria verse como la figura).

Test RED: misma albedo bajo 2 luces distintas da shading distinto y
albedo identico. Sin torch top-level, sin logging.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Checkpoint DPR real (espejo de `backend.dpr.DPR_T7_NAME`).
DPR_T7_NAME = "trained_model_03.t7"

# Cota del shading relativo (espejo de `backend/flame_texture.py::_dpr_normalize`).
SHADING_CLAMP: tuple[float, float] = (0.5, 2.0)


def sh_basis_9(normals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Base SH Sloan (9) sobre normales unitarias [...,3] -> [...,9].

    Espejo puro-numpy de `backend.dpr.sh_basis` (misma convencion).
    """
    n = np.asarray(normals, dtype=np.float64)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack(
        [
            np.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ],
        axis=-1,
    )


def shading_from_sh(normals: NDArray[np.float64], sh: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shading gris por texel: `Y(normal) . sh`, acotado a `SHADING_CLAMP`."""
    basis = sh_basis_9(np.asarray(normals, dtype=np.float64))
    coeffs = np.asarray(sh, dtype=np.float64).reshape(9)
    raw = (basis * coeffs[None, None, :]).sum(axis=-1) if basis.ndim == 3 else (basis * coeffs).sum(axis=-1)
    lo, hi = SHADING_CLAMP
    return np.clip(np.asarray(raw, dtype=np.float64), lo, hi)


def albedo_from_photo(
    photo: NDArray[np.float64], normals: NDArray[np.float64], sh: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Albedo = foto/shading SH9 con cota (tono preservado, puro numpy)."""
    img = np.asarray(photo, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    shading = shading_from_sh(n, np.asarray(sh, dtype=np.float64))
    ref = float(np.mean(shading)) if shading.size else 1.0
    if not np.isfinite(ref) or ref < 1e-6:
        ref = 1.0
    scale = np.clip(ref / np.maximum(shading, 1e-6), *SHADING_CLAMP)
    out = img * scale[..., None] if img.ndim == 3 else img * scale
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 255.0)


def relight(
    albedo: NDArray[np.float64], normals: NDArray[np.float64], sh: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Re-ilumina un albedo fijo bajo una luz SH nueva (neutral + 3).

    `out = albedo * shading(sh)/mean(shading)`: la misma albedo bajo 2
    luces distintas da shading distinto y albedo identico (el albedo no
    se toca, solo el shading cambia). Puro numpy.
    """
    alb = np.asarray(albedo, dtype=np.float64)
    shading = shading_from_sh(np.asarray(normals, dtype=np.float64), np.asarray(sh, dtype=np.float64))
    ref = float(np.mean(shading)) if shading.size else 1.0
    if not np.isfinite(ref) or ref < 1e-6:
        ref = 1.0
    gain = shading / ref
    out = alb * gain[..., None] if alb.ndim == 3 else alb * gain
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 255.0)
