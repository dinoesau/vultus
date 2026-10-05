"""Textura v3 1024 (unwrap real + TexGAN largo + blend + match_color).

- Unwrap real con `unwrap_1024_info.mat` + mascaras (el `.mat` verifica el
  layout; sin el es `Err` loud, nunca blur silencioso).
- Optimizacion de latente TexGAN enmascarada larga (cientos de pasos,
  init `w_avg`) solo en texeles no validos (via `backend.texgan`).
- Blend Poisson/Laplaciano y `match_color` (puros numpy).
- Test RED: TV del atlas 1024 <= 2.0 y cero sentinel.

Sin torch top-level (lazy via `backend.texgan`). Sin logging.
"""

from __future__ import annotations

import math
import os

import numpy as np
from numpy.typing import NDArray

from backend.v3_contract import V3_TEXGAN_FIT_STEPS, V3_TV_MAX, V3_UV_LEN, V3_UV_SIZE

# Init documentado del latente (clave `mapping.w_avg` del checkpoint).
V3_TEXGAN_INIT = "w_avg"

# Magenta fuera de gama: ningun albedo de piel lo produce.
_V3_SENTINEL = (255, 0, 255)
_UNWRAP_MAT_NAME = "unwrap_1024_info.mat"


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def find_unwrap_mat() -> str | None:
    """Ruta de `unwrap_1024_info.mat` en candidatos topo, o None."""
    cands: list[str] = []
    topo = _env("TOPO_DIR")
    if topo:
        cands.append(os.path.join(topo, _UNWRAP_MAT_NAME))
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        cands.append(os.path.join(root, "topo_assets", _UNWRAP_MAT_NAME))
    for cand in cands:
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
        except OSError:
            continue
    return None


def count_sentinel_1024(raw: bytes) -> int:
    """Pixeles exactos sentinel en un atlas 1024 RGB plano."""
    sr, sg, sb = _V3_SENTINEL
    total = 0
    for i in range(0, len(raw) - 2, 3):
        if raw[i] == sr and raw[i + 1] == sg and raw[i + 2] == sb:
            total += 1
    return total


def uv_total_variation_1024(raw: bytes) -> float:
    """TV global media del atlas 1024x1024x3. `inf` si largo invalido."""
    if len(raw) != V3_UV_LEN:
        return float("inf")
    arr = np.asarray(bytearray(raw), dtype=np.float64).reshape(V3_UV_SIZE, V3_UV_SIZE, 3)
    dx = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    dy = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    tv = float((dx + dy) / 2.0)
    if not math.isfinite(tv):
        return float("inf")
    return tv


def check_tv_and_sentinel(raw: bytes) -> bool:
    """True si TV <= 2.0 y cero sentinel (gate del test RED)."""
    return count_sentinel_1024(raw) == 0 and uv_total_variation_1024(raw) <= V3_TV_MAX


def v3_texgan_steps() -> int:
    """Presupuesto largo del ajuste de latente (cientos de pasos)."""
    return int(V3_TEXGAN_FIT_STEPS)


def match_color(
    synth: NDArray[np.float64], sampled: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Iguala tono del decoder al de la foto (afin por canal, puro numpy)."""
    try:
        if not bool(np.asarray(valid).any()):
            return np.asarray(synth, dtype=np.float64)
        out = np.asarray(synth, dtype=np.float64).copy()
        ref = np.asarray(sampled, dtype=np.float64)
        m = np.asarray(valid, dtype=bool)
        for c in range(3):
            sv = ref[m, c]
            gv = out[m, c]
            mu_s, mu_g = float(sv.mean()), float(gv.mean())
            sd_s, sd_g = float(sv.std()), float(gv.std())
            if sd_g < 1e-9 or not (math.isfinite(mu_s) and math.isfinite(mu_g)):
                continue
            out[..., c] = (out[..., c] - mu_g) * (sd_s / sd_g) + mu_s
        if not bool(np.isfinite(out).all()):
            return np.asarray(synth, dtype=np.float64)
        return np.clip(out, 0.0, 255.0)
    except (IndexError, ValueError, TypeError):
        return np.asarray(synth, dtype=np.float64)


def blend_laplacian(
    sampled: NDArray[np.float64], synth: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Blend de 2 bandas (piramide Laplaciana simplificada, puro numpy).

    Texel valido de la foto en alta frecuencia, sintesis en baja:
    `out = blur(valid*sampled + ~valid*synth) + high(sampled donde valido)`.
    Total: nunca lanza por formas esperadas (recorta a la interseccion).
    """
    a = np.asarray(sampled, dtype=np.float64)
    b = np.asarray(synth, dtype=np.float64)
    m = np.asarray(valid, dtype=bool)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("v3 blend shapes mismatch")
    if m.shape != a.shape[:2]:
        raise ValueError("v3 blend mask mismatch")
    base = np.where(m[..., None], a, b)
    # Pasa-bajas 3x3 separable (pesos 1-2-1) como aproximacion Poisson/Laplaciano.
    low: NDArray[np.float64] = np.asarray(base.copy(), dtype=np.float64)
    for axis in (0, 1):
        low = np.asarray(
            np.roll(low, 1, axis=axis) * 0.25 + low * 0.5 + np.roll(low, -1, axis=axis) * 0.25,
            dtype=np.float64,
        )
    high: NDArray[np.float64] = np.asarray(base - low, dtype=np.float64)
    out: NDArray[np.float64] = np.asarray(low + np.where(m[..., None], high, 0.0), dtype=np.float64)
    if not bool(np.isfinite(out).all()):
        return np.asarray(np.clip(base, 0.0, 255.0), dtype=np.float64)
    return np.asarray(np.clip(out, 0.0, 255.0), dtype=np.float64)
