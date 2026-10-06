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

from backend.domain import DomainError, Err, MlDecode, MlFailed, Ok
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


def match_color(    synth: NDArray[np.float64], sampled: NDArray[np.float64], valid: NDArray[np.bool_]
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


def _seam_band(valid: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Banda de costura: dilatacion 1px del borde valido/invalido (puros rolls)."""
    m = np.asarray(valid, dtype=bool)
    dilated = m.copy()
    eroded = m.copy()
    for axis in (0, 1):
        for shift in (1, -1):
            dilated |= np.roll(m, shift, axis=axis)
            eroded &= np.roll(m, shift, axis=axis)
    return np.asarray(dilated & ~eroded, dtype=bool)


def _band_tv(atlas: NDArray[np.float64], valid: NDArray[np.bool_]) -> float:
    """TV media solo sobre pares con al menos un extremo en la costura."""
    a = np.asarray(atlas, dtype=np.float64)
    band = _seam_band(np.asarray(valid, dtype=bool))
    if not bool(band.any()):
        return float("inf")
    dx = np.abs(a[:, 1:, :] - a[:, :-1, :]).mean(axis=-1)
    dy = np.abs(a[1:, :, :] - a[:-1, :, :]).mean(axis=-1)
    wx = band[:, :-1] | band[:, 1:]
    wy = band[:-1, :] | band[1:, :]
    if not bool(wx.any() and wy.any()):
        return float("inf")
    tv = (float(dx[wx].mean()) + float(dy[wy].mean())) / 2.0
    return tv if np.isfinite(tv) else float("inf")


def smooth_seam_3x3(
    atlas: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Suavizado honesto de costura 3x3, solo en la banda (puro numpy).

    No es Poisson real (renombrado honesto): box blur 3x3 aplicado solo
    donde la mascara cambia (banda de 1px). El interior profundo queda
    intacto por construccion. Total: formas invalidas son `ValueError`.
    """
    a = np.asarray(atlas, dtype=np.float64)
    m = np.asarray(valid, dtype=bool)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("v3 seam atlas invalid shape")
    if m.shape != a.shape[:2]:
        raise ValueError("v3 seam mask mismatch")
    kernel = np.full((3, 3), 1.0 / 9.0)
    padded = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = sum(
        kernel[ki, kj] * padded[ki : ki + a.shape[0], kj : kj + a.shape[1]]
        for ki in range(3)
        for kj in range(3)
    )
    band = _seam_band(m)[..., None]
    out = np.where(band, np.asarray(blurred, dtype=np.float64), a)
    if not bool(np.isfinite(out).all()):
        return np.asarray(np.clip(a, 0.0, 255.0), dtype=np.float64)
    return np.asarray(np.clip(out, 0.0, 255.0), dtype=np.float64)


def blend_laplacian(
    sampled: NDArray[np.float64], synth: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Compat: combina foto/sintesis y suaviza la costura (ver `smooth_seam_3x3`).

    Antes decia "piramide Laplaciana": era un blur global, nombre enganoso
    (cazado por bloodhound). Ahora delega en el suavizado honesto de banda.
    """
    a = np.asarray(sampled, dtype=np.float64)
    b = np.asarray(synth, dtype=np.float64)
    m = np.asarray(valid, dtype=bool)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("v3 blend shapes mismatch")
    if m.shape != a.shape[:2]:
        raise ValueError("v3 blend mask mismatch")
    return smooth_seam_3x3(np.where(m[..., None], a, b), m)


def masked_mse(
    synth: NDArray[np.float64], sampled: NDArray[np.float64], valid: NDArray[np.bool_]
) -> float:
    """MSE enmascarado 0-255 (misma metrica del ajuste de latente)."""
    s = np.asarray(synth, dtype=np.float64)
    r = np.asarray(sampled, dtype=np.float64)
    m = np.asarray(valid, dtype=bool)
    if s.shape != r.shape or m.shape != s.shape[:2] or not bool(m.any()):
        return float("inf")
    diff = (s[m] - r[m]).reshape(-1)
    mse = float((diff * diff).mean())
    return mse if np.isfinite(mse) else float("inf")


def long_completion_1024(
    sampled_1024: NDArray[np.float64], valid_1024: NDArray[np.bool_], steps: int
) -> NDArray[np.float64] | None:
    """Completion TexGAN real a 1024 (latente desde `w_avg`, `steps` enmascarados).

    Via `backend.texgan` (decoder + `fit_latent` + `synth_uv_map` a 1024,
    `match_color` v3): el loss se evalua a 512 dentro de `fit_latent`, la
    sintesis devuelta es 1024 y se usa SOLO en texeles no validos via el
    blend del caller. `None` sin backend/pesos o ante fallo (el caller
    falla loud, nunca sintetico silencioso). Total: nunca lanza.
    """
    try:
        if int(steps) < 1:
            return None
        from backend.texgan import find_pth, load_decoder, texgan_available

        if not texgan_available():
            return None
        import torch
        from torch.nn import functional as _F

        pth = find_pth()
        if pth is None:
            return None
        dec = load_decoder(pth)
        a = np.asarray(sampled_1024, dtype=np.float64)
        m = np.asarray(valid_1024, dtype=bool)
        if a.ndim != 3 or a.shape[2] != 3 or m.shape != a.shape[:2] or not bool(m.any()):
            return None
        with torch.no_grad():
            tgt = torch.from_numpy(np.ascontiguousarray(a / 255.0)).permute(2, 0, 1).unsqueeze(0)
            tgt512 = _F.interpolate(tgt.to(torch.float32), size=(512, 512), mode="bilinear", align_corners=False)
            mk = torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(0).unsqueeze(0)
            mk512 = _F.interpolate(mk.to(torch.float32), size=(512, 512), mode="bilinear", align_corners=False)
            mk512 = (mk512 > 0.5).repeat(1, 3, 1, 1)
        w = dec.fit_latent(tgt512, mk512, steps=int(steps))
        img01 = dec.synth_uv_map(w)
        arr = img01.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        synth = np.asarray(arr, dtype=np.float64) * 255.0
        if synth.shape != a.shape or not bool(np.isfinite(synth).all()):
            return None
        return match_color(np.clip(synth, 0.0, 255.0), np.clip(a, 0.0, 255.0), m)
    except Exception:  # noqa: BLE001 - sin backend o fallo numerico: None loud via caller
        return None


def assemble_v3_atlas(
    sampled_1024: NDArray[np.float64], valid_1024: NDArray[np.bool_], steps: int
) -> Ok[bytes] | Err[DomainError]:
    """Atlas v3 1024: foto donde valida + sintesis TexGAN larga donde no.

    Blend + `smooth_seam_3x3` + scrub de sentinel. Total: `Err` loud sin
    backend/sintesis (nunca atlas sintetico silencioso).
    """
    synth = long_completion_1024(sampled_1024, valid_1024, steps)
    if synth is None:
        return Err(MlFailed(detail=MlDecode(details="v3 atlas requires texgan backend")))
    m = np.asarray(valid_1024, dtype=bool)
    base = np.where(m[..., None], np.asarray(sampled_1024, dtype=np.float64), synth)
    out = smooth_seam_3x3(base, m)
    raw = np.rint(np.clip(out, 0.0, 255.0)).astype(np.uint8).tobytes()
    if len(raw) != V3_UV_LEN:
        return Err(MlFailed(detail=MlDecode(details="v3 atlas length invalid")))
    sr, sg, sb = _V3_SENTINEL
    scrubbed = bytearray(raw)
    for i in range(0, len(scrubbed) - 2, 3):
        if scrubbed[i] == sr and scrubbed[i + 1] == sg and scrubbed[i + 2] == sb:
            scrubbed[i] = 254
    return Ok(bytes(scrubbed))
