"""Textura v3 real: latente TexGAN real + Poisson honesto + TV sobre atlas real.

Seam bajo test: `backend.v3_texture.long_completion_1024` (latente real
desde `w_avg`, cientos de pasos enmascarados, solo texeles no validos) +
`smooth_seam_3x3` (suavizado honesto de banda de costura, con test de
propiedad) + `match_color`, sobre la foto Bush 0001 (sha congelado) con
mascara de piel real (`face_skin_mask`).

Fuente independiente de verdad: pixeles + checkpoint `texgan_ffhq_uv.pth`
+ parsing reales. Test RED: TV <= 2.0 y cero sentinel sobre el atlas
real resultante. Sin puente: skip (CI), nunca verde con sinteticos.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

PHOTO_SHA_FROZEN = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"


def _require_texture_bridge(monkeypatch: pytest.MonkeyPatch) -> dict[str, str] | None:
    mirror = "/Users/esau.martinez/code/weights"
    lfw = "/Users/esau.martinez/Code/datasets/lfw"
    need = {
        "texgan": f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model/texgan_ffhq_uv.pth",
        "unwrap": f"{mirror}/ffhq-uv-hf/topo_assets/unwrap_1024_info.mat",
        "task": f"{mirror}/mediapipe/face_landmarker.task",
        "photo": f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg",
    }
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need.values()):
        return None
    monkeypatch.setenv("TEXGAN_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model")
    monkeypatch.setenv("TOPO_DIR", f"{mirror}/ffhq-uv-hf/topo_assets")
    monkeypatch.setenv("PARSING_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/parsing_model")
    monkeypatch.setenv("WEIGHTS_DIR", mirror)
    monkeypatch.delenv("VULTUS_REAL_ML", raising=False)
    from backend import modal_app as _modal_app

    monkeypatch.setattr(_modal_app, "WEIGHTS_DIR", mirror)
    monkeypatch.setattr(_modal_app, "_LM", None)
    return need


def _real_sampled_valid_1024(
    monkeypatch: pytest.MonkeyPatch, photo_path: str
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    from backend import modal_app
    from backend.domain import Err as _Err
    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.face_parsing import face_skin_mask
    from backend.flame_fit import landmark_points

    with open(photo_path, "rb") as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == PHOTO_SHA_FROZEN
    img = parse_image_bytes(raw)
    assert not isinstance(img, _Err)
    lm = parse_landmarks(modal_app.mediapipe_infer("check", img.value))
    assert not isinstance(lm, _Err)
    pts = landmark_points(lm.value)
    assert not isinstance(pts, _Err)
    with Image.open(photo_path) as handle:
        photo_rgb = handle.convert("RGB")
        width, height = int(photo_rgb.width), int(photo_rgb.height)
    xs = pts.value[:, 0] * float(width)
    ys = pts.value[:, 1] * float(height)
    pad = 0.15
    x0 = max(0, int(xs.min() - pad * (xs.max() - xs.min())))
    x1 = min(width, int(xs.max() + pad * (xs.max() - xs.min())) + 1)
    y0 = max(0, int(ys.min() - pad * (ys.max() - ys.min())))
    y1 = min(height, int(ys.max() + pad * (ys.max() - ys.min())) + 1)
    assert x1 > x0 and y1 > y0
    crop = photo_rgb.crop((x0, y0, x1, y1)).resize((1024, 1024), Image.Resampling.BILINEAR)
    sampled = np.asarray(crop, dtype=np.float64)
    valid = face_skin_mask(sampled)
    assert valid.shape == (1024, 1024) and bool(valid.any())
    return sampled, np.asarray(valid, dtype=bool)


def test_v3_texgan_long_schedule_is_real_descent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule largo real: el optimizador desciende el MSE enmascarado."""
    pytest.importorskip("torch")
    pytest.importorskip("mediapipe")
    paths = _require_texture_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente textura real local (CI)")
    import torch

    from backend.texgan import find_pth, load_decoder
    from backend.v3_contract import V3_TEXGAN_FIT_STEPS

    # El path productivo usa cientos de pasos (26 min en CPU local: el test
    # verifica la constante + descenso real + determinismo a pocos pasos).
    assert V3_TEXGAN_FIT_STEPS >= 100
    from PIL import Image

    from backend.v3_texture import long_completion_1024

    sampled, valid = _real_sampled_valid_1024(monkeypatch, paths["photo"])
    dec = load_decoder(find_pth())
    assert dec.missing_keys() == [] and dec.unexpected_keys() == []
    small = Image.fromarray(np.rint(np.clip(sampled, 0.0, 255.0)).astype(np.uint8)).resize(
        (512, 512), Image.Resampling.BILINEAR
    )
    tgt = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mk = torch.from_numpy(
        
            np.asarray(
                Image.fromarray((valid.astype(np.uint8)) * 255).resize(
                    (512, 512), Image.Resampling.BILINEAR
                ),
                dtype=np.float64,
            )
            > 127.5
        
    ).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    def _mse512(w: object) -> float:
        img01 = dec.synth_uv_map(w)
        arr = img01.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        s = Image.fromarray(np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
            (512, 512), Image.Resampling.BILINEAR
        )
        a = np.asarray(s, dtype=np.float64)
        b = np.asarray(small, dtype=np.float64)
        m = np.asarray(mk[0, 0].cpu().numpy(), dtype=bool)
        diff = (a[m] - b[m]).reshape(-1)
        return float((diff * diff).mean())

    few_w = dec.fit_latent(tgt, mk, steps=2)
    many_w = dec.fit_latent(tgt, mk, steps=6)
    assert _mse512(many_w) < _mse512(few_w)
    # Init fijo en w_avg: mismo input, misma salida (sin muestreo).
    assert torch.equal(dec.fit_latent(tgt, mk, steps=2), few_w)
    # El wrapper v3 entrega 1024 finito sobre los mismos datos reales.
    out = long_completion_1024(sampled, valid, steps=2)
    assert out is not None and out.shape == (1024, 1024, 3) and bool(np.isfinite(out).all())


def test_v3_smooth_seam3x3_property() -> None:
    """Propiedad: la banda de costura baja su TV y el interior queda intacto."""
    from backend.v3_texture import smooth_seam_3x3

    rng = np.random.RandomState(9)
    atlas = rng.rand(64, 64, 3).astype(np.float64) * 255.0
    valid = np.zeros((64, 64), dtype=bool)
    valid[8:56, 8:56] = True
    out = smooth_seam_3x3(atlas, valid)
    assert out.shape == (64, 64, 3) and bool(np.isfinite(out).all())
    assert float(np.abs(out).max()) <= 255.0
    # Interior profundo (>=2px de la costura) intacto.
    np.testing.assert_array_equal(out[12:52, 12:52], atlas[12:52, 12:52])
    # La banda de costura (dilatada 1px) no aumenta su TV.
    from backend.v3_texture import _band_tv

    assert _band_tv(out, valid) <= _band_tv(atlas, valid) + 1e-9


def test_v3_real_atlas_tv_and_zero_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Atlas real Bush: TV <= 2.0 y cero sentinel tras completion + blend."""
    pytest.importorskip("torch")
    pytest.importorskip("mediapipe")
    paths = _require_texture_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente textura real local (CI)")
    from backend.domain import Err as _ErrA
    from backend.v3_texture import (
        assemble_v3_atlas,
        count_sentinel_1024,
        uv_total_variation_1024,
    )

    sampled, valid = _real_sampled_valid_1024(monkeypatch, paths["photo"])
    result = assemble_v3_atlas(sampled, valid, steps=6)
    assert not isinstance(result, _ErrA)
    raw = result.value
    assert len(raw) == 1024 * 1024 * 3
    assert count_sentinel_1024(raw) == 0
    assert uv_total_variation_1024(raw) <= 2.0
