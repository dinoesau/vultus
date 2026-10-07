"""Confianza + feather + TexGAN largo (fix parche gris / dientes / TV).

Seams bajo test: `compute_texel_confidence` ([0,1] por angulo,
profundidad, piel, boca excluida), `soften_confidence` +
`blend_with_feather` (nunca `np.where` binario), latente largo
(`w_avg` + `fit_latent` enmascarado + `match_color`, presupuesto pineado),
log por job `texgan=` + conf + TV + latente.

Fuente independiente: numpy puro para propiedades + foto Bush 0001
(sha congelado) + checkpoint `texgan_ffhq_uv.pth` + parsing reales.
Sin puente: skip (CI), nunca verde con sinteticos visuales.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

PHOTO_SHA_FROZEN = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"


def test_confidence_range_grazing_depth_skin_mouth() -> None:
    from backend.flame_texture import compute_texel_confidence

    normals_front = np.zeros((8, 8, 3), dtype=np.float64)
    normals_front[..., 2] = 1.0
    depth_flat = np.zeros((8, 8), dtype=np.float64)
    px = np.stack(np.meshgrid(np.arange(8), np.arange(8)), axis=-1).astype(np.float64)
    covered = np.ones((8, 8), dtype=bool)
    conf = compute_texel_confidence(
        normals_front, depth_flat, px, covered, 8, 8, None, None
    )
    assert conf.shape == (8, 8)
    assert bool((conf >= 0.0).all() and (conf <= 1.0).all())
    assert float(conf.mean()) > 0.99

    grazing = np.zeros((8, 8, 3), dtype=np.float64)
    grazing[..., 0] = 1.0
    conf_graze = compute_texel_confidence(grazing, depth_flat, px, covered, 8, 8, None, None)
    assert float(conf_graze.max()) == 0.0

    depth_step = np.zeros((8, 8), dtype=np.float64)
    depth_step[:, 4:] = 5.0
    conf_depth = compute_texel_confidence(normals_front, depth_step, px, covered, 8, 8, None, None)
    assert float(conf_depth[:, 3].mean()) > float(conf_depth[:, 4].mean())

    skin = np.ones((8, 8), dtype=bool)
    skin[:, 4:] = False
    conf_skin = compute_texel_confidence(normals_front, depth_flat, px, covered, 8, 8, skin, None)
    assert float(conf_skin[0, 0]) == 1.0
    assert float(conf_skin[0, 7]) == 0.0

    mouth = np.zeros((8, 8), dtype=bool)
    mouth[3:5, 3:5] = True
    conf_mouth = compute_texel_confidence(normals_front, depth_flat, px, covered, 8, 8, None, mouth)
    assert float(conf_mouth[4, 4]) == 0.0
    assert float(conf_mouth[0, 0]) == 1.0

    uncovered = np.zeros((8, 8), dtype=bool)
    conf_uncovered = compute_texel_confidence(
        normals_front, depth_flat, px, uncovered, 8, 8, None, None
    )
    assert float(conf_uncovered.max()) == 0.0


def test_mouth_interior_mask_excludes_center() -> None:
    from backend.flame_texture import mouth_interior_mask

    pts = np.zeros((478, 3), dtype=np.float64)
    pts[61] = [0.3, 0.5, 0.0]
    pts[291] = [0.7, 0.5, 0.0]
    pts[13] = [0.5, 0.4, 0.0]
    pts[14] = [0.5, 0.6, 0.0]
    mask = mouth_interior_mask(100, 100, pts)
    assert mask.shape == (100, 100)
    assert bool(mask[50, 50])
    assert not bool(mask[0, 0])
    assert 0.0 < float(mask.mean()) < 0.2

    deg = mouth_interior_mask(0, 0, pts)
    assert deg.shape == (0, 0)


def test_feather_half_mask_tv_band_le_interior() -> None:
    from backend.flame_texture import (
        band_tv_512,
        blend_with_feather,
        seam_band_512,
        soften_confidence,
    )

    rng = np.random.RandomState(11)
    base = np.full((64, 64, 3), 180.0, dtype=np.float64)
    base += (rng.rand(64, 64, 3) - 0.5) * 4.0
    synth = np.full((64, 64, 3), 182.0, dtype=np.float64)
    synth += (rng.rand(64, 64, 3) - 0.5) * 4.0
    valid = np.zeros((64, 64), dtype=bool)
    valid[:, :32] = True
    conf = np.asarray(valid, dtype=np.float64)
    sharp = np.where(valid[..., None], base, synth)
    soft = soften_confidence(conf, 4)
    feathered = blend_with_feather(base, synth, soft)
    assert feathered.shape == (64, 64, 3)
    assert float(band_tv_512(feathered, valid)) <= float(band_tv_512(sharp, valid)) + 1e-9
    band = seam_band_512(valid)[..., None]
    interior = ~band & valid[..., None]
    assert bool(interior.any())
    np.testing.assert_array_equal(feathered[12:20, 8:16], base[12:20, 8:16])


def test_blend_with_feather_rejects_bad_shapes() -> None:
    import pytest

    from backend.flame_texture import blend_with_feather

    with pytest.raises(ValueError):
        blend_with_feather(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), np.zeros((5, 5)))
    with pytest.raises(ValueError):
        blend_with_feather(np.zeros((4, 4, 3)), np.zeros((5, 5, 3)), np.zeros((4, 4)))


def test_env_defaults_and_timeouts() -> None:
    from backend import modal_app
    from backend.flame_texture import (
        FEATHER_PX_DEFAULT,
        conf_depth_gap,
        conf_graze_max,
        conf_graze_min,
        conf_valid_min,
        feather_px,
    )
    from backend.texgan import TEXGAN_FIT_STEPS, fit_steps
    from backend.v3_contract import V3_TEXGAN_FIT_STEPS, V3_TEXTURE_TIMEOUT_SECS

    assert feather_px() == FEATHER_PX_DEFAULT == 4
    assert 0.0 <= conf_graze_min() < conf_graze_max() <= 1.0
    assert conf_depth_gap() > 0.0
    assert 0.0 <= conf_valid_min() <= 1.0
    assert 1 <= TEXGAN_FIT_STEPS <= 100
    assert 1 <= fit_steps() <= 100
    assert V3_TEXGAN_FIT_STEPS >= 100
    assert modal_app.TEXTURE_TIMEOUT_SECS == 30
    assert V3_TEXTURE_TIMEOUT_SECS >= 600


def test_branch_str_includes_conf_tv_latent() -> None:
    from backend import modal_app

    line = modal_app._texture_branch_str()
    assert "texgan=" in line
    assert "conf=" in line
    assert "tv=" in line
    assert "latent_ms=" in line


def _require_real_bridge(monkeypatch) -> dict[str, str] | None:  # type: ignore[no-untyped-def]
    import pytest as _pytest

    _pytest.importorskip("torch")
    mirror = "/Users/esau.martinez/Code/weights"
    lfw = "/Users/esau.martinez/Code/datasets/lfw"
    need = {
        "ffhq": f"{mirror}/ffhq-uv/FLAME_w_HIFI3D_UV.obj",
        "eye": f"{mirror}/ffhq-uv/eye_ball_tex.png",
        "unwrap": f"{mirror}/ffhq-uv-hf/topo_assets/unwrap_1024_info.mat",
        "texgan": f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model/texgan_ffhq_uv.pth",
        "photo": f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg",
        "task": f"{mirror}/mediapipe/face_landmarker.task",
        "parsing": f"{mirror}/ffhq-uv-hf/checkpoints/parsing_model/79999_iter.pth",
        "dpr": f"{mirror}/ffhq-uv-hf/checkpoints/dpr_model/trained_model_03.t7",
    }
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need.values()):
        return None
    monkeypatch.setenv("FFHQ_UV_DIR", f"{mirror}/ffhq-uv")
    monkeypatch.setenv("TOPO_DIR", f"{mirror}/ffhq-uv-hf/topo_assets")
    monkeypatch.setenv("TEXGAN_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model")
    monkeypatch.setenv("PARSING_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/parsing_model")
    monkeypatch.setenv("DPR_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/dpr_model")
    monkeypatch.setenv("TEXGAN_FIT_STEPS", "3")
    monkeypatch.delenv("WEIGHTS_ROOT", raising=False)
    monkeypatch.delenv("WEIGHTS_DIR", raising=False)
    from backend import modal_app as _modal_app

    monkeypatch.setattr(_modal_app, "WEIGHTS_DIR", mirror)
    monkeypatch.setattr(_modal_app, "_LM", None)
    return need


def test_real_unwrap_confidence_tv_texgan_deterministic(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import pytest as _pytest

    _pytest.importorskip("scipy.io")
    paths = _require_real_bridge(monkeypatch)
    if paths is None:
        _pytest.skip("sin puente textura real local (CI)")
    with open(paths["photo"], "rb") as fh:
        raw_photo = fh.read()
    assert hashlib.sha256(raw_photo).hexdigest() == PHOTO_SHA_FROZEN
    from backend import flame_texture as _texmod
    from backend.domain import Err as _Err
    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.flame_fit import fit_flame, landmark_points
    from backend.flame_texture import (
        _LAST_TEXTURE_STATS,
        count_sentinel,
        uv_total_variation,
    )

    _texmod._raster_cache = None
    img = parse_image_bytes(raw_photo)
    assert not isinstance(img, _Err)
    from backend import modal_app as _modal

    _modal._LM = None
    lm_raw = _modal.mediapipe_infer("check", img.value)
    lm = parse_landmarks(lm_raw)
    assert not isinstance(lm, _Err)
    pts = landmark_points(lm.value)
    assert not isinstance(pts, _Err)
    fit = fit_flame(img.value, lm.value)
    assert not isinstance(fit, _Err)
    from backend.flame_texture import run_unwrap_texture

    first = run_unwrap_texture(img.value, fit.value, lm.value)
    assert not isinstance(first, _Err)
    raw_a = first.value.as_bytes()
    assert count_sentinel(raw_a) == 0
    assert uv_total_variation(raw_a) <= 2.0
    assert float(_LAST_TEXTURE_STATS.get("texgan", 0.0)) == 1.0
    conf_mean = float(_LAST_TEXTURE_STATS.get("conf_mean", 0.0))
    assert 0.0 < conf_mean < 1.0
    assert float(_LAST_TEXTURE_STATS.get("tv", 0.0)) <= 2.0
    assert float(_LAST_TEXTURE_STATS.get("latent_ms", 0.0)) >= 0.0
    branch = _modal._texture_branch_str()
    assert "texgan=1" in branch and "conf=" in branch and "tv=" in branch
    second = run_unwrap_texture(img.value, fit.value, lm.value)
    assert not isinstance(second, _Err)
    assert second.value.as_bytes() == raw_a
    _texmod._raster_cache = None


def test_real_latent_descent_deterministic(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import pytest as _pytest

    torch = _pytest.importorskip("torch")
    _mediapipe = _pytest.importorskip("mediapipe")
    _ = (_mediapipe, torch)
    paths = _require_real_bridge(monkeypatch)
    if paths is None:
        _pytest.skip("sin puente textura real local (CI)")
    import hashlib as _hashlib

    with open(paths["photo"], "rb") as _fh:
        _raw = _fh.read()
    assert _hashlib.sha256(_raw).hexdigest() == PHOTO_SHA_FROZEN
    from PIL import Image as _Image

    from backend import modal_app as _modal
    from backend.domain import Err as _Err
    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.face_parsing import face_skin_mask
    from backend.flame_fit import landmark_points
    from backend.texgan import find_pth, load_decoder

    img = parse_image_bytes(_raw)
    assert not isinstance(img, _Err)
    lm = parse_landmarks(_modal.mediapipe_infer("descent", img.value))
    assert not isinstance(lm, _Err)
    pts = landmark_points(lm.value)
    assert not isinstance(pts, _Err)
    with _Image.open(paths["photo"]) as _handle:
        _rgb = _handle.convert("RGB")
        _w, _h = int(_rgb.width), int(_rgb.height)
    _xs = pts.value[:, 0] * float(_w)
    _ys = pts.value[:, 1] * float(_h)
    _pad = 0.15
    _x0 = max(0, int(_xs.min() - _pad * (_xs.max() - _xs.min())))
    _x1 = min(_w, int(_xs.max() + _pad * (_xs.max() - _xs.min())) + 1)
    _y0 = max(0, int(_ys.min() - _pad * (_ys.max() - _ys.min())))
    _y1 = min(_h, int(_ys.max() + _pad * (_ys.max() - _ys.min())) + 1)
    _crop = _rgb.crop((_x0, _y0, _x1, _y1)).resize((512, 512), _Image.Resampling.BILINEAR)
    sampled = np.asarray(_crop, dtype=np.float64)
    valid = np.asarray(face_skin_mask(sampled), dtype=bool)
    assert bool(valid.any())
    pth = find_pth()
    assert pth is not None
    dec = load_decoder(pth)
    assert dec.missing_keys() == [] and dec.unexpected_keys() == []
    tgt = torch.from_numpy(np.ascontiguousarray(sampled / 255.0)).permute(2, 0, 1).unsqueeze(0)
    mk = torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    def _mse(w: object) -> float:
        img01 = dec.synth_uv_map(w)
        arr = img01.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        full = np.rint(np.clip(np.asarray(arr, dtype=np.float64) * 255.0, 0.0, 255.0)).astype(np.uint8)
        small_synth = np.asarray(
            _Image.fromarray(full).resize((512, 512), _Image.Resampling.BILINEAR), dtype=np.float64
        )
        m = valid
        diff = (small_synth[m] - sampled[m]).reshape(-1)
        return float((diff * diff).mean())

    few_w = dec.fit_latent(tgt, mk, steps=2)
    many_w = dec.fit_latent(tgt, mk, steps=6)
    assert _mse(many_w) < _mse(few_w)
    assert torch.equal(dec.fit_latent(tgt, mk, steps=2), few_w)
