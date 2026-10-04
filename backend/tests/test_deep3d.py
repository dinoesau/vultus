"""Fase 2 (issue 94): forma real Deep3D-HiFi3D++ en `displaced_positions`.

RED: backend/deep3d.py aun no existe. Tests CI-safe (sin torch, sin
pesos, sin scipy): consts del layout 1049->253, keys del checkpoint
(puro python), split/encode puros y asset de transferencia (solo
numpy). Carga estricta + forward + personalizacion, gateados a
torch/pesos/mat (local/Modal), CI los salta.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def test_deep3d_layout_consts() -> None:
    from backend.deep3d import (
        DEEP3D_COEFF,
        DEEP3D_EXP,
        DEEP3D_ID,
        DEEP3D_TEX,
        FIT_ANGLE,
        FIT_EXP,
        FIT_ID,
        FIT_SPARE,
        FIT_TRANS,
    )

    assert (DEEP3D_ID, DEEP3D_EXP, DEEP3D_TEX) == (532, 45, 439)
    assert DEEP3D_COEFF == 532 + 45 + 439 + 3 + 27 + 2 + 1
    assert (FIT_ID, FIT_EXP, FIT_ANGLE, FIT_TRANS, FIT_SPARE) == (200, 45, 3, 3, 2)
    assert FIT_ID + FIT_EXP + FIT_ANGLE + FIT_TRANS + FIT_SPARE == 253


def test_deep3d_expected_keys() -> None:
    from backend.deep3d import check_net_recon_keys, expected_net_recon_keys

    keys = expected_net_recon_keys()
    assert len(keys) == 332
    assert "conv1.weight" in keys
    assert "layer4.2.bn3.running_mean" in keys
    assert "final_layers.0.weight" in keys
    assert "final_layers.1.0.weight" in keys
    assert "final_layers.6.bias" in keys
    missing, unexpected = check_net_recon_keys(keys)
    assert missing == frozenset() and unexpected == frozenset()
    dropped = set(keys)
    dropped.remove("final_layers.0.weight")
    missing2, _ = check_net_recon_keys(dropped)
    assert missing2 == frozenset({"final_layers.0.weight"})


def test_deep3d_split_encode_pure() -> None:
    from backend.deep3d import encode_fit253, split_coeff_vector

    vec = np.arange(1049, dtype=np.float64)
    parts = split_coeff_vector(vec)
    assert parts["id"].shape == (532,)
    assert parts["exp"].shape == (45,)
    assert parts["angle"].shape == (3,)
    assert parts["gamma"].shape == (27,)
    assert parts["trans"].shape == (3,)
    assert float(parts["id"][0]) == 0.0
    assert float(parts["trans"][-1]) == 1048.0
    out = encode_fit253(parts["id"], parts["exp"], parts["angle"], parts["trans"])
    assert len(out) == 253
    assert list(out[:3]) == [0.0, 1.0, 2.0]
    assert list(out[200:203]) == [532.0, 533.0, 534.0]
    assert list(out[245:248]) == [1016.0, 1017.0, 1018.0]
    assert list(out[248:251]) == [1046.0, 1047.0, 1048.0]
    assert list(out[251:253]) == [0.0, 0.0]
    with pytest.raises(ValueError):
        encode_fit253(np.zeros(10), parts["exp"], parts["angle"], parts["trans"])


def test_deep3d_transfer_asset_numpy_only() -> None:
    from backend.deep3d import TRANSFER_NAME, find_transfer

    path = find_transfer()
    assert path is not None, TRANSFER_NAME
    z = np.load(path)
    idx, w, scale = z["idx"], z["w"], z["scale"]
    assert idx.shape == (5023, 3) and idx.min() >= 0 and idx.max() < 20481
    assert w.shape == (5023, 3) and bool(((w >= 0.0) & (w <= 1.0)).all())
    np.testing.assert_allclose(w.sum(axis=1), np.ones(5023), rtol=1e-5)
    assert scale.shape == (3,) and bool((scale > 0.0).all())


def _bridge_env() -> dict[str, str]:
    mirror = "/Users/esau.martinez/code/weights"
    repo_weights = "/Users/esau.martinez/Code/vultus/weights"
    return {
        "DEEP3D_DIR": f"{mirror}/ffhq-uv-hf/checkpoints/deep3d_model",
        "TOPO_DIR": f"{mirror}/ffhq-uv-hf/topo_assets",
        "FLAME_ASSETS_DIR": repo_weights + "/flame",
    }


def _require_real(monkeypatch: pytest.MonkeyPatch) -> str | None:
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    for k, v in _bridge_env().items():
        monkeypatch.setenv(k, v)
    epoch = os.path.join(_bridge_env()["DEEP3D_DIR"], "epoch_latest.pth")
    mat = os.path.join(_bridge_env()["TOPO_DIR"], "hifi3dpp_model_info.mat")
    if not (os.path.isfile(epoch) and os.path.isfile(mat)):
        return None
    return epoch


def test_deep3d_strict_load_real_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    epoch = _require_real(monkeypatch)
    if epoch is None:
        pytest.skip("sin epoch_latest.pth/mat local (CI)")
    from backend.deep3d import load_recon

    recon = load_recon(epoch)
    assert recon.missing_keys() == [] and recon.unexpected_keys() == []


def test_deep3d_forward_deterministic_and_personalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    epoch = _require_real(monkeypatch)
    if epoch is None:
        pytest.skip("sin epoch_latest.pth/mat local (CI)")
    import torch

    from backend.deep3d import encode_fit253, load_recon, split_coeff_vector

    recon = load_recon(epoch)
    rng = np.random.RandomState(7)
    face_a = (rng.rand(224, 224, 3) * 0.5 + 0.25).astype(np.float32)
    face_b = np.clip(face_a + rng.normal(0, 0.05, face_a.shape), 0, 1).astype(np.float32)
    xa = torch.from_numpy(face_a).permute(2, 0, 1).unsqueeze(0)
    xb = torch.from_numpy(face_b).permute(2, 0, 1).unsqueeze(0)
    ca = recon.forward_coeffs(xa).numpy().reshape(-1)
    ca2 = recon.forward_coeffs(xa).numpy().reshape(-1)
    cb = recon.forward_coeffs(xb).numpy().reshape(-1)
    assert ca.shape == (1049,)
    assert bool(np.isfinite(ca).all())
    np.testing.assert_array_equal(ca, ca2)
    assert float(np.linalg.norm(ca - cb)) > 0.0
    pa, pb = split_coeff_vector(ca), split_coeff_vector(cb)
    fa = encode_fit253(pa["id"], pa["exp"], pa["angle"], pa["trans"])
    fb = encode_fit253(pb["id"], pb["exp"], pb["angle"], pb["trans"])
    assert len(fa) == 253 and len(fb) == 253


def test_deep3d_dense_reconstruction_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate Fase 4: la malla densa HiFi3D++ existe y es determinista.

    20481 verts + head_tri 40832 del .mat, finita, x2 identica. Solo
    para la decision de topologia (el contrato sirve FLAME 5023).
    Gateado a torch/mat (local/Modal), CI lo salta.
    """
    epoch = _require_real(monkeypatch)
    if epoch is None:
        pytest.skip("sin epoch_latest.pth/mat local (CI)")
    import torch

    from backend.deep3d import (
        load_hifi_basis,
        load_recon,
        reconstruct_dense,
        split_coeff_vector,
    )

    mirror = "/Users/esau.martinez/code/weights"
    monkeypatch.setenv("TOPO_DIR", f"{mirror}/ffhq-uv-hf/topo_assets")
    recon = load_recon(epoch)
    rng = np.random.RandomState(7)
    face = (rng.rand(224, 224, 3) * 0.5 + 0.25).astype(np.float32)
    x = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
    vec = recon.forward_coeffs(x).numpy().reshape(-1)
    parts = split_coeff_vector(vec)
    first = reconstruct_dense(parts["id"], parts["exp"])
    second = reconstruct_dense(parts["id"], parts["exp"])
    assert first is not None and second is not None
    id_shape, exp_shape = first
    assert id_shape.shape == (20481, 3) and exp_shape.shape == (20481, 3)
    assert bool(np.isfinite(id_shape).all() and np.isfinite(exp_shape).all())
    np.testing.assert_array_equal(id_shape, second[0])
    np.testing.assert_array_equal(exp_shape, second[1])
    basis = load_hifi_basis()
    assert basis is not None
    assert basis["head_tri"].shape == (40832, 3)


def _lfw_bridge(monkeypatch: pytest.MonkeyPatch) -> dict[str, str] | None:
    mirror = "/Users/esau.martinez/code/weights"
    repo_weights = "/Users/esau.martinez/Code/vultus/weights"
    lfw = "/Users/esau.martinez/Code/datasets/lfw"
    need = [
        f"{mirror}/ffhq-uv-hf/checkpoints/deep3d_model/epoch_latest.pth",
        f"{mirror}/ffhq-uv-hf/topo_assets/hifi3dpp_model_info.mat",
        f"{mirror}/deca/deca_model.tar",
        f"{mirror}/flame/flame2023_Open.pkl",
        f"{mirror}/mediapipe/face_landmarker.task",
        f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg",
    ]
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need):
        return None
    monkeypatch.setenv("DEEP3D_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/deep3d_model")
    monkeypatch.setenv("TOPO_DIR", f"{mirror}/ffhq-uv-hf/topo_assets")
    monkeypatch.setenv("DECA_DIR", f"{mirror}/deca")
    monkeypatch.setenv("FLAME_ASSETS_DIR", f"{mirror}/flame")
    monkeypatch.setenv("WEIGHTS_DIR", mirror)
    monkeypatch.delenv("VULTUS_REAL_ML", raising=False)
    # modal_app congela WEIGHTS_DIR a import-time: parchar la const para
    # hermeticidad (en suite completa el import ocurre en collection con
    # env vacio) y resetear el singleton del landmarker.
    from backend import modal_app as _modal_app

    monkeypatch.setattr(_modal_app, "WEIGHTS_DIR", mirror)
    monkeypatch.setattr(_modal_app, "_LM", None)
    return {"mirror": mirror, "lfw": lfw, "repo_weights": repo_weights}


def _real_fit_for(monkeypatch: pytest.MonkeyPatch, jpg_path: str):
    from backend import modal_app
    from backend.domain import Err as _Err
    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.flame_fit import fit_flame

    with open(jpg_path, "rb") as fh:
        raw = fh.read()
    img = parse_image_bytes(raw)
    assert not isinstance(img, _Err)
    lm = parse_landmarks(modal_app.mediapipe_infer("check", img.value))
    assert not isinstance(lm, _Err)
    fit = fit_flame(img.value, lm.value)
    assert not isinstance(fit, _Err)
    return img.value, fit.value, lm.value


def test_deep3d_real_fits_margin_same_lt_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate Fase 2: fit real con landmarks reales, misma < distinta."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _lfw_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente LFW local (CI)")
    lfw = paths["lfw"]
    from backend.flame_fit import _LAST_FIT_STATS, flame_distance

    _, fit_a, _ = _real_fit_for(monkeypatch, f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg")
    _, fit_b, _ = _real_fit_for(monkeypatch, f"{lfw}/George_W_Bush/George_W_Bush_0002.jpg")
    _, fit_c, _ = _real_fit_for(monkeypatch, f"{lfw}/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    assert float(_LAST_FIT_STATS.get("deep3d", 0.0)) == 1.0
    d_same = flame_distance(fit_a.coeffs, fit_b.coeffs)
    d_diff = flame_distance(fit_a.coeffs, fit_c.coeffs)
    assert d_same < d_diff


def test_deep3d_displaced_personalizes_and_stays_smooth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate Fase 2: malla personalizada por persona, sin picos >5mm."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _lfw_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente LFW local (CI)")
    lfw = paths["lfw"]
    from backend.gnm_assemble import displaced_positions, load_flame_template

    _, fit_a, _ = _real_fit_for(monkeypatch, f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg")
    _, fit_c, _ = _real_fit_for(monkeypatch, f"{lfw}/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    from backend.domain import Err as _Err

    da = displaced_positions(fit_a)
    dc = displaced_positions(fit_c)
    assert not isinstance(da, _Err) and not isinstance(dc, _Err)
    pa = np.array(da.value)
    pc = np.array(dc.value)
    assert pa.shape == (5023, 3) and pc.shape == (5023, 3)
    assert float(np.abs(pa - pc).max()) > 1e-6
    tpl = load_flame_template()
    assert not isinstance(tpl, _Err)
    _, _, tris = tpl.value
    deltas = pa - np.array([p for p in tpl.value[0]])
    peak = 0.0
    for a, b, c in tris:
        peak = max(peak, float(np.abs(deltas[a] - deltas[b]).max()), float(np.abs(deltas[b] - deltas[c]).max()))
    assert peak <= 0.005 + 1e-9
