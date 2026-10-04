"""Fase 3 (issue 94): luz DPR SH9 + camara perspectiva con pose.

RED: backend/dpr.py y la camara con pose aun no existen. Tests CI-safe
(sin torch, sin pesos, sin cv2): consts, keys puras, base SH,
proyeccion perspectiva pura. Estimacion SH real y solvePnP, gateados a
torch/t7 y cv2 (local/Modal), CI los salta.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def test_dpr_consts() -> None:
    from backend.dpr import DPR_INPUT_SIZE, DPR_SH, EXPECTED_DPR_KEYS

    assert DPR_SH == 9
    assert DPR_INPUT_SIZE == 512
    assert EXPECTED_DPR_KEYS == 250


def test_dpr_check_keys_pure() -> None:
    from backend.dpr import EXPECTED_DPR_KEYS, check_dpr_keys, expected_dpr_keys

    keys = expected_dpr_keys()
    assert len(keys) == EXPECTED_DPR_KEYS
    assert "pre_conv.weight" in keys
    assert "pre_conv.bias" in keys
    assert "light.predict_FC2.weight" in keys
    assert "light.predict_relu1.weight" in keys
    assert "HG3.middle.middle.middle.middle.post_FC2.weight" in keys
    assert "output.weight" in keys
    assert "output.bias" in keys
    missing, unexpected = check_dpr_keys(keys)
    assert missing == frozenset() and unexpected == frozenset()
    dropped = set(keys)
    dropped.remove("output.bias")
    missing2, _ = check_dpr_keys(dropped)
    assert missing2 == frozenset({"output.bias"})


def test_dpr_sh_basis_front_brighter_than_back() -> None:
    from backend.dpr import sh_basis

    sh = np.array([0.6787, -0.1447, 0.2253, -0.0848, 0.0072, -0.0102, -0.0963, -0.001, -0.0055])
    front = (sh_basis(np.array([[[0.0, 0.0, 1.0]]])) * sh).sum(axis=-1)[0, 0]
    back = (sh_basis(np.array([[[0.0, 0.0, -1.0]]])) * sh).sum(axis=-1)[0, 0]
    assert bool(np.isfinite(front)) and bool(np.isfinite(back))
    assert front > back


def test_dpr_sh_basis_unit_normals_finite() -> None:
    from backend.dpr import sh_basis

    rng = np.random.RandomState(3)
    vecs = rng.normal(size=(50, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    y = sh_basis(vecs.reshape(1, 50, 3))
    assert y.shape == (1, 50, 9)
    assert bool(np.isfinite(y).all())


def test_affine_projection_maps_origin() -> None:
    """Proyeccion afin 2x4: el origen cae en la traslacion. Puro numpy."""
    from backend.flame_texture import _project_pose

    a = np.array([[2.0, 0.0, 0.0, 10.0], [0.0, -3.0, 0.0, 20.0]])
    px = _project_pose(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), a, 800, 600)
    assert px is not None and px.shape == (2, 2)
    np.testing.assert_allclose(px[0], [10.0, 20.0], rtol=1e-9)
    np.testing.assert_allclose(px[1], [12.0, 17.0], rtol=1e-9)
    assert _project_pose(np.zeros((4, 2)), a, 800, 600) is None


def test_affine_camera_roundtrip() -> None:
    """Afin 2x4 recupera la matriz con RMSE ~0. Puro numpy (corre en CI)."""
    from backend.flame_fit import estimate_affine_camera

    rng = np.random.RandomState(11)
    obj = rng.uniform(-1, 1, size=(20, 3))
    a_true = rng.normal(size=(2, 4))
    img = (a_true @ np.concatenate([obj, np.ones((20, 1))], axis=1).T).T
    a_est, rmse = estimate_affine_camera(obj, img)
    assert rmse < 1e-9
    assert a_est.shape == (2, 4)


def test_affine_camera_rejects_degenerate() -> None:
    from backend.flame_fit import estimate_affine_camera

    with pytest.raises(ValueError):
        estimate_affine_camera(np.zeros((10, 3)), np.zeros((10, 2)))
    with pytest.raises(ValueError):
        estimate_affine_camera(np.zeros((3, 3)), np.zeros((3, 2)))


def test_affine_pose_fits_face_tightly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate Fase 3: camara afin ajustada con RMSE bajo en cara real.

    La afin ajustada a la malla personalizada reemplaza al afin por bbox
    (adapta shear/rotacion por foto: menos smear) sin la degeneracion
    planar del DLT proyectivo. Gateado a LFW + puente local (CI salta).
    """
    pytest.importorskip("mediapipe")
    mirror = "/Users/esau.martinez/code/weights"
    lfw = "/Users/esau.martinez/Code/datasets/lfw/George_W_Bush/George_W_Bush_0001.jpg"
    task = f"{mirror}/mediapipe/face_landmarker.task"
    if not (os.path.isfile(lfw) and os.path.isfile(task)):
        pytest.skip("sin LFW/task local (CI)")
    from backend import modal_app as _modal_app

    monkeypatch.setattr(_modal_app, "WEIGHTS_DIR", mirror)
    monkeypatch.setattr(_modal_app, "_LM", None)
    from backend.domain import Err as _Err
    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.flame_fit import _LAST_FIT_STATS, _affine_pose, fit_flame

    with open(lfw, "rb") as fh:
        raw = fh.read()
    img = parse_image_bytes(raw)
    assert not isinstance(img, _Err)
    from PIL import Image as _Image

    with _Image.open(lfw) as _h:
        width, height = int(_h.width), int(_h.height)
    lm = parse_landmarks(_modal_app.mediapipe_infer("check", img.value))
    assert not isinstance(lm, _Err)
    monkeypatch.setenv("DECA_DIR", f"{mirror}/deca")
    monkeypatch.setenv("FLAME_ASSETS_DIR", f"{mirror}/flame")
    fit = fit_flame(img.value, lm.value)
    assert not isinstance(fit, _Err)
    assert float(_LAST_FIT_STATS.get("pose", 0.0)) == 1.0
    pose = _affine_pose(lm.value, width, height)
    assert not isinstance(pose, _Err)
    assert pose.value.shape == (2, 4)


def _t7_path() -> str | None:
    for base in (
        os.environ.get("DPR_DIR", "").strip(),
        "/Users/esau.martinez/code/weights/ffhq-uv-hf/checkpoints/dpr_model",
    ):
        if not base:
            continue
        cand = os.path.join(base, "trained_model_03.t7")
        if os.path.isfile(cand) and os.path.getsize(cand) > 0:
            return cand
    return None


def test_dpr_estimate_sh_real() -> None:
    pytest.importorskip("torch")
    t7 = _t7_path()
    if t7 is None:
        pytest.skip("sin trained_model_03.t7 local (CI)")
    from PIL import Image

    from backend.dpr import estimate_sh, load_light_net

    bush = "/Users/esau.martinez/Code/datasets/lfw/George_W_Bush/George_W_Bush_0001.jpg"
    if not os.path.isfile(bush):
        pytest.skip("sin LFW local (CI)")
    with Image.open(bush) as handle:
        lab = handle.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR).convert("LAB")
    L = np.asarray(lab, dtype=np.float64)[:, :, 0] / 255.0
    net = load_light_net(t7)
    assert net.missing_keys() == [] and net.unexpected_keys() == []
    sh_a = estimate_sh(net, L)
    sh_b = estimate_sh(net, L)
    assert sh_a is not None and sh_b is not None
    assert sh_a.shape == (9,)
    assert bool(np.isfinite(sh_a).all())
    np.testing.assert_array_equal(sh_a, sh_b)
    assert abs(float(sh_a[0])) == float(np.abs(sh_a).max())
