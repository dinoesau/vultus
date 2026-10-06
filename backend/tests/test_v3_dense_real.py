"""Densa v3 real: `.mat` + ReconNet + foto con sha congelado, nunca sinteticos.

Seam bajo test: `backend.v3_dense.load_hifi_basis_from` + `compute_shape_numpy`
+ `add_eyeballs` sobre base HiFi3D++ real y coefs ReconNet reales de la foto
Bush 0001 (sha congelado, misma fuente que `scripts/e2e-flame-real.py`).

Fuente independiente de verdad: bytes congelados upstream (`.mat` sha
`9b501bb8...`, foto `SHA_A`, checkpoint `epoch_latest.pth`) + formula
upstream (`mean + idBase@id + exBase@exp`). El test recomputa por via
directa (scipy crudo + matmul en el cuerpo) y exige igualdad con el modulo:
detecta drift del loader. Sin puente local el test hace skip (CI), nunca
verde falso con sinteticos.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

# Bytes congelados: verificados localmente contra mirror de solo lectura.
MAT_SHA_FROZEN = "9b501bb8a1c38d65a1706add89a9418c7fe7b729bcc5e5ff38e675a61b57963b"
PHOTO_SHA_FROZEN = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"


def _require_v3_dense_bridge(monkeypatch: pytest.MonkeyPatch) -> dict[str, str] | None:
    mirror = "/Users/esau.martinez/code/weights"
    lfw = "/Users/esau.martinez/Code/datasets/lfw"
    need = {
        "epoch": f"{mirror}/ffhq-uv-hf/checkpoints/deep3d_model/epoch_latest.pth",
        "mat": f"{mirror}/ffhq-uv-hf/topo_assets/hifi3dpp_model_info.mat",
        "task": f"{mirror}/mediapipe/face_landmarker.task",
        "photo": f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg",
    }
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need.values()):
        return None
    monkeypatch.setenv("DEEP3D_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/deep3d_model")
    monkeypatch.setenv("TOPO_DIR", f"{mirror}/ffhq-uv-hf/topo_assets")
    monkeypatch.setenv("WEIGHTS_DIR", mirror)
    monkeypatch.delenv("VULTUS_REAL_ML", raising=False)
    from backend import modal_app as _modal_app

    monkeypatch.setattr(_modal_app, "WEIGHTS_DIR", mirror)
    monkeypatch.setattr(_modal_app, "_LM", None)
    return need


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_v3_dense_mat_bytes_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """El `.mat` es el byte congelado upstream, no un sustituto."""
    paths = _require_v3_dense_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente .mat local (CI)")
    assert _sha256_file(paths["mat"]) == MAT_SHA_FROZEN


def _real_id_exp(monkeypatch: pytest.MonkeyPatch, photo_path: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from backend import modal_app
    from backend.deep3d import (
        face_crop224,
        find_epoch,
        load_recon,
        photo_array,
        split_coeff_vector,
    )
    from backend.domain import Err as _Err
    from backend.domain import parse_image_bytes, parse_landmarks
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
    photo = photo_array(img.value)
    assert not isinstance(photo, _Err)
    crop = face_crop224(photo.value, pts.value[:, 0], pts.value[:, 1])
    epoch = find_epoch()
    assert epoch is not None
    recon = load_recon(epoch)
    assert recon.missing_keys() == [] and recon.unexpected_keys() == []
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = recon.forward_coeffs(tensor)
    vec = np.asarray(out.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    parts = split_coeff_vector(vec)
    assert parts["id"].shape == (532,) and parts["exp"].shape == (45,)
    assert bool(np.isfinite(parts["id"]).all() and np.isfinite(parts["exp"]).all())
    return parts["id"], parts["exp"]


def test_v3_dense_from_real_photo_finite_deterministic_x2(monkeypatch: pytest.MonkeyPatch) -> None:
    """`compute_shape` con base real + coefs ReconNet reales: 20481v finita x2."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _require_v3_dense_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente denso real local (CI)")
    from scipy.io import loadmat

    from backend.domain import Err as _Err
    from backend.v3_dense import compute_shape_numpy, load_hifi_basis_from

    basis = load_hifi_basis_from(paths["mat"])
    assert not isinstance(basis, _Err)
    mean, id_base, ex_base, head_tri = basis.value.mean, basis.value.id_base, basis.value.ex_base, basis.value.head_tri
    assert mean.shape == (20481, 3)
    assert id_base.shape == (61443, 532)
    assert ex_base.shape == (61443, 45)
    assert head_tri.shape == (40832, 3)
    id_coeffs, exp_coeffs = _real_id_exp(monkeypatch, paths["photo"])
    first = compute_shape_numpy(mean, id_base, ex_base, id_coeffs, exp_coeffs)
    second = compute_shape_numpy(mean, id_base, ex_base, id_coeffs, exp_coeffs)
    assert not isinstance(first, _Err) and not isinstance(second, _Err)
    face, neutral = first.value
    assert face.shape == (20481, 3) and neutral.shape == (20481, 3)
    assert bool(np.isfinite(face).all() and np.isfinite(neutral).all())
    np.testing.assert_array_equal(face, second.value[0])
    np.testing.assert_array_equal(neutral, second.value[1])
    # Via directa independiente (scipy crudo + matmul en el cuerpo del test):
    # el modulo debe coincidir con los bytes del .mat, no consigo mismo.
    m = loadmat(paths["mat"])
    ref_mean = np.asarray(m["meanshape"], dtype=np.float64).reshape(-1, 3)
    ref_face = ref_mean + (np.asarray(m["idBase"], dtype=np.float64) @ id_coeffs).reshape(-1, 3)
    ref_face = ref_face + (np.asarray(m["exBase"], dtype=np.float64) @ exp_coeffs).reshape(-1, 3)
    np.testing.assert_allclose(face, ref_face, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(m["head_tri"], dtype=np.int64) - 1, np.asarray(head_tri, dtype=np.int64)
    )


def test_v3_dense_eyeballs_on_real_mesh_append_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ojos sobre malla real: append-only de `head_tri` real, finito x2."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _require_v3_dense_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente denso real local (CI)")
    from backend.domain import Err as _Err
    from backend.v3_dense import add_eyeballs, compute_shape_numpy, load_hifi_basis_from

    basis = load_hifi_basis_from(paths["mat"])
    assert not isinstance(basis, _Err)
    mean, id_base, ex_base, head_tri = basis.value.mean, basis.value.id_base, basis.value.ex_base, basis.value.head_tri
    id_coeffs, exp_coeffs = _real_id_exp(monkeypatch, paths["photo"])
    face_result = compute_shape_numpy(mean, id_base, ex_base, id_coeffs, exp_coeffs)
    assert not isinstance(face_result, _Err)
    face, _ = face_result.value
    first = add_eyeballs(face, np.asarray(head_tri, dtype=np.int64))
    second = add_eyeballs(face, np.asarray(head_tri, dtype=np.int64))
    assert not isinstance(first, _Err) and not isinstance(second, _Err)
    v1, f1 = first.value
    np.testing.assert_array_equal(v1[:20481], face)
    np.testing.assert_array_equal(f1[:40832], np.asarray(head_tri, dtype=np.int64))
    assert v1.shape[0] > 20481 and f1.shape[0] > 40832
    assert bool(np.isfinite(v1).all())
    np.testing.assert_array_equal(v1, second.value[0])
    np.testing.assert_array_equal(f1, second.value[1])
