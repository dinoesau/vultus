"""Pose v3 real: RMSE decrece sobre landmarks reales de foto con sha congelado.

Seam bajo test: `backend.v3_pose.refine_pose` sobre correspondencias
reales 3D-2D: 3D = malla densa neutra real en `keypoints` 68 del `.mat`
congelado, 2D = landmarks 68 reales (MediaPipe 478 via `MP_68_MAP`) de la
foto Bush 0001 (sha congelado, misma fuente que `scripts/e2e-flame-real.py`).

Fuente independiente de verdad: pixeles + landmarks + base upstream.
La propiedad (RMSE final < inicial, finito) no puede pasar con datos
inventados: con correspondencias aleatorias el descenso no converge.
Nota honesta: la foto LFW es 250px < barra FFHQ 1024, asi que el gate de
entrada la rechaza ruidoso (verificado aqui); el ajuste se testea con
focal = max(W,H) como mecanica sobre datos reales. Sin puente: skip (CI).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

PHOTO_SHA_FROZEN = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"


def _require_pose_bridge(monkeypatch: pytest.MonkeyPatch) -> dict[str, str] | None:
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


def _real_obj68_img68(
    monkeypatch: pytest.MonkeyPatch, photo_path: str
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
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
    from backend.flame_fit import landmark68, landmark_points
    from backend.v3_dense import compute_shape_numpy, load_hifi_basis_from

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
    height, width = int(photo.value.shape[0]), int(photo.value.shape[1])
    pts68 = landmark68(lm.value)
    assert not isinstance(pts68, _Err)
    img68 = np.asarray(pts68.value[:, :2], dtype=np.float64) * np.array([float(width), float(height)])
    basis = load_hifi_basis_from(
        f"{os.environ.get('TOPO_DIR')}/hifi3dpp_model_info.mat"
    )
    assert not isinstance(basis, _Err)
    crop = face_crop224(photo.value, pts.value[:, 0], pts.value[:, 1])
    epoch = find_epoch()
    assert epoch is not None
    recon = load_recon(epoch)
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = recon.forward_coeffs(tensor)
    vec = np.asarray(out.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    parts = split_coeff_vector(vec)
    face_result = compute_shape_numpy(
        basis.value.mean, basis.value.id_base, basis.value.ex_base, parts["id"], parts["exp"]
    )
    assert not isinstance(face_result, _Err)
    _, neutral = face_result.value
    obj68 = np.asarray(neutral[basis.value.keypoints], dtype=np.float64)
    assert obj68.shape == (68, 3) and bool(np.isfinite(obj68).all())
    return obj68, np.asarray(img68, dtype=np.float64), float(max(width, height)), width, height


def test_v3_pose_gate_rejects_250px_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """La LFW 250px queda bajo la barra FFHQ: rechazo ruidoso, no silencio."""
    paths = _require_pose_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente pose real local (CI)")
    from backend.domain import Err as _Err
    from backend.v3_pose import gate_entry

    assert isinstance(gate_entry(250, 250, 5.0), _Err)


def test_v3_pose_rmse_decreases_on_real_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refinamiento iterativo: RMSE final < inicial sobre foto real."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _require_pose_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente pose real local (CI)")
    from backend.v3_pose import refine_pose, reprojection_error

    obj68, img68, focal, _, _ = _real_obj68_img68(monkeypatch, paths["photo"])
    pose, errors = refine_pose(obj68, img68, focal, steps=8)
    assert pose.shape == (6,) and bool(np.isfinite(pose).all())
    assert len(errors) == 9
    assert all(float(e) >= 0.0 and np.isfinite(float(e)) for e in errors)
    assert float(errors[-1]) < float(errors[0])
    # El error reportado es el de la pose devuelta (sin maquillaje).
    assert float(errors[-1]) == pytest.approx(reprojection_error(obj68, img68, focal, pose))
