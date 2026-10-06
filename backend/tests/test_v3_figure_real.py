"""Figura v3 real: GLB denso + runner con linea de ramas siempre + E2E trio.

Seam bajo test: `backend.v3_bundle.build_dense_glb` (malla densa real +
ojos + UVs del `.mat` + NORMAL suave) y `backend.v3_worker.run_v3_subject`
(orquestador: densa -> pose -> atlas -> relights -> bundle, con `report`
de ramas OBLIGATORIO incluso en rechazo del gate).

Regla de ramas: cada sujeto emite exactamente una linea con las 5 claves,
pase o rechace (prohibido fallback sin linea). El gate por defecto es la
barra FFHQ 1024: el trio LFW 250px se rechaza ruidoso; el E2E verifica el
rechazo Y la cadena completa con `min_side_px` explicito y logueado.
Sin puente: skip (CI).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

SHA_A = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"
SHA_B = "f04d53698da366ca8562b2d24ad9ed058116621b8fd0d51fcb46a6e5e470e0f3"
SHA_C = "b68ed8d50ba85209d826b962987077bc8e1826f7f2f325469f20738e1bc8bad2"


def _require_figure_bridge(monkeypatch: pytest.MonkeyPatch) -> dict[str, str] | None:
    mirror = "/Users/esau.martinez/code/weights"
    lfw = "/Users/esau.martinez/Code/datasets/lfw"
    need = {
        "epoch": f"{mirror}/ffhq-uv-hf/checkpoints/deep3d_model/epoch_latest.pth",
        "mat": f"{mirror}/ffhq-uv-hf/topo_assets/hifi3dpp_model_info.mat",
        "task": f"{mirror}/mediapipe/face_landmarker.task",
        "texgan": f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model/texgan_ffhq_uv.pth",
        "unwrap": f"{mirror}/ffhq-uv-hf/topo_assets/unwrap_1024_info.mat",
        "t7": f"{mirror}/ffhq-uv-hf/checkpoints/dpr_model/trained_model_03.t7",
        "a": f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg",
    }
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need.values()):
        return None
    monkeypatch.setenv("DEEP3D_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/deep3d_model")
    monkeypatch.setenv("TOPO_DIR", f"{mirror}/ffhq-uv-hf/topo_assets")
    monkeypatch.setenv("TEXGAN_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model")
    monkeypatch.setenv("PARSING_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/parsing_model")
    monkeypatch.setenv("DPR_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/dpr_model")
    monkeypatch.setenv("WEIGHTS_DIR", mirror)
    monkeypatch.delenv("VULTUS_REAL_ML", raising=False)
    from backend import modal_app as _modal_app

    monkeypatch.setattr(_modal_app, "WEIGHTS_DIR", mirror)
    monkeypatch.setattr(_modal_app, "_LM", None)
    return need


def _real_dense_with_eyes(monkeypatch: pytest.MonkeyPatch, photo_path: str, sha: str):
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
    from backend.v3_dense import (
        add_eyeballs,
        compute_shape_numpy,
        dense_uv01_from_mat,
        load_hifi_basis_from,
    )

    with open(photo_path, "rb") as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == sha
    img = parse_image_bytes(raw)
    assert not isinstance(img, _Err)
    lm = parse_landmarks(modal_app.mediapipe_infer("check", img.value))
    assert not isinstance(lm, _Err)
    pts = landmark_points(lm.value)
    assert not isinstance(pts, _Err)
    photo = photo_array(img.value)
    assert not isinstance(photo, _Err)
    basis = load_hifi_basis_from(f"{os.environ.get('TOPO_DIR')}/hifi3dpp_model_info.mat")
    assert not isinstance(basis, _Err)
    crop = face_crop224(photo.value, pts.value[:, 0], pts.value[:, 1])
    recon = load_recon(find_epoch())
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        vec = np.asarray(recon.forward_coeffs(tensor).detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    parts = split_coeff_vector(vec)
    face_result = compute_shape_numpy(
        basis.value.mean, basis.value.id_base, basis.value.ex_base, parts["id"], parts["exp"]
    )
    assert not isinstance(face_result, _Err)
    eyes = add_eyeballs(face_result.value[0], np.asarray(basis.value.head_tri, dtype=np.int64))
    assert not isinstance(eyes, _Err)
    uvs = dense_uv01_from_mat(f"{os.environ.get('TOPO_DIR')}/hifi3dpp_model_info.mat")
    assert not isinstance(uvs, _Err)
    return eyes.value[0], eyes.value[1], uvs.value


def test_v3_dense_glb_real_magic_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """GLB denso real: magia glTF, conteos y determinismo x2."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _require_figure_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente figura real local (CI)")
    from backend.domain import Err as _Err
    from backend.v3_bundle import build_dense_glb

    verts, tris, uvs = _real_dense_with_eyes(monkeypatch, paths["a"], SHA_A)
    first = build_dense_glb(verts, tris, uvs)
    second = build_dense_glb(verts, tris, uvs)
    assert not isinstance(first, _Err) and not isinstance(second, _Err)
    assert first.value[:4] == b"glTF"
    assert len(first.value) > 12
    np.testing.assert_array_equal(np.frombuffer(first.value, dtype=np.uint8), np.frombuffer(second.value, dtype=np.uint8))


def test_v3_run_subject_gate_rejects_and_still_reports_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rechazo ruidoso del gate CON linea de ramas (prohibido silencio)."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _require_figure_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente figura real local (CI)")
    from backend.domain import Err as _Err
    from backend.v3_worker import V3_BRANCH_KEYS, run_v3_subject

    lines: list[tuple[str, dict[str, float]]] = []

    def _rec(job: str, stats: dict[str, float]) -> None:
        lines.append((job, stats))

    result = run_v3_subject("job-gate", paths["a"], _rec)
    assert isinstance(result, _Err)
    assert len(lines) == 1
    _, stats = lines[0]
    assert all(k in stats for k in V3_BRANCH_KEYS)


def test_v3_run_subject_eval_builds_bundle_and_reports_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cadena completa bajo barra (min_side explicito): bundle + linea."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _require_figure_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente figura real local (CI)")
    from backend.domain import ZIP_NAMES as _ZIP6
    from backend.domain import Err as _Err
    from backend.v3_bundle import V3Bundle
    from backend.v3_worker import V3_BRANCH_KEYS, run_v3_subject

    lines: list[tuple[str, dict[str, float]]] = []

    def _rec(job: str, stats: dict[str, float]) -> None:
        lines.append((job, stats))

    result = run_v3_subject("job-eval", paths["a"], _rec, min_side_px=250, texgan_steps=2)
    assert not isinstance(result, _Err)
    assert isinstance(result.value, V3Bundle)
    assert not (set(result.value.names()) & set(_ZIP6))
    assert len(lines) == 1
    _, stats = lines[0]
    assert all(k in stats for k in V3_BRANCH_KEYS)
    assert stats["deep3d"] == 1.0 and stats["texgan"] == 1.0 and stats["dpr"] == 1.0
