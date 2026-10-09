"""Orientacion V del GLB: upright en visores spec-compliant (`flipY=false`).

Seam bajo test: `build_personalized_glb` (UVs del GLB) contra el atlas PNG
del bake (convencion documentada pixel<->UV directa, sin flip del visor).

Golden con template real (`backend/assets/flame_template.bin`):
- vertice chin (min-y piel) con v > 0.9, frente (embed dlib 19/24) con
  v < 0.35, ojos en rango 2-3. El doble flip actual da chin ~0.03,
  frente ~0.71 y ojos negativos.
Render-test con bake real de foto Bush (sha congelado, skip sin puente):
raster frontal del GLB muestreando el atlas; el tercio superior del
render debe parecerse al tercio superior del atlas (pelo arriba), no al
inferior. Sin numeros magicos: comparacion relativa a<b.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

from backend.domain import (
    UV_LEN,
    Err,
    parse_camera_params,
    parse_gnm_coeffs,
    parse_rendered_image,
)

CHIN_V_MIN = 0.9
BROW_V_MAX = 0.35


def _glb_uvs(albedo_bytes: bytes) -> np.ndarray:
    from backend.domain import FitResult
    from backend.gnm_assemble import EYE_VERT_START, VERT_COUNT, build_personalized_glb

    coeffs = parse_gnm_coeffs([0.0] * 253)
    assert not isinstance(coeffs, Err)
    camera = parse_camera_params([0.0] * 11 + [1.0])
    assert not isinstance(camera, Err)
    albedo = parse_rendered_image(albedo_bytes)
    assert not isinstance(albedo, Err)
    glb = build_personalized_glb(FitResult(coeffs=coeffs.value, camera=camera.value), albedo.value)
    assert not isinstance(glb, Err)
    raw = glb.value.as_bytes()
    assert raw[:4] == b"glTF"
    (json_len,) = struct.unpack_from("<I", raw, 12)
    doc = json.loads(raw[20 : 20 + json_len].decode("utf-8"))
    uv_acc = doc["accessors"][1]
    view = doc["bufferViews"][uv_acc["bufferView"]]
    assert uv_acc["count"] == VERT_COUNT and uv_acc["type"] == "VEC2"
    bin_off = 12 + 8 + ((json_len + 3) // 4) * 4 + 8
    start = bin_off + int(view["byteOffset"])
    flat = struct.unpack_from(f"<{VERT_COUNT * 2}f", raw, start)
    uvs = np.asarray(flat, dtype=np.float64).reshape(VERT_COUNT, 2)
    assert uvs.shape[0] > EYE_VERT_START
    return uvs


def test_glb_chin_down_brow_up_eyes_in_range() -> None:
    """Golden V con template real: chin v≈1, frente v≈0, ojos en 2-3."""
    from backend.domain import Err as _Err
    from backend.gnm_assemble import (
        EYE_VERT_START,
        load_flame68_embed,
        load_flame_template,
    )

    loaded = load_flame_template()
    assert not isinstance(loaded, _Err)
    positions, _template_uvs, _ = loaded.value
    skin = [i for i in range(EYE_VERT_START)]
    chin = min(skin, key=lambda i: positions[i][1])
    uvs = _glb_uvs(bytes(UV_LEN))
    assert float(uvs[chin][1]) > CHIN_V_MIN
    embed = load_flame68_embed()
    assert not isinstance(embed, _Err)
    rows, tri, w = embed.value
    brow_vs: list[float] = []
    for dlib in (19, 24):
        for i, r in enumerate(rows):
            if int(r) == dlib:
                (a, b, c) = (int(tri[i][0]), int(tri[i][1]), int(tri[i][2]))
                v = float(w[i][0]) * float(uvs[a][1]) + float(w[i][1]) * float(uvs[b][1]) + float(w[i][2]) * float(uvs[c][1])
                brow_vs.append(v)
    assert brow_vs
    assert all(v < BROW_V_MAX for v in brow_vs)
    for eye in (EYE_VERT_START, 4500, 5022):
        assert 2.0 <= float(uvs[eye][0]) <= 3.0


def _require_bake_bridge(monkeypatch: pytest.MonkeyPatch) -> dict[str, str] | None:
    mirror = "/Users/esau.martinez/code/weights"
    lfw = "/Users/esau.martinez/Code/datasets/lfw"
    need = {
        "ffhq": f"{mirror}/ffhq-uv",
        "unwrap": f"{mirror}/ffhq-uv-hf/topo_assets/unwrap_1024_info.mat",
        "texgan": f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model/texgan_ffhq_uv.pth",
        "task": f"{mirror}/mediapipe/face_landmarker.task",
        "photo": f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg",
    }
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need.values() if p != need["ffhq"]):
        return None
    if not os.path.isdir(need["ffhq"]):
        return None
    monkeypatch.setenv("FFHQ_UV_DIR", need["ffhq"])
    monkeypatch.setenv("TOPO_DIR", f"{mirror}/ffhq-uv-hf/topo_assets")
    monkeypatch.setenv("TEXGAN_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model")
    monkeypatch.setenv("WEIGHTS_DIR", mirror)
    monkeypatch.delenv("VULTUS_REAL_ML", raising=False)
    from backend import modal_app as _modal_app

    monkeypatch.setattr(_modal_app, "WEIGHTS_DIR", mirror)
    monkeypatch.setattr(_modal_app, "_LM", None)
    return need


def _front_render(verts: np.ndarray, uvs: np.ndarray, tris: list[tuple[int, int, int]], atlas: np.ndarray) -> np.ndarray:
    """Delega al raster productivo (seam unica, sin duplicar)."""
    from backend.gnm_assemble import smooth_vertex_normals
    from backend.render_preview import raster_frontal_array

    verts_list = [(float(r[0]), float(r[1]), float(r[2])) for r in verts.tolist()]
    normals = np.asarray(smooth_vertex_normals(verts_list, tris), dtype=np.float64)
    return raster_frontal_array(verts, uvs, normals, tris, atlas, 256)


def test_glb_render_matches_atlas_orientation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render frontal del GLB sobre atlas real: arriba del render ~= arriba del atlas."""
    pytest.importorskip("torch")
    pytest.importorskip("mediapipe")
    paths = _require_bake_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente bake real local (CI)")
    import hashlib

    from backend import modal_app
    from backend.domain import Err as _Err
    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.flame_fit import fit_flame
    from backend.flame_texture import run_unwrap_texture
    from backend.gnm_assemble import (
        EYE_VERT_START,
        displaced_positions,
        load_flame_template,
    )

    with open(paths["photo"], "rb") as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"
    img = parse_image_bytes(raw)
    assert not isinstance(img, _Err)
    lm = parse_landmarks(modal_app.mediapipe_infer("check", img.value))
    assert not isinstance(lm, _Err)
    fit = fit_flame(img.value, lm.value)
    assert not isinstance(fit, _Err)
    baked = run_unwrap_texture(img.value, fit.value, lm.value)
    assert not isinstance(baked, _Err)
    atlas = np.asarray(bytearray(baked.value.as_bytes()), dtype=np.float64).reshape(512, 512, 3)
    disp = displaced_positions(fit.value)
    assert not isinstance(disp, _Err)
    loaded = load_flame_template()
    assert not isinstance(loaded, _Err)
    _, _template_uvs, template_tris = loaded.value
    glb_uvs = _glb_uvs(baked.value.as_bytes())
    verts = np.asarray(disp.value, dtype=np.float64)
    skin_tris = [t for t in template_tris if t[0] < EYE_VERT_START and t[1] < EYE_VERT_START and t[2] < EYE_VERT_START]
    render = _front_render(verts, glb_uvs, skin_tris, atlas)
    top = render[:85].reshape(-1, 3).mean(axis=0)
    atlas_top = atlas[:170].reshape(-1, 3).mean(axis=0)
    atlas_bottom = atlas[342:].reshape(-1, 3).mean(axis=0)
    assert float(np.linalg.norm(atlas_top - atlas_bottom)) > 1e-6
    assert float(np.linalg.norm(top - atlas_top)) < float(np.linalg.norm(top - atlas_bottom))
