"""Textura: bake 1024 real + seam build_albedo, goldens congelados a mano."""

from __future__ import annotations

import io
import json

import numpy as np

from backend.domain import UV_LEN, Err, Ok, parse_image_bytes, parse_landmarks
from backend.gnm_fit import fit_gnm
from backend.gnm_texture import (
    ATLAS_SIZE,
    NO_DATA,
    _sample_atlas,
    bake_1024,
    build_albedo,
    inpaint_occluded,
    project_texture,
    warp_with_landmarks,
)


def _image(marker: int):
    from PIL import Image as _Image

    img = _Image.new("RGB", (16, 16), (marker, marker, marker))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    parsed = parse_image_bytes(buf.getvalue())
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks():
    pts = [[0.1, 0.2, 0.3]] * 478
    raw = json.dumps(pts).encode("utf-8")
    parsed = parse_landmarks(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _fit():
    fit = fit_gnm(_image(0xA1), _landmarks())
    assert isinstance(fit, Ok)
    return fit.value


def _micro_quad(z_front: float = 0.0):
    mesh = np.asarray(
        [[0.0, 0.0, z_front], [1.0, 0.0, z_front], [1.0, 1.0, z_front], [0.0, 1.0, z_front]],
        dtype=np.float64,
    )
    normals = np.asarray([[0.0, 0.0, 1.0]] * 4, dtype=np.float64)
    tris = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    tri_uvs = np.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=np.float64,
    )
    return mesh, normals, tris, tri_uvs


def _micro_photo() -> np.ndarray:
    photo = np.zeros((4, 4, 3), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            photo[r, c] = (r * 4 + c, r * 4 + c, r * 4 + c)
    return photo


_IDENTITY_CAM = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_micro_fixture_samples_exact_hand_literals() -> None:
    mesh, normals, tris, tri_uvs = _micro_quad()
    atlas, evidence = _sample_atlas(_micro_photo(), mesh, normals, tris, tri_uvs, _IDENTITY_CAM, 4)
    assert atlas.shape == (4, 4, 3)
    # ix=1,iy=2 -> u=1/3,v=1/3 -> foto (1.0,1.0) = valor 5.
    assert list(atlas[2, 1]) == [5, 5, 5]
    # ix=2,iy=1 -> u=2/3,v=2/3 -> foto (2.0,2.0) = valor 10.
    assert list(atlas[1, 2]) == [10, 10, 10]
    assert evidence > 0.0


def test_micro_backface_is_honest_gray() -> None:
    mesh, _, tris, tri_uvs = _micro_quad()
    back_normals = np.asarray([[0.0, 0.0, -1.0]] * 4, dtype=np.float64)
    atlas, _ = _sample_atlas(_micro_photo(), mesh, back_normals, tris, tri_uvs, _IDENTITY_CAM, 4)
    assert bool((atlas == np.asarray(NO_DATA, dtype=np.uint8)).all())


def test_micro_occluded_layer_stays_gray() -> None:
    front_mesh, front_n, front_tris, front_uvs = _micro_quad(z_front=0.0)
    back_mesh, back_n, _, _ = _micro_quad(z_front=-0.5)
    mesh = np.vstack([front_mesh, back_mesh])
    normals = np.vstack([front_n, back_n])
    back_tris = front_tris + 4
    # La capa trasera reutiliza las mismas UVs (mismo XY): el z-buffer la oculta.
    tris = np.vstack([front_tris, back_tris])
    tri_uvs = np.vstack([front_uvs, front_uvs])
    atlas, _ = _sample_atlas(_micro_photo(), mesh, normals, tris, tri_uvs, _IDENTITY_CAM, 4)
    # El punto comparte UVs entre capas: gana la frontal (valor 5, no gris).
    assert list(atlas[2, 1]) == [5, 5, 5]


def test_build_albedo_sin_pesos_es_gris_honesto() -> None:
    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
        has_weights = True
    except RuntimeError:
        has_weights = False
    if has_weights:
        import pytest

        pytest.skip("con pesos el bake muestrea de verdad; este golden es solo CI sin pesos")
    out = build_albedo(_image(0xA1), _fit(), _landmarks())
    assert isinstance(out, Ok)
    raw = out.value.as_bytes()
    assert len(raw) == UV_LEN
    assert list(raw[:3]) == [128, 128, 128]
    assert set(raw) == {128}


def test_bake_sin_pesos_gris_completo() -> None:
    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
        has_weights = True
    except RuntimeError:
        has_weights = False
    if has_weights:
        import pytest

        pytest.skip("solo CI sin pesos")
    atlas, evidence, _ = bake_1024(_image(0xA1), _fit())
    assert atlas.shape == (ATLAS_SIZE, ATLAS_SIZE, 3)
    assert evidence == 0.0
    assert bool((atlas == 128).all())


def test_albedo_derives_from_real_photo_not_hallucinated() -> None:
    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
        has_weights = True
    except RuntimeError:
        has_weights = False
    if not has_weights:
        import pytest

        pytest.skip("sin pesos todo es gris honesto; nada que derivar")
    fit = _fit()
    landmarks = _landmarks()
    a = build_albedo(_image(0xA1), fit, landmarks)
    b = build_albedo(_image(0xB2), fit, landmarks)
    assert isinstance(a, Ok)
    assert isinstance(b, Ok)
    assert a.value.as_bytes() != b.value.as_bytes()


def test_inpaint_is_identity_never_adds_comb() -> None:
    """Regresion del peine de prod: el inpaint no toca ningun byte.

    El hash anterior corrompia 1/8 texels con periodo 32B y ningun gate lo
    veia porque los stubs solidos son punto fijo del promediado. Por eso el
    input es estructurado (gradiente + alterno): cualquier muestreo parcial
    dejaria diff != 0 o periodicidad a lag 32.
    """
    from backend.domain import parse_complete_uv

    landmarks = _landmarks()
    structured = bytes((i * 7 + (i // 3) * 13) % 256 for i in range(UV_LEN))
    parsed = parse_complete_uv(structured)
    assert isinstance(parsed, Ok)
    inpainted = inpaint_occluded(parsed.value, landmarks)
    assert isinstance(inpainted, Ok)
    assert inpainted.value.as_bytes() == structured


def test_warp_is_deterministic() -> None:
    fit = _fit()
    landmarks = _landmarks()
    projected = project_texture(_image(0xA1), fit)
    assert isinstance(projected, Ok)
    first = warp_with_landmarks(projected.value, landmarks)
    second = warp_with_landmarks(projected.value, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.as_bytes() == second.value.as_bytes()


def test_warp_is_identity_until_tps_lands() -> None:
    fit = _fit()
    landmarks = _landmarks()
    projected = project_texture(_image(0xA1), fit)
    assert isinstance(projected, Ok)
    warped = warp_with_landmarks(projected.value, landmarks)
    assert isinstance(warped, Ok)
    assert warped.value.as_bytes() == projected.value.as_bytes()


def test_garbage_lengths_rejected_at_parse() -> None:
    assert isinstance(project_texture(_image(0xA1), _fit()), Ok)
    assert isinstance(parse_image_bytes(bytes([1, 2, 3])), Err)
