"""CPU compartido: goldens literales a mano, nunca recomputados.

Wave 4 Step 5 (FLAME): sin heatmap, sin compute_heatmap, zip-6 unico.
El template GNM legacy vive para geometria hasta el cutover (Wave 6);
el template FLAME sucesor vive en `gnm_assemble` (5023 verts).
"""

from __future__ import annotations

import io
import json
import zipfile

from backend.domain import (
    UV_LEN,
    ZIP_NAMES,
    CompleteUv,
    FitResult,
    Ok,
    ZipBundle,
    parse_camera_params,
    parse_complete_uv,
    parse_gnm_coeffs,
    parse_landmarks,
)
from backend.gnm import build_result_zip, load_template, uv_to_png
from backend.gnm_assemble import build_personalized_glb


def _golden(head: bytes, fill: int) -> CompleteUv:
    raw = bytes(head) + bytes([fill]) * (UV_LEN - len(head))
    result = parse_complete_uv(raw)
    assert isinstance(result, Ok)
    return result.value


def test_compute_heatmap_import_fails() -> None:
    import importlib

    mod = importlib.import_module("backend.gnm")
    assert not hasattr(mod, "compute_heatmap")
    try:
        from backend.gnm import (
            compute_heatmap,  # type: ignore[attr-defined]  # noqa: F401
        )

        raise AssertionError("compute_heatmap import should fail")
    except ImportError:
        pass


def test_template_con_cuentas_canonicas() -> None:
    from backend.domain import TEMPLATE_TRIS, TEMPLATE_VERTS

    positions, uvs, indices = load_template()
    assert len(positions) == TEMPLATE_VERTS == 17821
    assert len(uvs) == TEMPLATE_VERTS
    assert len(indices) == TEMPLATE_TRIS == 35324


def _proven_fit(marker: float) -> FitResult:
    lm = parse_landmarks(json.dumps([[0.0, 0.0, 0.0]] * 478).encode("utf-8"))
    assert isinstance(lm, Ok)
    coeffs = parse_gnm_coeffs([marker] * 253)
    camera = parse_camera_params([0.0] * 12)
    assert isinstance(coeffs, Ok)
    assert isinstance(camera, Ok)
    return FitResult(coeffs=coeffs.value, camera=camera.value)


def test_zip_con_6_nombres_exactos() -> None:
    a = _golden(bytes([10, 200]), 0)
    b = _golden(bytes([4, 210]), 0)
    a_png = uv_to_png(a)
    b_png = uv_to_png(b)
    ma = build_personalized_glb(_proven_fit(0.1), a)
    mb = build_personalized_glb(_proven_fit(0.2), b)
    assert isinstance(ma, Ok)
    assert isinstance(mb, Ok)
    bundle = ZipBundle(
        uv_a_png=a_png,
        uv_b_png=b_png,
        mesh_a_glb=ma.value.as_bytes(),
        mesh_b_glb=mb.value.as_bytes(),
        pbr_a=a_png,
        pbr_b=b_png,
    )
    blob = build_result_zip(bundle)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
    assert names == list(ZIP_NAMES)
    assert len(names) == 6
    assert "heatmap.png" not in names


def test_wrong_uv_length_rejected_at_parse() -> None:
    from backend.domain import Err

    assert isinstance(parse_complete_uv(bytes([1, 2])), Err)
