"""CPU compartido: goldens literales a mano, nunca recomputados.

La via BFM (LUT + bake + GLB neutro) se retiro en S9; el template vive
para la geometria personalizada de `gnm_assemble` y el zip de 5 nombres
para `local_runner`.
"""

from __future__ import annotations

import io
import zipfile

from backend.domain import UV_LEN, ZIP_NAMES, CompleteUv, ZipBundle, parse_complete_uv
from backend.gnm import (
    build_result_zip,
    compute_heatmap,
    load_template,
    uv_to_png,
)


def _golden(head: bytes, fill: int) -> CompleteUv:
    raw = bytes(head) + bytes([fill]) * (UV_LEN - len(head))
    result = parse_complete_uv(raw)
    assert isinstance(result, object)
    from backend.domain import Ok

    assert isinstance(result, Ok)
    return result.value


def test_heatmap_golden_dos_bytes_mas_relleno() -> None:
    from backend.domain import Ok

    a = _golden(bytes([10, 200]), 7)
    b = _golden(bytes([4, 210]), 7)
    heat = compute_heatmap(a, b)
    assert len(heat.as_bytes()) == UV_LEN
    assert heat.as_bytes()[0:2] == bytes([6, 10])
    assert all(x == 0 for x in heat.as_bytes()[2:])
    _ = Ok


def test_template_con_cuentas_canonicas() -> None:
    from backend.domain import TEMPLATE_TRIS, TEMPLATE_VERTS

    positions, uvs, indices = load_template()
    assert len(positions) == TEMPLATE_VERTS == 17821
    assert len(uvs) == TEMPLATE_VERTS
    assert len(indices) == TEMPLATE_TRIS == 35324


def test_zip_con_5_nombres_exactos() -> None:
    a = _golden(bytes([10, 200]), 0)
    b = _golden(bytes([4, 210]), 0)
    heat = compute_heatmap(a, b)
    a_png = uv_to_png(a)
    b_png = uv_to_png(b)
    h_png = uv_to_png(heat)
    bundle = ZipBundle(
        uv_a_png=a_png,
        uv_b_png=b_png,
        heatmap_png=h_png,
        mesh_a_glb=a_png,
        mesh_b_glb=b_png,
        pbr_a=a_png,
        pbr_b=b_png,
    )
    blob = build_result_zip(bundle)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
    assert names == list(ZIP_NAMES)


def test_wrong_uv_length_rejected_at_parse() -> None:
    from backend.domain import Err

    assert isinstance(parse_complete_uv(bytes([1, 2])), Err)
