"""CPU compartido: goldens literales a mano, nunca recomputados."""

from __future__ import annotations

import io
import zipfile

from backend.domain import UV_LEN, CompleteUv, parse_complete_uv
from backend.gnm import (
    bake_bfm_to_gnm,
    build_gnm_glb,
    build_result_zip,
    compute_heatmap,
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


def test_bake_no_identidad_con_texel0_dorado() -> None:
    a = _golden(bytes([10, 200]), 7)
    baked = bake_bfm_to_gnm(a)
    assert len(baked.as_bytes()) == UV_LEN
    assert baked.as_bytes() != a.as_bytes()
    assert baked.as_bytes()[0:3] == bytes([10, 176, 7])
    assert baked.as_bytes()[21:24] == bytes([7, 7, 7])
    assert baked.as_bytes()[6156:6159] == bytes([7, 7, 7])


def test_glb_con_magic_y_tamano_acotado() -> None:
    baked = _golden(bytes([10, 200]), 0)
    mesh = build_gnm_glb(baked)
    data = mesh.as_bytes()
    assert data[0:4] == b"glTF"
    assert 100_000 < len(data) < 2_000_000
    total = int.from_bytes(data[8:12], "little")
    assert total == len(data)


def test_zip_con_5_nombres_exactos() -> None:
    a = _golden(bytes([10, 200]), 0)
    b = _golden(bytes([4, 210]), 0)
    heat = compute_heatmap(a, b)
    mesh_a = build_gnm_glb(a)
    mesh_b = build_gnm_glb(b)
    a_png = uv_to_png(a.as_bytes())
    b_png = uv_to_png(b.as_bytes())
    h_png = uv_to_png(heat.as_bytes())
    blob = build_result_zip(a_png, b_png, h_png, mesh_a.as_bytes(), mesh_b.as_bytes())
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = sorted(z.namelist())
    assert names == ["heatmap.png", "mesh_a.glb", "mesh_b.glb", "uv_a.png", "uv_b.png"]


def test_wrong_uv_length_rejected_at_parse() -> None:
    from backend.domain import Err

    assert isinstance(parse_complete_uv(bytes([1, 2])), Err)
