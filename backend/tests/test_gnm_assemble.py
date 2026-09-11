"""Assemble: 5 islas + PBR + GLB personalizado, goldens congelados a mano."""

from __future__ import annotations

import io
import json
import zipfile

from backend.domain import (
    UV_LEN,
    Ok,
    parse_gnm_mesh,
    parse_image_bytes,
    parse_landmarks,
)
from backend.gnm_assemble import (
    ISLAND_COUNT,
    ISLAND_LAYOUT_VERSION,
    albedo_from_islands,
    assemble_islands,
    build_full_zip,
    build_personalized_glb,
    island_bounds,
    pbr_from_albedo,
)
from backend.gnm_fit import fit_gnm
from backend.gnm_texture import build_albedo


def _image(marker: int):
    import io as _io

    from PIL import Image as _Image

    img = _Image.new("RGB", (16, 16), (marker, marker, marker))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    parsed = parse_image_bytes(buf.getvalue())
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks(marker: int):
    # Identidad geometrica por marker sin pesos: grilla con o sin warp
    # no-lineal en x (la camara de similaridad no absorbe el warp, asi el
    # fit real da mallas distintas; en CI el doble difiere por hash).
    warped = marker != 0xA1
    pts: list[list[float]] = []
    for i in range(478):
        gx = (i % 32) / 31.0
        gy = ((i // 32) % 15) / 14.0
        if warped:
            gx = gx**1.5
        pts.append([0.2 + 0.6 * gx, 0.2 + 0.6 * gy, 0.0])
    raw = json.dumps(pts).encode("utf-8")
    parsed = parse_landmarks(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _fit_and_albedo(marker: int):
    image = _image(marker)
    landmarks = _landmarks(marker)
    fit = fit_gnm(image, landmarks)
    assert isinstance(fit, Ok)
    albedo = build_albedo(image, fit.value, landmarks)
    assert isinstance(albedo, Ok)
    return fit.value, albedo.value


def test_five_islands_cover_full_uv_without_overlap() -> None:
    assert ISLAND_COUNT == 5
    assert ISLAND_LAYOUT_VERSION == 2
    _, albedo = _fit_and_albedo(0xA1)
    parts = assemble_islands(albedo)
    assert sorted(parts.keys()) == [1, 2, 3, 4, 5]
    total = sum(len(v) for v in parts.values())
    assert total == UV_LEN
    prev_end = 0
    for island in range(1, 6):
        start, end = island_bounds(island)
        assert end > start
        assert start == prev_end
        prev_end = end
    assert prev_end == 512
    joined = albedo_from_islands(parts)
    assert isinstance(joined, Ok)
    assert joined.value.as_bytes() == albedo.as_bytes()


def test_pbr_maps_ride_albedo_deterministically() -> None:
    _, albedo_a = _fit_and_albedo(0xA1)
    _, albedo_b = _fit_and_albedo(0xB2)
    first = pbr_from_albedo(albedo_a)
    second = pbr_from_albedo(albedo_a)
    other = pbr_from_albedo(albedo_b)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert isinstance(other, Ok)
    assert len(first.value) == UV_LEN
    assert first.value == second.value
    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
    except RuntimeError:
        import pytest

        pytest.skip("sin pesos ambos albedos son gris honesto; nada que distinguir")
    assert first.value != other.value


def test_personalized_glb_parses_and_names_islands() -> None:
    fit, albedo = _fit_and_albedo(0xA1)
    out = build_personalized_glb(fit, albedo)
    assert isinstance(out, Ok)
    data = out.value.as_bytes()
    assert data[0:4] == b"glTF"
    assert parse_gnm_mesh(data) == out
    assert b"personalized" in data
    assert b"islands" in data


def test_personalized_mesh_differs_per_identity() -> None:
    fit_a, albedo_a = _fit_and_albedo(0xA1)
    fit_b, _ = _fit_and_albedo(0xB2)
    ma = build_personalized_glb(fit_a, albedo_a)
    mb = build_personalized_glb(fit_b, albedo_a)
    assert isinstance(ma, Ok)
    assert isinstance(mb, Ok)
    assert ma.value.as_bytes() != mb.value.as_bytes()


def test_personalized_glb_reports_real_counts() -> None:
    import struct

    from backend.domain import UV_LEN

    fit, albedo = _fit_and_albedo(0xA1)
    out = build_personalized_glb(fit, albedo)
    assert isinstance(out, Ok)
    data = out.value.as_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    n_exp = doc["accessors"][0]["count"]
    assert doc["accessors"][1]["count"] == n_exp
    assert doc["accessors"][2]["count"] == doc["extras"]["tris"] * 3 == 35324 * 3
    assert doc["extras"]["verts"] == n_exp
    assert doc["extras"]["layout"] == ISLAND_LAYOUT_VERSION == 2
    # Con pesos hay seams reales: mas vertices que los 17821 del template.
    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
        assert n_exp > 17821
    except RuntimeError:
        assert n_exp == 17821
    pbr = pbr_from_albedo(albedo)
    assert isinstance(pbr, Ok)
    assert len(pbr.value) == UV_LEN


def test_glb_seams_unique_emissive_flipped_v() -> None:
    import struct

    fit, albedo = _fit_and_albedo(0xA1)
    out = build_personalized_glb(fit, albedo)
    assert isinstance(out, Ok)
    data = out.value.as_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    assert doc["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [0, 0, 0, 1]
    assert doc["materials"][0]["emissiveTexture"] == {"index": 0}
    assert doc["materials"][0]["emissiveFactor"] == [1, 1, 1]
    n_exp = doc["accessors"][0]["count"]
    pos_len = doc["bufferViews"][0]["byteLength"]
    uv_len = doc["bufferViews"][1]["byteLength"]
    assert pos_len == n_exp * 12
    assert uv_len == n_exp * 8
    bin_start = 20 + json_len + 8
    bin_buf = data[bin_start:]
    uv_off = doc["bufferViews"][1]["byteOffset"]
    idx_off = doc["bufferViews"][2]["byteOffset"]
    n_idx = doc["accessors"][2]["count"]
    uvs = struct.unpack(f"<{n_exp * 2}f", bin_buf[uv_off : uv_off + uv_len])
    comp = doc["accessors"][2]["componentType"]
    fmt = f"<{n_idx}I" if comp == 5125 else f"<{n_idx}H"
    idx = struct.unpack(fmt, bin_buf[idx_off : idx_off + n_idx * (4 if comp == 5125 else 2)])
    assert max(idx) < n_exp and min(idx) >= 0
    # Unicidad (v,vt): cada vertice de export aparece con una sola UV.
    seen: dict[int, tuple[float, float]] = {}
    for v in idx:
        uv = (uvs[2 * v], uvs[2 * v + 1])
        if v in seen:
            assert seen[v] == uv
        else:
            seen[v] = uv
    # Flip a glTF: la exportada es 1 - v_uv del npz (no last-wins).
    try:
        from backend.gnm_head import load_gnm_head

        head = load_gnm_head()
        import numpy as np

        raw_uv = np.asarray(head.triangle_uvs, dtype=np.float64)
        exp_v = np.sort((1.0 - raw_uv[:, :, 1]).ravel())
        got_v = np.asarray(uvs, dtype=np.float64)[1::2]
        # Cercania con tolerancia float32 (el redondeo a N decimales cruza
        # fronteras en casos borde): vecino mas cercano a < 1e-6.
        pos = np.searchsorted(exp_v, got_v)
        lo = np.clip(pos - 1, 0, exp_v.size - 1)
        hi = np.clip(pos, 0, exp_v.size - 1)
        best = np.minimum(np.abs(exp_v[lo] - got_v), np.abs(exp_v[hi] - got_v))
        assert bool((best < 1e-6).all())
    except RuntimeError:
        pass


def test_full_zip_lists_island_and_pbr_names() -> None:
    fit_a, albedo_a = _fit_and_albedo(0xA1)
    fit_b, albedo_b = _fit_and_albedo(0xB2)
    from backend.gnm import compute_heatmap, uv_to_png

    heat = compute_heatmap(albedo_a, albedo_b)
    ma = build_personalized_glb(fit_a, albedo_a)
    mb = build_personalized_glb(fit_b, albedo_b)
    pa = pbr_from_albedo(albedo_a)
    pb = pbr_from_albedo(albedo_b)
    assert isinstance(ma, Ok) and isinstance(mb, Ok)
    assert isinstance(pa, Ok) and isinstance(pb, Ok)
    blob = build_full_zip(
        uv_to_png(albedo_a.as_bytes()),
        uv_to_png(albedo_b.as_bytes()),
        uv_to_png(heat.as_bytes()),
        ma.value.as_bytes(),
        mb.value.as_bytes(),
        uv_to_png(pa.value),
        uv_to_png(pb.value),
    )
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = sorted(z.namelist())
    assert len(names) == 7
    assert names == [
        "heatmap.png",
        "mesh_a.glb",
        "mesh_b.glb",
        "pbr_a.png",
        "pbr_b.png",
        "uv_a.png",
        "uv_b.png",
    ]
