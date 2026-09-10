"""Ensamblaje GNM: 5 islas + PBR + GLB personalizado.

Islas = 5 bandas horizontales del UV 512 (layout v2, fronteras derivadas
del layout UV real).
Malla = template + combinacion lineal de la base de identidad (eval_mesh).
Sin torch, sin FastAPI, sin logging.
"""

from __future__ import annotations

import io
import struct
import zipfile

from backend.domain import (
    TEMPLATE_TRIS,
    TEMPLATE_VERTS,
    UV_HEIGHT,
    UV_LEN,
    UV_WIDTH,
    CompleteUv,
    DomainError,
    Err,
    FitResult,
    GnmMesh,
    MlDecode,
    MlFailed,
    Ok,
    UvRegion,
    parse_complete_uv,
    parse_gnm_mesh,
    parse_uv_region,
)

ISLAND_LAYOUT_VERSION = 2
ISLAND_COUNT = 5

# Fronteras v2 por filas UV (0..512), derivadas del layout UV real:
# cuantiles v de vertex_uvs del npz v3_0 (gnm_head.npz) para 5 islas
# balanceadas por vertices (~3564 verts/isla). Calculado una vez con numpy
# (quantile v = [0.29296, 0.48537, 0.60463, 0.69963] -> filas
# [149, 248, 309, 358]; script inline python, 2026-09-10). Particiona las
# 512 filas sin overlap cubriendo 0..512; roundtrip assemble/albedo intacto.
_ISLAND_ROW_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 149),
    (149, 248),
    (248, 309),
    (309, 358),
    (358, 512),
)

ZIP_PBR_A = "pbr_a.png"
ZIP_PBR_B = "pbr_b.png"


def island_bounds(island: int) -> tuple[int, int]:
    if isinstance(island, bool) or not isinstance(island, int):
        raise TypeError(f"isla no entera: {island!r}")
    if island < 1 or island > ISLAND_COUNT:
        raise ValueError(f"isla {island} fuera de 1..{ISLAND_COUNT}")
    return _ISLAND_ROW_BOUNDS[island - 1]


def island_albedo(albedo: CompleteUv, region: UvRegion) -> bytes:
    raw = albedo.as_bytes()
    start_row, end_row = island_bounds(region.island())
    row_len = UV_WIDTH * 3
    return raw[start_row * row_len : end_row * row_len]


def assemble_islands(albedo: CompleteUv) -> dict[int, bytes]:
    out: dict[int, bytes] = {}
    for island in range(1, ISLAND_COUNT + 1):
        region = parse_uv_region(island)
        assert isinstance(region, Ok)
        out[island] = island_albedo(albedo, region.value)
    return out


def pbr_from_albedo(albedo: CompleteUv) -> Ok[bytes] | Err[DomainError]:
    try:
        raw = albedo.as_bytes()
        out = bytearray(UV_LEN)
        for i in range(0, UV_LEN, 3):
            lum = (raw[i] + raw[i + 1] + raw[i + 2]) // 3
            inv = 255 - lum
            out[i] = inv
            out[i + 1] = inv
            out[i + 2] = inv
        return Ok(bytes(out))
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"pbr failed: {exc}")))


def displaced_positions(fit: FitResult) -> list[tuple[float, float, float]]:
    from backend.gnm_head import eval_mesh

    mesh = eval_mesh(fit.coeffs.as_tuple())
    return [(float(p[0]), float(p[1]), float(p[2])) for p in mesh]


def build_personalized_glb(fit: FitResult, albedo: CompleteUv) -> Ok[GnmMesh] | Err[DomainError]:
    try:
        from backend.gnm import load_template

        _, uvs, indices = load_template()
        positions = displaced_positions(fit)
        if len(positions) != TEMPLATE_VERTS or len(uvs) != TEMPLATE_VERTS:
            raise ValueError(
                f"mesh counts {len(positions)}/{len(uvs)} != {TEMPLATE_VERTS}"
            )
        if len(indices) != TEMPLATE_TRIS:
            raise ValueError(f"tris {len(indices)} != {TEMPLATE_TRIS}")
        from PIL import Image as _Image

        img = _Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(albedo.as_bytes()))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        pos_buf = struct.pack(f"<{len(positions) * 3}f", *[c for p in positions for c in p])
        uv_buf = struct.pack(f"<{len(uvs) * 2}f", *[c for t in uvs for c in t])
        flat_idx = [v for tri in indices for v in tri]
        if not flat_idx or max(flat_idx) >= TEMPLATE_VERTS or min(flat_idx) < 0:
            raise ValueError("indice fuera de rango")
        idx_buf = struct.pack(f"<{len(flat_idx)}H", *flat_idx)
        pos_len, uvb_len, idx_len = len(pos_buf), len(uv_buf), len(idx_buf)
        uv_off = pos_len
        idx_off = pos_len + uvb_len
        png_off = idx_off + idx_len
        bin_buf = pos_buf + uv_buf + idx_buf + png
        while len(bin_buf) % 4 != 0:
            bin_buf += b"\x00"
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        idx_count = len(indices) * 3
        json_str = (
            '{"asset":{"version":"2.0","generator":"vultus-gnm-fit"},"scene":0,'
            '"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"VultusFacePersonalized"}],'
            '"meshes":[{"name":"FacePersonalized","primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]}],'
            '"materials":[{"name":"SkinPBR","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":0.9,"baseColorTexture":{"index":0}}}],'
            '"textures":[{"source":0,"sampler":0}],"samplers":[{"magFilter":9729,"minFilter":9729}],'
            '"images":[{"bufferView":3,"mimeType":"image/png"}],'
            f'"buffers":[{{"byteLength":{len(bin_buf)}}}],'
            '"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%d,"target":34962},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d,"target":34962},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d,"target":34963},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d}]'
            % (pos_len, uv_off, uvb_len, idx_off, idx_len, png_off, len(png))
            + ',"accessors":[{"bufferView":0,"componentType":5126,"count":'
            + f"{TEMPLATE_VERTS}"
            + ',"type":"VEC3",'
            + f'"max":[{max(xs)},{max(ys)},{max(zs)}],"min":[{min(xs)},{min(ys)},{min(zs)}]'
            + "},"
            + '{"bufferView":1,"componentType":5126,"count":'
            + f"{TEMPLATE_VERTS}"
            + ',"type":"VEC2"},'
            + '{"bufferView":2,"componentType":5123,"count":'
            + f"{idx_count}"
            + ',"type":"SCALAR"}],'
            + '"extras":{"personalized":true,"islands":[1,2,3,4,5],"layout":'
            + f"{ISLAND_LAYOUT_VERSION}"
            + ',"verts":'
            + f"{TEMPLATE_VERTS}"
            + ',"tris":'
            + f"{TEMPLATE_TRIS}"
            + "}}"
        )
        json_bytes = json_str.encode("utf-8")
        while len(json_bytes) % 4 != 0:
            json_bytes += b" "
        total = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
        out = b"glTF" + struct.pack("<I", 2) + struct.pack("<I", total)
        out += struct.pack("<I", len(json_bytes)) + b"JSON" + json_bytes
        out += struct.pack("<I", len(bin_buf)) + b"BIN\x00" + bin_buf
        parsed = parse_gnm_mesh(out)
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"glb failed: {exc}")))


def build_full_zip(
    a_png: bytes,
    b_png: bytes,
    h_png: bytes,
    mesh_a: bytes,
    mesh_b: bytes,
    pbr_a: bytes,
    pbr_b: bytes,
) -> bytes:
    from backend.domain import ZIP_HEATMAP, ZIP_MESH_A, ZIP_MESH_B, ZIP_UV_A, ZIP_UV_B

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(ZIP_UV_A, a_png)
        z.writestr(ZIP_UV_B, b_png)
        z.writestr(ZIP_HEATMAP, h_png)
        z.writestr(ZIP_MESH_A, mesh_a)
        z.writestr(ZIP_MESH_B, mesh_b)
        z.writestr(ZIP_PBR_A, pbr_a)
        z.writestr(ZIP_PBR_B, pbr_b)
    return buf.getvalue()


def albedo_from_islands(parts: dict[int, bytes]) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        ordered = b"".join(parts[i] for i in range(1, ISLAND_COUNT + 1))
        assert len(ordered) == UV_LEN
        parsed = parse_complete_uv(ordered)
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"islands join failed: {exc}")))
