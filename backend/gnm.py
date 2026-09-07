"""Modulo CPU compartido: bake, heatmap, GLB y zip puros sin I/O.

Unica fuente para API local y sidecar. Sin torch, sin FastAPI, sin logging.
Entradas ya probadas (dominio), salidas probadas. Infallible salvo assets corruptos.
"""

from __future__ import annotations

import io
import math
import os
import struct
import zipfile

from backend.domain import (
    TEMPLATE_TRIS,
    TEMPLATE_VERTS,
    UV_CHANNELS,
    UV_HEIGHT,
    UV_LEN,
    UV_WIDTH,
    ZIP_HEATMAP,
    ZIP_MESH_A,
    ZIP_MESH_B,
    ZIP_UV_A,
    ZIP_UV_B,
    CompleteUv,
    GnmMesh,
    Heatmap,
)

_LUT_V2_LEN = 512 * 512 * 16

_template_cache: tuple[list[tuple[float, float, float]], list[tuple[float, float]], list[tuple[int, int, int]]] | None = None
_lut_v2_cache: bytes | None = None
_bake_cache: tuple[object, object, object, object, object, object] | None = None


def _assets_dir() -> str:
    env_dir = os.environ.get("GNM_ASSETS_DIR", "").strip()
    if env_dir:
        return env_dir
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets")


def _resolve_asset(name: str) -> str:
    primary = os.path.join(_assets_dir(), name)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", name)
    if os.path.exists(fallback):
        return fallback
    raise RuntimeError(f"gnm asset missing: {name} (buscado en {primary} y {fallback})")


def load_template() -> tuple[list[tuple[float, float, float]], list[tuple[float, float]], list[tuple[int, int, int]]]:
    global _template_cache
    if _template_cache is not None:
        return _template_cache
    path = _resolve_asset("gnm_template.bin")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise RuntimeError(f"gnm template truncado: {path}")
    verts, tris = struct.unpack_from("<II", data, 0)
    if verts != TEMPLATE_VERTS or tris != TEMPLATE_TRIS:
        raise RuntimeError(f"gnm template counts {verts}/{tris} != {TEMPLATE_VERTS}/{TEMPLATE_TRIS}")
    expect = 8 + verts * 12 + verts * 8 + tris * 12
    if len(data) != expect:
        raise RuntimeError(f"gnm template len {len(data)} != {expect}")
    off = 8
    positions: list[tuple[float, float, float]] = []
    for _ in range(verts):
        x, y, z = struct.unpack_from("<3f", data, off)
        off += 12
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise RuntimeError("gnm template posicion no finita")
        positions.append((x, y, z))
    uvs: list[tuple[float, float]] = []
    for _ in range(verts):
        u, v = struct.unpack_from("<2f", data, off)
        off += 8
        if not (math.isfinite(u) and math.isfinite(v)):
            raise RuntimeError("gnm template uv no finita")
        uvs.append((u, v))
    indices: list[tuple[int, int, int]] = []
    for _ in range(tris):
        a, b, c = struct.unpack_from("<III", data, off)
        off += 12
        if a >= verts or b >= verts or c >= verts:
            raise RuntimeError("gnm template indice fuera de rango")
        indices.append((a, b, c))
    _template_cache = (positions, uvs, indices)
    return _template_cache


def load_lut_v2() -> bytes:
    global _lut_v2_cache
    if _lut_v2_cache is not None:
        return _lut_v2_cache
    path = _resolve_asset("bfm_to_gnm_v2.bin")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != _LUT_V2_LEN:
        raise RuntimeError(f"gnm lut v2 len {len(data)} != {_LUT_V2_LEN}: {path}")
    for t in range(0, _LUT_V2_LEN, 16):
        tri, w0, w1, w2 = struct.unpack_from("<I3f", data, t)
        if tri >= TEMPLATE_TRIS:
            raise RuntimeError(f"gnm lut v2 tri fuera de rango en offset {t}")
        if not (math.isfinite(w0) and math.isfinite(w1) and math.isfinite(w2)):
            raise RuntimeError(f"gnm lut v2 peso no finito en offset {t}")
    _lut_v2_cache = data
    return data


def _bake_tables() -> tuple[object, object, object, object, object, object]:
    global _bake_cache
    if _bake_cache is not None:
        return _bake_cache
    import array as _arr

    _, uvs, indices = load_template()
    lut = load_lut_v2()
    n = 512 * 512
    src0 = _arr.array("I", [0]) * n
    src1 = _arr.array("I", [0]) * n
    src2 = _arr.array("I", [0]) * n
    w0a = _arr.array("f", [0.0]) * n
    w1a = _arr.array("f", [0.0]) * n
    w2a = _arr.array("f", [0.0]) * n
    for t in range(n):
        tri, w0, w1, w2 = struct.unpack_from("<I3f", lut, t * 16)
        a, b, c = indices[tri]
        for arr, vi in ((src0, a), (src1, b), (src2, c)):
            u, v = uvs[vi]
            sx = int(u * 512.0)
            sx = min(sx, 511)
            sy = int(v * 512.0)
            sy = min(sy, 511)
            arr[t] = sy * 512 + sx
        w0a[t] = w0
        w1a[t] = w1
        w2a[t] = w2
    _bake_cache = (src0, src1, src2, w0a, w1a, w2a)
    return _bake_cache


def compute_heatmap(uv_a: CompleteUv, uv_b: CompleteUv) -> Heatmap:
    raw = bytes(x - y if x >= y else y - x for x, y in zip(uv_a.as_bytes(), uv_b.as_bytes()))
    return Heatmap(_value=raw)


def bake_bfm_to_gnm(uv_bfm: CompleteUv) -> CompleteUv:
    src_bytes = uv_bfm.as_bytes()
    try:
        import numpy as _np

        src0, src1, src2, w0a, w1a, w2a = _bake_tables()
        src = _np.frombuffer(bytes(src_bytes), dtype=_np.uint8).reshape(-1, 3).astype(_np.float32)
        w0 = _np.asarray(w0a, dtype=_np.float32)[:, None]
        w1 = _np.asarray(w1a, dtype=_np.float32)[:, None]
        w2 = _np.asarray(w2a, dtype=_np.float32)[:, None]
        i0 = _np.asarray(src0, dtype=_np.int64)
        i1 = _np.asarray(src1, dtype=_np.int64)
        i2 = _np.asarray(src2, dtype=_np.int64)
        val = w0 * src[i0] + w1 * src[i1] + w2 * src[i2]
        out = _np.floor(val + 0.5).clip(0, 255).astype(_np.uint8).tobytes()
        return CompleteUv(_value=bytes(out))
    except ImportError:
        pass
    _, uvs, indices = load_template()
    lut = load_lut_v2()
    out_arr = bytearray(UV_LEN)
    for t in range(512 * 512):
        tri, w0, w1, w2 = struct.unpack_from("<I3f", lut, t * 16)
        a, b, c = indices[tri]
        offs = []
        for vi in (a, b, c):
            u, v = uvs[vi]
            sx = min(511, int(u * 512.0))
            sy = min(511, int(v * 512.0))
            offs.append((sy * 512 + sx) * 3)
        for k in range(3):
            val_f = w0 * src_bytes[offs[0] + k] + w1 * src_bytes[offs[1] + k] + w2 * src_bytes[offs[2] + k]
            out_arr[t * 3 + k] = min(255, max(0, int(val_f + 0.5)))
    return CompleteUv(_value=bytes(out_arr))


def build_gnm_glb(baked: CompleteUv) -> GnmMesh:
    positions, uvs, indices = load_template()
    from PIL import Image as _Image

    img = _Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(baked.as_bytes()))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    pos_buf = struct.pack(f"<{len(positions) * 3}f", *[c for p in positions for c in p])
    uv_buf = struct.pack(f"<{len(uvs) * 2}f", *[c for t in uvs for c in t])
    flat_idx = [v for tri in indices for v in tri]
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
        '{"asset":{"version":"2.0","generator":"vultus-gnm-real"},"scene":0,'
        '"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"VultusFaceNeutral"}],'
        '"meshes":[{"name":"FaceNeutral","primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]}],'
        '"materials":[{"name":"BakedSkin","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":0.9,"baseColorTexture":{"index":0}}}],'
        '"textures":[{"source":0,"sampler":0}],"samplers":[{"magFilter":9729,"minFilter":9729}],'
        '"images":[{"bufferView":3,"mimeType":"image/png"}],'
        f'"buffers":[{{"byteLength":{len(bin_buf)}}}],'
        '"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%d,"target":34962},'
        '{"buffer":0,"byteOffset":%d,"byteLength":%d,"target":34962},'
        '{"buffer":0,"byteOffset":%d,"byteLength":%d,"target":34963},'
        '{"buffer":0,"byteOffset":%d,"byteLength":%d}]'
        % (pos_len, uv_off, uvb_len, idx_off, idx_len, png_off, len(png))
        + ',"accessors":[{"bufferView":0,"componentType":5126,"count":4225,"type":"VEC3",'
        f'"max":[{max(xs)},{max(ys)},{max(zs)}],"min":[{min(xs)},{min(ys)},{min(zs)}]}},'
        '{"bufferView":1,"componentType":5126,"count":4225,"type":"VEC2"},'
        f'{{"bufferView":2,"componentType":5123,"count":{idx_count},"type":"SCALAR"}}],'
        f'"extras":{{"expression":"neutral","bakedLen":{UV_LEN},"verts":4225,"tris":8192}}}}'
    )
    json_bytes = json_str.encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    out = b"glTF" + struct.pack("<I", 2) + struct.pack("<I", total)
    out += struct.pack("<I", len(json_bytes)) + b"JSON" + json_bytes
    out += struct.pack("<I", len(bin_buf)) + b"BIN\x00" + bin_buf
    return GnmMesh(_value=out)


def uv_to_png(raw: bytes) -> bytes:
    from PIL import Image

    img = Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(raw))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_result_zip(a_png: bytes, b_png: bytes, h_png: bytes, mesh_a: bytes, mesh_b: bytes) -> bytes:
    _ = (UV_WIDTH, UV_HEIGHT, UV_CHANNELS)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(ZIP_UV_A, a_png)
        z.writestr(ZIP_UV_B, b_png)
        z.writestr(ZIP_HEATMAP, h_png)
        z.writestr(ZIP_MESH_A, mesh_a)
        z.writestr(ZIP_MESH_B, mesh_b)
    return buf.getvalue()
