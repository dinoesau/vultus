"""Modulo CPU compartido: template GNM legacy, PNG y zip-8 puros sin I/O.

Wave 4 Step 5 (FLAME): sin mapa termico, sin ZIP_FULL legacy.
El zip canonico es 8 piezas (`ZIP_NAMES`) via la unica seam `build_result_zip`.
La geometria FLAME sucesora (5023 verts, ojos [3931:5023)) vive en
`backend/gnm_assemble.py`; este modulo conserva el template GNM (17821)
hasta el cutover de pesos (Wave 6) mas los helpers PNG/zip sin I/O.
Sin torch, sin FastAPI, sin logging.
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
    UV_HEIGHT,
    UV_WIDTH,
    ZIP_NAMES,
    CompleteUv,
    EyeTexture,
    RenderedImage,
    ZipBundle,
)

_template_cache: (
    tuple[
        list[tuple[float, float, float]],
        list[tuple[float, float]],
        list[tuple[int, int, int]],
    ]
    | None
) = None


def _assets_dir() -> str:
    env_dir = os.environ.get("GNM_ASSETS_DIR", "").strip()
    if env_dir:
        return env_dir
    weights_dir = os.environ.get("WEIGHTS_DIR", "").strip()
    if weights_dir:
        return os.path.join(weights_dir, "gnm")
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


def load_template() -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float]],
    list[tuple[int, int, int]],
]:
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
        raise RuntimeError(
            f"gnm template counts {verts}/{tris} != {TEMPLATE_VERTS}/{TEMPLATE_TRIS}"
        )
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


def uv_to_png(uv: CompleteUv | RenderedImage | EyeTexture) -> bytes:
    from PIL import Image

    img = Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(uv.as_bytes()))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_result_zip(bundle: ZipBundle) -> bytes:
    """Zip 8 archivos en orden canonico ZIP_NAMES. Unica seam de zip.

    Entradas probadas (ZipBundle 8 campos); salidas sin I/O. Sin heatmap
    (ADR-008). Renders frontales al final (ADR-012).
    Compresion STORED para bytes deterministas.
    """
    payloads = (
        bundle.uv_a_png,
        bundle.uv_b_png,
        bundle.mesh_a_glb,
        bundle.mesh_b_glb,
        bundle.pbr_a,
        bundle.pbr_b,
        bundle.preview_a_png,
        bundle.preview_b_png,
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in zip(ZIP_NAMES, payloads):
            z.writestr(name, data)
    return buf.getvalue()
