"""Modulo CPU compartido: template, heatmap, PNG y zip puros sin I/O.

Unica fuente para API local y sidecar. Sin torch, sin FastAPI, sin logging.
Entradas ya probadas (dominio), salidas probadas. Infallible salvo assets corruptos.
El template personaliza su geometria en `gnm_assemble` via `load_template`.
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
    Heatmap,
    ZipBundle,
    _heatmap_from_diff,
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
    # En Modal los .bin viven en el Volume (`/weights/gnm`), no junto al
    # codigo (la imagen solo lleva .py). Sin este default el consumer muere
    # con `gnm asset missing` en prod aunque local pase (repo `assets/`).
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


def compute_heatmap(uv_a: CompleteUv, uv_b: CompleteUv) -> Heatmap:
    """Diferencia |a-b| por byte, infalible por construccion.

    Ambos inputs cargan el invariante UV_LEN via parse_complete_uv,
    el zip nunca puede fallar por longitud; no se retorna Result.
    Unico mint via dominio interno _heatmap_from_diff.
    """
    return _heatmap_from_diff(uv_a, uv_b)


def uv_to_png(uv: CompleteUv | Heatmap) -> bytes:
    from PIL import Image

    img = Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(uv.as_bytes()))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_result_zip(bundle: ZipBundle) -> bytes:
    """Zip 5 archivos ZIP_NAMES desde el mismo ZipBundle de 7.

    Delega a build_full_zip (dueno unico del orden ZIP_FULL) y
    emite solo el subset ZIP_NAMES en orden canonico.
    """
    from backend.gnm_assemble import build_full_zip

    full = build_full_zip(bundle)
    with zipfile.ZipFile(io.BytesIO(full)) as src:
        payloads = {name: src.read(name) for name in ZIP_NAMES}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for name in ZIP_NAMES:
            z.writestr(name, payloads[name])
    return buf.getvalue()
