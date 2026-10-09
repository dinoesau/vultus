"""Render frontal 2D determinista 512x512 para previews del zip-8.

Puro sin I/O, sin torch/mediapipe/logging/FastAPI. Reutiliza
`displaced_positions`, `load_flame_template` y `smooth_vertex_normals`
de `backend/gnm_assemble` (nada duplicado). Raster ortografico frontal
512, z-buffer max-z, sample bilineal del atlas 512, Lambert
`0.65+0.35*n.z`, fondo `#1a2027`, tris piel+ojos, encode PIL PNG.
Total con `Result`: fallo es `Err(MlFailed)`, nunca PNG vacio fallback.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray

from backend.domain import (
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    UV_HEIGHT,
    UV_WIDTH,
    CompleteUv,
    DomainError,
    Err,
    FitResult,
    MlDecode,
    MlFailed,
    Ok,
    RenderedImage,
)
from backend.gnm_assemble import (
    displaced_positions,
    load_flame_template,
    smooth_vertex_normals,
)

PREVIEW_BG_R = 0x1A
PREVIEW_BG_G = 0x20
PREVIEW_BG_B = 0x27
LAMBERT_BASE = 0.65
LAMBERT_SCALE = 0.35


def raster_frontal_array(
    verts: NDArray[np.float64],
    uvs: NDArray[np.float64],
    normals: NDArray[np.float64],
    tris: list[tuple[int, int, int]],
    atlas: NDArray[np.float64],
    side: int = PREVIEW_WIDTH,
) -> NDArray[np.float64]:
    """Raster ortografico frontal puro: x->col, y max->row 0, depth max-z.

    Seams compartida entre `build_preview_png` y el test de orientacion:
    el test importa de aqui, nunca duplica el raster. Bilineal sobre el
    atlas 512 + Lambert `0.65+0.35*n.z` + fondo `#1a2027`.
    """
    side_w = side
    side_h = side
    img = np.zeros((side_h, side_w, 3), dtype=np.float64)
    img[:, :, 0] = float(PREVIEW_BG_R)
    img[:, :, 1] = float(PREVIEW_BG_G)
    img[:, :, 2] = float(PREVIEW_BG_B)
    depth = np.full((side_h, side_w), -np.inf)
    xs = verts[:, 0]
    ys = verts[:, 1]
    zs = verts[:, 2]
    x0 = float(xs.min())
    x1 = float(xs.max())
    y0 = float(ys.min())
    y1 = float(ys.max())
    cols = (xs - x0) / max(x1 - x0, 1e-9) * float(side_w - 1)
    rows = (y1 - ys) / max(y1 - y0, 1e-9) * float(side_h - 1)
    count = int(verts.shape[0])
    for tri in tris:
        a = int(tri[0])
        b = int(tri[1])
        c = int(tri[2])
        if a < 0 or b < 0 or c < 0:
            continue
        if a >= count or b >= count or c >= count:
            continue
        px = np.array([cols[a], cols[b], cols[c]])
        py = np.array([rows[a], rows[b], rows[c]])
        pz = np.array([zs[a], zs[b], zs[c]])
        xa = int(max(0, px.min()))
        xb = int(min(side_w - 1, px.max()))
        ya = int(max(0, py.min()))
        yb = int(min(side_h - 1, py.max()))
        if xb < xa or yb < ya:
            continue
        denom = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
        if abs(float(denom)) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(xa, xb + 1), np.arange(ya, yb + 1))
        w0 = ((py[1] - py[2]) * (gx - px[2]) + (px[2] - px[1]) * (gy - py[2])) / denom
        w1 = ((py[2] - py[0]) * (gx - px[2]) + (px[0] - px[2]) * (gy - py[2])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not bool(inside.any()):
            continue
        z = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
        u = (w0 * uvs[a][0] + w1 * uvs[b][0] + w2 * uvs[c][0]) * 511.0
        v = (w0 * uvs[a][1] + w1 * uvs[b][1] + w2 * uvs[c][1]) * 511.0
        u = np.clip(u, 0.0, 511.0)
        v = np.clip(v, 0.0, 511.0)
        x_lo = np.clip(u.astype(int), 0, 510)
        y_lo = np.clip(v.astype(int), 0, 510)
        fx = (u - x_lo)[..., None]
        fy = (v - y_lo)[..., None]
        sample = (
            atlas[y_lo, x_lo] * (1 - fx) * (1 - fy)
            + atlas[y_lo, x_lo + 1] * fx * (1 - fy)
            + atlas[y_lo + 1, x_lo] * (1 - fx) * fy
            + atlas[y_lo + 1, x_lo + 1] * fx * fy
        )
        nz = w0 * normals[a][2] + w1 * normals[b][2] + w2 * normals[c][2]
        nz = np.clip(nz, -1.0, 1.0)
        shade = LAMBERT_BASE + LAMBERT_SCALE * nz
        lit = sample * shade[..., None]
        region_depth = depth[ya : yb + 1, xa : xb + 1]
        update = inside & (z > region_depth)
        region_depth[update] = z[update]
        region = img[ya : yb + 1, xa : xb + 1]
        region[update] = lit[update]
    return img


def build_preview_png(
    fit: FitResult, albedo: CompleteUv | RenderedImage
) -> Ok[bytes] | Err[DomainError]:
    """Raster frontal del fit sobre el albedo. Total: `Ok[png]` o `Err(MlFailed)`."""
    displaced = displaced_positions(fit)
    if isinstance(displaced, Err):
        return displaced
    loaded = load_flame_template()
    if isinstance(loaded, Err):
        return loaded
    try:
        positions = displaced.value
        _, template_uvs, template_tris = loaded.value
        verts = np.asarray(positions, dtype=np.float64)
        uvs = np.asarray(template_uvs, dtype=np.float64)
        normals = np.asarray(
            smooth_vertex_normals(positions, template_tris), dtype=np.float64
        )
        atlas = (
            np.frombuffer(albedo.as_bytes(), dtype=np.uint8)
            .reshape(UV_HEIGHT, UV_WIDTH, 3)
            .astype(np.float64)
        )
        img = raster_frontal_array(
            verts, uvs, normals, template_tris, atlas, PREVIEW_WIDTH
        )
        _ = PREVIEW_HEIGHT
        clipped = np.clip(np.rint(img), 0.0, 255.0).astype(np.uint8)
        from PIL import Image

        out_img = Image.fromarray(clipped, mode="RGB")
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return Ok(buf.getvalue())
    except Exception as exc:  # noqa: BLE001 - raster invalido: via Err(MlFailed), nunca PNG vacio
        return Err(MlFailed(detail=MlDecode(details=f"preview failed: {exc}")))
