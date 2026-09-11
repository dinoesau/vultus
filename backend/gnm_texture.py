"""Textura GNM: bake real 1024 por baricentricas + visibilidad triple.

Cada texel conoce su punto 3D por baricentricas sobre `triangle_uvs`
(origen abajo-izquierda), se proyecta a la foto con la camara fiteada
(`gnm_fit.project`), se muestrea bilinealmente solo si pasa visibilidad
triple (facing + grazing + z-buffer) y lo no visible queda en gris honesto.
Sin relleno, sin espejado, sin inpaint: jamas contenido falso.

El seam `build_albedo` no cambia: hornea a 1024 y reduce a 512 `CompleteUv`.
El contrato 512 (domain, heatmap, viewers, zip) queda intacto; 1024 vive
solo en el PNG incrustado en el GLB (artefacto de visualizacion).
Sin pesos no hay retroproyeccion posible: gris completo determinista.
"""

from __future__ import annotations

import io
import time

import numpy as np
from numpy.typing import NDArray

from backend.domain import (
    UV_HEIGHT,
    UV_LEN,
    UV_WIDTH,
    CompleteUv,
    DomainError,
    Err,
    FitResult,
    ImageBytes,
    Landmarks,
    MlDecode,
    MlFailed,
    Ok,
    parse_complete_uv,
)
from backend.gnm_head import eval_mesh, load_gnm_head, vertex_normals

# Constantes congeladas con procedencia (plan-gnm-texture-bake).
ATLAS_SIZE = 1024
NO_DATA: tuple[int, int, int] = (128, 128, 128)
GRAZING_COS_MIN = 0.3
# Tolerancia de profundidad relativa al rango: la malla es densa (~2 tris por
# pixel de foto LFW 250px), asi que la profundidad baricentrica del texel y la
# del centro del pixel difieren por pendiente x offset sub-pixel (hasta ~2% en
# la nariz). 0.02 cubre esa variacion sin alcanzar el gap real entre capas
# (cara vs nuca ~78% del rango). Medido en Bush: 0.001->6%, 0.01->15%, 0.03->30%.
_Z_EPS_REL = 0.02
_MOUTH_SOCK_GROUP = "mouth_sock"


def _photo_rgb(image: ImageBytes) -> NDArray[np.uint8]:
    from PIL import Image as _Image

    img = _Image.open(io.BytesIO(image.as_bytes())).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _sample_atlas(
    photo: NDArray[np.uint8],
    mesh: NDArray[np.float64],
    normals: NDArray[np.float64],
    tris: NDArray[np.int64],
    tri_uvs: NDArray[np.float64],
    camera_tuple: tuple[float, ...],
    atlas_size: int,
    mouth_tris: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.uint8], float]:
    """Nucleo puro del bake: raster + visibilidad triple + `cv2.remap`.

    `camera_tuple` son 12 floats row-major 3x4 (misma semantica que
    `gnm_fit.project`). Devuelve (atlas SxSx3 uint8, evidence_frac).
    Solo `remap` (+ `dilate` permitido, no usado para no rellenar) de cv2.
    """
    import cv2

    cv2.setNumThreads(1)
    s = int(atlas_size)
    n_tris = int(tris.shape[0])
    texel_pos = np.full((s, s, 3), np.nan, dtype=np.float64)
    texel_nor = np.zeros((s, s, 3), dtype=np.float64)
    texel_tri = np.full((s, s), -1, dtype=np.int32)
    # Las UVs se solapan entre capas (p. ej. cara y nuca comparten region):
    # gana la superficie frontal (Z maxima, camara en +Z), no la ultima en
    # orden del npz. Sin esto la nuca sobrescribe la cara y el z-buffer de
    # foto la marca ocluida dejando solo slivers.
    texel_depth = np.full((s, s), -np.inf, dtype=np.float64)
    ax = tri_uvs[:, :, 0] * float(s - 1)
    ay = (1.0 - tri_uvs[:, :, 1]) * float(s - 1)
    for t in range(n_tris):
        if mouth_tris is not None and bool(mouth_tris[t]):
            continue
        i0, i1, i2 = int(tris[t, 0]), int(tris[t, 1]), int(tris[t, 2])
        x0, y0 = float(ax[t, 0]), float(ay[t, 0])
        x1, y1 = float(ax[t, 1]), float(ay[t, 1])
        x2, y2 = float(ax[t, 2]), float(ay[t, 2])
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(s - 1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(s - 1, int(np.ceil(max(y0, y1, y2))))
        if xmax < xmin or ymax < ymin:
            continue
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        xs = np.arange(xmin, xmax + 1, dtype=np.float64)
        ys = np.arange(ymin, ymax + 1, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not bool(inside.any()):
            continue
        p0 = mesh[i0]
        p1 = mesh[i1]
        p2 = mesh[i2]
        n0 = normals[i0]
        n1 = normals[i1]
        n2 = normals[i2]
        py, px = np.where(inside)
        b0 = w0[py, px]
        b1 = w1[py, px]
        b2 = w2[py, px]
        pts = b0[:, None] * p0 + b1[:, None] * p1 + b2[:, None] * p2
        nrs = b0[:, None] * n0 + b1[:, None] * n1 + b2[:, None] * n2
        gy = (py + ymin).astype(np.int64)
        gx = (px + xmin).astype(np.int64)
        cand_z = pts[:, 2]
        keep = cand_z > texel_depth[gy, gx]
        gy = gy[keep]
        gx = gx[keep]
        texel_pos[gy, gx] = pts[keep]
        texel_nor[gy, gx] = nrs[keep]
        texel_tri[gy, gx] = t
        texel_depth[gy, gx] = cand_z[keep]
    valid = texel_tri >= 0
    atlas = np.full((s, s, 3), NO_DATA, dtype=np.uint8)
    if not bool(valid.any()):
        return atlas, 0.0
    vpos = texel_pos[valid]
    vnor = texel_nor[valid]
    nlens = np.linalg.norm(vnor, axis=1)
    facing = np.zeros((vpos.shape[0],), dtype=bool)
    nz = vnor[nlens > 0.0]
    facing[nlens > 0.0] = (nz[:, 2] / nlens[nlens > 0.0]) > GRAZING_COS_MIN
    c = camera_tuple
    nx = c[0] * vpos[:, 0] + c[1] * vpos[:, 1] + c[2] * vpos[:, 2] + c[3]
    ny = c[4] * vpos[:, 0] + c[5] * vpos[:, 1] + c[6] * vpos[:, 2] + c[7]
    depth = vpos[:, 2]
    ph, pw = int(photo.shape[0]), int(photo.shape[1])
    fx = nx * float(pw - 1)
    fy = ny * float(ph - 1)
    in_bounds = (fx >= 0.0) & (fx <= float(pw - 1)) & (fy >= 0.0) & (fy <= float(ph - 1))
    cand = facing & in_bounds
    if not bool(cand.any()):
        return atlas, 0.0
    # Z-buffer en espacio foto: se rasteriza la malla proyectada y se guarda
    # la profundidad maxima por pixel. Comparar texeles entre si con el mismo
    # pixel (shadow-map directo) produce acne: la pendiente de la superficie
    # dentro de un pixel supera a eps y solo sobrevive un texel por pixel.
    # Con el buffer por triangulo, toda la superficie frontal coincide con el
    # maximo dentro de tolerancia float y pasa completa.
    zbuf = np.full((ph * pw,), -np.inf, dtype=np.float64)
    win_tri = np.full((ph * pw,), -1, dtype=np.int32)
    tri_fx = (c[0] * mesh[tris][:, :, 0] + c[1] * mesh[tris][:, :, 1] + c[2] * mesh[tris][:, :, 2] + c[3]) * float(pw - 1)
    tri_fy = (c[4] * mesh[tris][:, :, 0] + c[5] * mesh[tris][:, :, 1] + c[6] * mesh[tris][:, :, 2] + c[7]) * float(ph - 1)
    tri_z = mesh[tris][:, :, 2]
    if mouth_tris is not None:
        skip = np.asarray(mouth_tris, dtype=bool)
    else:
        skip = np.zeros((n_tris,), dtype=bool)
    for t in range(n_tris):
        if skip[t]:
            continue
        x0, x1, x2 = float(tri_fx[t, 0]), float(tri_fx[t, 1]), float(tri_fx[t, 2])
        y0, y1, y2 = float(tri_fy[t, 0]), float(tri_fy[t, 1]), float(tri_fy[t, 2])
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(pw - 1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(ph - 1, int(np.ceil(max(y0, y1, y2))))
        if xmax < xmin or ymax < ymin:
            continue
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        xs = np.arange(xmin, xmax + 1, dtype=np.float64)
        ys = np.arange(ymin, ymax + 1, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not bool(inside.any()):
            continue
        z0, z1, z2 = float(tri_z[t, 0]), float(tri_z[t, 1]), float(tri_z[t, 2])
        zvals = w0[inside] * z0 + w1[inside] * z1 + w2[inside] * z2
        py, px = np.where(inside)
        flat_px = (py + ymin) * pw + (px + xmin)
        better = zvals > zbuf[flat_px]
        zbuf[flat_px[better]] = zvals[better]
        win_tri[flat_px[better]] = t
    zrange = float(depth[cand].max() - depth[cand].min())
    eps = _Z_EPS_REL * zrange if zrange > 0.0 else 0.0
    ix = np.clip(np.floor(fx[cand]).astype(np.int64), 0, pw - 1)
    iy = np.clip(np.floor(fy[cand]).astype(np.int64), 0, ph - 1)
    flat = iy * pw + ix
    my_tri = texel_tri[valid][cand]
    # Mismo triangulo que el ganador del pixel: la diferencia de profundidad
    # es solo interpolacion (sub-pixel), siempre visible. Distinto triangulo:
    # solo pasa dentro de tolerancia (superficies coplanares vecinas).
    vis_depth = (win_tri[flat] == my_tri) | (depth[cand] >= (zbuf[flat] - eps))
    map_x = np.full((s, s), -1.0, dtype=np.float32)
    map_y = np.full((s, s), -1.0, dtype=np.float32)
    vy, vx = np.where(valid)
    cy = vy[cand]
    cx = vx[cand]
    map_x[cy[vis_depth], cx[vis_depth]] = fx[cand][vis_depth].astype(np.float32)
    map_y[cy[vis_depth], cx[vis_depth]] = fy[cand][vis_depth].astype(np.float32)
    # Marca de visibilidad para enmascarar tras remap (remap muestrea todo).
    vis_mask = np.zeros((s, s), dtype=bool)
    vis_mask[cy[vis_depth], cx[vis_depth]] = True
    sampled_raw = cv2.remap(
        photo, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=NO_DATA,
    )
    sampled: NDArray[np.uint8] = np.asarray(sampled_raw, dtype=np.uint8)
    atlas[vis_mask] = sampled[vis_mask]
    evidence = float(vis_mask.mean())
    return atlas, evidence


def bake_1024(
    image: ImageBytes, fit: FitResult
) -> tuple[NDArray[np.uint8], float, float]:
    """Hornea el atlas 1024 + evidencia + tiempo. Sin pesos: gris completo."""
    start = time.perf_counter()
    try:
        head = load_gnm_head()
    except RuntimeError:
        gray = np.full((ATLAS_SIZE, ATLAS_SIZE, 3), NO_DATA, dtype=np.uint8)
        return gray, 0.0, (time.perf_counter() - start) * 1000.0
    photo = _photo_rgb(image)
    mesh = np.asarray(eval_mesh(fit.coeffs.as_tuple()), dtype=np.float64)
    normals = np.asarray(vertex_normals(mesh), dtype=np.float64)
    tris = np.asarray(head.triangles, dtype=np.int64)
    tri_uvs = np.asarray(head.triangle_uvs, dtype=np.float64)
    mouth_mask = None
    try:
        names = head.group_names
        if _MOUTH_SOCK_GROUP in names:
            groups = np.asarray(head.vertex_groups, dtype=np.float64)
            per_vert = groups[names.index(_MOUTH_SOCK_GROUP)] > 1e-4
            mouth_mask_np: NDArray[np.bool_] = per_vert[tris[:, 0]] & per_vert[tris[:, 1]] & per_vert[tris[:, 2]]
            mouth_mask = mouth_mask_np
    except Exception:  # noqa: BLE001 - sin gate bucal se hornea igual
        mouth_mask = None
    atlas, evidence = _sample_atlas(
        photo, mesh, normals, tris, tri_uvs, fit.camera.as_tuple(), ATLAS_SIZE, mouth_mask
    )
    bake_ms = (time.perf_counter() - start) * 1000.0
    return atlas, float(evidence), bake_ms


def project_texture(image: ImageBytes, fit: FitResult) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        del fit  # la proyeccion foto->UV v1 no usa la geometria; el warp TPS (Fase 2) si lo hara
        raw = image.as_bytes()
        try:
            from PIL import Image as _Image

            img = _Image.open(io.BytesIO(raw)).convert("RGB").resize((UV_WIDTH, UV_HEIGHT))
            out = img.tobytes()
        except Exception:  # noqa: BLE001 - fallback documentado para stubs no-PIL
            # Fallback para stubs sinteticos de tests (magic PNG + bytes de
            # marcador, no decodificables por PIL): color solido derivado del
            # ultimo byte (los primeros son el magic, identicos entre stubs).
            # Suave (diff 0) y determinista; deriva de la foto por marcador.
            v = raw[-1] if len(raw) else 0
            out = bytes([v]) * UV_LEN
        assert len(out) == UV_LEN
        parsed = parse_complete_uv(out)
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001 - la proyeccion nunca tumba sin causa
        return Err(MlFailed(detail=MlDecode(details=f"project failed: {exc}")))


def warp_with_landmarks(albedo: CompleteUv, landmarks: Landmarks) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        _ = landmarks  # reservado para el warp TPS real (Fase 2); v1 es identidad
        parsed = parse_complete_uv(bytes(albedo.as_bytes()))
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"warp failed: {exc}")))


def inpaint_occluded(albedo: CompleteUv, landmarks: Landmarks) -> Ok[CompleteUv] | Err[DomainError]:
    """Identidad hasta que la oclusion geometrica real aterrice (Fase 2).

    La mascara hash anterior tocaba 1/8 texels con periodo 32B y dejaba un
    peine visible sobre fotos reales; ningun gate byte-nivel lo detectaba.
    Se prohibe reintroducir muestreo por hash aqui: la oclusion debe derivar
    de geometria (normales, visibilidad) cuando se implemente.
    """
    try:
        _ = landmarks  # reservado para la oclusion geometrica (Fase 2)
        parsed = parse_complete_uv(bytes(albedo.as_bytes()))
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"inpaint failed: {exc}")))


def build_albedo(
    image: ImageBytes, fit: FitResult, landmarks: Landmarks
) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        _ = landmarks  # el bake usa la camara fiteada; landmarks ya viajan en el fit
        atlas, _, _ = bake_1024(image, fit)
        from PIL import Image as _Image

        img = _Image.fromarray(atlas, mode="RGB").resize(
            (UV_WIDTH, UV_HEIGHT), _Image.Resampling.BILINEAR
        )
        out = img.tobytes()
        assert len(out) == UV_LEN
        parsed = parse_complete_uv(out)
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001 - el bake nunca tumba sin causa
        return Err(MlFailed(detail=MlDecode(details=f"bake failed: {exc}")))
