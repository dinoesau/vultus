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
import logging
import time
from dataclasses import dataclass

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
# S2 plan-gnm-face-coverage: el ovalo foto-espacial sustituye al angulo como
# guardian contra fondo, asi que el grazing se relaja a hemisferio (0.0).
# Antes 0.3 (plan-gnm-texture-bake, medido como unico freno anti-fondo).
GRAZING_COS_MIN = 0.0
# Ovalo MediaPipe FaceMesh (FACEMESH_FACE_OVAL): anillo de 36 landmarks que
# delimita la silueta facial en espacio foto. Orden del loop original.
OVAL_INDICES: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
# Margen del ovalo en px foto: calibrado en Bush S2 (barrido 0/4/8/12/16:
# 8px cubre mejillas/sienes sin tocar fondo; ver PR). Constante nombrada,
# cambio en una linea si el checkpoint humano pide recalibrar.
OVAL_MARGIN_PX = 8.0
# Tolerancia de profundidad relativa al rango: la malla es densa (~2 tris por
# pixel de foto LFW 250px), asi que la profundidad baricentrica del texel y la
# del centro del pixel difieren por pendiente x offset sub-pixel (hasta ~2% en
# la nariz). 0.02 cubre esa variacion sin alcanzar el gap real entre capas
# (cara vs nuca ~78% del rango). Medido en Bush: 0.001->6%, 0.01->15%, 0.03->30%.
_Z_EPS_REL = 0.02
# Paso de la grilla del z-buffer en espacio foto (S3c): a 1px las facetas
# sub-pixel (medido: faceta huerfana de 0.75px sin winner que la cubra en 3x3)
# dejan texeles sin plano contra el que compararse. A 0.5px la faceta cubre
# muestras cercanas a cada texel. Coste ~4x en el loop de raster foto
# (limitado por timeout texture 30s; bake Bush ~3s). Sin filosofia: resolucion.
_Z_SS = 2
# Tolerancia por diedro S3c: la discrepancia entre facetas adyacentes de la
# misma superficie lisa esta acotada por su angulo diedro (vecinos lisos con
# diedro < ~10 grados discrepan ~0.015 medido en Bush); una oclusion real
# (aleta nasal, labio, oreja, nuca) excede esa cota por ordenes de magnitud.
# K alcanza el mismatch liso medido a ~10 grados; CAP satura para que ningun
# solape medio/grande pueda pasar por angulo. Capas paralelas (diedro 0,
# frente vs fondo) reciben solo el eps base: el contra-guardrail
# `test_micro_occluded_layer_stays_gray` lo fija. Calibrado rojo-primero en
# Bush (guardrail de cobertura), no adivinado.
_DIHEDRAL_K_PER_RAD = 0.06
_DIHEDRAL_CAP_RAD = 0.2617993877991494  # 15 grados
_MOUTH_SOCK_GROUP = "mouth_sock"
# Eyeballs fuera de evidencia (S3): `eye_interiors` == union exacta de
# scleras+irises+pupils (medido: 770 verts / 1488 tris AND en ambos).
# Superficie interior disjunta de skin (0 tris con vert skin) y de boca;
# parpados (left/right_eye, 1536 tris con algun vert fuera) se conservan.
_EYE_BALL_GROUP = "eye_interiors"

logger = logging.getLogger("gnm-texture")


@dataclass(frozen=True, slots=True)
class BakeCounters:
    """Rechazos por compuerta sobre texeles validos (S1 plan-gnm-face-coverage).

    `mouth_tris_skipped` cuenta tris excluidos pre-raster (no entran en
    `total_valid`); las otras compuertas + `sampled` particionan
    `total_valid` y deben sumar exacto (eval S1/S2).
    S2 agrega `oval_rejected` entre grazing y bounds.
    """

    total_valid: int
    facing_rejected: int
    grazing_rejected: int
    oval_rejected: int
    bounds_rejected: int
    z_rejected: int
    mouth_tris_skipped: int
    eye_tris_skipped: int
    sampled: int
    # Tolerancia z efectiva del bake: referencia INDEPENDIENTE DE COMPUERTAS
    # (rango sobre texeles validos post-raster anatomico, no sobre el cand
    # filtrado). Si derivara del cand, cada compuerta recalibraria el z-buffer
    # acoplando calibraciones (S2: el ovalo partia el rango y el eps a la
    # mitad, matando ~95k texeles en Bush). Se loguea en cada bake para que
    # el proximo acople sea visible sin adivinar.
    z_eps: float
    # Modo de comparacion S3b: admitidos por plano-subpixel vs por fallback
    # de pixel (fuera del footprint del ganador). Suman `sampled`.
    z_subpixel: int
    z_fallback: int


def _empty_counters(mouth_tris_skipped: int = 0, eye_tris_skipped: int = 0) -> BakeCounters:
    return BakeCounters(
        total_valid=0,
        facing_rejected=0,
        grazing_rejected=0,
        oval_rejected=0,
        bounds_rejected=0,
        z_rejected=0,
        mouth_tris_skipped=int(mouth_tris_skipped),
        eye_tris_skipped=int(eye_tris_skipped),
        sampled=0,
        z_eps=0.0,
        z_subpixel=0,
        z_fallback=0,
    )


def _rotate_normals(
    vnor: NDArray[np.float64], camera_tuple: tuple[float, ...]
) -> NDArray[np.float64]:
    """Normales a espacio camara antes del facing (S3).

    Filas normalizadas (tolerante a la escala del weak-perspective; el facing
    usa cosenos). Tercera fila degenerada (el fit actual deja 3a fila en
    ceros: camara frontal sin rotacion) = identidad, sin cambio de
    comportamiento para cams existentes. Cams futuras con rotacion real
    rotan el facing con ellas.
    """
    c = camera_tuple
    rows = np.asarray(
        [[c[0], c[1], c[2]], [c[4], c[5], c[6]], [c[8], c[9], c[10]]],
        dtype=np.float64,
    )
    if float(np.linalg.norm(rows[2])) < 1e-12:
        return np.asarray(vnor, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    rot = rows / norms
    return np.asarray(np.asarray(vnor, dtype=np.float64) @ rot.T, dtype=np.float64)


def _convex_hull(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hull convexo (monotone chain) CCW sin punto duplicado final.

    Entrada (N,2) finita; salida (M,2). Sin scipy (CI solo instala API deps).
    Se prueba solo a traves del seam `_sample_atlas`, nunca como unidad.
    """
    cleaned = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    cleaned = cleaned[np.all(np.isfinite(cleaned), axis=1)]
    order = np.lexsort((cleaned[:, 1], cleaned[:, 0]))
    sorted_pts = cleaned[order]
    unique: list[tuple[float, float]] = []
    for x, y in sorted_pts.tolist():
        if not unique or unique[-1] != (x, y):
            unique.append((x, y))
    if len(unique) < 3:
        return np.asarray(unique, dtype=np.float64).reshape(-1, 2)

    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64)


def _points_in_hull(
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    hull: NDArray[np.float64],
    margin_px: float,
) -> NDArray[np.bool_]:
    """Dentro del hull convexo CCW con margen uniforme en px.

    Distancia al borde = cross/|edge|; dentro con margen exige
    cross >= -margin*|edge| en cada arista. Vectorizado sobre aristas.
    Hull degenerado (<3 pts): rechaza todo (gris honesto, jamas falso).
    """
    xs = np.asarray(px, dtype=np.float64).ravel()
    ys = np.asarray(py, dtype=np.float64).ravel()
    poly = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3 or xs.shape != ys.shape:
        return np.zeros(xs.shape, dtype=bool)
    inside = np.ones(xs.shape, dtype=bool)
    m = float(margin_px)
    n = int(poly.shape[0])
    for i in range(n):
        x0, y0 = float(poly[i, 0]), float(poly[i, 1])
        x1, y1 = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        ex, ey = x1 - x0, y1 - y0
        elen = float(np.hypot(ex, ey))
        if elen <= 0.0:
            continue
        cross = ex * (ys - y0) - ey * (xs - x0)
        inside &= cross >= -m * elen
        if not bool(inside.any()):
            break
    return inside


def _oval_px_from_landmarks(
    landmarks: Landmarks, photo_w: int, photo_h: int
) -> NDArray[np.float64] | None:
    """Ovalo en px foto desde landmarks ya probados. None si degenerado.

    Los landmarks cruzan probados (parse_landmarks en el borde); esto es
    parse-en-borde ya pagado, no revalidacion del core.
    """
    pts = landmarks.as_tuple()
    oval = np.asarray(
        [[pts[i][0] * float(photo_w - 1), pts[i][1] * float(photo_h - 1)] for i in OVAL_INDICES],
        dtype=np.float64,
    )
    if oval.shape != (len(OVAL_INDICES), 2) or not bool(np.all(np.isfinite(oval))):
        return None
    return oval


def _photo_rgb(image: ImageBytes) -> NDArray[np.uint8]:
    from PIL import Image as _Image

    img = _Image.open(io.BytesIO(image.as_bytes())).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _remap_bilinear(
    photo: NDArray[np.uint8],
    map_x: NDArray[np.float32],
    map_y: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Muestreo bilineal con borde constante `NO_DATA`.

    Via `cv2.remap` cuando esta disponible (prod/GPU); fallback numpy puro
    cuando no (CI solo instala API deps, sin cv2). En coords enteras ambos
    son exactos; en fraccionarias coinciden dentro de redondeo.
    """
    try:
        import cv2

        cv2.setNumThreads(1)
        out = cv2.remap(
            photo, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=NO_DATA,
        )
        return np.asarray(out, dtype=np.uint8)
    except ImportError:
        pass
    src = np.asarray(photo, dtype=np.float64)
    ph, pw = src.shape[0], src.shape[1]
    mx = np.asarray(map_x, dtype=np.float64)
    my = np.asarray(map_y, dtype=np.float64)
    out = np.full(mx.shape + (3,), NO_DATA, dtype=np.float64)
    ok = (mx >= 0.0) & (mx <= float(pw - 1)) & (my >= 0.0) & (my <= float(ph - 1))
    if not bool(ok.any()):
        return np.asarray(out, dtype=np.uint8)
    x0 = np.floor(mx[ok]).astype(np.int64)
    y0 = np.floor(my[ok]).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, pw - 1)
    y1 = np.clip(y0 + 1, 0, ph - 1)
    x0c = np.clip(x0, 0, pw - 1)
    y0c = np.clip(y0, 0, ph - 1)
    fx = (mx[ok] - x0).astype(np.float64)
    fy = (my[ok] - y0).astype(np.float64)
    c00 = src[y0c, x0c]
    c10 = src[y0c, x1]
    c01 = src[y1, x0c]
    c11 = src[y1, x1]
    w = (fx * fy)[:, None]
    top = c00 * (1.0 - fx - fy + fx * fy)[:, None] + c10 * (fx - fx * fy)[:, None]
    bot = c01 * (fy - w[:, 0])[:, None] + c11 * w
    out[ok] = top + bot
    return np.asarray(np.round(out), dtype=np.uint8)


def _sample_atlas(
    photo: NDArray[np.uint8],
    mesh: NDArray[np.float64],
    normals: NDArray[np.float64],
    tris: NDArray[np.int64],
    tri_uvs: NDArray[np.float64],
    camera_tuple: tuple[float, ...],
    atlas_size: int,
    mouth_tris: NDArray[np.bool_] | None = None,
    eye_tris: NDArray[np.bool_] | None = None,
    oval_px: NDArray[np.float64] | None = None,
    oval_margin_px: float = OVAL_MARGIN_PX,
) -> tuple[NDArray[np.uint8], float, BakeCounters]:
    """Nucleo puro del bake: raster + visibilidad + remap bilineal.

    `camera_tuple` son 12 floats row-major 3x4 (misma semantica que
    `gnm_fit.project`). Devuelve (atlas SxSx3 uint8, evidence_frac, counters).
    Solo `remap` (+ `dilate` permitido, no usado para no rellenar) de cv2;
    sin cv2 (CI) usa el fallback numpy de `_remap_bilinear`.

    Compuertas S2: facing (hemisferio nz<=0) -> grazing (relajado a hemisferio,
    contador siempre 0; el ovalo guarda contra fondo) -> oval (hull MediaPipe
    + margen; None = sin filtro, preserva comportamiento S1 para fixtures
    viejas) -> bounds -> z. sampled+resto == total_valid.
    """
    s = int(atlas_size)
    n_tris = int(tris.shape[0])
    mouth_skipped = int(np.asarray(mouth_tris, dtype=bool).sum()) if mouth_tris is not None else 0
    eye_skipped = int(np.asarray(eye_tris, dtype=bool).sum()) if eye_tris is not None else 0
    if mouth_tris is not None:
        skip_raster = np.asarray(mouth_tris, dtype=bool)
    else:
        skip_raster = np.zeros((n_tris,), dtype=bool)
    if eye_tris is not None:
        skip_raster = skip_raster | np.asarray(eye_tris, dtype=bool)
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
        if bool(skip_raster[t]):
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
        return atlas, 0.0, _empty_counters(mouth_skipped, eye_skipped)
    # Referencia de tolerancia: rango sobre VALIDOS (post-raster), nunca sobre
    # el cand filtrado. Ver invariante en `BakeCounters.z_eps`.
    valid_range = float(texel_depth[valid].max() - texel_depth[valid].min())
    z_eps = _Z_EPS_REL * valid_range if valid_range > 0.0 else 0.0
    vpos = texel_pos[valid]
    vnor = _rotate_normals(texel_nor[valid], camera_tuple)
    nlens = np.linalg.norm(vnor, axis=1)
    cos_z = np.zeros((vpos.shape[0],), dtype=np.float64)
    has_n = nlens > 0.0
    cos_z[has_n] = vnor[has_n][:, 2] / nlens[has_n]
    facing_pass = cos_z > 0.0
    grazing_pass = cos_z > GRAZING_COS_MIN
    facing_rejected = int((~facing_pass).sum())
    grazing_rejected = int((facing_pass & ~grazing_pass).sum())
    grazed = facing_pass & grazing_pass
    c = camera_tuple
    nx = c[0] * vpos[:, 0] + c[1] * vpos[:, 1] + c[2] * vpos[:, 2] + c[3]
    ny = c[4] * vpos[:, 0] + c[5] * vpos[:, 1] + c[6] * vpos[:, 2] + c[7]
    depth = vpos[:, 2]
    ph, pw = int(photo.shape[0]), int(photo.shape[1])
    fx = nx * float(pw - 1)
    fy = ny * float(ph - 1)
    in_bounds = (fx >= 0.0) & (fx <= float(pw - 1)) & (fy >= 0.0) & (fy <= float(ph - 1))
    total_valid = int(valid.sum())
    if oval_px is not None:
        hull = _convex_hull(np.asarray(oval_px, dtype=np.float64))
        if hull.shape[0] < 3:
            # Landmarks degenerados (p. ej. dummy de tests con 478 pts
            # identicos): sin poligono no hay mascara; se hornea igual sin
            # filtro para no destruir evidencia por un borde patologico.
            # MediaPipe real nunca es degenerado.
            logger.warning("oval gate degradado: hull degenerado, sin filtro de ovalo")
            oval_rejected = 0
            ovo = grazed
        else:
            in_oval_full = _points_in_hull(fx, fy, hull, float(oval_margin_px))
            oval_rejected = int((grazed & ~in_oval_full).sum())
            ovo = grazed & in_oval_full
    else:
        oval_rejected = 0
        ovo = grazed
    bounds_rejected = int((ovo & ~in_bounds).sum())
    cand = ovo & in_bounds
    if not bool(cand.any()):
        counters = BakeCounters(
            total_valid=total_valid,
            facing_rejected=facing_rejected,
            grazing_rejected=grazing_rejected,
            oval_rejected=oval_rejected,
            bounds_rejected=bounds_rejected,
            z_rejected=0,
            mouth_tris_skipped=mouth_skipped,
            eye_tris_skipped=eye_skipped,
            sampled=0,
            z_eps=z_eps,
            z_subpixel=0,
            z_fallback=0,
        )
        return atlas, 0.0, counters
    # Z-buffer en espacio foto: se rasteriza la malla proyectada y se guarda
    # la profundidad maxima por pixel. Comparar texeles entre si con el mismo
    # pixel (shadow-map directo) produce acne: la pendiente de la superficie
    # dentro de un pixel supera a eps y solo sobrevive un texel por pixel.
    # Con el buffer por triangulo, toda la superficie frontal coincide con el
    # maximo dentro de tolerancia float y pasa completa.
    # S3c: grilla al doble de resolucion (ver `_Z_SS`). El texel consulta en
    # semipixeles; una faceta sub-pixel cubre muestras a <=0.5px del punto.
    ss = int(_Z_SS)
    pw2 = int(pw * ss)
    ph2 = int(ph * ss)
    zbuf = np.full((ph2 * pw2,), -np.inf, dtype=np.float64)
    win_tri = np.full((ph2 * pw2,), -1, dtype=np.int32)
    tri_fx = (c[0] * mesh[tris][:, :, 0] + c[1] * mesh[tris][:, :, 1] + c[2] * mesh[tris][:, :, 2] + c[3]) * float(pw - 1) * float(ss)
    tri_fy = (c[4] * mesh[tris][:, :, 0] + c[5] * mesh[tris][:, :, 1] + c[6] * mesh[tris][:, :, 2] + c[7]) * float(ph - 1) * float(ss)
    tri_z = mesh[tris][:, :, 2]
    skip = skip_raster
    for t in range(n_tris):
        if skip[t]:
            continue
        x0, x1, x2 = float(tri_fx[t, 0]), float(tri_fx[t, 1]), float(tri_fx[t, 2])
        y0, y1, y2 = float(tri_fy[t, 0]), float(tri_fy[t, 1]), float(tri_fy[t, 2])
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(pw2 - 1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(ph2 - 1, int(np.ceil(max(y0, y1, y2))))
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
        flat_px = (py + ymin) * pw2 + (px + xmin)
        # Empate (>=, gana el ultimo): a igualdad exacta de profundidad el
        # maximo no cambia, pero el test sub-pixel S3b necesita ver un tri
        # cuyo footprint contenga el punto. Con `>` estricto un texel podia
        # quedar huerfano (su tri pierde todos los empates con un coplanar
        # vecino y el ganador no cubre su sub-posicion). El empate exacto
        # entre superficies distintas es medida cero; entre coplanares el
        # plano es el mismo y el test decide igual.
        better = zvals >= zbuf[flat_px]
        zbuf[flat_px[better]] = zvals[better]
        win_tri[flat_px[better]] = t
    # S3b: el maximo por pixel se evalua en los ENTEROS del pixel, pero el texel
    # vive en su SUB-posicion continua. En escorzo (15 texeles/pixel medido)
    # esa diferencia es pendiente legitima, no oclusion, y el atajo same_tri
    # solo rescata al tri ganador exacto (2.6% medido). Por eso cada candidato
    # se compara contra los planos GANADORES del bloque 3x3 que rodea su
    # (fx,fy), interpolados en el punto exacto: la misma superficie coincide
    # por construccion (tol z_eps sin estrechar: no se recalibra nada); la
    # oclusion real (capas distintas) sigue difiriendo >> eps. Un solo pixel
    # no basta: con facetas <1px el ganador de floor(fx,fy) puede no cubrir
    # el punto (medido en fixture: faceta de 0.75px huerfana). Sin ningun
    # footprint que lo contenga (siluetas, discretizacion) rige el test por
    # pixel; jamas gris automatico. Todo vectorizado: loop constante sobre
    # 9 esquinas de la grilla 2x, nunca por texel.
    fss = float(ss)
    hxc = fx[cand] * fss
    hyc = fy[cand] * fss
    hx0 = np.clip(np.floor(hxc).astype(np.int64), 0, pw2 - 1)
    hy0 = np.clip(np.floor(hyc).astype(np.int64), 0, ph2 - 1)
    dc = depth[cand]
    my_tri = texel_tri[valid][cand]
    # Normales por cara para el diedro S3c (una vez por bake, no por texel;
    # degenerados quedan en cero y reciben solo el eps base).
    _tpts = mesh[tris]
    _fn_raw = np.cross(_tpts[:, 1] - _tpts[:, 0], _tpts[:, 2] - _tpts[:, 0])
    _fn_len = np.linalg.norm(_fn_raw, axis=1, keepdims=True)
    face_nor: NDArray[np.float64] = _fn_raw / np.where(_fn_len > 1e-12, _fn_len, 1.0)
    vis_sub = np.zeros((dc.shape[0],), dtype=bool)
    inside_any = np.zeros((dc.shape[0],), dtype=bool)
    for dx in (-1, 0, 1):
        qx = np.clip(hx0 + dx, 0, pw2 - 1)
        for dy in (-1, 0, 1):
            qy = np.clip(hy0 + dy, 0, ph2 - 1)
            w_idx = win_tri[qy * pw2 + qx]
            w_safe = np.where(w_idx >= 0, w_idx, 0)
            gx0 = tri_fx[w_safe, 0]
            gx1 = tri_fx[w_safe, 1]
            gx2 = tri_fx[w_safe, 2]
            gy0 = tri_fy[w_safe, 0]
            gy1 = tri_fy[w_safe, 1]
            gy2 = tri_fy[w_safe, 2]
            gz0 = tri_z[w_safe, 0]
            gz1 = tri_z[w_safe, 1]
            gz2 = tri_z[w_safe, 2]
            w_denom = (gy1 - gy2) * (gx0 - gx2) + (gx2 - gx1) * (gy0 - gy2)
            w_ok = np.abs(w_denom) >= 1e-12
            w_denom_safe: NDArray[np.float64] = np.where(w_ok, w_denom, 1.0)
            wb0 = ((gy1 - gy2) * (hxc - gx2) + (gx2 - gx1) * (hyc - gy2)) / w_denom_safe
            wb1 = ((gy2 - gy0) * (hxc - gx2) + (gx0 - gx2) * (hyc - gy2)) / w_denom_safe
            wb2 = 1.0 - wb0 - wb1
            inside_win = w_ok & (w_idx >= 0) & (wb0 >= -1e-9) & (wb1 >= -1e-9) & (wb2 >= -1e-9)
            dw = wb0 * gz0 + wb1 * gz1 + wb2 * gz2
            # Diedro S3c entre el tri del texel y el ganador: la tolerancia
            # crece con el angulo hasta saturar (ver `_DIHEDRAL_*`).
            gA = face_nor[my_tri]
            gB = face_nor[w_safe]
            denom_n = np.linalg.norm(gA, axis=1) * np.linalg.norm(gB, axis=1)
            ok_n = denom_n > 1e-24
            cosang = np.where(
                ok_n,
                np.sum(gA * gB, axis=1) / np.where(ok_n, denom_n, 1.0),
                1.0,
            )
            ang = np.arccos(np.clip(cosang, -1.0, 1.0))
            allow = z_eps + _DIHEDRAL_K_PER_RAD * np.minimum(ang, _DIHEDRAL_CAP_RAD)
            inside_any |= inside_win
            vis_sub |= inside_win & (np.abs(dc - dw) <= allow)
    flat = hy0 * pw2 + hx0
    vis_pix = (win_tri[flat] == my_tri) | (dc >= (zbuf[flat] - z_eps))
    vis_depth = vis_sub | (~inside_any & vis_pix)
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
    sampled: NDArray[np.uint8] = _remap_bilinear(photo, map_x, map_y)
    atlas[vis_mask] = sampled[vis_mask]
    evidence = float(vis_mask.mean())
    z_rejected = int(cand.sum() - int(vis_depth.sum()))
    counters = BakeCounters(
        total_valid=total_valid,
        facing_rejected=facing_rejected,
        grazing_rejected=grazing_rejected,
        oval_rejected=oval_rejected,
        bounds_rejected=bounds_rejected,
        z_rejected=z_rejected,
        mouth_tris_skipped=mouth_skipped,
        eye_tris_skipped=eye_skipped,
        sampled=int(vis_depth.sum()),
        z_eps=z_eps,
        z_subpixel=int(vis_sub.sum()),
        z_fallback=int((~inside_any & vis_pix).sum()),
    )
    return atlas, evidence, counters


def bake_1024(
    image: ImageBytes, fit: FitResult, landmarks: Landmarks | None = None
) -> tuple[NDArray[np.uint8], float, float, BakeCounters]:
    """Hornea el atlas 1024 + evidencia + tiempo + contadores. Sin pesos: gris completo.

    `landmarks` alimenta solo la compuerta `in_oval` (hull + margen); None la
    omite (compat con callers viejos y fixtures sin landmarks).
    """
    start = time.perf_counter()
    try:
        head = load_gnm_head()
    except RuntimeError:
        gray = np.full((ATLAS_SIZE, ATLAS_SIZE, 3), NO_DATA, dtype=np.uint8)
        return gray, 0.0, (time.perf_counter() - start) * 1000.0, _empty_counters()
    photo = _photo_rgb(image)
    mesh = np.asarray(eval_mesh(fit.coeffs), dtype=np.float64)
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
    except Exception as exc:  # noqa: BLE001 - sin gate bucal se hornea igual, pero ruidoso
        logger.warning("mouth gate degradado: sin exclusion bucal por %s", exc)
        mouth_mask = None
    # S3: medida, no adivinada. mouth_sock 406 verts / 752 tris AND, disjunto
    # de dientes/lengua/ojos; labios (upper/lower_lip) intactos fuera del sock.
    # Se conserva la regla AND sin mover threshold (ver PR con numeros Bush).
    eye_mask = None
    try:
        names = head.group_names
        if _EYE_BALL_GROUP in names:
            groups = np.asarray(head.vertex_groups, dtype=np.float64)
            per_vert_eye = groups[names.index(_EYE_BALL_GROUP)] > 1e-4
            eye_mask_np: NDArray[np.bool_] = per_vert_eye[tris[:, 0]] & per_vert_eye[tris[:, 1]] & per_vert_eye[tris[:, 2]]
            eye_mask = eye_mask_np
    except Exception as exc:  # noqa: BLE001 - sin gate ocular se hornea igual, pero ruidoso
        logger.warning("eye gate degradado: sin exclusion ocular por %s", exc)
        eye_mask = None
    oval: NDArray[np.float64] | None = None
    if landmarks is not None:
        oval = _oval_px_from_landmarks(landmarks, int(photo.shape[1]), int(photo.shape[0]))
    atlas, evidence, counters = _sample_atlas(
        photo, mesh, normals, tris, tri_uvs, fit.camera.as_tuple(), ATLAS_SIZE, mouth_mask, eye_mask,
        oval_px=oval, oval_margin_px=OVAL_MARGIN_PX,
    )
    bake_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "bake done evidence=%.4f sampled=%d facing=%d grazing=%d oval=%d bounds=%d z=%d mouth_tris=%d eye_tris=%d eps=%.5f zsub=%d zfb=%d ms=%.1f",
        float(evidence),
        counters.sampled,
        counters.facing_rejected,
        counters.grazing_rejected,
        counters.oval_rejected,
        counters.bounds_rejected,
        counters.z_rejected,
        counters.mouth_tris_skipped,
        counters.eye_tris_skipped,
        counters.z_eps,
        counters.z_subpixel,
        counters.z_fallback,
        bake_ms,
    )
    return atlas, float(evidence), bake_ms, counters


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
        if len(out) != UV_LEN:
            return Err(MlFailed(detail=MlDecode(details=f"project len {len(out)} != {UV_LEN}")))
        parsed = parse_complete_uv(out)
        if isinstance(parsed, Err):
            return parsed
        return parsed
    except Exception as exc:  # noqa: BLE001 - la proyeccion nunca tumba sin causa
        return Err(MlFailed(detail=MlDecode(details=f"project failed: {exc}")))


def warp_with_landmarks(albedo: CompleteUv, landmarks: Landmarks) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        _ = landmarks  # reservado para el warp TPS real (Fase 2); v1 es identidad
        parsed = parse_complete_uv(bytes(albedo.as_bytes()))
        if isinstance(parsed, Err):
            return parsed
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
        if isinstance(parsed, Err):
            return parsed
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"inpaint failed: {exc}")))


def build_albedo(
    image: ImageBytes, fit: FitResult, landmarks: Landmarks
) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        atlas, _, _, _ = bake_1024(image, fit, landmarks)
        from PIL import Image as _Image

        img = _Image.fromarray(atlas, mode="RGB").resize(
            (UV_WIDTH, UV_HEIGHT), _Image.Resampling.BILINEAR
        )
        out = img.tobytes()
        if len(out) != UV_LEN:
            return Err(MlFailed(detail=MlDecode(details=f"bake len {len(out)} != {UV_LEN}")))
        parsed = parse_complete_uv(out)
        if isinstance(parsed, Err):
            return parsed
        return parsed
    except Exception as exc:  # noqa: BLE001 - el bake nunca tumba sin causa
        return Err(MlFailed(detail=MlDecode(details=f"bake failed: {exc}")))
