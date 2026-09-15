"""Textura: bake 1024 real + seam build_albedo, goldens congelados a mano."""

from __future__ import annotations

import io
import json

import numpy as np

from backend.domain import UV_LEN, Err, Ok, parse_image_bytes, parse_landmarks
from backend.gnm_fit import fit_gnm
from backend.gnm_texture import (
    ATLAS_SIZE,
    NO_DATA,
    BakeCounters,
    _sample_atlas,
    bake_1024,
    build_albedo,
    inpaint_occluded,
    project_texture,
    warp_with_landmarks,
)


def _image(marker: int):
    from PIL import Image as _Image

    img = _Image.new("RGB", (16, 16), (marker, marker, marker))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    parsed = parse_image_bytes(buf.getvalue())
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks():
    pts = [[0.1, 0.2, 0.3]] * 478
    raw = json.dumps(pts).encode("utf-8")
    parsed = parse_landmarks(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _fit():
    fit = fit_gnm(_image(0xA1), _landmarks())
    assert isinstance(fit, Ok)
    return fit.value


def _micro_quad(z_front: float = 0.0):
    mesh = np.asarray(
        [[0.0, 0.0, z_front], [1.0, 0.0, z_front], [1.0, 1.0, z_front], [0.0, 1.0, z_front]],
        dtype=np.float64,
    )
    normals = np.asarray([[0.0, 0.0, 1.0]] * 4, dtype=np.float64)
    tris = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    tri_uvs = np.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=np.float64,
    )
    return mesh, normals, tris, tri_uvs


def _micro_photo() -> np.ndarray:
    photo = np.zeros((4, 4, 3), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            photo[r, c] = (r * 4 + c, r * 4 + c, r * 4 + c)
    return photo


_IDENTITY_CAM = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_micro_fixture_samples_exact_hand_literals() -> None:
    mesh, normals, tris, tri_uvs = _micro_quad()
    atlas, evidence, _ = _sample_atlas(_micro_photo(), mesh, normals, tris, tri_uvs, _IDENTITY_CAM, 4)
    assert atlas.shape == (4, 4, 3)
    # ix=1,iy=2 -> u=1/3,v=1/3 -> foto (1.0,1.0) = valor 5.
    assert list(atlas[2, 1]) == [5, 5, 5]
    # ix=2,iy=1 -> u=2/3,v=2/3 -> foto (2.0,2.0) = valor 10.
    assert list(atlas[1, 2]) == [10, 10, 10]
    assert evidence > 0.0


_YAW90_CAM = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0)


def test_rotated_camera_marks_edge_on_face_gray() -> None:
    """Facing en espacio camara: cara frontal vista de canto (yaw 90) es gris."""
    mesh, normals, tris, tri_uvs = _micro_quad()
    atlas, _, counters = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs, _YAW90_CAM, 4
    )
    assert bool((atlas == np.asarray(NO_DATA, dtype=np.uint8)).all())
    assert counters.sampled == 0
    assert counters.facing_rejected == counters.total_valid


def test_rotated_camera_samples_turned_face() -> None:
    """La cara girada hacia la camara (normal -X ante yaw 90) si muestrea."""
    mesh = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    normals = np.asarray([[-1.0, 0.0, 0.0]] * 4, dtype=np.float64)
    tris = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    tri_uvs = np.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=np.float64,
    )
    atlas, _, counters = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs, _YAW90_CAM, 4
    )
    assert counters.sampled > 0
    # Vertice-exacto: atlas[0,0] -> (u=0,v=1) -> m3=(0,0,1) -> fx=3,fy=0
    # -> foto[0,3] = valor 3. Sin interpolacion, sin redondeo.
    assert list(atlas[0, 0]) == [3, 3, 3]


def test_eye_tris_are_skipped_not_sampled() -> None:
    """Eyeballs fuera de evidencia (regla AND como mouth_sock, S3)."""
    mesh, normals, tris, tri_uvs = _micro_quad()
    eye = np.asarray([False, True])
    atlas, _, counters = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs, _IDENTITY_CAM, 4,
        eye_tris=eye,
    )
    assert counters.eye_tris_skipped == 1
    assert counters.sampled > 0
    # (ix=0,iy=0) es vertice exclusivo de tri 1 -> gris por ojo...
    assert list(atlas[0, 0]) == list(NO_DATA)
    # ...pero tri 0 sigue muestreando ((ix=3,iy=0): u=1,v=1 -> foto (3,3)=15).
    assert list(atlas[0, 3]) == [15, 15, 15]


def test_micro_counters_sum_to_valid_texels() -> None:
    mesh, normals, tris, tri_uvs = _micro_quad()
    _, _, counters = _sample_atlas(_micro_photo(), mesh, normals, tris, tri_uvs, _IDENTITY_CAM, 4)
    assert isinstance(counters, BakeCounters)
    total = (
        counters.sampled
        + counters.facing_rejected
        + counters.grazing_rejected
        + counters.oval_rejected
        + counters.bounds_rejected
        + counters.z_rejected
    )
    assert total == counters.total_valid
    assert counters.total_valid > 0
    assert counters.sampled > 0
    assert counters.mouth_tris_skipped == 0
    assert counters.eye_tris_skipped == 0
    assert counters.oval_rejected == 0


def test_micro_backface_counts_as_facing() -> None:
    mesh, _, tris, tri_uvs = _micro_quad()
    back_normals = np.asarray([[0.0, 0.0, -1.0]] * 4, dtype=np.float64)
    _, _, counters = _sample_atlas(_micro_photo(), mesh, back_normals, tris, tri_uvs, _IDENTITY_CAM, 4)
    assert counters.sampled == 0
    assert counters.facing_rejected == counters.total_valid
    assert counters.total_valid > 0


def _slanted_normals(nz: float = 0.2) -> np.ndarray:
    nx = float(np.sqrt(max(0.0, 1.0 - nz * nz)))
    return np.asarray([[nx, 0.0, nz]] * 4, dtype=np.float64)


def _big_oval() -> np.ndarray:
    return np.asarray([[-10.0, -10.0], [14.0, -10.0], [14.0, 14.0], [-10.0, 14.0]], dtype=np.float64)


def _folded_front_back() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Dos capas con la MISMA proyeccion (camara identidad) en distintas UVs.
    # Delante: pliegue en la diagonal (dos planos, escorzo geometrico real).
    # Detras: plano z=-0.5 (oclusion real). Normales +Z para aislar el z-test.
    # Atlas s=4: delante en mitad UV izquierda (u<=0.5), detras en derecha.
    front = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.6], [1.0, 1.0, 1.0], [0.0, 1.0, 0.2]],
        dtype=np.float64,
    )
    back = np.asarray(
        [[0.0, 0.0, -0.5], [1.0, 0.0, -0.5], [1.0, 1.0, -0.5], [0.0, 1.0, -0.5]],
        dtype=np.float64,
    )
    mesh = np.vstack([front, back])
    normals = np.asarray([[0.0, 0.0, 1.0]] * 8, dtype=np.float64)
    tris = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
    tri_uvs = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]],
            [[0.0, 0.0], [0.5, 1.0], [0.0, 1.0]],
            [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.0], [1.0, 1.0], [0.5, 1.0]],
        ],
        dtype=np.float64,
    )
    return mesh, normals, tris, tri_uvs


def test_folded_front_occludes_back() -> None:
    """El pliegue de delante muestrea 100% (incluido escorzo) contra su
    propio plano interpolado; la capa de detras queda en gris exacto."""
    mesh, normals, tris, tri_uvs = _folded_front_back()
    atlas, _, counters = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 4, oval_px=_big_oval(), oval_margin_px=0.0,
    )
    gray = np.asarray(NO_DATA, dtype=np.uint8)
    front = atlas[:, 0:2]
    back = atlas[:, 2:4]
    assert not bool((front == gray).all(axis=2).any())
    assert bool((back == gray).all(axis=2).all())
    # Literales vertice-exactos: (u=0,v=1)->pos(0,1)->foto(0,3)=12;
    # (u=1/3,v=0)->x=2/3 (el tri cubre u 0..0.5 sobre x 0..1)->foto(2,0)=2.
    assert list(atlas[0, 0]) == [12, 12, 12]
    assert list(atlas[3, 1]) == [2, 2, 2]
    assert counters.sampled == 8


def _arced_sheet_back() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Loma de 4 facetas (varias facetas por pixel de foto 4x4: el tri ganador
    # del pixel casi nunca es el del texel) + plano trasero oclusor.
    # Normales +Z para aislar el z-test del facing.
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    verts: list[list[float]] = []
    for x in xs:
        z = float(np.sin(np.pi * x) * 0.5)
        verts.append([x, 0.0, z])
        verts.append([x, 1.0, z])
    tris: list[list[int]] = []
    uvs: list[list[list[float]]] = []
    for i in range(4):
        a, b, c, d = 2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1
        tris.append([a, b, c])
        tris.append([a, c, d])
        u0, u1 = i / 8.0, (i + 1) / 8.0
        uvs.append([[u0, 0.0], [u1, 0.0], [u1, 1.0]])
        uvs.append([[u0, 0.0], [u1, 1.0], [u0, 1.0]])
    n0 = len(verts)
    verts += [[0.0, 0.0, -0.5], [1.0, 0.0, -0.5], [1.0, 1.0, -0.5], [0.0, 1.0, -0.5]]
    tris.append([n0, n0 + 1, n0 + 2])
    tris.append([n0, n0 + 2, n0 + 3])
    uvs.append([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0]])
    uvs.append([[0.5, 0.0], [1.0, 1.0], [0.5, 1.0]])
    mesh = np.asarray(verts, dtype=np.float64)
    normals = np.asarray([[0.0, 0.0, 1.0]] * len(verts), dtype=np.float64)
    return mesh, normals, np.asarray(tris, dtype=np.int64), np.asarray(uvs, dtype=np.float64)


def test_arced_sheet_samples_fully_back_occluded() -> None:
    """S3b: la loma continua muestrea integra (cada texel contra el plano del
    ganador interpolado en su sub-pixel); el plano trasero, gris exacto."""
    mesh, normals, tris, tri_uvs = _arced_sheet_back()
    photo = np.zeros((4, 4, 3), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            photo[r, c] = (r * 4 + c, r * 4 + c, r * 4 + c)
    atlas, _, counters = _sample_atlas(
        photo, mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 8, oval_px=_big_oval(), oval_margin_px=0.0,
    )
    gray = np.asarray(NO_DATA, dtype=np.uint8)
    assert not bool((atlas[:, 0:4] == gray).all(axis=2).any())
    assert bool((atlas[:, 4:8] == gray).all(axis=2).all())
    # Vertice-exacto: (u=0,v=1)->(x=0,y=1,z=0)->foto(0,3)=12.
    assert list(atlas[0, 0]) == [12, 12, 12]
    assert counters.sampled == 32


def _gentle_crease() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Pliegue suave (~8 grados): izquierda z=0, derecha z=0.14*(x-0.5).
    # Continuacion lisa: debe muestrear integra (pin de direccion S3c).
    mesh = np.asarray(
        [
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0], [1.0, 0.0, 0.07], [1.0, 1.0, 0.07], [0.5, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    normals = np.asarray([[0.0, 0.0, 1.0]] * 8, dtype=np.float64)
    tris = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
    tri_uvs = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]],
            [[0.0, 0.0], [0.5, 1.0], [0.0, 1.0]],
            [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.0], [1.0, 1.0], [0.5, 1.0]],
        ],
        dtype=np.float64,
    )
    return mesh, normals, tris, tri_uvs


def test_gentle_crease_samples_fully() -> None:
    """Pin S3c (direccion): pliegue suave ~8 grados sin ningun gris."""
    mesh, normals, tris, tri_uvs = _gentle_crease()
    photo = np.zeros((4, 4, 3), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            photo[r, c] = (r * 4 + c, r * 4 + c, r * 4 + c)
    atlas, _, counters = _sample_atlas(
        photo, mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 8, oval_px=_big_oval(), oval_margin_px=0.0,
    )
    gray = np.asarray(NO_DATA, dtype=np.uint8)
    assert not bool((atlas == gray).all(axis=2).any())
    assert counters.sampled == counters.total_valid


def test_steep_tilted_back_stays_gray() -> None:
    """Pin S3c (restriccion): trasera inclinada 45 grados con gap >= 0.2
    queda gris aunque el diedro saturaria la tolerancia (el gap manda)."""
    mesh = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 0.0, -1.2], [1.0, 0.0, -0.2], [1.0, 1.0, -0.2], [0.0, 1.0, -1.2]],
        dtype=np.float64,
    )
    normals = np.asarray([[0.0, 0.0, 1.0]] * 8, dtype=np.float64)
    tris = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
    tri_uvs = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]],
            [[0.0, 0.0], [0.5, 1.0], [0.0, 1.0]],
            [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.0], [1.0, 1.0], [0.5, 1.0]],
        ],
        dtype=np.float64,
    )
    atlas, _, _ = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 4, oval_px=_big_oval(), oval_margin_px=0.0,
    )
    gray = np.asarray(NO_DATA, dtype=np.uint8)
    assert not bool((atlas[:, 0:2] == gray).all(axis=2).any())
    assert bool((atlas[:, 2:4] == gray).all(axis=2).all())


def test_s3b_keeps_outside_oval_gray() -> None:
    """S3b no filtra por z nada fuera del ovalo: todo gris exacto."""
    mesh, normals, tris, tri_uvs = _folded_front_back()
    far = np.asarray([[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]], dtype=np.float64)
    atlas, _, counters = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 4, oval_px=far, oval_margin_px=0.0,
    )
    assert bool((atlas == np.asarray(NO_DATA, dtype=np.uint8)).all())
    assert counters.sampled == 0
    assert counters.oval_rejected == counters.total_valid


def _half_oval() -> np.ndarray:
    # Cubre solo la mitad izquierda del fixture (x < 0.5 en foto 4x4: fx < 1.5).
    return np.asarray([[-10.0, -10.0], [1.5, -10.0], [1.5, 14.0], [-10.0, 14.0]], dtype=np.float64)


def _ramp_quad() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Rampa en profundidad (z crece con x): el oval que corta la mitad honda
    # cambia el rango del cand pero NO el de validos. Normales +Z puras para
    # aislar el comportamiento de z/eps del facing.
    mesh = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    normals = np.asarray([[0.0, 0.0, 1.0]] * 4, dtype=np.float64)
    tris = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    tri_uvs = np.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=np.float64,
    )
    return mesh, normals, tris, tri_uvs


def _curved_patch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Parche curvo 3x3 (z = 0.25*x^2, normals +Z): superficie continua sin
    # pliegues donde el acne de profundidad abriria agujeros interiores.
    xs = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    verts = []
    for y in xs:
        for x in xs:
            verts.append([x, y, 0.25 * x * x])
    mesh = np.asarray(verts, dtype=np.float64)
    normals = np.asarray([[0.0, 0.0, 1.0]] * 9, dtype=np.float64)
    tris = np.asarray(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
         [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]],
        dtype=np.int64,
    )
    uvs = np.asarray(
        [[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]], [[0.0, 0.0], [0.5, 0.5], [0.0, 0.5]],
         [[0.5, 0.0], [1.0, 0.0], [1.0, 0.5]], [[0.5, 0.0], [1.0, 0.5], [0.5, 0.5]],
         [[0.0, 0.5], [0.5, 0.5], [0.5, 1.0]], [[0.0, 0.5], [0.5, 1.0], [0.0, 1.0]],
         [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0]], [[0.5, 0.5], [1.0, 1.0], [0.5, 1.0]]],
        dtype=np.float64,
    )
    return mesh, normals, tris, uvs


def test_z_eps_invariant_to_oval_margin() -> None:
    """El eps no depende del conjunto filtrado (acople S2): mismo fixture,
    ovalo completo vs mitad cortada, identico z_eps y z_eps > 0 en rampa."""
    mesh, normals, tris, tri_uvs = _ramp_quad()
    _, _, c_all = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 4, oval_px=_big_oval(), oval_margin_px=0.0,
    )
    _, _, c_cut = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 4, oval_px=_half_oval(), oval_margin_px=0.0,
    )
    assert c_cut.oval_rejected > 0
    assert c_all.z_eps > 0.0
    assert c_cut.z_eps == c_all.z_eps


def test_curved_patch_has_zero_interior_gray() -> None:
    """Parche curvo: ningun texel valido dentro de la mitad conservada es gris."""
    mesh, normals, tris, tri_uvs = _curved_patch()
    photo = np.zeros((8, 8, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            photo[r, c] = (r * 8 + c, r * 8 + c, r * 8 + c)
    atlas, _, counters = _sample_atlas(
        photo, mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 8, oval_px=_half_oval(), oval_margin_px=0.0,
    )
    assert counters.sampled > 0
    gray = np.asarray(NO_DATA, dtype=np.uint8)
    # El ovalo corta por proyeccion foto (x): con camara identidad fx=x*7 y el
    # corte x>0.5 equivale a u>0.5 (UV = posicion). La mitad conservada son las
    # columnas atlas 0..1 (u<0.21); debe estar integra sin agujeros.
    kept = atlas[:, 0:2]
    assert not bool((kept == gray).all(axis=2).any())


def test_slanted_in_oval_samples_after_relax() -> None:
    """Rojo-primero S2: inclinado (nz=0.2) muestrea dentro del ovalo.

    Con grazing 0.3 este texel era gris; con hemisferio + ovalo es evidencia.
    Margen 0 para probar la compuerta exacta sin interferencia de escala.
    """
    mesh, _, tris, tri_uvs = _micro_quad()
    atlas, _, counters = _sample_atlas(
        _micro_photo(), mesh, _slanted_normals(0.2), tris, tri_uvs,
        _IDENTITY_CAM, 4, oval_px=_big_oval(), oval_margin_px=0.0,
    )
    assert counters.sampled > 0
    assert list(atlas[2, 1]) == [5, 5, 5]


def test_outside_oval_is_exact_gray() -> None:
    mesh, normals, tris, tri_uvs = _micro_quad()
    far = np.asarray([[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]], dtype=np.float64)
    atlas, _, counters = _sample_atlas(
        _micro_photo(), mesh, normals, tris, tri_uvs,
        _IDENTITY_CAM, 4, oval_px=far, oval_margin_px=0.0,
    )
    assert bool((atlas == np.asarray(NO_DATA, dtype=np.uint8)).all())
    assert counters.sampled == 0
    assert counters.oval_rejected == counters.total_valid


def test_micro_backface_is_honest_gray() -> None:
    mesh, _, tris, tri_uvs = _micro_quad()
    back_normals = np.asarray([[0.0, 0.0, -1.0]] * 4, dtype=np.float64)
    atlas, _, _ = _sample_atlas(_micro_photo(), mesh, back_normals, tris, tri_uvs, _IDENTITY_CAM, 4)
    assert bool((atlas == np.asarray(NO_DATA, dtype=np.uint8)).all())


def test_micro_occluded_layer_stays_gray() -> None:
    front_mesh, front_n, front_tris, front_uvs = _micro_quad(z_front=0.0)
    back_mesh, back_n, _, _ = _micro_quad(z_front=-0.5)
    mesh = np.vstack([front_mesh, back_mesh])
    normals = np.vstack([front_n, back_n])
    back_tris = front_tris + 4
    # La capa trasera reutiliza las mismas UVs (mismo XY): el z-buffer la oculta.
    tris = np.vstack([front_tris, back_tris])
    tri_uvs = np.vstack([front_uvs, front_uvs])
    atlas, _, _ = _sample_atlas(_micro_photo(), mesh, normals, tris, tri_uvs, _IDENTITY_CAM, 4)
    # El punto comparte UVs entre capas: gana la frontal (valor 5, no gris).
    assert list(atlas[2, 1]) == [5, 5, 5]


def test_build_albedo_sin_pesos_es_gris_honesto() -> None:
    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
        has_weights = True
    except RuntimeError:
        has_weights = False
    if has_weights:
        import pytest

        pytest.skip("con pesos el bake muestrea de verdad; este golden es solo CI sin pesos")
    out = build_albedo(_image(0xA1), _fit(), _landmarks())
    assert isinstance(out, Ok)
    raw = out.value.as_bytes()
    assert len(raw) == UV_LEN
    assert list(raw[:3]) == [128, 128, 128]
    assert set(raw) == {128}


def test_bake_sin_pesos_gris_completo() -> None:
    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
        has_weights = True
    except RuntimeError:
        has_weights = False
    if has_weights:
        import pytest

        pytest.skip("solo CI sin pesos")
    atlas, evidence, _, _ = bake_1024(_image(0xA1), _fit())
    assert atlas.shape == (ATLAS_SIZE, ATLAS_SIZE, 3)
    assert evidence == 0.0
    assert bool((atlas == 128).all())


def test_albedo_derives_from_real_photo_not_hallucinated() -> None:
    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
        has_weights = True
    except RuntimeError:
        has_weights = False
    if not has_weights:
        import pytest

        pytest.skip("sin pesos todo es gris honesto; nada que derivar")
    fit = _fit()
    landmarks = _landmarks()
    a = build_albedo(_image(0xA1), fit, landmarks)
    b = build_albedo(_image(0xB2), fit, landmarks)
    assert isinstance(a, Ok)
    assert isinstance(b, Ok)
    assert a.value.as_bytes() != b.value.as_bytes()


def test_inpaint_is_identity_never_adds_comb() -> None:
    """Regresion del peine de prod: el inpaint no toca ningun byte.

    El hash anterior corrompia 1/8 texels con periodo 32B y ningun gate lo
    veia porque los stubs solidos son punto fijo del promediado. Por eso el
    input es estructurado (gradiente + alterno): cualquier muestreo parcial
    dejaria diff != 0 o periodicidad a lag 32.
    """
    from backend.domain import parse_complete_uv

    landmarks = _landmarks()
    structured = bytes((i * 7 + (i // 3) * 13) % 256 for i in range(UV_LEN))
    parsed = parse_complete_uv(structured)
    assert isinstance(parsed, Ok)
    inpainted = inpaint_occluded(parsed.value, landmarks)
    assert isinstance(inpainted, Ok)
    assert inpainted.value.as_bytes() == structured


def test_warp_is_deterministic() -> None:
    fit = _fit()
    landmarks = _landmarks()
    projected = project_texture(_image(0xA1), fit)
    assert isinstance(projected, Ok)
    first = warp_with_landmarks(projected.value, landmarks)
    second = warp_with_landmarks(projected.value, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.as_bytes() == second.value.as_bytes()


def test_warp_is_identity_until_tps_lands() -> None:
    fit = _fit()
    landmarks = _landmarks()
    projected = project_texture(_image(0xA1), fit)
    assert isinstance(projected, Ok)
    warped = warp_with_landmarks(projected.value, landmarks)
    assert isinstance(warped, Ok)
    assert warped.value.as_bytes() == projected.value.as_bytes()


def test_garbage_lengths_rejected_at_parse() -> None:
    assert isinstance(project_texture(_image(0xA1), _fit()), Ok)
    assert isinstance(parse_image_bytes(bytes([1, 2, 3])), Err)
