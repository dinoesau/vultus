"""Textura: bake 1024 real + seam build_albedo, goldens congelados a mano."""

from __future__ import annotations

import io
import json

import numpy as np

from backend.domain import UV_LEN, Err, Ok, parse_image_bytes, parse_landmarks
from backend.gnm_fit import fit_gnm
from backend.gnm_texture import (
    NO_DATA,
    BakeCounters,
    _sample_atlas,
    bake_1024,
    build_albedo,
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


def test_bake_sin_pesos_es_loud_no_gris() -> None:
    """Sin pesos el bake GNM legacy es loud, nunca gris silencioso (ADR-008).

    Gris honesto abandonado: sin caller productivo (solo tests), el fallback
    gris se borro. La completion visual vive en `flame_texture`.
    """
    import pytest as _pytest

    try:
        from backend.gnm_head import load_gnm_head

        load_gnm_head()
        has_weights = True
    except RuntimeError:
        has_weights = False
    if has_weights:
        import pytest

        pytest.skip("solo CI sin pesos")
    with _pytest.raises(RuntimeError, match="requires weights"):
        bake_1024(_image(0xA1), _fit())


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


def test_garbage_lengths_rejected_at_parse() -> None:
    assert isinstance(parse_image_bytes(bytes([1, 2, 3])), Err)


# --- Wave 3 Step 4: textura FFHQ-UV sin gris (flame_texture) ---
# Seam run_pair+sink preferida (ver test_pipeline.py); aqui unidad del bake:
# double/fixture da albedo sin SKIN_SENTINEL, bytes-identicos-x2.


def _flame_fit_value():  # type: ignore[no-untyped-def]
    from backend.flame_fit import fit_flame

    fit = fit_flame(_image(0xA1), _landmarks())
    assert isinstance(fit, Ok)
    return fit.value


def test_flame_bake_deterministic_bytes_identical_x2() -> None:
    from backend.flame_texture import bake_flame

    fit = _flame_fit_value()
    landmarks = _landmarks()
    first = bake_flame(_image(0xA1), fit, landmarks)
    second = bake_flame(_image(0xA1), fit, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.as_bytes() == second.value.as_bytes()
    assert len(first.value.as_bytes()) == UV_LEN


def test_flame_bake_zero_sentinel_in_skin() -> None:
    from backend.flame_texture import SKIN_SENTINEL, bake_flame, count_sentinel

    fit = _flame_fit_value()
    out = bake_flame(_image(0xA1), fit, _landmarks())
    assert isinstance(out, Ok)
    raw = out.value.as_bytes()
    assert count_sentinel(raw) == 0
    # Barrido independiente: ningun triple es el magenta fuera de gama.
    sr, sg, sb = SKIN_SENTINEL
    for i in range(0, len(raw), 3):
        assert not (raw[i] == sr and raw[i + 1] == sg and raw[i + 2] == sb)


def test_flame_bake_evidence_over_threshold() -> None:
    from backend.flame_texture import EVIDENCE_MIN, bake_flame, texture_evidence

    fit = _flame_fit_value()
    out = bake_flame(_image(0xA1), fit, _landmarks())
    assert isinstance(out, Ok)
    evidence = texture_evidence(out.value.as_bytes())
    assert evidence >= EVIDENCE_MIN
    assert evidence > 0.0


def test_flame_bake_differs_per_identity() -> None:
    from backend.flame_texture import bake_flame

    fit = _flame_fit_value()
    landmarks = _landmarks()
    a = bake_flame(_image(0xA1), fit, landmarks)
    b = bake_flame(_image(0xB2), fit, landmarks)
    assert isinstance(a, Ok)
    assert isinstance(b, Ok)
    assert a.value.as_bytes() != b.value.as_bytes()


def test_flame_bake_request_bad_payload_fails_loudly() -> None:
    from backend.flame_texture import bake_flame_from_request

    assert isinstance(bake_flame_from_request(b"\x00\x01"), Err)
    assert isinstance(bake_flame_from_request(b""), Err)


def test_flame_bake_request_v2_roundtrip_ok() -> None:
    from backend.flame_fit import fit_flame
    from backend.flame_texture import bake_flame_from_request
    from backend.pipeline_local import encode_texture_request

    fit_res = fit_flame(_image(0xA1), _landmarks())
    assert isinstance(fit_res, Ok)
    blob = encode_texture_request(_image(0xA1), fit_res.value, _landmarks())
    assert isinstance(bake_flame_from_request(blob), Ok)


def test_flame_bake_real_required_without_weights_fails_loudly(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from backend.domain import domain_to_status
    from backend.flame_texture import bake_flame

    # Fit valido antes de forzar REAL=1 (fit_flame tambien falla loud con REAL=1 sin pesos).
    fit = _flame_fit_value()
    landmarks = _landmarks()
    image = _image(0xA1)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("VULTUS_REAL_ML", "1")
    monkeypatch.setenv("FFHQ_UV_DIR", str(empty))
    result = bake_flame(image, fit, landmarks)
    assert isinstance(result, Err)
    assert domain_to_status(result.error) == 500


def test_flame_determinism_flags_and_fallback_diff() -> None:
    from backend.flame_texture import (
        bake_flame,
        ensure_deterministic_texture,
        max_abs_diff,
    )

    # Sin torch el ensure es no-op total; con torch aplica flags sin tumbar.
    ensure_deterministic_texture()
    fit = _flame_fit_value()
    a = bake_flame(_image(0xA1), fit, _landmarks())
    b = bake_flame(_image(0xA1), fit, _landmarks())
    assert isinstance(a, Ok)
    assert isinstance(b, Ok)
    assert max_abs_diff(a.value.as_bytes(), b.value.as_bytes()) == 0
    assert max_abs_diff(bytes([0, 10, 20]), bytes([1, 12, 19])) == 2


# --- Wave 6-fix Lane A: vectores reales (completion FFHQ-UV) vs dobles ---
# Dobles (sin VULTUS_REAL_ML/LANDMARKS_REAL): solo propiedades
# deterministicas (x2 identico, cero sentinel, evidence>=MIN, s_self==1.0).
# Reales (ambos flags + puente canonico + triple LFW + MediaPipe):
# albedo de la foto con SSIM(misma)>SSIM(distinta) y ojos en bake
# separado con el mapa real (pbr != uv a nivel de modulo).
# Sin flags o sin assets los tests reales hacen skip con motivo; el gate
# prod (scripts/e2e-flame-real.py, Lane B) los exige en verde.


def _require_real_texture_vectors(monkeypatch):  # type: ignore[no-untyped-def]
    """Activa el puente FFHQ-UV y devuelve el triple LFW, o skip con motivo."""
    import os as _os
    from pathlib import Path as _Path

    if _os.environ.get("VULTUS_REAL_ML") != "1" or _os.environ.get("LANDMARKS_REAL") != "1":
        import pytest

        pytest.skip("vectores reales solo con VULTUS_REAL_ML=1 + LANDMARKS_REAL=1")
    home = _Path.home()
    monkeypatch.setenv("DECA_DIR", _os.environ.get("DECA_DIR", str(home / "Code" / "weights" / "deca")))
    monkeypatch.setenv(
        "FLAME_ASSETS_DIR",
        _os.environ.get("FLAME_ASSETS_DIR", str(home / "Code" / "weights" / "flame")),
    )
    monkeypatch.setenv(
        "FFHQ_UV_DIR",
        _os.environ.get("FFHQ_UV_DIR", str(home / "Code" / "weights" / "ffhq-uv")),
    )
    from backend.flame_texture import weights_present

    if not weights_present():
        import pytest

        pytest.skip("sin puente FFHQ-UV canonico no hay completion real local")
    dataset = _Path("/Users/esau.martinez/Code/datasets/lfw")
    paths = {
        "A": dataset / "George_W_Bush" / "George_W_Bush_0001.jpg",
        "B": dataset / "George_W_Bush" / "George_W_Bush_0002.jpg",
        "C": dataset / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg",
    }
    for tag, path in paths.items():
        if not path.is_file():
            import pytest as _pytest

            _pytest.skip(f"sin triple LFW no hay vectores reales ({tag})")
    return {tag: path.read_bytes() for tag, path in paths.items()}


def _ssim_global(a: bytes, b: bytes) -> float:
    """SSIM global self-contained (mirror del gate e2e, sin skimage)."""
    import numpy as _np

    if len(a) != len(b) or len(a) == 0:
        return 0.0
    x = _np.frombuffer(a, dtype=_np.uint8).astype(_np.float64)
    y = _np.frombuffer(b, dtype=_np.uint8).astype(_np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mx = float(x.mean())
    my = float(y.mean())
    vx = float(((x - mx) ** 2).mean())
    vy = float(((y - my) ** 2).mean())
    cov = float(((x - mx) * (y - my)).mean())
    num = (2.0 * mx * my + c1) * (2.0 * cov + c2)
    den = (mx * mx + my * my + c1) * (vx + vy + c2)
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return num / den


def test_flame_double_ssim_self_is_one() -> None:
    """Propiedad doble: SSIM de un albedo consigo mismo es 1.0."""
    from backend.flame_texture import bake_flame

    fit = _flame_fit_value()
    out = bake_flame(_image(0xA1), fit, _landmarks())
    assert isinstance(out, Ok)
    assert _ssim_global(out.value.as_bytes(), out.value.as_bytes()) == 1.0


def test_flame_texture_junk_weights_not_real(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Junk files no habilitan el real: solo obj UV + mapa ocular."""
    from backend.flame_texture import weights_present

    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "random.txt").write_text("junk")
    monkeypatch.setenv("FFHQ_UV_DIR", str(junk))
    assert weights_present() is False


def test_flame_eye_bake_missing_weights_loud(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Ojos sin puente fallan loud 500, nunca sintetico."""
    from backend.domain import domain_to_status
    from backend.flame_texture import bake_eye_texture

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("FFHQ_UV_DIR", str(empty))
    result = bake_eye_texture()
    assert isinstance(result, Err)
    assert domain_to_status(result.error) == 500


def _real_fit_and_landmarks(raw: bytes):  # type: ignore[no-untyped-def]
    """Fit real + landmarks reales sobre un JPEG del triple."""
    import io as _io
    import json as _json

    try:
        import mediapipe as _mp
        from mediapipe.tasks.python import BaseOptions as _BaseOptions
        from mediapipe.tasks.python import vision as _vision
    except ImportError:
        import pytest

        pytest.skip("sin mediapipe no hay landmarks reales")
    from pathlib import Path as _Path

    import numpy as _np
    from PIL import Image as _Image

    from backend.flame_fit import fit_flame

    task = _Path.home() / "Code" / "weights" / "mediapipe" / "face_landmarker.task"
    if not task.is_file():
        import pytest

        pytest.skip("sin face_landmarker.task no hay landmarks reales")
    parsed_img = parse_image_bytes(raw)
    assert isinstance(parsed_img, Ok)
    rgb = _np.asarray(_Image.open(_io.BytesIO(raw)).convert("RGB"))
    opts = _vision.FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(task)),
        running_mode=_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with _vision.FaceLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
    assert result.face_landmarks, "MediaPipe sin cara en foto congelada (gate rojo)"
    parsed_lm = parse_landmarks(
        _json.dumps([[float(p.x), float(p.y), float(p.z)] for p in result.face_landmarks[0]]).encode(
            "utf-8"
        )
    )
    assert isinstance(parsed_lm, Ok)
    out = fit_flame(parsed_img.value, parsed_lm.value)
    assert isinstance(out, Ok)
    return parsed_img.value, out.value, parsed_lm.value


def test_flame_real_bake_zero_sentinel_evidence_x2(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Bake real: cero sentinel, evidence>=MIN, bytes-identicos-x2."""
    from backend.flame_texture import (
        EVIDENCE_MIN,
        bake_flame,
        count_sentinel,
        max_abs_diff,
        texture_evidence,
    )

    raws = _require_real_texture_vectors(monkeypatch)
    image, fit, landmarks = _real_fit_and_landmarks(raws["A"])
    first = bake_flame(image, fit, landmarks)
    second = bake_flame(image, fit, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    raw = first.value.as_bytes()
    assert len(raw) == UV_LEN
    assert count_sentinel(raw) == 0
    assert texture_evidence(raw) >= EVIDENCE_MIN
    assert max_abs_diff(raw, second.value.as_bytes()) == 0
    assert raw == second.value.as_bytes()


def test_flame_real_ssim_same_beats_diff(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Evidencia real: SSIM(misma)>SSIM(distinta), self==1.0."""
    from backend.flame_texture import bake_flame

    raws = _require_real_texture_vectors(monkeypatch)
    albedos = {}
    for tag in ("A", "B", "C"):
        image, fit, landmarks = _real_fit_and_landmarks(raws[tag])
        out = bake_flame(image, fit, landmarks)
        assert isinstance(out, Ok)
        albedos[tag] = out.value.as_bytes()
    s_self = _ssim_global(albedos["A"], albedos["A"])
    s_same = _ssim_global(albedos["A"], albedos["B"])
    s_diff = _ssim_global(albedos["A"], albedos["C"])
    assert s_self >= 0.999
    assert s_same > s_diff


def test_flame_eye_bake_real_map_differs_from_skin(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Segundo bake ocular: mapa real, x2 identico, pbr != uv en modulo."""
    from backend.flame_texture import bake_eye_texture, bake_flame, max_abs_diff

    raws = _require_real_texture_vectors(monkeypatch)
    image, fit, landmarks = _real_fit_and_landmarks(raws["A"])
    skin = bake_flame(image, fit, landmarks)
    assert isinstance(skin, Ok)
    first = bake_eye_texture()
    second = bake_eye_texture()
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert len(first.value.as_bytes()) == UV_LEN
    assert first.value.as_bytes() == second.value.as_bytes()
    assert max_abs_diff(first.value.as_bytes(), skin.value.as_bytes()) > 0


def test_unwrap_extras_missing_fails_loud_when_real(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Sin unwrap mat, unwrap real es Err loud (nunca blur silencioso)."""
    from pathlib import Path as _Path

    from backend import flame_texture as _tex
    from backend.domain import Err as _Err

    assert _tex.TEXGAN_NAME == "texgan_ffhq_uv.pth"
    assert _tex.UNWRAP_MAT_NAME == "unwrap_1024_info.mat"
    repo = _Path(__file__).resolve().parent.parent.parent
    monkeypatch.setenv("FFHQ_UV_DIR", str(repo / "weights" / "ffhq-uv"))
    empty = tmp_path / "topo-vacio"
    empty.mkdir()
    monkeypatch.setenv("TOPO_DIR", str(empty))
    monkeypatch.setenv("WEIGHTS_ROOT", str(tmp_path / "vacio"))
    assert _tex._find_unwrap_mat() is None
    img = _image(0xA1)
    fit = _flame_fit_value()
    res = _tex.run_unwrap_texture(img, fit, _landmarks())
    assert isinstance(res, _Err)


def _require_local_unwrap_env(monkeypatch):  # type: ignore[no-untyped-def]
    """Env a pesos locales del repo (puente 4 + unwrap mat), o skip con motivo."""
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parent.parent.parent
    monkeypatch.setenv("DECA_DIR", str(repo / "weights" / "deca"))
    monkeypatch.setenv("FLAME_ASSETS_DIR", str(repo / "weights" / "flame"))
    monkeypatch.setenv("FFHQ_UV_DIR", str(repo / "weights" / "ffhq-uv"))
    monkeypatch.setenv("TOPO_DIR", str(repo / "weights" / "topo_assets"))
    monkeypatch.setenv("TEXGAN_DIR", str(repo / "weights" / "checkpoints" / "texgan_model"))
    from backend.flame_texture import weights_present as _wp

    if not _wp():
        import pytest

        pytest.skip("sin puente FFHQ-UV local no hay unwrap real")
    dataset = _Path("/Users/esau.martinez/Code/datasets/lfw")
    path_a = dataset / "George_W_Bush" / "George_W_Bush_0001.jpg"
    if not path_a.is_file():
        import pytest as _pytest

        _pytest.skip("sin LFW no hay foto para unwrap")
    return path_a.read_bytes()


def test_unwrap_projection_ok_sentinel_evidence_x2(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """GWT2 RED: unwrap por proyeccion Ok, cero sentinel, evidence>=MIN, x2 identico."""
    import json as _json

    import mediapipe as _mp
    from mediapipe.tasks import python as _base
    from mediapipe.tasks.python import vision as _vis

    from backend.domain import Ok as _Ok
    from backend.domain import parse_image_bytes as _pimg
    from backend.domain import parse_landmarks as _plm
    from backend.flame_fit import fit_flame
    from backend.flame_texture import (
        EVIDENCE_MIN,
        count_sentinel,
        max_abs_diff,
        run_unwrap_texture,
        texture_evidence,
    )

    raw = _require_local_unwrap_env(monkeypatch)
    task = "/Users/esau.martinez/Code/weights/mediapipe/face_landmarker.task"
    opts = _vis.FaceLandmarkerOptions(base_options=_base.BaseOptions(model_asset_path=task), num_faces=1)
    import io as _io

    import numpy as _np
    from PIL import Image as _Image

    with _vis.FaceLandmarker.create_from_options(opts) as landmarker:
        rgb = _np.asarray(_Image.open(_io.BytesIO(raw)).convert("RGB"))
        res = landmarker.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
        pts = [[float(q.x), float(q.y), float(q.z)] for q in res.face_landmarks[0]]
    img = _pimg(raw)
    lms = _plm(_json.dumps(pts).encode())
    assert isinstance(img, _Ok) and isinstance(lms, _Ok)
    fit = fit_flame(img.value, lms.value)
    assert isinstance(fit, _Ok)
    first = run_unwrap_texture(img.value, fit.value, lms.value)
    second = run_unwrap_texture(img.value, fit.value, lms.value)
    assert isinstance(first, _Ok)
    assert isinstance(second, _Ok)
    out = first.value.as_bytes()
    assert len(out) == 786432
    assert count_sentinel(out) == 0
    assert texture_evidence(out) >= EVIDENCE_MIN
    assert max_abs_diff(out, second.value.as_bytes()) == 0


def test_unwrap_differs_from_blur_completion(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """GWT2 RED: unwrap proyectado no es el blur 64->512 (proyeccion UV real)."""
    import json as _json

    import mediapipe as _mp
    from mediapipe.tasks import python as _base
    from mediapipe.tasks.python import vision as _vis

    from backend.domain import Ok as _Ok
    from backend.domain import parse_image_bytes as _pimg
    from backend.domain import parse_landmarks as _plm
    from backend.flame_fit import fit_flame
    from backend.flame_texture import (
        _decode_photo,
        _face_crop,
        _photo_base,
        run_unwrap_texture,
    )

    raw = _require_local_unwrap_env(monkeypatch)
    task = "/Users/esau.martinez/Code/weights/mediapipe/face_landmarker.task"
    opts = _vis.FaceLandmarkerOptions(base_options=_base.BaseOptions(model_asset_path=task), num_faces=1)
    import io as _io

    import numpy as _np
    from PIL import Image as _Image

    with _vis.FaceLandmarker.create_from_options(opts) as landmarker:
        rgb = _np.asarray(_Image.open(_io.BytesIO(raw)).convert("RGB"))
        res = landmarker.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
        pts = [[float(q.x), float(q.y), float(q.z)] for q in res.face_landmarks[0]]
    img = _pimg(raw)
    lms = _plm(_json.dumps(pts).encode())
    assert isinstance(img, _Ok) and isinstance(lms, _Ok)
    fit = fit_flame(img.value, lms.value)
    assert isinstance(fit, _Ok)
    unwrapped = run_unwrap_texture(img.value, fit.value, lms.value)
    assert isinstance(unwrapped, _Ok)
    photo = _decode_photo(img.value)
    assert isinstance(photo, _Ok)
    crop = _face_crop(photo.value, lms.value)
    assert isinstance(crop, _Ok)
    blur = _photo_base(crop.value).astype(_np.float64)
    uw = _np.frombuffer(unwrapped.value.as_bytes(), dtype=_np.uint8).astype(_np.float64).reshape(512, 512, 3)
    assert float(_np.abs(uw - blur).mean()) > 1.0
