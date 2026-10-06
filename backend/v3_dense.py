"""Forma densa v3 (port de FFHQ-UV sin torch en runtime caliente).

Porta `ParametricFaceModel.compute_shape` (mean + `idBase@id` + `exBase@exp`
-> 20481v) + `Mesh_Add_EyeBall` (append-only de 2 esferas) con solo
numpy/pure-python en runtime caliente. Sin torch top-level (lazy + guards
donde haga falta, aqui ni siquiera lazy: numpy puro).

Referencia upstream en mirror (solo lectura):
`DataSet_Step4_UV_Texture/face3d_recon/parametric_face_model.py`
(`compute_shape`: `id_part + exp_part + mean_shape`, `id_part + mean_shape`).

Sin logging. Entradas validadas una vez, salidas finitas. Total: nunca
lanza por inputs esperados; retorna `Result`, fail-loud sin Error Hiding.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from backend.domain import DomainError, Err, MlDecode, MlFailed, Ok

# Topologia densa canonica (espejo del .mat HiFi3D++).
DENSE_VERTS = 20481
DENSE_TRIS = 40832

# Ojos v3: 2 esferas lat-long 26x21 (546 verts/bola, 1040 tris/bola),
# mismos defaults geometricos que el fixture forense documentado.
_EYE_COLS = 26
_EYE_ROWS = 21
_EYE_RADIUS = 0.055
_EYE_CENTERS = ((-0.115, 0.06, 0.30), (0.115, 0.06, 0.30))


def _is_finite_array(arr: NDArray[np.float64]) -> bool:
    try:
        return bool(np.isfinite(np.asarray(arr)).all())
    except (ValueError, TypeError):
        return False


def compute_shape_numpy(
    mean: NDArray[np.float64],
    id_base: NDArray[np.float64],
    ex_base: NDArray[np.float64],
    id_coeffs: NDArray[np.float64],
    exp_coeffs: NDArray[np.float64],
) -> Ok[tuple[NDArray[np.float64], NDArray[np.float64]]] | Err[DomainError]:
    """Cara densa + neutra: `mean + idBase@id + exBase@exp`, `mean + idBase@id`.

    Espejo numpy de `ParametricFaceModel.compute_shape` (sin batch, sin
    torch, sin recentrado implicito: el caller pasa `mean` ya recentrado
    como el upstream). Determinista x2 por construccion (matmul puro).
    Total: `Err` si formas/dims no cuadran o sale no-finito; nunca lanza
    por inputs esperados.
    """
    try:
        m = np.asarray(mean, dtype=np.float64).reshape(DENSE_VERTS, 3)
        ib = np.asarray(id_base, dtype=np.float64)
        eb = np.asarray(ex_base, dtype=np.float64)
        iv = np.asarray(id_coeffs, dtype=np.float64).reshape(-1)
        ev = np.asarray(exp_coeffs, dtype=np.float64).reshape(-1)
    except (ValueError, TypeError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense shape invalid input: {exc}")))
    n_id = int(iv.shape[0])
    n_exp = int(ev.shape[0])
    if ib.shape != (DENSE_VERTS * 3, n_id) or eb.shape != (DENSE_VERTS * 3, n_exp):
        return Err(
            MlFailed(
                detail=MlDecode(
                    details=f"v3 dense basis mismatch: idBase {ib.shape} exBase {eb.shape} "
                    f"id {n_id} exp {n_exp}"
                )
            )
        )
    if n_id <= 0 or n_exp <= 0:
        return Err(MlFailed(detail=MlDecode(details="v3 dense empty coeffs")))
    if not (_is_finite_array(m) and _is_finite_array(ib) and _is_finite_array(eb)):
        return Err(MlFailed(detail=MlDecode(details="v3 dense basis non-finite")))
    if not (_is_finite_array(iv) and _is_finite_array(ev)):
        return Err(MlFailed(detail=MlDecode(details="v3 dense coeffs non-finite")))
    try:
        id_part = (ib @ iv).reshape(DENSE_VERTS, 3)
        exp_part = (eb @ ev).reshape(DENSE_VERTS, 3)
        neutral = m + id_part
        face = neutral + exp_part
    except (ValueError, TypeError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense matmul failed: {exc}")))
    if not (_is_finite_array(neutral) and _is_finite_array(face)):
        return Err(MlFailed(detail=MlDecode(details="v3 dense result non-finite")))
    return Ok((np.asarray(face, dtype=np.float64), np.asarray(neutral, dtype=np.float64)))


def _eye_sphere(center: tuple[float, float, float]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Esfera lat-long determinista (puros `math`, sin aleatoriedad)."""
    cx, cy, cz = center
    verts: list[tuple[float, float, float]] = []
    for row in range(_EYE_ROWS):
        phi = math.pi * (float(row) + 0.5) / float(_EYE_ROWS)
        sin_phi = math.sin(phi)
        for col in range(_EYE_COLS):
            theta = 2.0 * math.pi * float(col) / float(_EYE_COLS)
            x = cx + _EYE_RADIUS * sin_phi * math.cos(theta)
            y = cy + _EYE_RADIUS * math.cos(phi)
            z = cz + _EYE_RADIUS * sin_phi * math.sin(theta)
            verts.append((x, y, z))
    v = np.asarray(verts, dtype=np.float64)
    tris: list[tuple[int, int, int]] = []
    for row in range(_EYE_ROWS - 1):
        for col in range(_EYE_COLS):
            c2 = (col + 1) % _EYE_COLS
            v0 = row * _EYE_COLS + col
            v1 = row * _EYE_COLS + c2
            v2 = (row + 1) * _EYE_COLS + col
            v3 = (row + 1) * _EYE_COLS + c2
            tris.append((v0, v1, v2))
            tris.append((v1, v3, v2))
    return v, np.asarray(tris, dtype=np.int64)


def add_eyeballs(
    verts: NDArray[np.float64],
    tris: NDArray[np.int64],
) -> Ok[tuple[NDArray[np.float64], NDArray[np.int64]]] | Err[DomainError]:
    """Append-only de 2 globos oculares (port `Mesh_Add_EyeBall`).

    Los `N` verts y `M` caras de entrada quedan intactos al inicio;
    los ojos se agregan con indices desplazados. Determinista x2,
    finito por construccion. Total: `Err` si formas invalidas.
    """
    try:
        v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(tris, dtype=np.int64)
    except (ValueError, TypeError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 eyeball invalid input: {exc}")))
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] <= 0:
        return Err(MlFailed(detail=MlDecode(details="v3 eyeball verts invalid shape")))
    if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] <= 0:
        return Err(MlFailed(detail=MlDecode(details="v3 eyeball tris invalid shape")))
    if not _is_finite_array(v):
        return Err(MlFailed(detail=MlDecode(details="v3 eyeball verts non-finite")))
    if int(f.min()) < 0 or int(f.max()) >= int(v.shape[0]):
        return Err(MlFailed(detail=MlDecode(details="v3 eyeball tri index out of range")))
    try:
        parts_v: list[NDArray[np.float64]] = [v]
        parts_f: list[NDArray[np.int64]] = [np.asarray(f, dtype=np.int64)]
        offset = int(v.shape[0])
        for center in _EYE_CENTERS:
            sv, sf = _eye_sphere(center)
            parts_v.append(sv)
            parts_f.append(sf + offset)
            offset += int(sv.shape[0])
        out_v = np.concatenate(parts_v, axis=0).astype(np.float64)
        out_f = np.concatenate(parts_f, axis=0).astype(np.int64)
    except (ValueError, TypeError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 eyeball assemble failed: {exc}")))
    if not _is_finite_array(out_v):
        return Err(MlFailed(detail=MlDecode(details="v3 eyeball result non-finite")))
    return Ok((out_v, out_f))
