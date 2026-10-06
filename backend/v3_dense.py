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

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from backend.domain import DomainError, Err, MlDecode, MlFailed, Ok

# Topologia densa canonica (espejo del .mat HiFi3D++).
DENSE_VERTS = 20481
DENSE_TRIS = 40832
DENSE_ID_DIMS = 532
DENSE_EXP_DIMS = 45

# Byte congelado upstream de `hifi3dpp_model_info.mat` (mirror solo lectura).
# `load_hifi_basis_from` falla ruidoso si el archivo difiere: nunca un
# sustituto silencioso (p.ej. `data/HIFI3D.obj` v1, ya descartado).
MAT_SHA_FROZEN = "9b501bb8a1c38d65a1706add89a9418c7fe7b729bcc5e5ff38e675a61b57963b"
MAT_NAME = "hifi3dpp_model_info.mat"

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


@dataclass(frozen=True, slots=True)
class HifiBasis:
    """Base HiFi3D++ real del `.mat`. Solo via `load_hifi_basis_from`."""

    mean: NDArray[np.float64]
    id_base: NDArray[np.float64]
    ex_base: NDArray[np.float64]
    head_tri: NDArray[np.int64]
    keypoints: NDArray[np.int64]


def _read_mat(mat_path: str) -> dict[object, object]:
    from scipy.io import loadmat  # type: ignore[import-untyped]

    return dict(loadmat(mat_path))


def _sha256_file(path: str) -> Ok[str] | Err[DomainError]:
    try:
        with open(path, "rb") as fh:
            return Ok(hashlib.sha256(fh.read()).hexdigest())
    except OSError as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense mat unreadable: {exc}")))


def _env(name: str) -> str:
    import os

    return os.environ.get(name, "").strip()


def find_dense_mat() -> str | None:
    """Ruta del `.mat` congelado en candidatos (`TOPO_DIR` + `WEIGHTS_*`)."""
    import os

    cands: list[str] = []
    topo = _env("TOPO_DIR")
    if topo:
        cands.append(os.path.join(topo, MAT_NAME))
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        cands.append(os.path.join(root, "topo_assets", MAT_NAME))
    for cand in cands:
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
        except OSError:
            continue
    return None


def load_hifi_basis_from(mat_path: object) -> Ok[HifiBasis] | Err[DomainError]:
    """Borde `.mat`: verifica el byte congelado y parsea una vez a `HifiBasis`.

    Total: `Err` ruidoso si el path no es el byte congelado, si falta
    scipy, o si las formas no son las upstream exactas
    (mean 20481x3, idBase 61443x532, exBase 61443x45, head_tri 40832x3,
    keypoints 68). Nunca sustituto silencioso, nunca lanza por inputs esperados.
    """
    if not isinstance(mat_path, str) or not mat_path:
        return Err(MlFailed(detail=MlDecode(details="v3 dense mat path invalid")))
    digest = _sha256_file(mat_path)
    if isinstance(digest, Err):
        return digest
    if digest.value != MAT_SHA_FROZEN:
        return Err(MlFailed(detail=MlDecode(details="v3 dense mat sha mismatch (no es el byte congelado)")))
    try:
        m = _read_mat(mat_path)
        mean = np.asarray(m["meanshape"], dtype=np.float64).reshape(DENSE_VERTS, 3)
        id_base = np.asarray(m["idBase"], dtype=np.float64)
        ex_base = np.asarray(m["exBase"], dtype=np.float64)
        # Convencion MATLAB 1-based -> 0-based una vez en el borde (espejo
        # del upstream `head_tri - 1` en `parametric_face_model.py`).
        head_tri = np.asarray(m["head_tri"], dtype=np.int64) - 1
        keypoints = np.asarray(m["keypoints"], dtype=np.int64).reshape(-1) - 1
    except (ImportError, KeyError, ValueError, TypeError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense mat unreadable: {exc}")))
    if id_base.shape != (DENSE_VERTS * 3, DENSE_ID_DIMS):
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense idBase shape {id_base.shape} != upstream")))
    if ex_base.shape != (DENSE_VERTS * 3, DENSE_EXP_DIMS):
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense exBase shape {ex_base.shape} != upstream")))
    if head_tri.shape != (DENSE_TRIS, 3):
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense head_tri shape {head_tri.shape} != upstream")))
    if int(head_tri.min()) < 0 or int(head_tri.max()) >= DENSE_VERTS:
        return Err(MlFailed(detail=MlDecode(details="v3 dense head_tri index out of range")))
    if keypoints.shape != (68,):
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense keypoints shape {keypoints.shape} != upstream")))
    if int(keypoints.min()) < 0 or int(keypoints.max()) >= DENSE_VERTS:
        return Err(MlFailed(detail=MlDecode(details="v3 dense keypoints index out of range")))
    if not (_is_finite_array(mean) and _is_finite_array(id_base) and _is_finite_array(ex_base)):
        return Err(MlFailed(detail=MlDecode(details="v3 dense basis non-finite")))
    return Ok(HifiBasis(mean=mean, id_base=id_base, ex_base=ex_base, head_tri=head_tri, keypoints=keypoints))


def dense_uv01_from_mat(mat_path: object) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Borde UV: `vtx_vt` 0..512 del `.mat` a 0..1 para la cabeza (20481,2).

    Reusa el byte congelado (`MAT_SHA_FROZEN` verificado aqui tambien: mismo
    archivo, misma garantia). Total: `Err` ruidoso si forma/rango difieren.
    """
    if not isinstance(mat_path, str) or not mat_path:
        return Err(MlFailed(detail=MlDecode(details="v3 dense uv path invalid")))
    digest = _sha256_file(mat_path)
    if isinstance(digest, Err):
        return digest
    if digest.value != MAT_SHA_FROZEN:
        return Err(MlFailed(detail=MlDecode(details="v3 dense uv sha mismatch (no es el byte congelado)")))
    try:
        m = _read_mat(mat_path)
        vt = np.asarray(m["vtx_vt"], dtype=np.float64)
    except (ImportError, KeyError, ValueError, TypeError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense uv unreadable: {exc}")))
    if vt.shape != (DENSE_VERTS, 2):
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense vtx_vt shape {vt.shape} != upstream")))
    if not bool(np.isfinite(vt).all()) or float(vt.min()) < 0.0 or float(vt.max()) > 512.0:
        return Err(MlFailed(detail=MlDecode(details="v3 dense vtx_vt range != 0..512")))
    return Ok(np.asarray(vt / 512.0, dtype=np.float64))


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
