"""Cabeza GNM real (17821 verts): loader validado una vez + evaluacion lineal.

Step 1 del plan-gnm-real-fit. Lee el npz real
`weights/gnm/versions/v3_0/gnm_head.npz` y el mapa disperso
`weights/gnm/landmarks/head_sparse_68.txt`.
Solo numpy, sin torch, sin FastAPI, sin logging.
Debe importar en Python 3.10 (viaja a Modal como `domain.py` y `gnm.py`).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

TEMPLATE_VERTS_REAL = 17821
TEMPLATE_TRIS_REAL = 35324
IDENTITY_DIM = 253
LANDMARKS68 = 68
MEDIAPIPE_POINTS = 478

_NPZ_REL = os.path.join("weights", "gnm", "versions", "v3_0", "gnm_head.npz")
_LM_REL = os.path.join("weights", "gnm", "landmarks", "head_sparse_68.txt")
_ABS_WEIGHTS = "/Users/esau.martinez/Code/weights/gnm"

_GROUP_THRESHOLD = 1e-4

# Convencion V unica documentada: las UVs de GNM viven con origen
# abajo-izquierda (su propio renderer flipea V). El bake trabaja en espacio
# pixel (flipea al rasterizar, como blob) y el export escribe `v = 1 - v_uv`
# a convencion glTF (origen arriba-izquierda). Un solo flip total por
# construccion; doble-flip y cero-flip fallan el gate de orientacion (S5).
GNM_UV_ORIGIN_BOTTOM_LEFT = True

_ISLAND_MIN = 1
_ISLAND_MAX = 5

_ISLAND_SKIN = ("skin",)
_ISLAND_LEFT_EYE = ("left_eye",)
_ISLAND_RIGHT_EYE = ("right_eye",)
_ISLAND_TEETH = ("upper_teeth_and_gums", "lower_teeth_and_gums")
_ISLAND_TONGUE = ("tongue",)

# MP_68_MAP: 68 indices MediaPipe (0-477) en orden dlib-68 que aproximan los
# 68 landmarks del mapa disperso GNM. Mapa v1 aproximado a mano: puntos
# faciales estables (ovalo, cejas, nariz, ojos, boca exterior e interior).
# La lateralidad sigue el orden x de imagen (dlib 36-41 sobre el anillo
# MP 33; dlib 42-47 sobre el anillo MP 362). Un flip espejo queda fuera
# del alcance de v1 y se calibrara en el Step 2 con el fit real.
MP_68_MAP: tuple[int, ...] = (
    # 0-16 mandibula (ovalo inferior, de sien izquierda a derecha).
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
    365, 397,
    # 17-21 ceja A y 22-26 ceja B (5 puntos estables por ceja).
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    # 27-30 puente nasal hasta la punta (MP 1 = punta).
    168, 6, 197, 1,
    # 31-35 base nasal (aletas, fosas y centro bajo la punta).
    129, 98, 2, 327, 358,
    # 36-41 ojo del anillo MP 33 y 42-47 ojo del anillo MP 362.
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    # 48-59 boca exterior (esquina L, labio sup. a esquina R, labio inf.).
    61, 185, 40, 0, 267, 269, 291, 375, 321, 17, 84, 91,
    # 60-67 boca interior (esquinas 78/308, centros sup. 13 e inf. 14).
    78, 80, 13, 312, 308, 318, 14, 87,
)


@dataclass(frozen=True, slots=True)
class GnmHead:
    """Cabeza GNM real validada una vez en `load_gnm_head`."""

    template_positions: NDArray[np.float32]
    triangles: NDArray[np.int32]
    triangle_uvs: NDArray[np.float32]
    vertex_uvs: NDArray[np.float32]
    identity_basis: NDArray[np.float32]
    vertex_groups: NDArray[np.float32]
    group_names: tuple[str, ...]
    landmark_indices: NDArray[np.int32]
    landmark_weights: NDArray[np.float32]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dedupe(paths: list[str]) -> list[str]:
    seen: list[str] = []
    for cand in paths:
        if cand and cand not in seen:
            seen.append(cand)
    return seen


def _candidate_npz_paths() -> list[str]:
    cands: list[str] = []
    direct = os.environ.get("GNM_NPZ_PATH", "").strip()
    if direct:
        cands.append(direct)
    assets_dir = os.environ.get("GNM_ASSETS_DIR", "").strip()
    if assets_dir:
        cands.append(os.path.join(assets_dir, "versions", "v3_0", "gnm_head.npz"))
    weights_dir = os.environ.get("WEIGHTS_DIR", "").strip()
    if weights_dir:
        cands.append(os.path.join(weights_dir, "gnm", "versions", "v3_0", "gnm_head.npz"))
    cands.append(os.path.join(_repo_root(), _NPZ_REL))
    cands.append(os.path.join(_ABS_WEIGHTS, "versions", "v3_0", "gnm_head.npz"))
    return _dedupe(cands)


def _candidate_landmark_paths() -> list[str]:
    cands: list[str] = []
    direct = os.environ.get("GNM_NPZ_PATH", "").strip()
    if direct:
        sibling = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(direct))),
            "landmarks",
            "head_sparse_68.txt",
        )
        cands.append(sibling)
    assets_dir = os.environ.get("GNM_ASSETS_DIR", "").strip()
    if assets_dir:
        cands.append(os.path.join(assets_dir, "landmarks", "head_sparse_68.txt"))
    weights_dir = os.environ.get("WEIGHTS_DIR", "").strip()
    if weights_dir:
        cands.append(os.path.join(weights_dir, "gnm", "landmarks", "head_sparse_68.txt"))
    cands.append(os.path.join(_repo_root(), _LM_REL))
    cands.append(os.path.join(_ABS_WEIGHTS, "landmarks", "head_sparse_68.txt"))
    return _dedupe(cands)


def _resolve_first(cands: list[str], label: str) -> str:
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(f"gnm head {label} missing (buscado en {', '.join(cands)})")


def _need_f32(data: Any, key: str) -> NDArray[np.float32]:
    arr: NDArray[np.float32] = np.asarray(data[key], dtype=np.float32)
    if arr.dtype != np.float32:
        raise RuntimeError(f"gnm head {key} dtype {arr.dtype} != float32")
    return arr


def _build_vertex_uvs(quads: NDArray[np.int64], quad_uvs: NDArray[np.float32]) -> NDArray[np.float32]:
    """Replica GNM vertex_uvs: last-wins sobre quads aplanados.

    Ante vertices compartidos (seams) gana la ultima ocurrencia, resuelta
    con unique sobre el orden inverso. Mismo algoritmo que
    `scripts/extract_gnm_template.py`.
    """
    flat_idx = np.asarray(quads.reshape(-1), dtype=np.int64)
    flat_uv = np.asarray(quad_uvs.reshape(-1, 2), dtype=np.float32)
    rev_idx = flat_idx[::-1]
    rev_uv = flat_uv[::-1]
    uniq: NDArray[np.int64]
    first: NDArray[np.intp]
    uniq, first = np.unique(rev_idx, return_index=True)
    vertex_uvs = np.zeros((TEMPLATE_VERTS_REAL, 2), dtype=np.float32)
    vertex_uvs[uniq] = rev_uv[first]
    if len(uniq) != TEMPLATE_VERTS_REAL:
        raise RuntimeError(
            f"gnm head quads cubren {len(uniq)} verts != {TEMPLATE_VERTS_REAL}"
        )
    return vertex_uvs


def _load_landmarks(path: str) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    with open(path, encoding="utf-8") as f:
        lines = [line.split() for line in f.read().splitlines() if line.split()]
    if len(lines) != LANDMARKS68:
        raise RuntimeError(f"gnm head landmarks {len(lines)} filas != {LANDMARKS68}")
    idx_rows: list[list[int]] = []
    w_rows: list[list[float]] = []
    for lineno, parts in enumerate(lines, start=1):
        if len(parts) != 6:
            raise RuntimeError(f"gnm head landmarks linea {lineno}: 6 campos != {len(parts)}")
        try:
            ids = [int(parts[0]), int(parts[2]), int(parts[4])]
            ws = [float(parts[1]), float(parts[3]), float(parts[5])]
        except ValueError:
            raise RuntimeError(f"gnm head landmarks linea {lineno} no numerica")
        for v in ids:
            if v < 0 or v >= TEMPLATE_VERTS_REAL:
                raise RuntimeError(f"gnm head landmarks linea {lineno} indice {v} fuera de rango")
        for w in ws:
            if not math.isfinite(w) or w < 0.0:
                raise RuntimeError(f"gnm head landmarks linea {lineno} peso no valido")
        idx_rows.append(ids)
        w_rows.append(ws)
    indices: NDArray[np.int32] = np.asarray(idx_rows, dtype=np.int32)
    weights: NDArray[np.float32] = np.asarray(w_rows, dtype=np.float32)
    peak = np.asarray(indices[np.arange(LANDMARKS68), np.asarray(weights.argmax(axis=1))])
    if len(set(peak.tolist())) != LANDMARKS68:
        raise RuntimeError("gnm head landmarks no cubren 68 verts distintos")
    return indices, weights


def _load_uncached() -> GnmHead:
    npz_path = _resolve_first(_candidate_npz_paths(), "gnm_head.npz")
    lm_path = _resolve_first(_candidate_landmark_paths(), "head_sparse_68.txt")
    with np.load(npz_path) as data:
        positions = _need_f32(data, "template_vertex_positions")
        if positions.shape != (TEMPLATE_VERTS_REAL, 3):
            raise RuntimeError(f"gnm head positions shape {positions.shape}")
        if not bool(np.all(np.isfinite(positions))):
            raise RuntimeError("gnm head posicion no finita")
        basis = _need_f32(data, "vertex_identity_basis")
        if basis.shape != (IDENTITY_DIM, TEMPLATE_VERTS_REAL, 3):
            raise RuntimeError(f"gnm head basis shape {basis.shape}")
        if not bool(np.all(np.isfinite(basis))):
            raise RuntimeError("gnm head basis no finita")
        quads: NDArray[np.int64] = np.asarray(data["quads"], dtype=np.int64)
        if quads.shape != (17662, 4):
            raise RuntimeError(f"gnm head quads shape {quads.shape}")
        if int(quads.min()) < 0 or int(quads.max()) >= TEMPLATE_VERTS_REAL:
            raise RuntimeError("gnm head quads con indice fuera de rango")
        quad_uvs = _need_f32(data, "quad_uvs")
        if quad_uvs.shape != (17662, 4, 2):
            raise RuntimeError(f"gnm head quad_uvs shape {quad_uvs.shape}")
        if not bool(np.all(np.isfinite(quad_uvs))):
            raise RuntimeError("gnm head uv no finita")
        tri64: NDArray[np.int64] = np.asarray(data["triangles"], dtype=np.int64)
        if tri64.shape != (TEMPLATE_TRIS_REAL, 3):
            raise RuntimeError(f"gnm head triangles shape {tri64.shape}")
        if int(tri64.min()) < 0 or int(tri64.max()) >= TEMPLATE_VERTS_REAL:
            raise RuntimeError("gnm head triangles con indice fuera de rango")
        triangles: NDArray[np.int32] = tri64.astype(np.int32)
        tri_uvs = _need_f32(data, "triangle_uvs")
        if tri_uvs.shape != (TEMPLATE_TRIS_REAL, 3, 2):
            raise RuntimeError(f"gnm head triangle_uvs shape {tri_uvs.shape}")
        if not bool(np.all(np.isfinite(tri_uvs))):
            raise RuntimeError("gnm head triangle_uvs no finita")
        if float(tri_uvs.min()) < 0.0 or float(tri_uvs.max()) > 1.0:
            raise RuntimeError("gnm head triangle_uvs fuera de 0..1")
        groups = _need_f32(data, "vertex_groups")
        if groups.shape[1] != TEMPLATE_VERTS_REAL:
            raise RuntimeError(f"gnm head vertex_groups shape {groups.shape}")
        if not bool(np.all(np.isfinite(groups))):
            raise RuntimeError("gnm head vertex_groups no finito")
        names_raw: list[Any] = np.asarray(data["vertex_group_names"]).tolist()
        group_names = tuple(str(name) for name in names_raw)
        if len(group_names) != groups.shape[0]:
            raise RuntimeError("gnm head group_names/groups desalineados")
    vertex_uvs = _build_vertex_uvs(quads, quad_uvs)
    if not bool(np.all(np.isfinite(vertex_uvs))):
        raise RuntimeError("gnm head vertex_uvs derivada no finita")
    if float(vertex_uvs.min()) < 0.0 or float(vertex_uvs.max()) > 1.0:
        raise RuntimeError("gnm head vertex_uvs fuera de 0..1")
    for needed in _ISLAND_SKIN + _ISLAND_LEFT_EYE + _ISLAND_RIGHT_EYE + _ISLAND_TEETH + _ISLAND_TONGUE:
        if needed not in group_names:
            raise RuntimeError(f"gnm head sin grupo {needed}")
    landmark_indices, landmark_weights = _load_landmarks(lm_path)
    return GnmHead(
        template_positions=positions,
        triangles=triangles,
        triangle_uvs=np.ascontiguousarray(tri_uvs, dtype=np.float32),
        vertex_uvs=vertex_uvs,
        identity_basis=basis,
        vertex_groups=groups,
        group_names=group_names,
        landmark_indices=landmark_indices,
        landmark_weights=landmark_weights,
    )


_HEAD_CACHE: GnmHead | None = None


def load_gnm_head() -> GnmHead:
    """Devuelve la cabeza GNM real validada una vez (con cache)."""
    global _HEAD_CACHE
    if _HEAD_CACHE is None:
        _HEAD_CACHE = _load_uncached()
    return _HEAD_CACHE


def _as_coeff_vector(coeffs: tuple[float, ...]) -> NDArray[np.float64]:
    if len(coeffs) != IDENTITY_DIM:
        raise ValueError(f"coeffs len {len(coeffs)} != {IDENTITY_DIM}")
    vals: list[float] = []
    for value in coeffs:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("coef no numerico")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("coef no finito")
        vals.append(number)
    return np.asarray(vals, dtype=np.float64)


def _eval_mesh_np(coeffs: tuple[float, ...]) -> NDArray[np.float64]:
    head = load_gnm_head()
    vec = _as_coeff_vector(coeffs)
    template = np.asarray(head.template_positions, dtype=np.float64)
    basis = np.asarray(head.identity_basis, dtype=np.float64)
    mesh: NDArray[np.float64] = template + np.einsum("dnm,d->nm", basis, vec)
    return mesh


def eval_mesh(coeffs: tuple[float, ...]) -> list[list[float]]:
    """Malla personalizada: template + combinacion lineal de la base (253)."""
    out: list[list[float]] = _eval_mesh_np(coeffs).tolist()
    return out


def eval_landmarks68(coeffs: tuple[float, ...]) -> list[list[float]]:
    """68 landmarks 3D como combo baricentrico de la malla evaluada."""
    head = load_gnm_head()
    mesh = _eval_mesh_np(coeffs)
    idx = np.asarray(head.landmark_indices, dtype=np.int64)
    weights = np.asarray(head.landmark_weights, dtype=np.float64)
    combo = (mesh[idx] * weights[:, :, None]).sum(axis=1)
    out: list[list[float]] = combo.tolist()
    return out


def mediapipe478_to_gnm68_targets(landmarks_bytes_json_478: bytes) -> list[list[float]]:
    """Reduce 478 puntos MediaPipe JSON a 68 pares x,y segun MP_68_MAP.

    Unica funcion estrecha para este mapeo v1 aproximado: valida el JSON
    (478 puntos [x, y, ...] finitos) y proyecta por el mapa congelado.
    """
    try:
        text = bytes(landmarks_bytes_json_478).decode("utf-8")
    except ValueError:
        raise ValueError("landmarks no son utf-8")
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("landmarks no son json")
    if not isinstance(parsed, list) or len(parsed) != MEDIAPIPE_POINTS:
        got = len(parsed) if isinstance(parsed, list) else -1
        raise ValueError(f"se esperaban {MEDIAPIPE_POINTS} puntos, hay {got}")
    pts: list[list[float]] = []
    for point in parsed:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("punto sin par x,y")
        pair: list[float] = []
        for value in (point[0], point[1]):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("coordenada no finita")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("coordenada no finita")
            pair.append(number)
        pts.append(pair)
    return [pts[i] for i in MP_68_MAP]


def _group_mask(head: GnmHead, names: tuple[str, ...]) -> NDArray[np.bool_]:
    groups = np.asarray(head.vertex_groups, dtype=np.float64)
    mask: NDArray[np.bool_] = np.zeros((TEMPLATE_VERTS_REAL,), dtype=np.bool_)
    for name in names:
        mask = mask | (groups[head.group_names.index(name)] > _GROUP_THRESHOLD)
    return mask


def _island_ids(head: GnmHead) -> NDArray[np.int32]:
    """Isla por vertice: 1 skin, 2 left_eye, 3 right_eye, 4 dientes, 5 lengua.

    Los 5 grupos son disjuntos en v3_0 y cubren los 17821 verts sin overlap.
    Prioridad ojos/dientes/lengua sobre piel para robustez ante solapes
    futuros (p. ej. `gums`/`mouth_sock` viven dentro de dientes/piel).
    """
    ids: NDArray[np.int32] = np.full((TEMPLATE_VERTS_REAL,), 1, dtype=np.int32)
    ids[np.asarray(_group_mask(head, _ISLAND_TONGUE))] = 5
    ids[np.asarray(_group_mask(head, _ISLAND_TEETH))] = 4
    ids[np.asarray(_group_mask(head, _ISLAND_RIGHT_EYE))] = 3
    ids[np.asarray(_group_mask(head, _ISLAND_LEFT_EYE))] = 2
    return ids


def island_vertex_mask(island: int) -> list[bool]:
    """Mascara booleana de 17821 para la isla 1..5 (ValueError si no)."""
    if isinstance(island, bool) or not isinstance(island, int):
        raise TypeError(f"isla no entera: {island!r}")
    if island < _ISLAND_MIN or island > _ISLAND_MAX:
        raise ValueError(f"isla {island} fuera de 1..5")
    head = load_gnm_head()
    mask: list[bool] = (_island_ids(head) == island).tolist()
    return mask


def eye_mask() -> list[bool]:
    """Mascara left_eye + right_eye (islas 2+3 en v3_0)."""
    head = load_gnm_head()
    mask: list[bool] = (_group_mask(head, _ISLAND_LEFT_EYE + _ISLAND_RIGHT_EYE)).tolist()
    return mask


def teeth_mask() -> list[bool]:
    """Mascara dientes+encias sup./inf. (`teeth` y `gums` son subconjuntos)."""
    head = load_gnm_head()
    mask: list[bool] = _group_mask(head, _ISLAND_TEETH).tolist()
    return mask


def tongue_mask() -> list[bool]:
    """Mascara lengua (grupo `tongue`, disjunto de ojos y dientes)."""
    head = load_gnm_head()
    mask: list[bool] = _group_mask(head, _ISLAND_TONGUE).tolist()
    return mask


def raw_triangles() -> NDArray[np.int32]:
    """Passthrough crudo de `triangles` 35324x3 (sin copiar semantica)."""
    return np.asarray(load_gnm_head().triangles, dtype=np.int32)


def raw_triangle_uvs() -> NDArray[np.float32]:
    """Passthrough crudo de `triangle_uvs` 35324x3x2 en origen abajo-izquierda."""
    return np.asarray(load_gnm_head().triangle_uvs, dtype=np.float32)


def vertex_normals(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normales por vertice ponderadas por area, normalizadas.

    Entrada (17821,3); salida (17821,3) finita. Triangulos degenerados
    aportan cero; vertices aislados quedan en (0,0,0) para que el facing
    los marque no visibles en el bake.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape != (TEMPLATE_VERTS_REAL, 3):
        raise ValueError(f"positions shape {pos.shape} != {(TEMPLATE_VERTS_REAL, 3)}")
    if not bool(np.all(np.isfinite(pos))):
        raise ValueError("positions no finitas")
    tris = np.asarray(load_gnm_head().triangles, dtype=np.int64)
    acc: NDArray[np.float64] = np.zeros_like(pos)
    p0 = pos[tris[:, 0]]
    p1 = pos[tris[:, 1]]
    p2 = pos[tris[:, 2]]
    face = np.cross(p1 - p0, p2 - p0)
    np.add.at(acc, tris[:, 0], face)
    np.add.at(acc, tris[:, 1], face)
    np.add.at(acc, tris[:, 2], face)
    lens = np.linalg.norm(acc, axis=1)
    out: NDArray[np.float64] = np.zeros_like(acc)
    nz = lens > 0.0
    out[nz] = acc[nz] / lens[nz][:, None]
    return out
