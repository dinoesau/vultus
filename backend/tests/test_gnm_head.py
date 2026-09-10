"""Cabeza GNM real: loader validado una vez, algebra lineal y particion.

Goldens congelados a mano (17821/35324/253/68); la geometria se verifica
contra el npz, nunca se recomputa en el test salvo el algebra lineal.
"""

from __future__ import annotations

import ast
import json
import os

import numpy as np

from backend.gnm_head import (
    IDENTITY_DIM,
    LANDMARKS68,
    MEDIAPIPE_POINTS,
    MP_68_MAP,
    TEMPLATE_TRIS_REAL,
    TEMPLATE_VERTS_REAL,
    eval_landmarks68,
    eval_mesh,
    eye_mask,
    island_vertex_mask,
    load_gnm_head,
    mediapipe478_to_gnm68_targets,
    teeth_mask,
    tongue_mask,
)


def test_loader_valida_una_vez_y_cachea() -> None:
    head = load_gnm_head()
    assert load_gnm_head() is head
    assert head.template_positions.shape == (TEMPLATE_VERTS_REAL, 3) == (17821, 3)
    assert head.triangles.shape == (TEMPLATE_TRIS_REAL, 3) == (35324, 3)
    assert head.vertex_uvs.shape == (17821, 2)
    assert head.identity_basis.shape == (IDENTITY_DIM, 17821, 3) == (253, 17821, 3)
    assert head.vertex_groups.shape[1] == 17821
    assert len(head.group_names) == head.vertex_groups.shape[0]
    assert head.template_positions.dtype == np.float32
    assert head.identity_basis.dtype == np.float32
    assert head.vertex_groups.dtype == np.float32
    assert head.landmark_indices.shape == (LANDMARKS68, 3) == (68, 3)
    assert head.landmark_weights.shape == (68, 3)
    assert bool(np.all(np.isfinite(np.asarray(head.vertex_uvs, dtype=np.float64))))
    assert float(np.asarray(head.vertex_uvs).min()) >= 0.0
    assert float(np.asarray(head.vertex_uvs).max()) <= 1.0


def test_eval_mesh_con_ceros_es_template() -> None:
    head = load_gnm_head()
    out = eval_mesh(tuple([0.0] * IDENTITY_DIM))
    assert out == np.asarray(head.template_positions).tolist()


def test_eval_mesh_un_coef_suma_la_base() -> None:
    head = load_gnm_head()
    coeffs = [0.0] * IDENTITY_DIM
    coeffs[5] = 1.0
    out = np.asarray(eval_mesh(tuple(coeffs)), dtype=np.float64)
    template = np.asarray(head.template_positions, dtype=np.float64)
    basis5 = np.asarray(head.identity_basis[5], dtype=np.float64)
    assert bool(np.all(out == template + basis5))


def test_landmarks68_cubren_68_verts_y_combinan() -> None:
    head = load_gnm_head()
    idx = np.asarray(head.landmark_indices)
    assert int(idx.min()) >= 0 and int(idx.max()) < 17821
    peak = idx[np.arange(68), np.asarray(head.landmark_weights).argmax(axis=1)]
    assert len(set(peak.tolist())) == 68
    out = np.asarray(eval_landmarks68(tuple([0.0] * IDENTITY_DIM)), dtype=np.float64)
    assert out.shape == (68, 3)
    template = np.asarray(head.template_positions, dtype=np.float64)
    weights = np.asarray(head.landmark_weights, dtype=np.float64)
    manual = (template[np.asarray(idx, dtype=np.int64)] * weights[:, :, None]).sum(axis=1)
    assert bool(np.all(out == manual))


def test_islas_particionan_17821_sin_overlap() -> None:
    masks = [np.asarray(island_vertex_mask(i), dtype=bool) for i in (1, 2, 3, 4, 5)]
    assert all(len(m) == 17821 for m in masks)
    total = sum(int(m.sum()) for m in masks)
    assert total == 17821
    stacked = np.stack(masks, axis=1)
    assert bool(np.all(stacked.sum(axis=1) == 1))
    for bad in (0, 6):
        try:
            island_vertex_mask(bad)
        except ValueError:
            continue
        raise AssertionError(f"isla {bad} no rechazo")


def test_mascaras_ojo_dientes_lengua_no_solapan() -> None:
    eye = np.asarray(eye_mask(), dtype=bool)
    teeth = np.asarray(teeth_mask(), dtype=bool)
    tongue = np.asarray(tongue_mask(), dtype=bool)
    assert len(eye) == len(teeth) == len(tongue) == 17821
    assert int(eye.sum()) > 0 and int(teeth.sum()) > 0 and int(tongue.sum()) > 0
    assert int((eye & teeth).sum()) == 0
    assert int((eye & tongue).sum()) == 0
    assert int((teeth & tongue).sum()) == 0


def test_mediapipe478_a_68_por_mapa_congelado() -> None:
    assert len(MP_68_MAP) == 68
    assert len(set(MP_68_MAP)) == 68
    assert min(MP_68_MAP) >= 0 and max(MP_68_MAP) < MEDIAPIPE_POINTS == 478
    pts = [[float(i) * 0.01, float(i) * 0.02, 0.0] for i in range(478)]
    raw = json.dumps(pts).encode("utf-8")
    out = mediapipe478_to_gnm68_targets(raw)
    assert out == [[pts[i][0], pts[i][1]] for i in MP_68_MAP]
    for bad in (b"no-json", json.dumps([[0.0, 0.0]]).encode("utf-8")):
        try:
            mediapipe478_to_gnm68_targets(bad)
        except ValueError:
            continue
        raise AssertionError(f"input {bad[:12]} no rechazo")


def test_gnm_head_parsea_con_gramatica_py310() -> None:
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(backend_dir, "gnm_head.py"), encoding="utf-8") as f:
        src = f.read()
    ast.parse(src, filename=os.path.join("backend", "gnm_head.py"), feature_version=(3, 10))
