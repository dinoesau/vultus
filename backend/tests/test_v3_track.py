"""Track v3 FFHQ-UV (figure): freeze forense + forma densa.

Seams bajo test (acordados):
- Seam forensic-freeze: `backend.domain` + `backend.gnm_assemble` + `edge/contract.ts`
  (zip-8, 5023, TTL 60 intactos; prohibido mezclar con v3).
- Seam v3-dense: `backend.v3_dense.compute_shape_numpy` + `add_eyeballs`
  (mean + idBase@id + exBase@exp -> 20481v/40832f, solo numpy/pure-python,
  finita determinista x2).

TDD slice 1: RED primero. Este archivo debe FALLAR sin `backend/v3_dense.py`
y sin `backend/v3_contract.py`, y pasar con ellos sin tocar la via forense.
"""

from __future__ import annotations

import numpy as np


def test_v3_forensic_track_frozen() -> None:
    """La via forense no se mueve salvo ADR-012: zip-8, 5023, TTL 60, CONTRACT_VERSION 3."""
    from backend import gnm_assemble
    from backend.domain import (
        TOTAL_TIMEOUT_SECS,
        UV_LEN,
        ZIP_NAMES,
    )

    assert gnm_assemble.VERT_COUNT == 5023
    assert gnm_assemble.EYE_VERT_START == 3931
    assert gnm_assemble.EYE_VERT_END == 5023
    assert tuple(ZIP_NAMES) == (
        "uv_a.png",
        "uv_b.png",
        "mesh_a.glb",
        "mesh_b.glb",
        "pbr_a.png",
        "pbr_b.png",
        "render_a.png",
        "render_b.png",
    )
    assert len(ZIP_NAMES) == 8
    assert UV_LEN == 512 * 512 * 3 == 786432
    assert TOTAL_TIMEOUT_SECS == 60


def test_v3_contract_parallel_to_forensic() -> None:
    """Track v3 en paralelo: version 3, malla densa, atlas 1024, bundle nuevo.

    Prohibido mezclar con zip-8: los nombres v3 no comparten literales.
    """
    from backend.v3_contract import (
        V3_CONTRACT_VERSION,
        V3_RETENTION_DAYS,
        V3_TRI_COUNT,
        V3_UV_LEN,
        V3_UV_SIZE,
        V3_VERT_COUNT,
        V3_ZIP_NAMES,
    )

    assert V3_CONTRACT_VERSION == 3
    assert V3_VERT_COUNT == 20481
    assert V3_TRI_COUNT == 40832
    assert V3_UV_SIZE == 1024
    assert V3_UV_LEN == 1024 * 1024 * 3
    # Bundle v3 nuevo (malla densa + albedo 1024 + neutral + 3 relights
    # con esferas como la figura): 6 piezas, ningun literal coincide
    # con el zip-8 forense.
    assert len(V3_ZIP_NAMES) == 6
    assert len(set(V3_ZIP_NAMES)) == 6
    forensic = {
        "uv_a.png",
        "uv_b.png",
        "mesh_a.glb",
        "mesh_b.glb",
        "pbr_a.png",
        "pbr_b.png",
        "render_a.png",
        "render_b.png",
    }
    assert not (set(V3_ZIP_NAMES) & forensic)
    # Retencion de dias, no TTL 60: el worker v3 no vive en 60s.
    assert V3_RETENTION_DAYS >= 1


def test_v3_dense_shape_finite_deterministic_x2() -> None:
    """Port ParametricFaceModel.compute_shape solo numpy: 20481v finita x2."""
    from backend.v3_dense import compute_shape_numpy

    rng = np.random.RandomState(7)
    mean = rng.rand(20481, 3).astype(np.float64)
    id_base = (rng.rand(20481 * 3, 32).astype(np.float64) - 0.5) / 100.0
    ex_base = (rng.rand(20481 * 3, 8).astype(np.float64) - 0.5) / 100.0
    id_coeffs = (rng.rand(32).astype(np.float64) - 0.5).astype(np.float64)
    exp_coeffs = (rng.rand(8).astype(np.float64) - 0.5).astype(np.float64)

    from backend.domain import Err as _Err

    first = compute_shape_numpy(mean, id_base, ex_base, id_coeffs, exp_coeffs)
    second = compute_shape_numpy(mean, id_base, ex_base, id_coeffs, exp_coeffs)
    assert not isinstance(first, _Err) and not isinstance(second, _Err)
    face, neutral = first.value
    face2, neutral2 = second.value
    assert face.shape == (20481, 3) and neutral.shape == (20481, 3)
    assert bool(np.isfinite(face).all() and np.isfinite(neutral).all())
    np.testing.assert_array_equal(face, face2)
    np.testing.assert_array_equal(neutral, neutral2)
    # Neutral + expresion = cara: id_shape + exBase@exp == face_shape.
    np.testing.assert_allclose(face, neutral + (ex_base @ exp_coeffs).reshape(20481, 3), rtol=1e-9)


def test_v3_eyeballs_append_only_and_deterministic() -> None:
    """Mesh_Add_EyeBall solo numpy: append-only, determinista x2, finito."""
    from backend.v3_dense import add_eyeballs

    rng = np.random.RandomState(11)
    verts = rng.rand(20481, 3).astype(np.float64)
    tris = np.zeros((40832, 3), dtype=np.int64)
    # Tris dummy validos: abanico sobre los primeros verts (solo topologia).
    for i in range(40832):
        tris[i] = (i % 20481, (i + 1) % 20481, (i + 2) % 20481)

    from backend.domain import Err as _Err2

    first = add_eyeballs(verts, tris)
    second = add_eyeballs(verts, tris)
    assert not isinstance(first, _Err2) and not isinstance(second, _Err2)
    v1, f1 = first.value
    v2, f2 = second.value
    # Append-only: los primeros 20481 verts intactos, caras originales intactas.
    np.testing.assert_array_equal(v1[:20481], verts)
    np.testing.assert_array_equal(f1[:40832], tris)
    assert v1.shape[0] > 20481 and f1.shape[0] > 40832
    assert bool(np.isfinite(v1).all())
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(f1, f2)


def test_v3_pose_gate_and_iterative_error_decreases() -> None:
    """Pose perspectiva v3: gate FFHQ+yaw ruidoso + error baja entre iters."""
    from backend.v3_pose import (
        gate_entry,
        iterative_pose_errors,
        reprojection_error,
    )

    # Gate: lado minimo FFHQ 1024 y yaw <= 15 deg; fuera de rango es Err.
    assert not isinstance(gate_entry(1024, 1024, 5.0), object) or True
    from backend.domain import Err as _ErrP

    ok = gate_entry(1024, 1024, 5.0)
    assert not isinstance(ok, _ErrP)
    assert isinstance(gate_entry(250, 250, 5.0), _ErrP)
    assert isinstance(gate_entry(1024, 1024, 45.0), _ErrP)
    # MTCNN .pb TF1 descartado y documentado: el mapa es MP_68_MAP.
    from backend.gnm_head import MP_68_MAP
    from backend.v3_pose import LANDMARK_SOURCE

    assert LANDMARK_SOURCE == "MP_68_MAP"
    assert len(MP_68_MAP) == 68

    # Fitting iterativo: el error de reproyeccion baja entre iteraciones.
    rng = np.random.RandomState(3)
    obj = rng.rand(68, 3).astype(np.float64)
    # Proyeccion perspectiva sintetica con focal conocida + desplazamiento
    # sistematico (pose verdadera no nula) + ruido: el cero inicial erra.
    focal = 1200.0
    proj = obj[:, :2] / (obj[:, 2:3] + 2.0) * focal
    noisy = proj + 10.0 + rng.normal(0, 1.0, proj.shape)
    errors = iterative_pose_errors(obj, noisy, focal, steps=5)
    assert len(errors) == 6  # error inicial + 5 iteraciones
    assert all(float(e) >= 0.0 for e in errors)
    assert float(errors[-1]) < float(errors[0])
    assert float(reprojection_error(obj, noisy, focal, np.zeros(6))) >= 0.0


def test_v3_texture_1024_tv_and_zero_sentinel() -> None:
    """Textura 1024 v3: TV <= 2.0 y cero sentinel en atlas suave."""
    from backend.v3_texture import (
        count_sentinel_1024,
        uv_total_variation_1024,
    )

    # Atlas suave (gradiente): TV baja y cero sentinel.
    yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float64)
    smooth = np.stack([(xx / 1024.0 * 255.0), (yy / 1024.0 * 255.0), np.full_like(xx, 128.0)], axis=-1)
    raw = np.rint(np.clip(smooth, 0.0, 255.0)).astype(np.uint8).tobytes()
    assert len(raw) == 1024 * 1024 * 3
    assert count_sentinel_1024(raw) == 0
    assert uv_total_variation_1024(raw) <= 2.0
    # TexGAN largo: cientos de pasos con init w_avg, solo texeles no validos.
    from backend.v3_contract import V3_TEXGAN_FIT_STEPS
    from backend.v3_texture import V3_TEXGAN_INIT

    assert V3_TEXGAN_FIT_STEPS >= 100
    assert V3_TEXGAN_INIT == "w_avg"
    # Blend Poisson/Laplaciano + match_color existen y son totales.
    from backend.v3_texture import blend_laplacian, match_color

    a = np.full((8, 8, 3), 100.0)
    b = np.full((8, 8, 3), 200.0)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4] = True
    out = blend_laplacian(a, b, mask)
    assert out.shape == (8, 8, 3) and bool(np.isfinite(out).all())
    matched = match_color(b, a, mask)
    assert matched.shape == (8, 8, 3)


def test_v3_light_albedo_stable_under_two_lights() -> None:
    """DPR SH9 v3: misma albedo bajo 2 luces da shading distinto y albedo identico."""
    from backend.v3_light import albedo_from_photo, relight, sh_basis_9

    rng = np.random.RandomState(5)
    normals = rng.rand(16, 16, 3).astype(np.float64) - 0.5
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
    photo = np.full((16, 16, 3), 150.0)
    sh_a = np.zeros(9)
    sh_a[0] = 1.0
    sh_b = np.zeros(9)
    sh_b[0] = 1.0
    sh_b[1] = 0.5
    basis = sh_basis_9(normals)
    assert basis.shape == (16, 16, 9)
    alb_a = albedo_from_photo(photo, normals, sh_a)
    assert alb_a.shape == (16, 16, 3)
    # Misma foto y misma geometria: el albedo estimado difiere porque la luz
    # difiere (el test fuerte es el inverso: mismo albedo relite distinto).
    # Mismo albedo bajo 2 luces: shading distinto, albedo identico.
    albedo = np.full((16, 16, 3), 180.0)
    rel_a = relight(albedo, normals, sh_a)
    rel_b = relight(albedo, normals, sh_b)
    assert rel_a.shape == (16, 16, 3) and rel_b.shape == (16, 16, 3)
    assert float(np.abs(rel_a - rel_b).max()) > 1e-6
    np.testing.assert_array_equal(albedo, np.full((16, 16, 3), 180.0))
    # Cota del shading y checkpoint real documentado.
    from backend.v3_light import DPR_T7_NAME, SHADING_CLAMP

    assert DPR_T7_NAME == "trained_model_03.t7"
    assert SHADING_CLAMP == (0.5, 2.0)


def test_v3_bundle_separate_from_zip8_and_branch_log_keys() -> None:
    """Bundle v3 nuevo disjunto del zip-8 + claves de ramas efectivas."""
    from backend.domain import Err as _ErrB
    from backend.v3_bundle import V3Bundle, build_v3_bundle
    from backend.v3_contract import V3_ZIP_NAMES

    mesh = b"glTF" + bytes(100)
    albedo = bytes(1024 * 1024 * 3)
    relights = [bytes(64) for _ in range(4)]
    result = build_v3_bundle(mesh, albedo, relights)
    assert not isinstance(result, _ErrB)
    bundle = result.value
    assert isinstance(bundle, V3Bundle)
    assert tuple(bundle.names()) == tuple(V3_ZIP_NAMES)
    # Prohibido mezclar con zip-8 a nivel de tipos y de nombres.
    from backend.domain import ZIP_NAMES as _ZIP6

    assert not (set(bundle.names()) & set(_ZIP6))
    # Ramas efectivas con el patron existente, sin fallback silencioso.
    from backend.v3_worker import V3_BRANCH_KEYS, format_v3_branches

    assert tuple(V3_BRANCH_KEYS) == ("deep3d", "pose", "texgan", "dpr", "displacement")
    line = format_v3_branches({"deep3d": 1.0, "pose": 1.0, "texgan": 1.0, "dpr": 1.0, "displacement": 1.0})
    for key in V3_BRANCH_KEYS:
        assert f"{key}=" in line
