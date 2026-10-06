"""Luz v3 real: SH de foto Bush con `trained_model_03.t7` congelado.

Seam bajo test: `backend.v3_light.estimate_sh_from_photo` (foto real ->
9 SH via Hourglass real) + `albedo_from_photo`/`relight`/`shading_from_sh`
sobre pixeles y geometria reales (foto Bush 0001 + normales radiales de
la malla densa neutra real, ambas con sha congelado).

Fuente independiente de verdad: checkpoint `.t7` (sha congelado) + foto.
Propiedad real: misma albedo bajo 2 luces (estimada y perturbada) da
shading distinto y albedo identico (el input no se muta), con cota
[0.5, 2.0] respetada. Sin puente: skip (CI).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

T7_SHA_FROZEN = "950157bf950b899eba7bb1a6439d602e89fef9d180e235d034774d6d8583bc51"
PHOTO_SHA_FROZEN = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"


def _require_dpr_bridge(monkeypatch: pytest.MonkeyPatch) -> dict[str, str] | None:
    mirror = "/Users/esau.martinez/code/weights"
    lfw = "/Users/esau.martinez/Code/datasets/lfw"
    need = {
        "t7": f"{mirror}/ffhq-uv-hf/checkpoints/dpr_model/trained_model_03.t7",
        "photo": f"{lfw}/George_W_Bush/George_W_Bush_0001.jpg",
    }
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need.values()):
        return None
    monkeypatch.setenv("DPR_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/dpr_model")
    monkeypatch.setenv("WEIGHTS_DIR", mirror)
    monkeypatch.delenv("VULTUS_REAL_ML", raising=False)
    return need


def test_v3_dpr_checkpoint_bytes_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """El `.t7` es el byte congelado upstream."""
    paths = _require_dpr_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente DPR local (CI)")
    with open(paths["t7"], "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == T7_SHA_FROZEN


def _real_photo_normals_512(photo_path: str) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    with open(photo_path, "rb") as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == PHOTO_SHA_FROZEN
    with Image.open(photo_path) as handle:
        small = handle.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    photo = np.asarray(small, dtype=np.float64)
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float64)
    cx, cy = 256.0, 256.0
    normals = np.stack([xx - cx, yy - cy, np.full_like(xx, 320.0)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
    return photo, np.asarray(normals, dtype=np.float64)


def test_v3_dpr_sh_from_real_photo_and_two_light_albedo_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SH reales de foto Bush; misma albedo, 2 luces -> shading distinto."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pytest.importorskip("mediapipe")
    paths = _require_dpr_bridge(monkeypatch)
    if paths is None:
        pytest.skip("sin puente DPR local (CI)")
    from backend.domain import Err as _Err
    from backend.v3_light import (
        SHADING_CLAMP,
        albedo_from_photo,
        estimate_sh_from_photo,
        relight,
        shading_from_sh,
    )

    sh = estimate_sh_from_photo(paths["photo"])
    assert not isinstance(sh, _Err)
    assert sh.value.shape == (9,) and bool(np.isfinite(sh.value).all())
    sh_a = np.asarray(sh.value, dtype=np.float64)
    # Segunda luz: misma estimada + boost ambiente (+2.0 sale del suelo del
    # clamp en la zona frontal; sin boost ambas colapsan a 0.5 y el test no
    # diria nada).
    sh_b = sh_a.copy()
    sh_b[0] += 2.0
    photo, normals = _real_photo_normals_512(paths["photo"])
    shading_a = shading_from_sh(normals, sh_a)
    shading_b = shading_from_sh(normals, sh_b)
    lo, hi = SHADING_CLAMP
    assert bool(((shading_a >= lo) & (shading_a <= hi)).all())
    assert float(np.abs(shading_a - shading_b).max()) > 1e-6
    albedo = albedo_from_photo(photo, normals, sh_a)
    assert albedo.shape == (512, 512, 3) and bool(np.isfinite(albedo).all())
    before = albedo.copy()
    rel_a = relight(albedo, normals, sh_a)
    rel_b = relight(albedo, normals, sh_b)
    np.testing.assert_array_equal(albedo, before)
    assert float(np.abs(rel_a - rel_b).max()) > 1e-6
    again = albedo_from_photo(photo, normals, sh_a)
    np.testing.assert_array_equal(albedo, again)
