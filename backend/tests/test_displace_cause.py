"""Causa ruidosa del fallback de displacement (anti Error Hiding).

Seam bajo test: `deep3d.{load_hifi_basis,load_transfer,real_displacement}`
aceptan `note` opcional y anotan CUAL rama fallo (basis ausente/invalida/
OOM, transfer ausente/invalida, delta no finita/error); `displaced_positions`
lo copia a `_LAST_DISPLACE_STATS` y `modal_app._displace_branch_str` lo pone
en la linea del job. Prohibido `except -> None` sin motivo.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _no_mat_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> str:
    empty = str(tmp_path / "empty")
    os.makedirs(empty, exist_ok=True)
    monkeypatch.setenv("TOPO_DIR", empty)
    monkeypatch.setenv("FFHQ_UV_DIR", empty)
    monkeypatch.setenv("WEIGHTS_ROOT", empty)
    monkeypatch.setenv("WEIGHTS_DIR", empty)
    return empty


def _clear_caches() -> None:
    from backend import deep3d as _d3

    _d3._basis_cache.clear()
    _d3._transfer_cache.clear()


def test_basis_missing_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from backend import deep3d as _d3

    _no_mat_env(monkeypatch, tmp_path)
    _clear_caches()
    seen: list[str] = []
    assert _d3.load_hifi_basis(note=seen.append) is None
    assert seen == ["basis_missing"]


def test_basis_invalid_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from backend import deep3d as _d3

    empty = _no_mat_env(monkeypatch, tmp_path)
    (tmp_path / "empty" / "hifi3dpp_model_info.mat").write_bytes(b"not a mat file")
    _clear_caches()
    seen: list[str] = []
    assert _d3.load_hifi_basis(note=seen.append) is None
    assert seen == ["basis_invalid"]
    assert empty


def test_basis_oom_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from backend import deep3d as _d3

    _no_mat_env(monkeypatch, tmp_path)
    (tmp_path / "empty" / "hifi3dpp_model_info.mat").write_bytes(b"x" * 64)
    _clear_caches()
    import scipy.io

    monkeypatch.setattr(scipy.io, "loadmat", lambda *a, **k: (_ for _ in ()).throw(MemoryError()))
    seen: list[str] = []
    assert _d3.load_hifi_basis(note=seen.append) is None
    assert seen == ["basis_oom"]


def test_transfer_missing_and_invalid_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from backend import deep3d as _d3

    _no_mat_env(monkeypatch, tmp_path)
    monkeypatch.setattr(_d3, "find_transfer", lambda: None)
    _clear_caches()
    seen: list[str] = []
    assert _d3.load_transfer(note=seen.append) is None
    assert seen == ["transfer_missing"]
    junk = tmp_path / "junk.npz"
    np.savez(str(junk), idx=np.zeros((2, 2)), w=np.zeros((2, 2)), scale=np.zeros(2))
    monkeypatch.setattr(_d3, "find_transfer", lambda: str(junk))
    _clear_caches()
    seen2: list[str] = []
    assert _d3.load_transfer(note=seen2.append) is None
    assert seen2 == ["transfer_invalid"]


def test_delta_nonfinite_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from backend import deep3d as _d3

    _no_mat_env(monkeypatch, tmp_path)
    inf_basis = {
        "mean": np.zeros((20481, 3)),
        "id200": np.full((61443, 200), np.inf),
        "exp45": np.zeros((61443, 45)),
        "id_full": np.zeros((61443, 532), dtype=np.float32),
        "ex_full": np.zeros((61443, 45), dtype=np.float32),
        "head_tri": np.zeros((40832, 3), dtype=np.int64),
    }
    monkeypatch.setattr(_d3, "load_hifi_basis", lambda *a, **k: inf_basis)
    monkeypatch.setattr(
        _d3,
        "load_transfer",
        lambda *a, **k: {
            "idx": np.zeros((5023, 3), dtype=np.int64),
            "w": np.full((5023, 3), 1.0 / 3.0),
            "scale": np.ones(3),
        },
    )
    seen: list[str] = []
    assert _d3.real_displacement(tuple([0.0] * 253), note=seen.append) is None
    assert seen == ["delta_nonfinite"]


def test_displaced_positions_copies_cause_and_logs_it(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from backend import gnm_assemble as _asm
    from backend import modal_app
    from backend.domain import Err, FitResult, parse_camera_params, parse_gnm_coeffs

    _no_mat_env(monkeypatch, tmp_path)
    _clear_caches()
    coeffs = parse_gnm_coeffs([0.0] * 253)
    assert not isinstance(coeffs, Err)
    camera = parse_camera_params([0.0] * 11 + [1.0])
    assert not isinstance(camera, Err)
    out = _asm.displaced_positions(FitResult(coeffs=coeffs.value, camera=camera.value))
    assert not isinstance(out, Err)
    assert _asm._LAST_DISPLACE_STATS.get("legacy", 0.0) == 1.0
    assert _asm._LAST_DISPLACE_STATS.get("basis_missing", 0.0) == 1.0
    line = modal_app._displace_branch_str()
    assert line.startswith("displacement=legacy")
    assert "basis_missing" in line
