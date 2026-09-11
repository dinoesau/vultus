"""S7 prod-block en `backend/modal_app.py`.

Matriz:
- REAL_MODE=0 local -> fit pasa (doble cuando no hay npz).
- VULTUS_REAL_ML=1 sin npz -> fit retorna Err / falla ruidosa 500, nunca doble.
- con npz real -> fit real pasa (iterations==3 en stats y en el log).
- `_weights_present` exige task MediaPipe Y npz GNM (sin npz no hay auto-real).
- `_impl_fit` / `_impl_texture` propagan Err como RuntimeError (500);
  errores de decode van como ValueError (400).
"""

from __future__ import annotations

import json
import logging
import os

import pytest

from backend import gnm_fit, gnm_head, modal_app
from backend.domain import (
    EmptyPayload,
    Err,
    FitFailed,
    MlDecode,
    MlFailed,
    Ok,
)
from backend.pipeline_local import FIT_RESULT_LEN


def _stub_image(marker: int = 0xA1) -> bytes:
    import io as _io

    from PIL import Image as _Image

    img = _Image.new("RGB", (16, 16), (marker, marker, marker))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _landmarks_json() -> bytes:
    return json.dumps([[0.1, 0.2, 0.3]] * 478).encode("utf-8")


def _fit_payload() -> bytes:
    landmarks = _landmarks_json()
    return len(landmarks).to_bytes(4, "big") + landmarks + _stub_image()


@pytest.fixture
def _no_gnm_weights(monkeypatch, tmp_path):
    """Simula volumen sin pesos: neutraliza TODOS los candidatos de gnm_head.

    GNM_NPZ_PATH solo no basta (hay fallbacks a repo/abs); se parchean las
    listas de candidatos y se limpian los caches singleton.
    """
    missing_npz = str(tmp_path / "missing.npz")
    missing_lm = str(tmp_path / "missing.txt")
    monkeypatch.setattr(gnm_head, "_candidate_npz_paths", lambda: [missing_npz])
    monkeypatch.setattr(gnm_head, "_candidate_landmark_paths", lambda: [missing_lm])
    monkeypatch.setenv("GNM_NPZ_PATH", missing_npz)
    monkeypatch.setenv("GNM_ASSETS_DIR", str(tmp_path / "gnm"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("VULTUS_REAL_ML", "0")
    monkeypatch.setattr(modal_app, "REAL_MODE", "0")
    monkeypatch.setattr(modal_app, "WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setattr(modal_app, "GNM_ASSETS_DIR", str(tmp_path / "gnm"))
    gnm_head._HEAD_CACHE = None
    gnm_fit._LM_X0 = None
    gnm_fit._LM_BASIS = None
    yield tmp_path
    gnm_head._HEAD_CACHE = None
    gnm_fit._LM_X0 = None
    gnm_fit._LM_BASIS = None


def test_weights_present_requires_mediapipe_and_npz(_no_gnm_weights, monkeypatch):
    weights = str(_no_gnm_weights / "weights")
    task = os.path.join(weights, "mediapipe", "face_landmarker.task")
    assert modal_app._weights_present() is False
    os.makedirs(os.path.dirname(task), exist_ok=True)
    with open(task, "wb") as f:
        f.write(b"fake-task")
    # Solo task, sin npz: S7 exige npz -> sigue sin real (antes era True).
    assert modal_app._weights_present() is False
    monkeypatch.setattr(modal_app, "REAL_MODE", "auto")
    assert modal_app._use_real() is False


def test_weights_present_true_with_task_and_real_npz(monkeypatch, tmp_path):
    if not gnm_fit._real_fit_available():
        pytest.skip("sin pesos GNM: no hay npz real que exigir")
    weights = tmp_path / "weights"
    task = weights / "mediapipe" / "face_landmarker.task"
    task.parent.mkdir(parents=True)
    task.write_bytes(b"fake-task")
    monkeypatch.setattr(modal_app, "WEIGHTS_DIR", str(weights))
    assert modal_app._weights_present() is True
    monkeypatch.setattr(modal_app, "REAL_MODE", "auto")
    assert modal_app._use_real() is True


def test_use_real_modes(_no_gnm_weights, monkeypatch):
    monkeypatch.setattr(modal_app, "REAL_MODE", "0")
    assert modal_app._use_real() is False
    monkeypatch.setattr(modal_app, "REAL_MODE", "1")
    assert modal_app._use_real() is True
    monkeypatch.setattr(modal_app, "REAL_MODE", "auto")
    assert modal_app._use_real() is False


def test_impl_fit_local_double_passes(_no_gnm_weights):
    out = modal_app._impl_fit(_fit_payload())
    assert len(out) == FIT_RESULT_LEN
    assert modal_app._impl_fit(_fit_payload()) == out


def test_impl_fit_real_required_without_weights_fails_loudly(_no_gnm_weights, monkeypatch):
    monkeypatch.setattr(modal_app, "REAL_MODE", "1")
    monkeypatch.setenv("VULTUS_REAL_ML", "1")
    assert gnm_fit._real_fit_available() is False
    result = gnm_fit.fit_gnm_from_request(_fit_payload())
    assert isinstance(result, Err)
    with pytest.raises(RuntimeError, match="weights missing"):
        modal_app._impl_fit(_fit_payload())


def test_impl_fit_guard_blocks_silent_double(monkeypatch, tmp_path):
    """Aunque el seam devolviera Ok, con real exigido y sin pesos debe tronar."""
    if not gnm_fit._real_fit_available():
        pytest.skip("se necesita un FitResult real para simular el doble silencioso")
    ok_result = gnm_fit.fit_gnm_from_request(_fit_payload())
    assert isinstance(ok_result, Ok)
    missing = str(tmp_path / "missing.npz")
    monkeypatch.setattr(gnm_head, "_candidate_npz_paths", lambda: [missing])
    monkeypatch.setattr(gnm_head, "_candidate_landmark_paths", lambda: [missing])
    gnm_head._HEAD_CACHE = None
    gnm_fit._LM_X0 = None
    gnm_fit._LM_BASIS = None
    try:
        assert gnm_fit._real_fit_available() is False
        monkeypatch.setattr(modal_app, "REAL_MODE", "1")
        monkeypatch.setattr(gnm_fit, "fit_gnm_from_request", lambda _p: ok_result)
        with pytest.raises(RuntimeError, match="weights missing"):
            modal_app._impl_fit(_fit_payload())
    finally:
        gnm_head._HEAD_CACHE = None
        gnm_fit._LM_X0 = None
        gnm_fit._LM_BASIS = None


def test_impl_fit_err_mapping(monkeypatch):
    monkeypatch.setattr(modal_app, "REAL_MODE", "0")
    monkeypatch.setattr(
        gnm_fit,
        "fit_gnm_from_request",
        lambda _p: Err(FitFailed(detail=MlDecode(details="boom"))),
    )
    with pytest.raises(RuntimeError, match="internal error"):
        modal_app._impl_fit(_fit_payload())
    monkeypatch.setattr(
        gnm_fit, "fit_gnm_from_request", lambda _p: Err(EmptyPayload())
    )
    with pytest.raises(ValueError):
        modal_app._impl_fit(_fit_payload())


def test_impl_texture_mapping(monkeypatch):
    from backend import gnm_texture

    monkeypatch.setattr(modal_app, "REAL_MODE", "0")
    with pytest.raises(ValueError):
        modal_app._impl_texture(b"")
    monkeypatch.setattr(
        gnm_texture,
        "build_albedo",
        lambda _i, _f, _l: Err(MlFailed(detail=MlDecode(details="tex boom"))),
    )
    fit_out = modal_app._impl_fit(_fit_payload())
    pay = _fit_payload()
    tex_req = len(pay).to_bytes(4, "big") + pay + fit_out
    with pytest.raises(RuntimeError, match="internal error"):
        modal_app._impl_texture(tex_req)


def test_impl_texture_happy_path_uses_photo():
    if not gnm_fit._real_fit_available():
        pytest.skip("sin pesos GNM: el happy path de textura corre con fit real local")
    from backend.modal_app import UV_LEN

    fit_out = modal_app._impl_fit(_fit_payload())
    pay = _fit_payload()
    tex_req = len(pay).to_bytes(4, "big") + pay + fit_out
    out = modal_app._impl_texture(tex_req)
    assert len(out) == UV_LEN


def test_fit_infer_logs_iterations_and_loss(caplog):
    if not gnm_fit._real_fit_available():
        pytest.skip("sin pesos GNM: no hay stats reales que loguear")
    caplog.set_level(logging.INFO, logger="vultus-ml-sidecar")
    out = modal_app.fit_infer("job-s7", _fit_payload())
    assert len(out) == FIT_RESULT_LEN
    assert gnm_fit._LAST_FIT_STATS["iterations"] == 3.0
    line = next(r.getMessage() for r in caplog.records if "fit ok job=job-s7" in r.getMessage())
    assert "iterations=3" in line
    assert "loss=" in line
    assert "duration_ms=" in line


def test_fit_worker_timeout_stays_60():
    assert modal_app.FIT_TIMEOUT_SECS == 10
    assert modal_app.TOTAL_TIMEOUT_SECS == 60
