"""Borrar inputs a|b tras PUT result.zip en Modal, exito o fallo.

Seams bajo prueba (acordados en plan):
- (a) `_run_job_from_r2` orquestador (pipeline + olvido garantizado).
- (b) `_r2_client().delete_object` via `_delete_r2_inputs` / `_delete_r2_related`.

Semantica: sin conservas para reintento. Exito borra a|b y conserva el
zip; cualquier fallo borra a|b mas result.zip y propaga, sin reintento.
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest

from backend import gnm, gnm_assemble, modal_app, pipeline_local, render_preview
from backend.domain import Ok, parse_r2_key


def _stub_image() -> bytes:
    import io as _io

    from PIL import Image as _Image

    img = _Image.new("RGB", (16, 16), (0xA1, 0xA1, 0xA1))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _landmarks_bytes() -> bytes:
    return json.dumps([[0.1, 0.2, 0.3]] * 478).encode("utf-8")


def _fit_bytes() -> bytes:
    from backend.domain import FitResult as _Fit
    from backend.domain import parse_camera_params, parse_gnm_coeffs
    from backend.pipeline_local import encode_fit_result

    coeffs = parse_camera_params([0.0] * 12)
    # Nota: coeffs/camera se construyen via smart constructors, sin forja.
    c = parse_gnm_coeffs([0.0] * 253)
    assert isinstance(c, Ok)
    assert isinstance(coeffs, Ok)
    fit = _Fit(coeffs=c.value, camera=coeffs.value)
    return encode_fit_result(fit)


def _uv_bytes() -> bytes:
    from backend.modal_app import UV_LEN

    return bytes([10]) * UV_LEN


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _NotFoundLike(Exception):
    def __init__(self) -> None:
        super().__init__("NoSuchKey")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class FakeR2:
    """Doble en memoria de boto3 S3 para R2. Sin disco, sin red."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.put_fail = False
        self.delete_fail_keys: set[str] = set()

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict[str, str]:
        _ = (Bucket, ContentType)
        self.calls.append(("put", Key))
        if self.put_fail:
            raise RuntimeError("put failed")
        self.store[Key] = bytes(Body)
        return {}

    def get_object(self, Bucket: str, Key: str) -> dict[str, _Body]:
        _ = Bucket
        if Key not in self.store:
            raise _NotFoundLike()
        return {"Body": _Body(self.store[Key])}

    def delete_object(self, Bucket: str, Key: str) -> dict[str, str]:
        _ = Bucket
        self.calls.append(("delete", Key))
        if Key in self.delete_fail_keys:
            raise RuntimeError("delete failed")
        self.store.pop(Key, None)
        return {}


def _proven_keys(job_id: str):  # type: ignore[no-untyped-def]
    a = parse_r2_key(f"jobs/{job_id}/a")
    b = parse_r2_key(f"jobs/{job_id}/b")
    assert isinstance(a, Ok)
    assert isinstance(b, Ok)
    return a.value, b.value


def _install_run_stubs(monkeypatch, fake: FakeR2, progress_calls: list[tuple[str, float, str]]):  # type: ignore[no-untyped-def]
    """Cablea la cadena pesada a dobles rapidos; el I/O R2 queda real contra FakeR2."""
    import backend.domain as _d

    monkeypatch.setattr(modal_app, "HAVE_MODAL", False)
    monkeypatch.setattr(modal_app, "_job_alive", lambda _jid: True)

    def _report(job_id: str, progress: float, stage: str) -> None:
        progress_calls.append((job_id, progress, stage))

    monkeypatch.setattr(modal_app, "_report_progress", _report)
    monkeypatch.setattr(modal_app, "_r2_client", lambda: fake)
    monkeypatch.setattr(modal_app, "mediapipe_infer", lambda _jid, _img: _landmarks_bytes())
    monkeypatch.setattr(modal_app, "fit_infer", lambda _jid, _img, _lm: Ok(_fit_bytes()))
    monkeypatch.setattr(modal_app, "texture_infer", lambda _jid, _img, _fit, _lm, _hq=False: Ok(_uv_bytes()))

    def _fake_glb(_fit, _uv, eye_texture=None):  # type: ignore[no-untyped-def]
        _ = (_fit, _uv, eye_texture)
        return Ok(_d.GnmMesh(_value=b"glTF" + b"\x00" * 32))

    monkeypatch.setattr(gnm_assemble, "build_personalized_glb", _fake_glb)
    monkeypatch.setattr(gnm, "uv_to_png", lambda _uv: b"png-bytes")
    monkeypatch.setattr(pipeline_local, "resolve_pbr_pngs", lambda _a, _b: Ok((b"pbr-a", b"pbr-b")))
    monkeypatch.setattr(render_preview, "build_preview_png", lambda _fit, _uv: Ok(b"preview-png"))
    monkeypatch.setattr(gnm, "build_result_zip", lambda _bundle: b"PK\x03\x04fakezip")


def test_delete_inputs_after_put_ok_gives_404(monkeypatch, caplog) -> None:
    """Given PUT ok, When termina _run_job_from_r2, Then a y b dan 404."""
    caplog.set_level(logging.INFO, logger="vultus-ml-sidecar")
    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[f"jobs/{job_id}/a"] = _stub_image()
    fake.store[f"jobs/{job_id}/b"] = _stub_image()
    progress_calls: list[tuple[str, float, str]] = []
    _install_run_stubs(monkeypatch, fake, progress_calls)

    modal_app._run_job_from_r2(job_id, r2_a, r2_b)

    assert f"jobs/{job_id}/result.zip" in fake.store
    assert f"jobs/{job_id}/a" not in fake.store
    assert f"jobs/{job_id}/b" not in fake.store
    with pytest.raises(_NotFoundLike):
        fake.get_object("vultus-jobs", f"jobs/{job_id}/a")
    with pytest.raises(_NotFoundLike):
        fake.get_object("vultus-jobs", f"jobs/{job_id}/b")


def test_hq_run_deletes_inputs_after_put_ok(monkeypatch) -> None:
    """Given HQ con stubs, When termina _run_job_from_r2(..., hq=True), Then zip queda y a|b se borran."""
    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[f"jobs/{job_id}/a"] = _stub_image()
    fake.store[f"jobs/{job_id}/b"] = _stub_image()
    progress_calls: list[tuple[str, float, str]] = []
    _install_run_stubs(monkeypatch, fake, progress_calls)

    modal_app._run_job_from_r2(job_id, r2_a, r2_b, True)

    assert f"jobs/{job_id}/result.zip" in fake.store
    assert f"jobs/{job_id}/a" not in fake.store
    assert f"jobs/{job_id}/b" not in fake.store


def test_put_failure_deletes_everything_without_retry(monkeypatch) -> None:
    """Given PUT falla, Then a|b y result.zip se borran y el error propaga, sin reintento."""
    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[f"jobs/{job_id}/a"] = _stub_image()
    fake.store[f"jobs/{job_id}/b"] = _stub_image()
    fake.put_fail = True
    progress_calls: list[tuple[str, float, str]] = []
    _install_run_stubs(monkeypatch, fake, progress_calls)

    with pytest.raises(RuntimeError, match="put failed"):
        modal_app._run_job_from_r2(job_id, r2_a, r2_b)

    assert f"jobs/{job_id}/a" not in fake.store
    assert f"jobs/{job_id}/b" not in fake.store
    assert f"jobs/{job_id}/result.zip" not in fake.store
    deletes = [c for c in fake.calls if c[0] == "delete"]
    assert len(deletes) == 3


def test_pipeline_failure_before_put_deletes_inputs(monkeypatch) -> None:
    """Fallo previo al PUT (fit) tampoco deja huerfanos: a|b fuera, sin zip."""
    from backend.domain import Err as _Err
    from backend.domain import FitFailed as _FitFailed
    from backend.domain import MlDecode as _MlDecode

    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[f"jobs/{job_id}/a"] = _stub_image()
    fake.store[f"jobs/{job_id}/b"] = _stub_image()
    progress_calls: list[tuple[str, float, str]] = []
    _install_run_stubs(monkeypatch, fake, progress_calls)
    monkeypatch.setattr(
        modal_app,
        "fit_infer",
        lambda _jid, _img, _lm: _Err(_FitFailed(detail=_MlDecode(details="boom"))),
    )

    with pytest.raises(RuntimeError):
        modal_app._run_job_from_r2(job_id, r2_a, r2_b)

    assert f"jobs/{job_id}/a" not in fake.store
    assert f"jobs/{job_id}/b" not in fake.store
    assert f"jobs/{job_id}/result.zip" not in fake.store


def test_failure_clears_stale_result_zip(monkeypatch) -> None:
    """Todo lo relacionado: un zip previo del job tambien se purga en fallo."""
    from backend.domain import Err as _Err2
    from backend.domain import FitFailed as _FitFailed2
    from backend.domain import MlDecode as _MlDecode2

    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[f"jobs/{job_id}/a"] = _stub_image()
    fake.store[f"jobs/{job_id}/b"] = _stub_image()
    fake.store[f"jobs/{job_id}/result.zip"] = b"PK\x03\x04stale"
    progress_calls: list[tuple[str, float, str]] = []
    _install_run_stubs(monkeypatch, fake, progress_calls)
    monkeypatch.setattr(
        modal_app,
        "fit_infer",
        lambda _jid, _img, _lm: _Err2(_FitFailed2(detail=_MlDecode2(details="boom"))),
    )

    with pytest.raises(RuntimeError):
        modal_app._run_job_from_r2(job_id, r2_a, r2_b)

    assert f"jobs/{job_id}/a" not in fake.store
    assert f"jobs/{job_id}/b" not in fake.store
    assert f"jobs/{job_id}/result.zip" not in fake.store


def test_done_log_includes_delete_ms_without_bytes(monkeypatch, caplog) -> None:
    """Log con job_id sin bytes incluye delete_inputs_ms."""
    caplog.set_level(logging.INFO, logger="vultus-ml-sidecar")
    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[f"jobs/{job_id}/a"] = _stub_image()
    fake.store[f"jobs/{job_id}/b"] = _stub_image()
    progress_calls: list[tuple[str, float, str]] = []
    _install_run_stubs(monkeypatch, fake, progress_calls)

    modal_app._run_job_from_r2(job_id, r2_a, r2_b)

    done_lines = [r.getMessage() for r in caplog.records if "job done" in r.getMessage()]
    assert len(done_lines) == 1
    line = done_lines[0]
    assert f"job={job_id}" in line
    assert "delete_inputs_ms=" in line
    # Sin bytes de imagen en el log: ni PNG ni landmarks crudos.
    assert "PNG" not in line
    assert "0.1, 0.2" not in line


def test_no_delete_before_put_and_done(monkeypatch) -> None:
    """No se borra antes del assemble: orden PUT -> report done -> delete."""
    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[f"jobs/{job_id}/a"] = _stub_image()
    fake.store[f"jobs/{job_id}/b"] = _stub_image()
    progress_calls: list[tuple[str, float, str]] = []
    order: list[str] = []
    _install_run_stubs(monkeypatch, fake, progress_calls)

    orig_put = fake.put_object
    orig_delete = fake.delete_object
    orig_report = modal_app._report_progress

    def _put(*args, **kwargs):  # type: ignore[no-untyped-def]
        order.append("put")
        return orig_put(*args, **kwargs)

    def _delete(*args, **kwargs):  # type: ignore[no-untyped-def]
        order.append("delete")
        return orig_delete(*args, **kwargs)

    def _report(job_id_arg: str, progress: float, stage: str) -> None:
        if stage == "done":
            order.append("report-done")
        orig_report(job_id_arg, progress, stage)

    monkeypatch.setattr(fake, "put_object", _put)
    monkeypatch.setattr(fake, "delete_object", _delete)
    monkeypatch.setattr(modal_app, "_report_progress", _report)

    modal_app._run_job_from_r2(job_id, r2_a, r2_b)

    assert order.count("put") == 1
    assert order.count("report-done") == 1
    assert order.count("delete") == 2
    assert order.index("put") < order.index("report-done") < order.index("delete")


def test_delete_best_effort_keeps_done_on_single_failure(monkeypatch, caplog) -> None:
    """Delete parcial no tumba done: best-effort con warning, el otro key si se borra."""
    caplog.set_level(logging.WARNING, logger="vultus-ml-sidecar")
    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[f"jobs/{job_id}/a"] = _stub_image()
    fake.store[f"jobs/{job_id}/b"] = _stub_image()
    fake.delete_fail_keys.add(f"jobs/{job_id}/a")
    progress_calls: list[tuple[str, float, str]] = []
    _install_run_stubs(monkeypatch, fake, progress_calls)

    modal_app._run_job_from_r2(job_id, r2_a, r2_b)

    assert f"jobs/{job_id}/result.zip" in fake.store
    assert f"jobs/{job_id}/a" in fake.store
    assert f"jobs/{job_id}/b" not in fake.store
    warns = [r.getMessage() for r in caplog.records if "delete input failed" in r.getMessage()]
    assert len(warns) == 1
    assert job_id in warns[0]


def test_delete_helper_returns_ms_and_deletes_both(monkeypatch) -> None:
    """Seam (b): helper borra a+b y retorna ms entero sin lanzar."""
    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[r2_a.as_str()] = b"a-bytes"
    fake.store[r2_b.as_str()] = b"b-bytes"
    monkeypatch.setattr(modal_app, "_r2_client", lambda: fake)

    ms = modal_app._delete_r2_inputs("vultus-jobs", r2_a, r2_b, job_id)

    assert isinstance(ms, int)
    assert ms >= 0
    assert r2_a.as_str() not in fake.store
    assert r2_b.as_str() not in fake.store


def test_delete_related_removes_inputs_and_zip(monkeypatch) -> None:
    """Via de fallo: helper borra a|b mas result.zip sin lanzar."""
    job_id = str(uuid.uuid4())
    r2_a, r2_b = _proven_keys(job_id)
    fake = FakeR2()
    fake.store[r2_a.as_str()] = b"a-bytes"
    fake.store[r2_b.as_str()] = b"b-bytes"
    fake.store[f"jobs/{job_id}/result.zip"] = b"PK\x03\x04stale"
    monkeypatch.setattr(modal_app, "_r2_client", lambda: fake)

    ms = modal_app._delete_r2_related("vultus-jobs", r2_a, r2_b, job_id)

    assert isinstance(ms, int)
    assert ms >= 0
    assert r2_a.as_str() not in fake.store
    assert r2_b.as_str() not in fake.store
    assert f"jobs/{job_id}/result.zip" not in fake.store
