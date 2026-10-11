"""Paridad dev anti-basura: el runner deja tmpfs y R2 emulado sin inputs.

Seams bajo prueba (acordados):
- (a) `process_message` orquestador (fetch blobs + pipeline + PUT + done).
- (b) `HttpProgressSink.complete` via `_http` (PUT zip antes de POST done).

Semantica: tras `done` el R2 emulado conserva solo `result.zip`
(inputs `a|b` purgados como el gateway dev) y `job_dir` no existe.
En fallo no quedan huerfanos de tmp.
"""

from __future__ import annotations

import json
import uuid

from backend.domain import (
    CompleteUv,
    DomainError,
    Err,
    FitResult,
    ImageBytes,
    JobId,
    Landmarks,
    Ok,
    parse_image_bytes,
    parse_landmarks,
)
from backend.pipeline_local import job_dir


def _image(marker: int) -> ImageBytes:
    raw = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes([marker]) * 56
    parsed = parse_image_bytes(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks_bytes() -> bytes:
    pts = [[0.0, 0.0, 0.0]] * 478
    return json.dumps(pts).encode("utf-8")


def _golden_complete(head: bytes) -> bytes:
    from backend.domain import UV_LEN

    return bytes(head) + bytes(UV_LEN - len(head))


class _FakeMlOk:
    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        _ = (job_id, image)
        return parse_landmarks(_landmarks_bytes())

    def fit(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
        from backend.gnm_fit import fit_gnm

        _ = job_id
        return fit_gnm(image, landmarks)

    def texture(
        self, job_id: JobId, image: ImageBytes, fit: FitResult, landmarks: Landmarks
    ) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import parse_complete_uv

        _ = (job_id, fit, landmarks)
        head = bytes([10, 200]) if 0xA1 in image.as_bytes()[:16] else bytes([4, 210])
        return parse_complete_uv(_golden_complete(head))


class _FakeMlFail:
    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        _ = (job_id, image)
        return parse_landmarks(_landmarks_bytes())

    def fit(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
        from backend.domain import FitFailed, MlBadStatus

        _ = (job_id, image, landmarks)
        return Err(FitFailed(detail=MlBadStatus(status=500)))

    def texture(
        self, job_id: JobId, image: ImageBytes, fit: FitResult, landmarks: Landmarks
    ) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import parse_complete_uv

        _ = (job_id, image, fit, landmarks)
        return parse_complete_uv(_golden_complete(bytes([10, 200])))


class _FakeGateway:
    """Emula gateway dev + R2: blobs via GET, PUT borra a|b como worker.dev."""

    def __init__(self, job_id: str, blob_a: bytes, blob_b: bytes) -> None:
        self.job_id = job_id
        self.r2: dict[str, bytes] = {
            f"jobs/{job_id}/a": blob_a,
            f"jobs/{job_id}/b": blob_b,
        }
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, method: str, url: str, body: bytes | None = None, content_type: str = "application/json"
    ) -> tuple[int, bytes]:
        _ = content_type
        self.calls.append((method, url))
        if method == "GET" and url.endswith(f"/v1/jobs/{self.job_id}"):
            return 200, json.dumps({"status": "queued"}).encode()
        if method == "GET" and url.endswith(f"/dev/blobs/{self.job_id}/a"):
            key = f"jobs/{self.job_id}/a"
            if key not in self.r2:
                return 404, b""
            return 200, self.r2[key]
        if method == "GET" and url.endswith(f"/dev/blobs/{self.job_id}/b"):
            key = f"jobs/{self.job_id}/b"
            if key not in self.r2:
                return 404, b""
            return 200, self.r2[key]
        if method == "PUT" and url.endswith(f"/dev/results/{self.job_id}"):
            assert body is not None and body[:2] == b"PK"
            self.r2[f"jobs/{self.job_id}/result.zip"] = bytes(body)
            self.r2.pop(f"jobs/{self.job_id}/a", None)
            self.r2.pop(f"jobs/{self.job_id}/b", None)
            return 200, b'{"ok": true}'
        if method == "POST" and f"/v1/jobs/{self.job_id}/progress" in url:
            return 200, b'{"ok": true}'
        return 404, b""


def _envelope(job_id: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "r2_keys": {"image_a": f"jobs/{job_id}/a", "image_b": f"jobs/{job_id}/b"},
    }


def test_process_message_done_deja_tmp_vacio_y_r2_sin_inputs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Given job queued, When process_message completa, Then tmp vacio y R2 sin a|b."""
    import backend.local_runner as runner
    from backend.domain import new_job_id

    job_id = new_job_id().as_str()
    gateway = _FakeGateway(job_id, _image(0xA1).as_bytes(), _image(0xB2).as_bytes())
    monkeypatch.setattr(runner, "_http", gateway)
    monkeypatch.setattr(runner, "_sidecar_client", lambda: _FakeMlOk())

    outcome = runner.process_message(_envelope(job_id))

    assert outcome.action == "done"
    assert outcome.job_id == job_id
    # tmpfs vacio: el pipeline limpio el directorio del job.
    assert not job_dir(_job_id_of(job_id)).exists()
    # R2 emulado sin inputs, con zip.
    assert f"jobs/{job_id}/a" not in gateway.r2
    assert f"jobs/{job_id}/b" not in gateway.r2
    assert f"jobs/{job_id}/result.zip" in gateway.r2
    # Orden: PUT zip antes del POST done.
    puts = [i for i, (m, u) in enumerate(gateway.calls) if m == "PUT" and "/dev/results/" in u]
    posts = [i for i, (m, u) in enumerate(gateway.calls) if m == "POST" and "/progress" in u]
    assert len(puts) == 1
    assert len(posts) >= 1
    assert puts[0] < posts[-1]


def _job_id_of(job_id: str) -> JobId:
    from backend.domain import parse_job_id

    parsed = parse_job_id(job_id)
    assert isinstance(parsed, Ok)
    return parsed.value


def test_process_message_done_limpia_tmp_real(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Given pipeline real, When process_message ok, Then job_dir no existe."""
    import backend.local_runner as runner

    job_id = str(uuid.uuid4())
    gateway = _FakeGateway(job_id, _image(0xA1).as_bytes(), _image(0xB2).as_bytes())
    monkeypatch.setattr(runner, "_http", gateway)
    monkeypatch.setattr(runner, "_sidecar_client", lambda: _FakeMlOk())

    outcome = runner.process_message(_envelope(job_id))

    assert outcome.action == "done"
    assert not job_dir(_job_id_of(job_id)).exists()


def test_process_message_fallo_limpia_tmp_y_no_deja_zip(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Given fit falla, When process_message termina, Then tmp vacio y sin zip."""
    import backend.local_runner as runner

    job_id = str(uuid.uuid4())
    gateway = _FakeGateway(job_id, _image(0xA1).as_bytes(), _image(0xB2).as_bytes())
    monkeypatch.setattr(runner, "_http", gateway)
    monkeypatch.setattr(runner, "_sidecar_client", lambda: _FakeMlFail())

    outcome = runner.process_message(_envelope(job_id))

    assert outcome.action == "failed-pipeline"
    assert not job_dir(_job_id_of(job_id)).exists()
    assert f"jobs/{job_id}/result.zip" not in gateway.r2
