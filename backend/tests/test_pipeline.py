"""Worker local: pipeline par con sidecar falso, goldens literales a mano.

El pipeline consume el sink estrecho (`report`, `complete`, `fail`);
los tests inyectan el sink en memoria, el runner el sink HTTP.
"""

from __future__ import annotations

import json

from backend.domain import (
    CompareResult,
    CompleteUv,
    DomainError,
    Err,
    FitResult,
    ImageBytes,
    JobId,
    Landmarks,
    Ok,
    Progress,
    Stage,
    parse_image_bytes,
    parse_progress,
)
from backend.pipeline_local import ProgressSink, default_config, job_dir
from backend.pipeline_local import run_pair as run_pair_real

MARKER_A = 0xA1
MARKER_B = 0xB2


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


class InMemorySink:
    """Sink de tests: registra reportes y guarda el resultado."""

    def __init__(self) -> None:
        self.reports: list[tuple[Progress, Stage]] = []
        self.result: CompareResult | None = None
        self.failed = False

    def report(self, progress: Progress, stage: Stage) -> Ok[None] | Err[DomainError]:
        self.reports.append((progress, stage))
        return Ok(None)

    def complete(self, result: CompareResult) -> Ok[None] | Err[DomainError]:
        self.result = result
        return Ok(None)

    def fail(self) -> Ok[None] | Err[DomainError]:
        self.failed = True
        return Ok(None)


class FakeMlOk:
    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        from backend.domain import parse_landmarks

        return parse_landmarks(_landmarks_bytes())

    def fit(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
        from backend.gnm_fit import fit_gnm

        return fit_gnm(image, landmarks)

    def texture(
        self, job_id: JobId, image: ImageBytes, fit: FitResult, landmarks: Landmarks
    ) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import parse_complete_uv

        head = bytes([10, 200]) if MARKER_A in image.as_bytes() else bytes([4, 210])
        return parse_complete_uv(_golden_complete(head))


class FakeMlFailFit:
    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        from backend.domain import parse_landmarks

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


def test_pair_produces_canonical_uvs_and_golden_heatmap() -> None:
    from backend.domain import UV_LEN, new_job_id

    sink: ProgressSink = InMemorySink()
    image_a = _image(MARKER_A)
    image_b = _image(MARKER_B)
    job_id = new_job_id()
    out = run_pair_real(sink, FakeMlOk(), job_id, image_a, image_b, default_config())  # type: ignore[arg-type]
    assert isinstance(out, Ok)
    result = out.value
    assert len(result.uv_a.as_bytes()) == UV_LEN
    assert len(result.uv_b.as_bytes()) == UV_LEN
    assert result.uv_a.as_bytes()[0:2] == bytes([10, 200])
    assert result.uv_b.as_bytes()[0:2] == bytes([4, 210])
    assert all(b == 0 for b in result.uv_a.as_bytes()[2:])
    assert all(b == 0 for b in result.uv_b.as_bytes()[2:])
    assert result.heatmap.as_bytes()[0:2] == bytes([6, 10])
    assert all(b == 0 for b in result.heatmap.as_bytes()[2:])
    for mesh in (result.mesh_a, result.mesh_b):
        assert mesh.as_bytes()[0:4] == b"glTF"
        assert 100_000 < len(mesh.as_bytes()) < 2_000_000
    assert result.mesh_a != result.mesh_b
    assert isinstance(sink, InMemorySink)
    assert sink.result == result
    assert not sink.failed
    done = parse_progress(1.0)
    assert isinstance(done, Ok)
    assert [(p.value(), s) for p, s in sink.reports] == [
        (0.40, Stage.FIT),
        (0.75, Stage.TEXTURE),
        (0.95, Stage.ASSEMBLE),
    ]
    assert not job_dir(job_id).exists()


def test_sidecar_error_fails_job_and_cleans_tmp() -> None:
    from backend.domain import new_job_id

    sink: ProgressSink = InMemorySink()
    image_a = _image(MARKER_A)
    image_b = _image(MARKER_B)
    job_id = new_job_id()
    out = run_pair_real(sink, FakeMlFailFit(), job_id, image_a, image_b, default_config())  # type: ignore[arg-type]
    assert isinstance(out, Err)
    assert isinstance(sink, InMemorySink)
    assert sink.failed
    assert sink.result is None
    assert not job_dir(job_id).exists()


def test_pipeline_timeouts_match_decisions() -> None:
    cfg = default_config()
    assert cfg.landmarks_timeout_secs == 5.0
    assert cfg.fit_timeout_secs == 10.0
    assert cfg.texture_timeout_secs == 30.0
    assert cfg.total_timeout_secs == 60.0
