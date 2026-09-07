"""Worker local: pipeline par con sidecar falso, goldens literales a mano."""

from __future__ import annotations

import json

from backend.domain import (
    UV_LEN,
    CompleteUv,
    DomainError,
    EnqueueCommand,
    Err,
    FlawUv,
    ImageBytes,
    JobId,
    JobStatus,
    Landmarks,
    Ok,
    Stage,
    parse_image_bytes,
    parse_progress,
)
from backend.pipeline_local import QueueLike, default_config
from backend.pipeline_local import run_pair as run_pair_real
from backend.store import MemoryQueue, job_dir

MARKER_A = 0xA1
MARKER_B = 0xB2
FLAW_A = 0x11
FLAW_B = 0x22


def _image(marker: int) -> ImageBytes:
    raw = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes([marker]) * 56
    parsed = parse_image_bytes(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks_bytes() -> bytes:
    pts = [[0.0, 0.0, 0.0]] * 478
    return json.dumps(pts).encode("utf-8")


def _golden_complete(head: bytes) -> bytes:
    return bytes(head) + bytes(UV_LEN - len(head))


class FakeMlOk:
    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        from backend.domain import parse_landmarks

        return parse_landmarks(_landmarks_bytes())

    def flame(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FlawUv] | Err[DomainError]:
        from backend.domain import parse_flaw_uv

        flaw = FLAW_A if MARKER_A in image.as_bytes() else FLAW_B
        return parse_flaw_uv(bytes([flaw]) * UV_LEN)

    def freeuv(self, job_id: JobId, flaw: FlawUv) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import parse_complete_uv

        head = bytes([10, 200]) if flaw.as_bytes()[0] == FLAW_A else bytes([4, 210])
        return parse_complete_uv(_golden_complete(head))


class FakeMlFailLandmarks:
    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        from backend.domain import MlBadStatus, MlFailed

        return Err(MlFailed(detail=MlBadStatus(status=500)))

    def flame(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FlawUv] | Err[DomainError]:
        from backend.domain import parse_flaw_uv

        return parse_flaw_uv(bytes([FLAW_A]) * UV_LEN)

    def freeuv(self, job_id: JobId, flaw: FlawUv) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import parse_complete_uv

        return parse_complete_uv(_golden_complete(bytes([10, 200])))


def test_pair_produces_canonical_uvs_and_golden_heatmap() -> None:
    queue: QueueLike = MemoryQueue()
    image_a = _image(MARKER_A)
    image_b = _image(MARKER_B)
    job_id = queue.enqueue(EnqueueCommand(image_a=image_a, image_b=image_b)).job_id
    out = run_pair_real(queue, FakeMlOk(), job_id, image_a, image_b, default_config())  # type: ignore[arg-type]
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
    status = queue.status(job_id)
    assert isinstance(status, Ok)
    assert status.value == JobStatus.DONE
    progress = queue.progress(job_id)
    assert isinstance(progress, Ok)
    done = parse_progress(1.0)
    assert isinstance(done, Ok)
    assert progress.value[0].value() == done.value.value()
    assert progress.value[1] == Stage.DONE
    stored = queue.fetch_result(job_id)
    assert isinstance(stored, Ok)
    assert stored.value == result
    assert not job_dir(job_id).exists()


def test_sidecar_error_fails_job_and_cleans_tmp() -> None:
    queue: QueueLike = MemoryQueue()
    image_a = _image(MARKER_A)
    image_b = _image(MARKER_B)
    job_id = queue.enqueue(EnqueueCommand(image_a=image_a, image_b=image_b)).job_id
    out = run_pair_real(queue, FakeMlFailLandmarks(), job_id, image_a, image_b, default_config())  # type: ignore[arg-type]
    assert isinstance(out, Err)
    status = queue.status(job_id)
    assert isinstance(status, Ok)
    assert status.value == JobStatus.FAILED
    assert not job_dir(job_id).exists()


def test_pipeline_timeouts_match_decisions() -> None:
    cfg = default_config()
    assert cfg.landmarks_timeout_secs == 5.0
    assert cfg.flame_timeout_secs == 10.0
    assert cfg.freeuv_timeout_secs == 30.0
    assert cfg.total_timeout_secs == 60.0
