"""Cola: comportamiento por interface publica, relojes manuales sin sleeps."""

from __future__ import annotations

from backend.domain import (
    EnqueueCommand,
    Err,
    JobStatus,
    Ok,
    Stage,
    parse_image_bytes,
    parse_progress,
    parse_ttl_secs,
)
from backend.store import (
    ManualClock,
    MemoryQueue,
    R2PointerQueue,
    cleanup_job_dir,
    job_dir,
)


def _png() -> bytes:
    return bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes(56)


def _cmd() -> EnqueueCommand:
    a = parse_image_bytes(_png())
    b = parse_image_bytes(_png())
    assert isinstance(a, Ok) and isinstance(b, Ok)
    return EnqueueCommand(image_a=a.value, image_b=b.value)


def test_memory_queue_keeps_bytes_and_tracks_progress() -> None:
    q = MemoryQueue()
    job = q.enqueue(_cmd())
    assert not job.is_r2_pointer()
    lens = q.stored_lens(job.job_id)
    assert isinstance(lens, Ok)
    assert lens.value == (64, 64)
    current = q.progress(job.job_id)
    assert isinstance(current, Ok)
    assert current.value[1] == Stage.QUEUED
    progress = parse_progress(0.4)
    assert isinstance(progress, Ok)
    updated = q.set_progress(job.job_id, progress.value, Stage.FLAME)
    assert isinstance(updated, Ok)
    status = q.status(job.job_id)
    assert isinstance(status, Ok)
    assert status.value == JobStatus.PROCESSING


def test_r2_pointer_queue_returns_keys() -> None:
    q = R2PointerQueue()
    job = q.enqueue(_cmd())
    assert job.is_r2_pointer()
    assert job.r2_keys is not None
    assert job.job_id.as_str() in job.r2_keys.image_a.as_str()
    lens = q.stored_lens(job.job_id)
    assert isinstance(lens, Ok)
    assert lens.value == (64, 64)


def test_unknown_job_is_not_found() -> None:
    from backend.domain import new_job_id

    q = MemoryQueue()
    missing = new_job_id()
    assert isinstance(q.status(missing), Err)
    assert isinstance(q.progress(missing), Err)


def test_job_expires_after_ttl_and_lens_gone() -> None:
    ttl = parse_ttl_secs(1)
    assert isinstance(ttl, Ok)
    clock = ManualClock()
    q = MemoryQueue(ttl=ttl.value, clock=clock)
    job = q.enqueue(_cmd())
    status = q.status(job.job_id)
    assert isinstance(status, Ok)
    assert status.value == JobStatus.QUEUED
    clock.advance(2.0)
    expired = q.status(job.job_id)
    assert isinstance(expired, Ok)
    assert expired.value == JobStatus.EXPIRED
    assert isinstance(q.stored_lens(job.job_id), Err)
    assert isinstance(q.progress(job.job_id), Err)


def test_expired_job_is_purged_after_double_ttl_and_tmp_clean() -> None:
    ttl = parse_ttl_secs(1)
    assert isinstance(ttl, Ok)
    clock = ManualClock()
    q = MemoryQueue(ttl=ttl.value, clock=clock)
    job = q.enqueue(_cmd())
    directory = job_dir(job.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "x.bin").write_bytes(b"hi")
    clock.advance(3.0)
    purged = q.purge_expired()
    assert purged == 1
    assert isinstance(q.status(job.job_id), Err)
    assert not directory.exists()
    cleanup_job_dir(job.job_id)
