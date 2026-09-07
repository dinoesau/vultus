"""Cola en memoria con ciclo TTL, reloj inyectable y reaper a mitad del TTL.

Un solo Store compartido tras dos adapters (MemoryQueue / R2PointerQueue).
El ciclo de vida es uno solo; los adapters solo difieren en r2_keys.
Sin FastAPI ni logging aqui salvo warning de limpieza tmp.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from backend.domain import (
    CompareResult,
    EnqueueCommand,
    EnqueuedJob,
    JobId,
    JobStatus,
    NotFound,
    Ok,
    Progress,
    R2Keys,
    Stage,
    TtlSecs,
    default_ttl,
    new_job_id,
    parse_r2_key,
    zero_progress,
)
from backend.domain import Err as DomainErr

logger = logging.getLogger("vultus-store")


class SystemClock:
    def now(self) -> float:
        return time.monotonic()


class ManualClock:
    def __init__(self, start: float | None = None) -> None:
        self._lock = threading.Lock()
        self._now = start if start is not None else time.monotonic()

    def now(self) -> float:
        with self._lock:
            return self._now

    def advance(self, delta_secs: float) -> None:
        with self._lock:
            self._now += delta_secs

    def set(self, at: float) -> None:
        with self._lock:
            self._now = at


Clock = SystemClock | ManualClock


def job_dir(job_id: JobId) -> Path:
    return Path(tempfile.gettempdir()) / f"vultus-{job_id.as_str()}"


def cleanup_job_dir(job_id: JobId) -> None:
    directory = job_dir(job_id)
    try:
        shutil.rmtree(directory, ignore_errors=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("tmp cleanup failed job=%s path=%s err=%s", job_id.as_str(), str(directory), exc)


@dataclass
class _Entry:
    status: JobStatus
    progress: Progress
    stage: Stage
    image_a_len: int
    image_b_len: int
    result: CompareResult | None
    created_at: float
    ttl_secs: int


def _not_found(job_id: JobId) -> DomainErr[NotFound]:
    return DomainErr(NotFound(job_id=job_id.as_str()))


class Store:
    def __init__(self, ttl: TtlSecs | None = None, clock: Clock | None = None) -> None:
        self._ttl = ttl if ttl is not None else default_ttl()
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._lock = threading.RLock()
        self._inner: dict[str, _Entry] = {}

    def ttl(self) -> TtlSecs:
        return self._ttl

    def _is_expired(self, entry: _Entry, now: float) -> bool:
        return (now - entry.created_at) >= float(entry.ttl_secs)

    def _is_purgeable(self, entry: _Entry, now: float) -> bool:
        return (now - entry.created_at) >= float(entry.ttl_secs * 2)

    def _insert_queued(self, job_id: JobId, a_len: int, b_len: int) -> None:
        with self._lock:
            self._inner[job_id.as_str()] = _Entry(
                status=JobStatus.QUEUED,
                progress=zero_progress(),
                stage=Stage.QUEUED,
                image_a_len=a_len,
                image_b_len=b_len,
                result=None,
                created_at=self._clock.now(),
                ttl_secs=self._ttl.value(),
            )

    def status(self, job_id: JobId) -> Ok[JobStatus] | DomainErr[NotFound]:
        now = self._clock.now()
        with self._lock:
            entry = self._inner.get(job_id.as_str())
            if entry is None:
                return _not_found(job_id)
            if self._is_expired(entry, now):
                return Ok(JobStatus.EXPIRED)
            return Ok(entry.status)

    def progress(self, job_id: JobId) -> Ok[tuple[Progress, Stage]] | DomainErr[NotFound]:
        now = self._clock.now()
        with self._lock:
            entry = self._inner.get(job_id.as_str())
            if entry is None:
                return _not_found(job_id)
            if self._is_expired(entry, now):
                return _not_found(job_id)
            return Ok((entry.progress, entry.stage))

    def set_progress(self, job_id: JobId, progress: Progress, stage: Stage) -> Ok[None] | DomainErr[NotFound]:
        now = self._clock.now()
        with self._lock:
            entry = self._inner.get(job_id.as_str())
            if entry is None:
                return _not_found(job_id)
            if self._is_expired(entry, now):
                return _not_found(job_id)
            entry.progress = progress
            entry.stage = stage
            entry.status = JobStatus.PROCESSING
            return Ok(None)

    def stored_lens(self, job_id: JobId) -> Ok[tuple[int, int]] | DomainErr[NotFound]:
        now = self._clock.now()
        with self._lock:
            entry = self._inner.get(job_id.as_str())
            if entry is None:
                return _not_found(job_id)
            if self._is_expired(entry, now):
                return _not_found(job_id)
            return Ok((entry.image_a_len, entry.image_b_len))

    def complete_with_result(self, job_id: JobId, result: CompareResult) -> Ok[None] | DomainErr[NotFound]:
        now = self._clock.now()
        with self._lock:
            entry = self._inner.get(job_id.as_str())
            if entry is None:
                return _not_found(job_id)
            if self._is_expired(entry, now):
                return _not_found(job_id)
            from backend.domain import parse_progress

            done = parse_progress(1.0)
            assert isinstance(done, Ok)
            entry.result = result
            entry.progress = done.value
            entry.stage = Stage.DONE
            entry.status = JobStatus.DONE
            return Ok(None)

    def fetch_result(self, job_id: JobId) -> Ok[CompareResult] | DomainErr[NotFound]:
        now = self._clock.now()
        with self._lock:
            entry = self._inner.get(job_id.as_str())
            if entry is None:
                return _not_found(job_id)
            if self._is_expired(entry, now):
                return _not_found(job_id)
            if entry.result is None:
                return _not_found(job_id)
            return Ok(entry.result)

    def fail_job(self, job_id: JobId) -> Ok[None] | DomainErr[NotFound]:
        now = self._clock.now()
        with self._lock:
            entry = self._inner.get(job_id.as_str())
            if entry is None:
                return _not_found(job_id)
            if self._is_expired(entry, now):
                return _not_found(job_id)
            entry.status = JobStatus.FAILED
            return Ok(None)

    def purge_expired(self) -> int:
        now = self._clock.now()
        expired: list[JobId] = []
        with self._lock:
            for key, entry in list(self._inner.items()):
                if self._is_purgeable(entry, now):
                    expired.append(JobId(_value=key))
            for job_id in expired:
                self._inner.pop(job_id.as_str(), None)
        for job_id in expired:
            cleanup_job_dir(job_id)
        return len(expired)


class MemoryQueue:
    def __init__(self, ttl: TtlSecs | None = None, clock: Clock | None = None) -> None:
        self._store = Store(ttl=ttl, clock=clock)

    def ttl(self) -> TtlSecs:
        return self._store.ttl()

    def enqueue(self, cmd: EnqueueCommand) -> EnqueuedJob:
        job_id = new_job_id()
        a_len, b_len = cmd.stored_lens()
        self._store._insert_queued(job_id, a_len, b_len)
        return EnqueuedJob(job_id=job_id, r2_keys=None)

    def status(self, job_id: JobId) -> Ok[JobStatus] | DomainErr[NotFound]:
        return self._store.status(job_id)

    def progress(self, job_id: JobId) -> Ok[tuple[Progress, Stage]] | DomainErr[NotFound]:
        return self._store.progress(job_id)

    def set_progress(self, job_id: JobId, progress: Progress, stage: Stage) -> Ok[None] | DomainErr[NotFound]:
        return self._store.set_progress(job_id, progress, stage)

    def stored_lens(self, job_id: JobId) -> Ok[tuple[int, int]] | DomainErr[NotFound]:
        return self._store.stored_lens(job_id)

    def complete_with_result(self, job_id: JobId, result: CompareResult) -> Ok[None] | DomainErr[NotFound]:
        return self._store.complete_with_result(job_id, result)

    def fetch_result(self, job_id: JobId) -> Ok[CompareResult] | DomainErr[NotFound]:
        return self._store.fetch_result(job_id)

    def fail_job(self, job_id: JobId) -> Ok[None] | DomainErr[NotFound]:
        return self._store.fail_job(job_id)

    def purge_expired(self) -> int:
        return self._store.purge_expired()


class R2PointerQueue:
    def __init__(self, ttl: TtlSecs | None = None, clock: Clock | None = None) -> None:
        self._store = Store(ttl=ttl, clock=clock)

    def ttl(self) -> TtlSecs:
        return self._store.ttl()

    def enqueue(self, cmd: EnqueueCommand) -> EnqueuedJob:
        job_id = new_job_id()
        a_len, b_len = cmd.stored_lens()
        self._store._insert_queued(job_id, a_len, b_len)
        key_a = parse_r2_key(f"jobs/{job_id.as_str()}/a")
        key_b = parse_r2_key(f"jobs/{job_id.as_str()}/b")
        assert isinstance(key_a, Ok) and isinstance(key_b, Ok)
        return EnqueuedJob(job_id=job_id, r2_keys=R2Keys(image_a=key_a.value, image_b=key_b.value))

    def status(self, job_id: JobId) -> Ok[JobStatus] | DomainErr[NotFound]:
        return self._store.status(job_id)

    def progress(self, job_id: JobId) -> Ok[tuple[Progress, Stage]] | DomainErr[NotFound]:
        return self._store.progress(job_id)

    def set_progress(self, job_id: JobId, progress: Progress, stage: Stage) -> Ok[None] | DomainErr[NotFound]:
        return self._store.set_progress(job_id, progress, stage)

    def stored_lens(self, job_id: JobId) -> Ok[tuple[int, int]] | DomainErr[NotFound]:
        return self._store.stored_lens(job_id)

    def complete_with_result(self, job_id: JobId, result: CompareResult) -> Ok[None] | DomainErr[NotFound]:
        return self._store.complete_with_result(job_id, result)

    def fetch_result(self, job_id: JobId) -> Ok[CompareResult] | DomainErr[NotFound]:
        return self._store.fetch_result(job_id)

    def fail_job(self, job_id: JobId) -> Ok[None] | DomainErr[NotFound]:
        return self._store.fail_job(job_id)

    def purge_expired(self) -> int:
        return self._store.purge_expired()
