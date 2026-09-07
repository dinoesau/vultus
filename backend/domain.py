"""Dominio Python: tipos probados una vez en el borde, core sin revalidar.

Fuente de verdad del contrato HTTP en TypeScript (`edge/contract.ts`);
este modulo es dueno local de la invariante. Sin FastAPI ni logging aqui.
Historico: antes Rust (borrado en plan-remove-rust-stack).
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Never, TypeVar

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")
T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", covariant=True)

# --- Constantes canonicas (no inventar valores nuevos) ---

MAX_IMAGE_BYTES = 8 * 1024 * 1024
RESULT_TTL_SECONDS = 60
TTL_MIN_SECS = 1
TTL_MAX_SECS = 3600

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

LANDMARKS_LEN = 478
UV_WIDTH = 512
UV_HEIGHT = 512
UV_CHANNELS = 3
UV_LEN = UV_WIDTH * UV_HEIGHT * UV_CHANNELS

TEMPLATE_VERTS = 4225
TEMPLATE_TRIS = 8192

GLB_MAGIC = b"glTF"
GLB_MIN_LEN = 12
GLB_VERSION = 2

ZIP_UV_A = "uv_a.png"
ZIP_UV_B = "uv_b.png"
ZIP_HEATMAP = "heatmap.png"
ZIP_MESH_A = "mesh_a.glb"
ZIP_MESH_B = "mesh_b.glb"

ZIP_NAMES = (ZIP_UV_A, ZIP_UV_B, ZIP_HEATMAP, ZIP_MESH_A, ZIP_MESH_B)

PROGRESS_LANDMARKS = 0.15
PROGRESS_FLAME = 0.40
PROGRESS_FREEUV = 0.75
PROGRESS_BAKE = 0.95
PROGRESS_DONE = 1.0

LANDMARKS_TIMEOUT_SECS = 5
FLAME_TIMEOUT_SECS = 10
FREEUV_TIMEOUT_SECS = 30
TOTAL_TIMEOUT_SECS = 60


class Stage(str, Enum):
    QUEUED = "queued"
    LANDMARKS = "landmarks"
    FLAME = "flame"
    FREEUV = "freeuv"
    BAKE = "bake"
    DONE = "done"

    def as_str(self) -> str:
        return self.value


STAGES = ("queued", "landmarks", "flame", "freeuv", "bake", "done")


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    EXPIRED = "expired"

    def as_str(self) -> str:
        return self.value


TERMINAL_STATUSES = ("done", "failed", "expired")


# --- Result y combinadores (railway) ---


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E


type Result[T, E] = Ok[T] | Err[E]


def map_result(result: Result[T, E], fn: Callable[[T], U]) -> Result[U, E]:
    if isinstance(result, Ok):
        return Ok(fn(result.value))
    return result


def and_then(result: Result[T, E], fn: Callable[[T], Result[U, F]]) -> Result[U, E | F]:
    if isinstance(result, Ok):
        return fn(result.value)  # type: ignore[return-value]
    err: Err[E] = result
    return Err(err.error)


def map_err(result: Result[T, E], fn: Callable[[E], F]) -> Result[T, F]:
    if isinstance(result, Err):
        return Err(fn(result.error))
    return result


def assert_never(value: Never) -> Never:
    raise AssertionError(f"unhandled case: {value!r}")


# --- Errores estratificados (union exhaustiva de dominio) ---


@dataclass(frozen=True, slots=True)
class SizeOutOfRange:
    detail: str = "size out of range"


@dataclass(frozen=True, slots=True)
class UnsupportedFormat:
    detail: str = "not jpeg nor png"


type ImageError = SizeOutOfRange | UnsupportedFormat


@dataclass(frozen=True, slots=True)
class BadScheme:
    detail: str = "base_url must start with http:// or https://"


@dataclass(frozen=True, slots=True)
class EmptyUrl:
    detail: str = "base_url is empty"


type BaseUrlError = BadScheme | EmptyUrl


@dataclass(frozen=True, slots=True)
class MlTransport:
    details: str


@dataclass(frozen=True, slots=True)
class MlBadStatus:
    status: int


@dataclass(frozen=True, slots=True)
class MlDecode:
    details: str


@dataclass(frozen=True, slots=True)
class MlEmpty:
    detail: str = "ml empty payload"


type MlError = MlTransport | MlBadStatus | MlDecode | MlEmpty


@dataclass(frozen=True, slots=True)
class QueueBackend:
    details: str


@dataclass(frozen=True, slots=True)
class InvalidImage:
    detail: ImageError


@dataclass(frozen=True, slots=True)
class InvalidJobId:
    detail: str = "invalid job_id"


@dataclass(frozen=True, slots=True)
class InvalidProgress:
    detail: str = "invalid progress"


@dataclass(frozen=True, slots=True)
class InvalidR2Key:
    detail: str = "invalid r2_key"


@dataclass(frozen=True, slots=True)
class InvalidBaseUrl:
    detail: BaseUrlError


@dataclass(frozen=True, slots=True)
class EmptyPayload:
    detail: str = "empty payload"


@dataclass(frozen=True, slots=True)
class QueueFailed:
    detail: QueueBackend


@dataclass(frozen=True, slots=True)
class MlFailed:
    detail: MlError


@dataclass(frozen=True, slots=True)
class NotFound:
    job_id: str


@dataclass(frozen=True, slots=True)
class Invariant:
    detail: str


type DomainError = (
    InvalidImage
    | InvalidJobId
    | InvalidProgress
    | InvalidR2Key
    | InvalidBaseUrl
    | EmptyPayload
    | QueueFailed
    | MlFailed
    | NotFound
    | Invariant
)


def domain_to_status(error: DomainError) -> int:
    if isinstance(error, (InvalidImage, InvalidJobId, InvalidProgress, InvalidR2Key, InvalidBaseUrl, EmptyPayload)):
        return 400
    if isinstance(error, NotFound):
        return 404
    if isinstance(error, (QueueFailed, MlFailed, Invariant)):
        return 500
    assert_never(error)


def domain_to_message(error: DomainError) -> str:
    if isinstance(error, InvalidImage):
        detail = error.detail
        if isinstance(detail, SizeOutOfRange):
            return "invalid image: size out of range"
        return "invalid image: not jpeg nor png"
    if isinstance(error, InvalidJobId):
        return "invalid job_id"
    if isinstance(error, InvalidProgress):
        return "invalid progress"
    if isinstance(error, InvalidR2Key):
        return "invalid r2_key"
    if isinstance(error, InvalidBaseUrl):
        return f"invalid base_url: {error.detail.detail}"
    if isinstance(error, EmptyPayload):
        return "empty payload"
    if isinstance(error, NotFound):
        return f"not found: {error.job_id}"
    if isinstance(error, (QueueFailed, MlFailed, Invariant)):
        return "internal error"
    assert_never(error)


# --- Value objects con smart constructor unico ---


def _is_jpeg(raw: bytes) -> bool:
    return len(raw) >= 3 and raw[0:3] == JPEG_MAGIC


def _is_png(raw: bytes) -> bool:
    return len(raw) >= 8 and raw[0:8] == PNG_MAGIC


@dataclass(frozen=True, slots=True)
class ImageBytes:
    """Solo via parse_image_bytes."""

    _value: bytes

    def as_bytes(self) -> bytes:
        return self._value

    def __len__(self) -> int:
        return len(self._value)


def parse_image_bytes(raw: object) -> Result[ImageBytes, DomainError]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return Err(InvalidImage(detail=SizeOutOfRange()))
    data = bytes(raw)
    if len(data) == 0 or len(data) > MAX_IMAGE_BYTES:
        return Err(InvalidImage(detail=SizeOutOfRange()))
    if not (_is_jpeg(data) or _is_png(data)):
        return Err(InvalidImage(detail=UnsupportedFormat()))
    return Ok(ImageBytes(_value=data))


@dataclass(frozen=True, slots=True)
class JobId:
    """Solo via parse_job_id o new_job_id."""

    _value: str

    def as_str(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value


def new_job_id() -> JobId:
    return JobId(_value=str(uuid.uuid4()))


def parse_job_id(raw: object) -> Result[JobId, DomainError]:
    if not isinstance(raw, str):
        return Err(InvalidJobId())
    trimmed = raw.strip()
    if not trimmed:
        return Err(InvalidJobId())
    try:
        parsed = uuid.UUID(trimmed)
    except (ValueError, AttributeError, TypeError):
        return Err(InvalidJobId())
    return Ok(JobId(_value=str(parsed)))


@dataclass(frozen=True, slots=True)
class Progress:
    """Solo via parse_progress. Rango 0..1 finito."""

    _value: float

    def value(self) -> float:
        return self._value


def parse_progress(raw: object) -> Result[Progress, DomainError]:
    if isinstance(raw, bool):
        return Err(InvalidProgress())
    if not isinstance(raw, (int, float)):
        return Err(InvalidProgress())
    value = float(raw)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        return Err(InvalidProgress())
    return Ok(Progress(_value=value))


def zero_progress() -> Progress:
    return Progress(_value=0.0)


def parse_stage(raw: object) -> Result[Stage, DomainError]:
    if not isinstance(raw, str):
        return Err(InvalidProgress())
    try:
        return Ok(Stage(raw))
    except ValueError:
        return Err(InvalidProgress())


@dataclass(frozen=True, slots=True)
class TtlSecs:
    """Solo via parse_ttl_secs. Rango 1..3600."""

    _value: int

    def value(self) -> int:
        return self._value

    def reaper_interval_secs(self) -> int:
        doubled = (self._value + 1) // 2
        return max(doubled, 1)

    def purge_after_secs(self) -> int:
        return self._value * 2


def parse_ttl_secs(raw: object) -> Result[TtlSecs, DomainError]:
    if isinstance(raw, bool):
        return Err(Invariant(detail="ttl out of range"))
    if not isinstance(raw, int):
        return Err(Invariant(detail="ttl out of range"))
    if raw < TTL_MIN_SECS or raw > TTL_MAX_SECS:
        return Err(Invariant(detail="ttl out of range"))
    return Ok(TtlSecs(_value=raw))


def default_ttl() -> TtlSecs:
    return TtlSecs(_value=RESULT_TTL_SECONDS)


@dataclass(frozen=True, slots=True)
class R2Key:
    """Solo via parse_r2_key."""

    _value: str

    def as_str(self) -> str:
        return self._value


def parse_r2_key(raw: object) -> Result[R2Key, DomainError]:
    if not isinstance(raw, str):
        return Err(InvalidR2Key())
    trimmed = raw.strip()
    if not trimmed or len(trimmed) > 1024 or ".." in trimmed:
        return Err(InvalidR2Key())
    return Ok(R2Key(_value=trimmed))


@dataclass(frozen=True, slots=True)
class R2Keys:
    image_a: R2Key
    image_b: R2Key


@dataclass(frozen=True, slots=True)
class EnqueueCommand:
    image_a: ImageBytes
    image_b: ImageBytes

    def stored_lens(self) -> tuple[int, int]:
        return (len(self.image_a), len(self.image_b))


@dataclass(frozen=True, slots=True)
class EnqueuedJob:
    job_id: JobId
    r2_keys: R2Keys | None

    def is_r2_pointer(self) -> bool:
        return self.r2_keys is not None


@dataclass(frozen=True, slots=True)
class BaseUrl:
    """Solo via parse_base_url."""

    _value: str

    def as_str(self) -> str:
        return self._value

    def join(self, path: str) -> str:
        return f"{self._value}{path}"

    def __str__(self) -> str:
        return self._value


def parse_base_url(raw: object) -> Result[BaseUrl, DomainError]:
    if not isinstance(raw, str):
        return Err(InvalidBaseUrl(detail=BadScheme()))
    trimmed = raw.strip()
    if not trimmed:
        return Err(InvalidBaseUrl(detail=EmptyUrl()))
    if not trimmed.startswith(("http://", "https://")):
        return Err(InvalidBaseUrl(detail=BadScheme()))
    return Ok(BaseUrl(_value=trimmed.rstrip("/")))


@dataclass(frozen=True, slots=True)
class Landmarks:
    """Solo via parse_landmarks. JSON 478 puntos finitos."""

    _value: bytes

    def as_bytes(self) -> bytes:
        return self._value


def parse_landmarks(raw: object) -> Result[Landmarks, DomainError]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return Err(MlFailed(detail=MlDecode(details="expected landmarks bytes")))
    data = bytes(raw)
    if len(data) == 0:
        return Err(EmptyPayload())
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return Err(MlFailed(detail=MlDecode(details=f"invalid landmarks json: {exc}")))
    try:
        pts = json.loads(text)
    except json.JSONDecodeError as exc:
        return Err(MlFailed(detail=MlDecode(details=f"invalid landmarks json: {exc}")))
    if not isinstance(pts, list) or len(pts) != LANDMARKS_LEN:
        got = len(pts) if isinstance(pts, list) else -1
        return Err(MlFailed(detail=MlDecode(details=f"expected {LANDMARKS_LEN} points, got {got}")))
    for point in pts:
        if not isinstance(point, list) or len(point) != 3:
            return Err(MlFailed(detail=MlDecode(details="invalid landmark point, expected [x,y,z]")))
        for value in point:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return Err(MlFailed(detail=MlDecode(details="non-finite landmark")))
            if not math.isfinite(float(value)):
                return Err(MlFailed(detail=MlDecode(details="non-finite landmark")))
    return Ok(Landmarks(_value=data))


def _parse_uv_bytes(raw: object, label: str) -> Result[bytes, DomainError]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return Err(MlFailed(detail=MlDecode(details=f"expected {UV_LEN} uv bytes, got -1")))
    data = bytes(raw)
    if len(data) != UV_LEN:
        return Err(MlFailed(detail=MlDecode(details=f"expected {UV_LEN} uv bytes, got {len(data)}")))
    _ = label
    return Ok(data)


@dataclass(frozen=True, slots=True)
class FlawUv:
    """Solo via parse_flaw_uv."""

    _value: bytes

    def as_bytes(self) -> bytes:
        return self._value

    def __len__(self) -> int:
        return len(self._value)


def parse_flaw_uv(raw: object) -> Result[FlawUv, DomainError]:
    result = _parse_uv_bytes(raw, "flaw")
    if isinstance(result, Err):
        return result
    return Ok(FlawUv(_value=result.value))


@dataclass(frozen=True, slots=True)
class CompleteUv:
    """Solo via parse_complete_uv."""

    _value: bytes

    def as_bytes(self) -> bytes:
        return self._value

    def __len__(self) -> int:
        return len(self._value)


def parse_complete_uv(raw: object) -> Result[CompleteUv, DomainError]:
    result = _parse_uv_bytes(raw, "complete")
    if isinstance(result, Err):
        return result
    return Ok(CompleteUv(_value=result.value))


@dataclass(frozen=True, slots=True)
class Heatmap:
    """Solo via parse_heatmap o compute_heatmap."""

    _value: bytes

    def as_bytes(self) -> bytes:
        return self._value

    def __len__(self) -> int:
        return len(self._value)


def parse_heatmap(raw: object) -> Result[Heatmap, DomainError]:
    result = _parse_uv_bytes(raw, "heatmap")
    if isinstance(result, Err):
        return result
    return Ok(Heatmap(_value=result.value))


@dataclass(frozen=True, slots=True)
class GnmMesh:
    """Solo via parse_gnm_mesh. Magic glTF y longitud coherente."""

    _value: bytes

    def as_bytes(self) -> bytes:
        return self._value

    def __len__(self) -> int:
        return len(self._value)


def parse_gnm_mesh(raw: object) -> Result[GnmMesh, DomainError]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return Err(MlFailed(detail=MlDecode(details="glb truncated: got -1 bytes")))
    data = bytes(raw)
    if len(data) < GLB_MIN_LEN:
        return Err(MlFailed(detail=MlDecode(details=f"glb truncated: got {len(data)} bytes")))
    if data[0:4] != GLB_MAGIC:
        return Err(MlFailed(detail=MlDecode(details="glb missing glTF magic")))
    total = int.from_bytes(data[8:12], "little")
    if total != len(data):
        return Err(MlFailed(detail=MlDecode(details=f"glb length mismatch: header {total} != {len(data)}")))
    return Ok(GnmMesh(_value=data))


@dataclass(frozen=True, slots=True)
class CompareResult:
    uv_a: CompleteUv
    uv_b: CompleteUv
    heatmap: Heatmap
    mesh_a: GnmMesh
    mesh_b: GnmMesh


def encode_flame_payload(landmarks: Landmarks, image: ImageBytes) -> bytes:
    lm = landmarks.as_bytes()
    img = image.as_bytes()
    return len(lm).to_bytes(4, "big") + lm + img


def decode_flame_payload(raw: object) -> Result[tuple[Landmarks, ImageBytes], DomainError]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return Err(MlFailed(detail=MlDecode(details="flame payload <4 bytes")))
    data = bytes(raw)
    if len(data) < 4:
        return Err(MlFailed(detail=MlDecode(details="flame payload <4 bytes")))
    size = int.from_bytes(data[0:4], "big")
    if len(data) < 4 + size:
        return Err(MlFailed(detail=MlDecode(details="flame payload truncated")))
    lm_raw = data[4 : 4 + size]
    img_raw = data[4 + size :]
    lm_result = parse_landmarks(lm_raw)
    if isinstance(lm_result, Err):
        return lm_result
    img_result = parse_image_bytes(img_raw)
    if isinstance(img_result, Err):
        detail = domain_to_message(img_result.error)
        return Err(MlFailed(detail=MlDecode(details=detail)))
    return Ok((lm_result.value, img_result.value))


# --- Type-state del ciclo de job (estados separados, sin bool) ---


@dataclass(frozen=True, slots=True)
class JobQueued:
    job_id: JobId
    progress: Progress
    stage: Stage


@dataclass(frozen=True, slots=True)
class JobProcessing:
    job_id: JobId
    progress: Progress
    stage: Stage


@dataclass(frozen=True, slots=True)
class JobDone:
    job_id: JobId

    def receipt(self) -> str:
        return f"done {self.job_id.as_str()}"


@dataclass(frozen=True, slots=True)
class JobFailed:
    job_id: JobId


@dataclass(frozen=True, slots=True)
class JobExpired:
    job_id: JobId


def new_queued(job_id: JobId) -> JobQueued:
    return JobQueued(job_id=job_id, progress=zero_progress(), stage=Stage.QUEUED)


def start_job(job: JobQueued) -> JobProcessing:
    return JobProcessing(job_id=job.job_id, progress=job.progress, stage=Stage.LANDMARKS)


def set_job_progress(job: JobProcessing, progress: Progress, stage: Stage) -> JobProcessing:
    return JobProcessing(job_id=job.job_id, progress=progress, stage=stage)


def complete_job(job: JobProcessing) -> JobDone:
    return JobDone(job_id=job.job_id)


def fail_job_state(job: JobProcessing) -> JobFailed:
    return JobFailed(job_id=job.job_id)


def expire_job_state(job: JobProcessing) -> JobExpired:
    return JobExpired(job_id=job.job_id)
