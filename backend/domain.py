"""Dominio Python: tipos probados una vez en el borde, core sin revalidar.

Fuente de verdad del contrato HTTP en TypeScript (`edge/contract.ts`);
este modulo es dueno local de la invariante. Sin FastAPI ni logging aqui.
Historico: antes Rust (borrado en plan-remove-rust-stack).
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeAlias, TypeVar
from urllib.parse import unquote

# La imagen Modal pina Python 3.10: nada de sintaxis 3.11+ aqui
# (`type X =` PEP 695 ni `typing.Never`). Este modulo viaja a Modal
# via `add_local_python_source` y debe importar en 3.10.
if sys.version_info >= (3, 11):
    from typing import Never
else:
    from typing import NoReturn as Never

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")
T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", covariant=True)

# --- Constantes canonicas (no inventar valores nuevos) ---

MAX_IMAGE_BYTES = 8 * 1024 * 1024

R2_MAX_LEN = 1024

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

LANDMARKS_LEN = 478
UV_WIDTH = 512
UV_HEIGHT = 512
UV_CHANNELS = 3
UV_LEN = UV_WIDTH * UV_HEIGHT * UV_CHANNELS

TEMPLATE_VERTS = 17821
TEMPLATE_TRIS = 35324

GLB_MAGIC = b"glTF"
GLB_MIN_LEN = 12
GLB_VERSION = 2

ZIP_UV_A = "uv_a.png"
ZIP_UV_B = "uv_b.png"
ZIP_HEATMAP = "heatmap.png"
ZIP_MESH_A = "mesh_a.glb"
ZIP_MESH_B = "mesh_b.glb"
ZIP_PBR_A = "pbr_a.png"
ZIP_PBR_B = "pbr_b.png"

ZIP_NAMES = (ZIP_UV_A, ZIP_UV_B, ZIP_HEATMAP, ZIP_MESH_A, ZIP_MESH_B)

ZIP_FULL = (*ZIP_NAMES, ZIP_PBR_A, ZIP_PBR_B)


@dataclass(frozen=True, slots=True)
class ZipBundle:
    """Bundle 7 archivos en orden canonico ZIP_FULL. Solo via constructores shell."""

    uv_a_png: bytes
    uv_b_png: bytes
    heatmap_png: bytes
    mesh_a_glb: bytes
    mesh_b_glb: bytes
    pbr_a: bytes
    pbr_b: bytes

# GNM fit directo: 253 coeficientes finitos + camara 3x4 (12 finitos).
GNM_COEFFS_LEN = 253
CAMERA_PARAMS_LEN = 12

# Islas UV GNM publicas (1-5). Mas alla es investigacion fuera de alcance.
UV_ISLAND_MIN = 1
UV_ISLAND_MAX = 5

PROGRESS_DONE = 1.0

# Hitos GNM (espejo de edge/contract.ts): fit, texture, assemble.
PROGRESS_FIT = 0.40
PROGRESS_TEXTURE = 0.75
PROGRESS_ASSEMBLE = 0.95

LANDMARKS_TIMEOUT_SECS = 5
FIT_TIMEOUT_SECS = 10
TEXTURE_TIMEOUT_SECS = 30
TOTAL_TIMEOUT_SECS = 60


class Stage(str, Enum):
    QUEUED = "queued"
    FIT = "fit"
    TEXTURE = "texture"
    ASSEMBLE = "assemble"
    DONE = "done"

    def as_str(self) -> str:
        return self.value


# --- Result y combinadores (railway) ---


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E


Result: TypeAlias = Ok[T] | Err[E]


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


ImageError: TypeAlias = SizeOutOfRange | UnsupportedFormat


@dataclass(frozen=True, slots=True)
class BadScheme:
    detail: str = "base_url must start with http:// or https://"


@dataclass(frozen=True, slots=True)
class EmptyUrl:
    detail: str = "base_url is empty"


BaseUrlError: TypeAlias = BadScheme | EmptyUrl


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


MlError: TypeAlias = MlTransport | MlBadStatus | MlDecode | MlEmpty


@dataclass(frozen=True, slots=True)
class InvalidImage:
    detail: ImageError


@dataclass(frozen=True, slots=True)
class InvalidJobId:
    detail: str = "invalid job_id"


@dataclass(frozen=True, slots=True)
class InvalidR2Key:
    detail: str = "invalid r2 key"


@dataclass(frozen=True, slots=True)
class InvalidProgress:
    detail: str = "invalid progress"


@dataclass(frozen=True, slots=True)
class InvalidBaseUrl:
    detail: BaseUrlError


@dataclass(frozen=True, slots=True)
class InvalidCoeffs:
    detail: str = "invalid gnm coeffs"


@dataclass(frozen=True, slots=True)
class InvalidCamera:
    detail: str = "invalid camera params"


@dataclass(frozen=True, slots=True)
class FitFailed:
    detail: MlError


@dataclass(frozen=True, slots=True)
class EmptyPayload:
    detail: str = "empty payload"


@dataclass(frozen=True, slots=True)
class MlFailed:
    detail: MlError


@dataclass(frozen=True, slots=True)
class NotFound:
    job_id: str


@dataclass(frozen=True, slots=True)
class Invariant:
    detail: str


DomainError: TypeAlias = (
    InvalidImage
    | InvalidJobId
    | InvalidR2Key
    | InvalidProgress
    | InvalidBaseUrl
    | InvalidCoeffs
    | InvalidCamera
    | EmptyPayload
    | MlFailed
    | FitFailed
    | NotFound
    | Invariant
)


def domain_to_status(error: DomainError) -> int:
    if isinstance(
        error,
        (
            InvalidImage,
            InvalidJobId,
            InvalidR2Key,
            InvalidProgress,
            InvalidBaseUrl,
            EmptyPayload,
            InvalidCoeffs,
            InvalidCamera,
        ),
    ):
        return 400
    if isinstance(error, NotFound):
        return 404
    if isinstance(error, (MlFailed, FitFailed, Invariant)):
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
    if isinstance(error, InvalidR2Key):
        return "invalid r2 key"
    if isinstance(error, InvalidProgress):
        return "invalid progress"
    if isinstance(error, InvalidBaseUrl):
        return f"invalid base_url: {error.detail.detail}"
    if isinstance(error, InvalidCoeffs):
        return "invalid gnm coeffs"
    if isinstance(error, InvalidCamera):
        return "invalid camera params"
    if isinstance(error, EmptyPayload):
        return "empty payload"
    if isinstance(error, NotFound):
        return f"not found: {error.job_id}"
    if isinstance(error, (MlFailed, FitFailed, Invariant)):
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


def _has_controls(s: str) -> bool:
    for ch in s:
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            return True
    return False


@dataclass(frozen=True, slots=True)
class R2Key:
    """Solo via parse_r2_key."""

    _value: str

    def as_str(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value


def parse_r2_key(raw: object) -> Result[R2Key, DomainError]:
    if not isinstance(raw, str):
        return Err(InvalidR2Key())
    trimmed = raw.strip()
    if not trimmed:
        return Err(InvalidR2Key())
    if len(trimmed) > R2_MAX_LEN:
        return Err(InvalidR2Key())
    if trimmed.startswith("/"):
        return Err(InvalidR2Key())
    if "\\" in trimmed:
        return Err(InvalidR2Key())
    if _has_controls(trimmed):
        return Err(InvalidR2Key())
    decoded = trimmed
    for _ in range(2):
        nxt = unquote(decoded)
        if nxt == decoded:
            break
        decoded = nxt
        if len(decoded) > R2_MAX_LEN:
            return Err(InvalidR2Key())
    if len(decoded) > R2_MAX_LEN:
        return Err(InvalidR2Key())
    if "%2e" in decoded.lower():
        return Err(InvalidR2Key())
    if "\\" in decoded:
        return Err(InvalidR2Key())
    if decoded.startswith("/"):
        return Err(InvalidR2Key())
    if _has_controls(decoded):
        return Err(InvalidR2Key())
    for seg in decoded.split("/"):
        if seg in ("..", "."):
            return Err(InvalidR2Key())
    return Ok(R2Key(_value=trimmed))


@dataclass(frozen=True, slots=True)
class QueueJob:
    """Producto borde cola: job + punteros R2. Solo via parse_queue_envelope.

    Wave 5: sin campo `deprecated`; legacy `a/b` es Err, solo canonico.
    """

    job_id: JobId
    r2_a: R2Key
    r2_b: R2Key


def parse_queue_envelope(raw: object) -> Result[QueueJob, DomainError]:
    """Borde unico de cola. Solo canonico `job_id` + `r2_keys.image_a/image_b`.

    Wave 5 flip: legacy `a/b` es Err InvalidR2Key, sin path deprecated.
    Extras policy: legacy a/b -> Err; other unknown extras ignored for
    forward-compat (no se rechazan claves desconocidas fuera de a/b).
    `body`/`message` solo con fallback None-estricto: si `body` es None se
    usa `message`; falsy no-None nunca cae al otro campo.
    R2Key guarda el trim exacto; lo decodificado solo valida, no se almacena.
    """
    body: object = raw
    if isinstance(raw, dict) and ("body" in raw or "message" in raw):
        cand: object = raw.get("body")
        if cand is None:
            cand = raw.get("message")
        if isinstance(cand, str):
            try:
                cand = json.loads(cand)
            except json.JSONDecodeError:
                return Err(InvalidJobId(detail="queue body not json"))
        if not isinstance(cand, dict):
            return Err(InvalidJobId(detail="queue body not object"))
        body = cand
    if not isinstance(body, dict):
        return Err(InvalidJobId(detail="queue body not object"))
    job_raw: object = body.get("job_id")
    if job_raw is None:
        job_raw = body.get("jobId")
    job_result = parse_job_id(job_raw)
    if isinstance(job_result, Err):
        return job_result
    keys_raw: object = body.get("r2_keys")
    if keys_raw is None:
        keys_raw = body.get("r2Keys")
    if not isinstance(keys_raw, dict):
        return Err(InvalidR2Key())
    a_canon: object = keys_raw.get("image_a")
    b_canon: object = keys_raw.get("image_b")
    if "a" in keys_raw or "b" in keys_raw:
        return Err(InvalidR2Key())
    if not isinstance(a_canon, str) or not isinstance(b_canon, str):
        return Err(InvalidR2Key())
    a_parsed = parse_r2_key(a_canon)
    if isinstance(a_parsed, Err):
        return a_parsed
    b_parsed = parse_r2_key(b_canon)
    if isinstance(b_parsed, Err):
        return b_parsed
    return Ok(QueueJob(job_id=job_result.value, r2_a=a_parsed.value, r2_b=b_parsed.value))


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

    def as_tuple(self) -> tuple[tuple[float, float, float], ...]:
        pts: list[list[float]] = json.loads(self._value.decode("utf-8"))
        return tuple((float(p[0]), float(p[1]), float(p[2])) for p in pts)


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


def _heatmap_from_diff(a: CompleteUv, b: CompleteUv) -> Heatmap:
    raw = bytes(x - y if x >= y else y - x for x, y in zip(a.as_bytes(), b.as_bytes()))
    return Heatmap(_value=raw)


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


# --- GNM fit directo: tipos probados en el borde del fitter ---


def _parse_finite_floats(raw: object, expect: int) -> Result[tuple[float, ...], None]:
    if not isinstance(raw, (list, tuple)):
        return Err(None)
    if len(raw) != expect:
        return Err(None)
    out: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return Err(None)
        number = float(value)
        if not math.isfinite(number):
            return Err(None)
        out.append(number)
    return Ok(tuple(out))


@dataclass(frozen=True, slots=True)
class GnmCoeffs:
    """Solo via parse_gnm_coeffs. 253 floats finitos."""

    _value: tuple[float, ...]

    def as_tuple(self) -> tuple[float, ...]:
        return self._value

    def __len__(self) -> int:
        return len(self._value)


def parse_gnm_coeffs(raw: object) -> Result[GnmCoeffs, DomainError]:
    result = _parse_finite_floats(raw, GNM_COEFFS_LEN)
    if isinstance(result, Err):
        return Err(InvalidCoeffs())
    return Ok(GnmCoeffs(_value=result.value))


@dataclass(frozen=True, slots=True)
class CameraParams:
    """Solo via parse_camera_params. Matriz 3x4 aplanada, 12 finitos."""

    _value: tuple[float, ...]

    def as_tuple(self) -> tuple[float, ...]:
        return self._value


def parse_camera_params(raw: object) -> Result[CameraParams, DomainError]:
    result = _parse_finite_floats(raw, CAMERA_PARAMS_LEN)
    if isinstance(result, Err):
        return Err(InvalidCamera())
    return Ok(CameraParams(_value=result.value))


@dataclass(frozen=True, slots=True)
class UvRegion:
    """Solo via parse_uv_region. Isla GNM 1-5."""

    _value: int

    def island(self) -> int:
        return self._value


def parse_uv_region(raw: object) -> Result[UvRegion, DomainError]:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return Err(InvalidProgress())
    if raw < UV_ISLAND_MIN or raw > UV_ISLAND_MAX:
        return Err(InvalidProgress())
    return Ok(UvRegion(_value=raw))


@dataclass(frozen=True, slots=True)
class FitResult:
    """Seam fit->texture: coefs + camara ya probados."""

    coeffs: GnmCoeffs
    camera: CameraParams


@dataclass(frozen=True, slots=True)
class FittedMesh:
    """Geometria personalizada derivada de un FitResult."""

    fit: FitResult


def encode_fit_request(image: ImageBytes, landmarks: Landmarks) -> bytes:
    lm = landmarks.as_bytes()
    img = image.as_bytes()
    return len(lm).to_bytes(4, "big") + lm + img


def decode_fit_request(raw: object) -> Result[tuple[Landmarks, ImageBytes], DomainError]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return Err(FitFailed(detail=MlDecode(details="fit payload <4 bytes")))
    data = bytes(raw)
    if len(data) < 4:
        return Err(FitFailed(detail=MlDecode(details="fit payload <4 bytes")))
    size = int.from_bytes(data[0:4], "big")
    if len(data) < 4 + size:
        return Err(FitFailed(detail=MlDecode(details="fit payload truncated")))
    lm_raw = data[4 : 4 + size]
    img_raw = data[4 + size :]
    lm_result = parse_landmarks(lm_raw)
    if isinstance(lm_result, Err):
        return lm_result
    img_result = parse_image_bytes(img_raw)
    if isinstance(img_result, Err):
        return Err(img_result.error)
    return Ok((lm_result.value, img_result.value))
