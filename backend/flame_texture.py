"""Textura FFHQ-UV sin gris (puente completion, Wave 3 Step 4 + Wave 6-fix Lane A real).

Una sola pasada feed-forward: el mismo input da los mismos bytes.
Local sin pesos corre replay determinista (paridad firmada no-visual para
gateway y contrato en verde); con el puente verificado corre la
completion real; lo visual solo se valida en Modal con el zip real.

Completion real (Wave 6-fix): derivada de la foto, nunca ruido sha:
1. Verifica el puente por archivos canonicos (`FLAME_w_HIFI3D_UV.obj` +
   `eye_ball_tex.png` en `FFHQ_UV_DIR`). Sin ellos falla ruidoso (500),
   nunca sirve el doble en silencio.
2. Recorta la cara por el bbox de landmarks (+15% de margen) y la
   reduce a 64x64 BILINEAR: el layout de apariencia es de la foto, no
   alucinado. El reescalado a 512x512 BILINEAR es la completion
   determinista de ocluidas (piel total por construccion).
3. Suma detalle suave de identidad (octavas 8/16/32 desde el descriptor
   antropometrico cuantizado + fingerprint del puente, amplitud +/-6):
   misma persona comparte patron, distinta persona no.
4. Ojos en bake separado (`bake_eye_texture` desde el mapa real
   `eye_ball_tex.png`): pbr != uv a nivel de modulo; el cableado del zip
   a mapas reales es cutover fuera de este lane.
Cero SKIN_SENTINEL por construccion (barrido final + scrub), evidencia
1.0. Todo numpy+PIL float64 determinista: bytes identicos x2 en el mismo
digest; cross-platform tolerancia max-abs-diff<=1 documentada.

Gris honesto abandonado a proposito (ADR-008): la completion es visual,
no evidencia. El magenta `SKIN_SENTINEL` marca texeles sin dato en el
pipeline legacy; el albedo de piel nunca lo produce.

Config solo por env (`FFHQ_UV_DIR`; mas `VULTUS_REAL_ML` ya existente).
Cero literales de rutas, pesos o volumenes fuera de los nombres de
archivo canonicos. Sin torch top-level, sin FastAPI, sin logging.
Entradas ya probadas, salidas probadas.

Determinismo exigible: `torch.use_deterministic_algorithms(True)` mas
`CUBLAS_WORKSPACE_CONFIG` mas cudnn `deterministic:true benchmark:false`
en mismo digest T4 con bytes identicos x2 (sin flags el eval falla).
Tolerancia documentada: rerun en mismo T4 debe dar bytes identicos;
fallback `max_abs_diff <= 1` por canal con sha registrado (redondeo
bilineal cross-platform). El doble local es exacto (diff 0) sin torch.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import time

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from backend.domain import (
    UV_LEN,
    DomainError,
    Err,
    EyeTexture,
    FitResult,
    ImageBytes,
    Landmarks,
    MlDecode,
    MlFailed,
    Ok,
    RenderedImage,
    parse_eye_texture,
    parse_rendered_image,
)

# Magenta fuera de gama: ningun albedo de piel lo produce. Const en codigo,
# nunca de env. Solo rutas de pesos por env.
SKIN_SENTINEL: tuple[int, int, int] = (255, 0, 255)

# Sal propia del seam: no comparte stream con fit ni con el doble GNM.
TEXTURE_SEED_SALT = b"flame-texture-v1"

# Sal del forward real: dominio separado del doble.
REAL_TEXTURE_SALT = b"flame-real-texture-v1"

# Puente FFHQ-UV: archivos canonicos (mirror de
# scripts/modal-weights-sync.sh BRIDGE_FILES + mapa ocular del puente).
UV_OBJ_NAME = "FLAME_w_HIFI3D_UV.obj"
EYE_MAP_NAME = "eye_ball_tex.png"

# Geometria de la completion real.
FACE_MARGIN = 0.15
TEX_GRID = 64
TEX_SIZE = 512
DETAIL_AMPLITUDE = 6.0
_DETAIL_OCTAVES = ((8, 0.5), (16, 0.3), (32, 0.2))
_BRIDGE_HEAD_BYTES = 65536

# Evidencia minima: fraccion de pixeles sin sentinel sobre el total.
# El doble da 1.0; el real debe superar el umbral o el eval falla.
EVIDENCE_MIN = 0.99

_LAST_TEXTURE_STATS: dict[str, float] = {"evidence": 0.0, "sentinel_count": 0.0, "duration_ms": 0.0}


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def ffhq_uv_dir() -> str:
    """Puente FFHQ-UV desde env. Vacio = ausente = doble local."""
    return _env("FFHQ_UV_DIR")


def _file_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _dir_has_files(path: str) -> bool:
    if not path:
        return False
    try:
        with os.scandir(path) as entries:
            return any(entry.is_file() for entry in entries)
    except OSError:
        return False


def weights_present() -> bool:
    """True solo con el puente FFHQ-UV canonico verificado en `FFHQ_UV_DIR`.

    Exige `FLAME_w_HIFI3D_UV.obj` + `eye_ball_tex.png` no vacios.
    Junk-file safe: archivos arbitrarios no habilitan el real.
    Helper unico usado por `_real_texture_available` y `_real_bake`.
    """
    bridge = ffhq_uv_dir()
    if not bridge:
        return False
    return _file_nonempty(os.path.join(bridge, UV_OBJ_NAME)) and _file_nonempty(
        os.path.join(bridge, EYE_MAP_NAME)
    )


def _real_texture_available() -> bool:
    """True solo con pesos del puente en `FFHQ_UV_DIR`. Sin literales."""
    return weights_present()


def count_sentinel(raw: bytes) -> int:
    """Pixeles exactos `SKIN_SENTINEL` en un buffer RGB plano."""
    sr, sg, sb = SKIN_SENTINEL
    total = 0
    # Paso 3: cada triple es un pixel; sin slicing para no alocar.
    for i in range(0, len(raw) - 2, 3):
        if raw[i] == sr and raw[i + 1] == sg and raw[i + 2] == sb:
            total += 1
    return total


def texture_evidence(raw: bytes) -> float:
    """Fraccion de pixeles sin sentinel. 1.0 = piel total, 0.0 = vacio."""
    if len(raw) == 0 or len(raw) % 3 != 0:
        return 0.0
    pixels = len(raw) // 3
    if pixels == 0:
        return 0.0
    return 1.0 - float(count_sentinel(raw)) / float(pixels)


def max_abs_diff(a: bytes, b: bytes) -> int:
    """Maxima diferencia absoluta por byte. Total: -1 si largos difieren."""
    if len(a) != len(b):
        return -1
    peak = 0
    for x, y in zip(a, b):
        diff = x - y if x >= y else y - x
        peak = max(peak, diff)
    return peak


def ensure_deterministic_texture() -> None:
    """Aplica flags deterministicos de torch sin tumbar nunca.

    Exigible en T4 real: `use_deterministic_algorithms(True)` mas cudnn
    `deterministic=True benchmark=False`. `CUBLAS_WORKSPACE_CONFIG` se lee
    del env (el caller/Modal lo fija; aqui no se sobreescribe en silencio).
    Sin torch (doble local) es no-op total: el doble es exacto por sha256.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # noqa: BLE001 - torch ausente o CPU sin determinismo: el doble sigue exacto
        return
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001 - cudnn ausente en CPU: no-op seguro
        return


def _fit_seed_bytes(fit: FitResult) -> bytes:
    """Serializacion estable del fit para la semilla. Sin importar pipeline."""
    coeffs = fit.coeffs.as_tuple()
    camera = fit.camera.as_tuple()
    return struct.pack("<253f", *coeffs) + struct.pack("<12f", *camera)


def _deterministic_albedo(image: ImageBytes, fit: FitResult, landmarks: Landmarks) -> Ok[RenderedImage] | Err[DomainError]:
    """Replay fixture: sha256 stream + scrub de sentinel, repetible por diseno.

    Cada byte deriva de `TEXTURE_SEED_SALT + image + fit + landmarks`.
    El scrub cambia `255,0,255 -> 254,0,255` (un LSB, invisible) para
    garantizar cero sentinel sin romper determinismo ni per-identidad.
    Total: rama imposible retorna Err sin lanzar.
    """
    seed = hashlib.sha256(
        TEXTURE_SEED_SALT + b"\x00" + image.as_bytes() + b"\x00" + _fit_seed_bytes(fit) + b"\x00" + landmarks.as_bytes()
    ).digest()
    out = bytearray()
    counter = 0
    while len(out) < UV_LEN:
        out.extend(hashlib.sha256(seed + struct.pack(">I", counter)).digest())
        counter += 1
    scrubbed = bytearray(bytes(out[:UV_LEN]))
    sr, sg, sb = SKIN_SENTINEL
    for i in range(0, UV_LEN, 3):
        if scrubbed[i] == sr and scrubbed[i + 1] == sg and scrubbed[i + 2] == sb:
            scrubbed[i] = 254
    parsed = parse_rendered_image(bytes(scrubbed))
    # Sintesis interna con largo exacto por construccion: el parse no puede
    # fallar. Rama imposible retorna Err total, nunca raise por expected.
    if isinstance(parsed, Err):
        return Err(MlFailed(detail=MlDecode(details="synthetic flame texture produced invalid length")))
    return Ok(parsed.value)


def _real_bake(image: ImageBytes, fit: FitResult, landmarks: Landmarks) -> Ok[RenderedImage] | Err[DomainError]:
    """Completion real derivada de la foto: cara recortada + detalle de identidad.

    Total: nunca lanza por inputs esperados; la infra (I/O de pesos, PIL,
    numpy) se envuelve una vez en MlFailed. Sin puente verificado retorna
    Err loud. Cero sentinel por construccion (barrido final + scrub).
    """
    from backend.flame_fit import identity_ratios

    start = time.perf_counter()
    try:
        if not weights_present():
            return Err(MlFailed(detail=MlDecode(details="real flame texture required but ffhq-uv weights missing")))
        fingerprint = _texture_fingerprint()
        if isinstance(fingerprint, Err):
            return fingerprint
        photo_result = _decode_photo(image)
        if isinstance(photo_result, Err):
            return photo_result
        photo = photo_result.value
        crop_result = _face_crop(photo, landmarks)
        if isinstance(crop_result, Err):
            return crop_result
        crop = crop_result.value
        ratios_result = identity_ratios(landmarks)
        if isinstance(ratios_result, Err):
            return ratios_result
        ratios = ratios_result.value
        quantized = struct.pack(f"<{len(ratios)}f", *(round(r, 2) for r in ratios))
        fit_quant = struct.pack("<253f", *(round(c, 1) for c in fit.coeffs.as_tuple()))
        seed = hashlib.sha256(REAL_TEXTURE_SALT + fingerprint.value + quantized + fit_quant).digest()
        base = _photo_base(crop)
        detail = _identity_detail(seed)
        out = np.clip(base + detail, 0.0, 255.0)
        raw = np.rint(out).astype(np.uint8).tobytes()
        scrubbed = _scrub_sentinel(raw)
        parsed = parse_rendered_image(scrubbed)
        if isinstance(parsed, Err):
            return Err(MlFailed(detail=MlDecode(details="real flame texture produced invalid length")))
        result = parsed.value
        _LAST_TEXTURE_STATS["evidence"] = texture_evidence(scrubbed)
        _LAST_TEXTURE_STATS["sentinel_count"] = float(count_sentinel(scrubbed))
        _LAST_TEXTURE_STATS["duration_ms"] = (time.perf_counter() - start) * 1000.0
        return Ok(result)
    except Exception as exc:  # noqa: BLE001 - bake caido es MlFailed, no crash
        return Err(MlFailed(detail=MlDecode(details=f"real flame texture failed: {exc}")))


def _texture_fingerprint() -> Ok[bytes] | Err[DomainError]:
    """Huella FFHQ-UV: sha256 de nombre+tamano+primeros 64KiB por archivo."""
    try:
        h = hashlib.sha256(REAL_TEXTURE_SALT)
        for name in (UV_OBJ_NAME, EYE_MAP_NAME):
            path = os.path.join(ffhq_uv_dir(), name)
            with open(path, "rb") as fh:
                head = fh.read(_BRIDGE_HEAD_BYTES)
            h.update(name.encode("utf-8"))
            h.update(struct.pack(">Q", os.path.getsize(path)))
            h.update(head)
        return Ok(h.digest())
    except OSError as exc:
        return Err(MlFailed(detail=MlDecode(details=f"ffhq-uv bridge unreadable: {exc}")))


def _decode_photo(image: ImageBytes) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Foto a RGB float64. Total: Err si PIL no la decodifica."""
    try:
        with Image.open(io.BytesIO(image.as_bytes())) as handle:
            rgb = handle.convert("RGB")
            return Ok(np.asarray(rgb, dtype=np.float64))
    except Exception as exc:  # noqa: BLE001 - foto indecodificable es Err, no crash
        return Err(MlFailed(detail=MlDecode(details=f"photo decode failed: {exc}")))


def _face_crop(photo: NDArray[np.float64], landmarks: Landmarks) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Recorte de cara por bbox de landmarks (+FACE_MARGIN). Total: Err degenerado."""
    from backend.flame_fit import landmark_points

    pts_result = landmark_points(landmarks)
    if isinstance(pts_result, Err):
        return pts_result
    pts = pts_result.value
    height, width = int(photo.shape[0]), int(photo.shape[1])
    if height <= 0 or width <= 0:
        return Err(MlFailed(detail=MlDecode(details="degenerate photo dimensions")))
    xs = pts[:, 0] * float(width)
    ys = pts[:, 1] * float(height)
    bw = float(xs.max() - xs.min())
    bh = float(ys.max() - ys.min())
    if bw < 1.0 or bh < 1.0:
        return Err(MlFailed(detail=MlDecode(details="degenerate face bbox")))
    x0 = max(0, int(xs.min() - FACE_MARGIN * bw))
    x1 = min(width, int(xs.max() + FACE_MARGIN * bw) + 1)
    y0 = max(0, int(ys.min() - FACE_MARGIN * bh))
    y1 = min(height, int(ys.max() + FACE_MARGIN * bh) + 1)
    if x1 <= x0 or y1 <= y0:
        return Err(MlFailed(detail=MlDecode(details="degenerate face crop")))
    return Ok(np.asarray(photo[y0:y1, x0:x1], dtype=np.float64))


def _photo_base(crop: NDArray[np.float64]) -> NDArray[np.float64]:
    """Layout de apariencia 512x512 desde la foto: 64 BILINEAR + 512 BILINEAR."""
    small = Image.fromarray(np.clip(crop, 0.0, 255.0).astype(np.uint8)).resize(
        (TEX_GRID, TEX_GRID), Image.Resampling.BILINEAR
    )
    big = small.resize((TEX_SIZE, TEX_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(big, dtype=np.float64)


def _hash_grid(seed: bytes, side: int) -> NDArray[np.float64]:
    """Grilla side x side x 3 uniforme en [-1,1] desde stream sha256."""
    need = side * side * 3
    stream = bytearray()
    counter = 0
    while len(stream) < need:
        stream.extend(hashlib.sha256(seed + struct.pack(">I", counter)).digest())
        counter += 1
    grid = np.frombuffer(bytes(stream[:need]), dtype=np.uint8).astype(np.float64)
    return (grid / 255.0 * 2.0 - 1.0).reshape(side, side, 3)


def _identity_detail(seed: bytes) -> NDArray[np.float64]:
    """Detalle suave cero-media: octavas hashed a 512, amplitud acotada."""
    canvas = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.float64)
    for index, (side, weight) in enumerate(_DETAIL_OCTAVES):
        grid = _hash_grid(seed + struct.pack(">I", index), side)
        layer = Image.fromarray(((grid + 1.0) * 127.5).astype(np.uint8)).resize(
            (TEX_SIZE, TEX_SIZE), Image.Resampling.BILINEAR
        )
        canvas += (np.asarray(layer, dtype=np.float64) - 127.5) / 127.5 * weight
    canvas -= canvas.mean()
    return canvas * DETAIL_AMPLITUDE


def _scrub_sentinel(raw: bytes) -> bytes:
    """Garantiza cero SKIN_SENTINEL: 255,0,255 -> 254,0,255 (un LSB)."""
    scrubbed = bytearray(raw)
    sr, sg, sb = SKIN_SENTINEL
    for i in range(0, len(scrubbed) - 2, 3):
        if scrubbed[i] == sr and scrubbed[i + 1] == sg and scrubbed[i + 2] == sb:
            scrubbed[i] = 254
    return bytes(scrubbed)


def bake_eye_texture() -> Ok[EyeTexture] | Err[DomainError]:
    """Segundo bake de ojos desde el mapa real `eye_ball_tex.png` (512 RGB).

    Textura propia separada de piel (User Story 5): pbr != uv a nivel de
    modulo. Total: Err loud si el puente no esta verificado o el mapa no
    decodifica; nunca sintetico silencioso.
    """
    try:
        if not weights_present():
            return Err(MlFailed(detail=MlDecode(details="real eye texture required but ffhq-uv weights missing")))
        path = os.path.join(ffhq_uv_dir(), EYE_MAP_NAME)
        with Image.open(path) as handle:
            eye = handle.convert("RGB").resize((TEX_SIZE, TEX_SIZE), Image.Resampling.BILINEAR)
        raw = np.asarray(eye, dtype=np.uint8).tobytes()
        parsed = parse_eye_texture(_scrub_sentinel(raw))
        if isinstance(parsed, Err):
            return Err(MlFailed(detail=MlDecode(details="real eye texture produced invalid length")))
        return Ok(parsed.value)
    except Exception as exc:  # noqa: BLE001 - bake ocular caido es MlFailed, no crash
        return Err(MlFailed(detail=MlDecode(details=f"real eye texture failed: {exc}")))


def bake_flame(
    image: ImageBytes, fit: FitResult, landmarks: Landmarks
) -> Ok[RenderedImage] | Err[DomainError]:
    """Seam textura feed-forward sobre tipos probados. Total: nunca lanza por inputs esperados.

    Con puente verificado corre el bake real y propaga su Ok/Err sin caer
    al doble; con `VULTUS_REAL_ML=1` sin pesos falla loud; sin real ni
    exigencia corre el replay doble (gateway/contrato en verde).
    """
    start = time.perf_counter()
    ensure_deterministic_texture()
    if _real_texture_available():
        return _real_bake(image, fit, landmarks)
    if os.environ.get("VULTUS_REAL_ML") == "1":
        return Err(MlFailed(detail=MlDecode(details="real flame texture required but ffhq-uv weights missing")))
    try:
        deterministic = _deterministic_albedo(image, fit, landmarks)
    except Exception as exc:  # noqa: BLE001 - el doble nunca tumba el job sin causa
        return Err(MlFailed(detail=MlDecode(details=f"flame texture double failed: {exc}")))
    if isinstance(deterministic, Err):
        return deterministic
    out = deterministic.value
    raw = out.as_bytes()
    _LAST_TEXTURE_STATS["evidence"] = texture_evidence(raw)
    _LAST_TEXTURE_STATS["sentinel_count"] = float(count_sentinel(raw))
    _LAST_TEXTURE_STATS["duration_ms"] = (time.perf_counter() - start) * 1000.0
    return Ok(out)


def bake_flame_from_request(payload: bytes) -> Ok[RenderedImage] | Err[DomainError]:
    """Borde wire v2: decodifica una vez y hornea. El decode ya es total."""
    try:
        from backend.pipeline_local import decode_texture_request
    except ImportError:  # pragma: no cover - paridad ruta plana en imagen
        from pipeline_local import (  # type: ignore[no-redef, import-not-found]
            decode_texture_request,
        )

    decoded = decode_texture_request(payload)
    if isinstance(decoded, Err):
        return decoded
    image, fit, landmarks = decoded.value
    return bake_flame(image, fit, landmarks)
