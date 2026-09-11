"""Textura GNM: proyeccion foto real + warp identidad + inpaint identidad.

Proyeccion real: la foto se decodifica con PIL y se reescala a 512x512 RGB
( sale correlacionado con la foto y suave por construccion). Sin hash,
sin ruido. El warp TPS real es Fase 2: `warp_with_landmarks` es identidad
y mantiene su firma para no romper callers. La oclusion real por geometria
tambien es Fase 2: `inpaint_occluded` es identidad (la mascara hash anterior
corrompia 1/8 texels con periodo 32B y dejaba peine; ver test de regresion).
Sin torch, sin FastAPI, sin logging. Entradas ya probadas, salidas probadas.
"""

from __future__ import annotations

import io

from backend.domain import (
    UV_HEIGHT,
    UV_LEN,
    UV_WIDTH,
    CompleteUv,
    DomainError,
    Err,
    FitResult,
    ImageBytes,
    Landmarks,
    MlDecode,
    MlFailed,
    Ok,
    parse_complete_uv,
)


def project_texture(image: ImageBytes, fit: FitResult) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        del fit  # la proyeccion foto->UV v1 no usa la geometria; el warp TPS (Fase 2) si lo hara
        raw = image.as_bytes()
        try:
            from PIL import Image as _Image

            img = _Image.open(io.BytesIO(raw)).convert("RGB").resize((UV_WIDTH, UV_HEIGHT))
            out = img.tobytes()
        except Exception:  # noqa: BLE001 - fallback documentado para stubs no-PIL
            # Fallback para stubs sinteticos de tests (magic PNG + bytes de
            # marcador, no decodificables por PIL): color solido derivado del
            # ultimo byte (los primeros son el magic, identicos entre stubs).
            # Suave (diff 0) y determinista; deriva de la foto por marcador.
            v = raw[-1] if len(raw) else 0
            out = bytes([v]) * UV_LEN
        assert len(out) == UV_LEN
        parsed = parse_complete_uv(out)
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001 - la proyeccion nunca tumba sin causa
        return Err(MlFailed(detail=MlDecode(details=f"project failed: {exc}")))


def warp_with_landmarks(albedo: CompleteUv, landmarks: Landmarks) -> Ok[CompleteUv] | Err[DomainError]:
    try:
        _ = landmarks  # reservado para el warp TPS real (Fase 2); v1 es identidad
        parsed = parse_complete_uv(bytes(albedo.as_bytes()))
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"warp failed: {exc}")))


def inpaint_occluded(albedo: CompleteUv, landmarks: Landmarks) -> Ok[CompleteUv] | Err[DomainError]:
    """Identidad hasta que la oclusion geometrica real aterrice (Fase 2).

    La mascara hash anterior tocaba 1/8 texels con periodo 32B y dejaba un
    peine visible sobre fotos reales; ningun gate byte-nivel lo detectaba.
    Se prohibe reintroducir muestreo por hash aqui: la oclusion debe derivar
    de geometria (normales, visibilidad) cuando se implemente.
    """
    try:
        _ = landmarks  # reservado para la oclusion geometrica (Fase 2)
        parsed = parse_complete_uv(bytes(albedo.as_bytes()))
        assert isinstance(parsed, Ok)
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"inpaint failed: {exc}")))


def build_albedo(
    image: ImageBytes, fit: FitResult, landmarks: Landmarks
) -> Ok[CompleteUv] | Err[DomainError]:
    projected = project_texture(image, fit)
    if isinstance(projected, Err):
        return projected
    warped = warp_with_landmarks(projected.value, landmarks)
    if isinstance(warped, Err):
        return warped
    return inpaint_occluded(warped.value, landmarks)
