"""Slice 1: previews frontales 512 deterministas via `build_preview_png`."""

from __future__ import annotations

import io
import json

import numpy as np

from backend.domain import (
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    UV_LEN,
    Err,
    Ok,
    parse_camera_params,
    parse_gnm_coeffs,
    parse_landmarks,
    parse_preview_png,
    parse_rendered_image,
)


def _fit(marker: float):  # type: ignore[no-untyped-def]
    from backend.domain import FitResult

    lm = parse_landmarks(json.dumps([[0.0, 0.0, 0.0]] * 478).encode("utf-8"))
    assert isinstance(lm, Ok)
    coeffs = parse_gnm_coeffs([marker] * 253)
    camera = parse_camera_params([0.0] * 12)
    assert isinstance(coeffs, Ok)
    assert isinstance(camera, Ok)
    return FitResult(coeffs=coeffs.value, camera=camera.value)


def _albedo(marker: int):  # type: ignore[no-untyped-def]
    raw = bytes([marker & 0xFF]) * UV_LEN
    parsed = parse_rendered_image(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def test_preview_png_valid_and_deterministic_x2() -> None:
    from backend.render_preview import build_preview_png

    first = build_preview_png(_fit(0.1), _albedo(0xA1))
    second = build_preview_png(_fit(0.1), _albedo(0xA1))
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value == second.value
    assert first.value[:8] == b"\x89PNG\r\n\x1a\n"
    parsed = parse_preview_png(first.value)
    assert isinstance(parsed, Ok)
    assert len(parsed.value.as_bytes()) == len(first.value)
    other = build_preview_png(_fit(0.1), _albedo(0xB2))
    assert isinstance(other, Ok)
    assert other.value != first.value


def test_preview_is_512x512() -> None:
    from PIL import Image

    from backend.render_preview import build_preview_png

    out = build_preview_png(_fit(0.0), _albedo(0x11))
    assert isinstance(out, Ok)
    img = Image.open(io.BytesIO(out.value))
    assert img.size == (PREVIEW_WIDTH, PREVIEW_HEIGHT) == (512, 512)
    assert img.mode == "RGB"


def test_preview_chin_down_relative() -> None:
    """Chin abajo relativo: atlas con arriba blanco/abajo negro pinta arriba claro."""
    from PIL import Image

    from backend.render_preview import build_preview_png

    rows = np.zeros((512, 512, 3), dtype=np.uint8)
    rows[:256] = 230
    rows[256:] = 20
    raw = rows.tobytes()
    assert len(raw) == UV_LEN
    parsed = parse_rendered_image(raw)
    assert isinstance(parsed, Ok)
    out = build_preview_png(_fit(0.0), parsed.value)
    assert isinstance(out, Ok)
    img = np.asarray(Image.open(io.BytesIO(out.value)).convert("RGB"), dtype=np.float64)
    assert img.shape == (512, 512, 3)
    top = img[:170].reshape(-1, 3).mean(axis=0)
    bottom = img[342:].reshape(-1, 3).mean(axis=0)
    assert float(top.mean()) > float(bottom.mean())


def test_preview_rejects_non_png_and_wrong_size() -> None:
    assert isinstance(parse_preview_png(b"not-a-png"), Err)
    assert isinstance(parse_preview_png(b""), Err)
    assert isinstance(parse_preview_png(123), Err)
