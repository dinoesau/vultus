"""Textura: seam build_albedo sobre el fit real, goldens congelados a mano."""

from __future__ import annotations

import json

from backend.domain import UV_LEN, Err, Ok, parse_image_bytes, parse_landmarks
from backend.gnm_fit import fit_gnm
from backend.gnm_texture import (
    build_albedo,
    inpaint_occluded,
    occlusion_mask,
    project_texture,
    warp_with_landmarks,
)


def _image(marker: int):
    raw = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes([marker]) * 56
    parsed = parse_image_bytes(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks():
    pts = [[0.1, 0.2, 0.3]] * 478
    raw = json.dumps(pts).encode("utf-8")
    parsed = parse_landmarks(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _fit():
    fit = fit_gnm(_image(0xA1), _landmarks())
    assert isinstance(fit, Ok)
    return fit.value


def test_build_albedo_golden_frozen_head() -> None:
    out = build_albedo(_image(0xA1), _fit(), _landmarks())
    assert isinstance(out, Ok)
    raw = out.value.as_bytes()
    assert len(raw) == UV_LEN
    assert list(raw[:4]) == [161, 161, 161, 161]


def test_albedo_derives_from_real_photo_not_hallucinated() -> None:
    fit = _fit()
    landmarks = _landmarks()
    a = build_albedo(_image(0xA1), fit, landmarks)
    b = build_albedo(_image(0xB2), fit, landmarks)
    assert isinstance(a, Ok)
    assert isinstance(b, Ok)
    assert a.value.as_bytes()[:64] != b.value.as_bytes()[:64]
    assert list(b.value.as_bytes()[:4]) == [178, 178, 178, 178]


def test_inpaint_touches_only_occluded_texels() -> None:
    from backend.domain import parse_complete_uv

    landmarks = _landmarks()
    # Patron alterno no plano: el fallback solido de project_texture es punto
    # fijo del promediado de vecinos, asi que el inpaint se prueba sobre un
    # albedo alterno donde cada texel ocluido cambia si o si (vecinos iguales
    # entre si y distintos del centro) y cada no ocluido queda intacto.
    pattern = bytes(255 if i % 2 else 0 for i in range(UV_LEN))
    parsed = parse_complete_uv(pattern)
    assert isinstance(parsed, Ok)
    inpainted = inpaint_occluded(parsed.value, landmarks)
    assert isinstance(inpainted, Ok)
    before = parsed.value.as_bytes()
    after = inpainted.value.as_bytes()
    mask = occlusion_mask(landmarks)
    assert any(mask)
    assert not all(mask)
    for i in (0, 1, 2, 3):
        if not mask[i]:
            assert before[i] == after[i]
    for i in range(UV_LEN):
        if not mask[i]:
            assert before[i] == after[i]
    assert any(before[i] != after[i] for i in range(UV_LEN) if mask[i])


def test_warp_is_deterministic() -> None:
    fit = _fit()
    landmarks = _landmarks()
    projected = project_texture(_image(0xA1), fit)
    assert isinstance(projected, Ok)
    first = warp_with_landmarks(projected.value, landmarks)
    second = warp_with_landmarks(projected.value, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.as_bytes() == second.value.as_bytes()


def test_warp_is_identity_until_tps_lands() -> None:
    fit = _fit()
    landmarks = _landmarks()
    projected = project_texture(_image(0xA1), fit)
    assert isinstance(projected, Ok)
    warped = warp_with_landmarks(projected.value, landmarks)
    assert isinstance(warped, Ok)
    assert warped.value.as_bytes() == projected.value.as_bytes()


def test_garbage_lengths_rejected_at_parse() -> None:
    assert isinstance(project_texture(_image(0xA1), _fit()), Ok)
    assert isinstance(parse_image_bytes(bytes([1, 2, 3])), Err)
