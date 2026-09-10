"""Fitter: seam fit_gnm con fit real ridge sobre la cabeza GNM."""

from __future__ import annotations

import json
import math
import time

import pytest

from backend.domain import (
    FIT_TIMEOUT_SECS,
    Err,
    ImageBytes,
    Landmarks,
    Ok,
    parse_image_bytes,
    parse_landmarks,
)
from backend.gnm_fit import (
    _LAST_FIT_STATS,
    _real_fit_available,
    coeff_distance,
    fit_gnm,
    fit_gnm_from_request,
)
from backend.gnm_head import MP_68_MAP, eval_landmarks68


def _image(marker: int) -> ImageBytes:
    raw = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes([marker]) * 56
    parsed = parse_image_bytes(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks() -> Landmarks:
    pts = [[0.1, 0.2, 0.3]] * 478
    raw = json.dumps(pts).encode("utf-8")
    parsed = parse_landmarks(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks_from_identity(coef_idx: int, coef_val: float, scale: float, tx: float, ty: float) -> Landmarks:
    """478 sinteticos: proyecta coefs conocidos a los 68 del mapa, resto relleno."""
    coefs = [0.0] * 253
    coefs[coef_idx] = coef_val
    projected = eval_landmarks68(tuple(coefs))
    pts: list[list[float]] = [[0.5, 0.5, 0.0] for _ in range(478)]
    for k, mp_idx in enumerate(MP_68_MAP):
        pts[mp_idx] = [scale * projected[k][0] + tx, scale * projected[k][1] + ty, 0.0]
    parsed = parse_landmarks(json.dumps(pts).encode("utf-8"))
    assert isinstance(parsed, Ok)
    return parsed.value


def test_fit_deterministic_real() -> None:
    image = _image(0xA1)
    landmarks = _landmarks()
    first = fit_gnm(image, landmarks)
    second = fit_gnm(image, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.coeffs.as_tuple() == second.value.coeffs.as_tuple()
    assert first.value.camera.as_tuple() == second.value.camera.as_tuple()
    assert len(first.value.coeffs.as_tuple()) == 253
    assert len(first.value.camera.as_tuple()) == 12
    assert all(math.isfinite(v) for v in first.value.coeffs.as_tuple())
    assert all(math.isfinite(v) for v in first.value.camera.as_tuple())


def test_fit_differs_per_identity_and_distance_positive() -> None:
    image = _image(0xA1)
    landmarks_a = _landmarks_from_identity(0, 2.0, 3.0, 0.5, 0.1)
    landmarks_b = _landmarks_from_identity(0, -2.0, 3.0, 0.5, 0.1)
    ra = fit_gnm(image, landmarks_a)
    rb = fit_gnm(image, landmarks_b)
    assert isinstance(ra, Ok)
    assert isinstance(rb, Ok)
    assert ra.value.coeffs.as_tuple() != rb.value.coeffs.as_tuple()
    assert coeff_distance(ra.value.coeffs, rb.value.coeffs) > 0.0
    assert coeff_distance(ra.value.coeffs, ra.value.coeffs) == 0.0


def test_fit_request_bad_payload_fails_loudly() -> None:
    assert isinstance(fit_gnm_from_request(b"\x00\x01"), Err)
    assert isinstance(fit_gnm_from_request(b""), Err)


def test_fit_p95_inside_fit_timeout() -> None:
    image = _image(0xA1)
    landmarks = _landmarks_from_identity(0, 2.0, 3.0, 0.5, 0.1)
    durations: list[float] = []
    for _ in range(21):
        start = time.perf_counter()
        out = fit_gnm(image, landmarks)
        durations.append(time.perf_counter() - start)
        assert isinstance(out, Ok)
    durations.sort()
    p95 = durations[int(0.95 * (len(durations) - 1))]
    # Gate: el fit real debe caber en FIT_TIMEOUT_SECS; en local corre en ms.
    assert p95 < FIT_TIMEOUT_SECS


def test_fit_uses_real_head_basis() -> None:
    if not _real_fit_available():
        pytest.skip("sin pesos GNM: no hay base real que verificar")
    from backend.gnm_head import eval_mesh, load_gnm_head

    head = load_gnm_head()
    mesh = eval_mesh(tuple([0.0] * 253))
    template = head.template_positions.tolist()
    assert len(mesh) == len(template)
    for row_m, row_t in zip(mesh, template):
        for vm, vt in zip(row_m, row_t):
            assert abs(float(vm) - float(vt)) < 1e-4
    zeros = eval_landmarks68(tuple([0.0] * 253))
    assert len(zeros) == 68
    landmarks = _landmarks_from_identity(0, 2.0, 3.0, 0.5, 0.1)
    out = fit_gnm(_image(0xA1), landmarks)
    assert isinstance(out, Ok)
    coefs = out.value.coeffs.as_tuple()
    assert len(coefs) == 253
    assert all(math.isfinite(v) for v in coefs)
    assert any(abs(v) > 1e-6 for v in coefs)
    grid = fit_gnm(_image(0xA1), _landmarks())
    assert isinstance(grid, Ok)
    assert coeff_distance(out.value.coeffs, grid.value.coeffs) > 0.0
    assert _LAST_FIT_STATS["iterations"] == 3.0
    assert math.isfinite(_LAST_FIT_STATS["loss"]) and _LAST_FIT_STATS["loss"] >= 0.0
    assert _LAST_FIT_STATS["duration_ms"] > 0.0
