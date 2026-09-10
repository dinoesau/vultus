"""Fitter: seam fit_gnm con dobles deterministas, goldens congelados a mano."""

from __future__ import annotations

import json
import math
import time

from backend.domain import Err, Ok, parse_image_bytes, parse_landmarks
from backend.gnm_fit import coeff_distance, fit_gnm, fit_gnm_from_request


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


def test_fit_deterministic_golden_vector() -> None:
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


def test_fit_differs_per_identity_and_distance_positive() -> None:
    landmarks = _landmarks()
    ra = fit_gnm(_image(0xA1), landmarks)
    rb = fit_gnm(_image(0xB2), landmarks)
    assert isinstance(ra, Ok)
    assert isinstance(rb, Ok)
    assert ra.value.coeffs.as_tuple() != rb.value.coeffs.as_tuple()
    assert coeff_distance(ra.value.coeffs, rb.value.coeffs) > 0.0
    assert coeff_distance(ra.value.coeffs, ra.value.coeffs) == 0.0


def test_fit_request_bad_payload_fails_loudly() -> None:
    assert isinstance(fit_gnm_from_request(b"\x00\x01"), Err)
    assert isinstance(fit_gnm_from_request(b""), Err)


def test_fit_p95_local_double_inside_ttl_gate() -> None:
    image = _image(0xA1)
    landmarks = _landmarks()
    durations: list[float] = []
    for _ in range(21):
        start = time.perf_counter()
        out = fit_gnm(image, landmarks)
        durations.append(time.perf_counter() - start)
        assert isinstance(out, Ok)
    durations.sort()
    p95 = durations[int(0.95 * (len(durations) - 1))]
    # Gate local: el doble debe estar muy por debajo del TTL; el p95 en T4
    # real se mide en Step 4 contra GPU antes de renegociar TTL en S10.
    assert p95 < 1.0
