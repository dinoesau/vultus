"""Borde landmarks: 478 puntos finitos via parse_landmarks mas as_tuple.

Formas adversas (no-json, no-utf8, 477, no-triple, NaN) dan Err tipado
en el borde; el core solo recibe Landmarks probados.
"""

from __future__ import annotations

import json

from backend.domain import LANDMARKS_LEN, EmptyPayload, Err, Ok, parse_landmarks


def _valid_raw() -> bytes:
    pts = [[0.0, 1.0, 2.0]] * LANDMARKS_LEN
    return json.dumps(pts).encode("utf-8")


def test_valid_478_roundtrips_tuple() -> None:
    parsed = parse_landmarks(_valid_raw())
    assert isinstance(parsed, Ok)
    tup = parsed.value.as_tuple()
    assert isinstance(tup, tuple)
    assert len(tup) == 478
    assert tup[0] == (0.0, 1.0, 2.0)
    assert all(isinstance(p, tuple) and len(p) == 3 for p in tup)
    assert parsed.value.as_bytes() == _valid_raw()


def test_no_json_and_non_utf8_are_err() -> None:
    assert isinstance(parse_landmarks(b"no-json"), Err)
    assert isinstance(parse_landmarks(b"\xff\xfe\x00"), Err)
    assert isinstance(parse_landmarks(b'{"todo":"x"}'), Err)


def test_wrong_length_is_err() -> None:
    assert isinstance(parse_landmarks(json.dumps([[0.0, 1.0, 2.0]] * 477).encode()), Err)
    assert isinstance(parse_landmarks(json.dumps([[0.0, 1.0, 2.0]] * 479).encode()), Err)
    assert isinstance(parse_landmarks(json.dumps([0.0] * 478).encode()), Err)


def test_single_non_triple_point_is_err() -> None:
    pts = [[0.0, 1.0, 2.0]] * 477 + [[0.0, 1.0]]
    assert isinstance(parse_landmarks(json.dumps(pts).encode()), Err)
    pts2 = [[0.0, 1.0, 2.0]] * 477 + ["x"]
    assert isinstance(parse_landmarks(json.dumps(pts2).encode()), Err)


def test_nan_and_inf_are_err() -> None:
    pts = [[0.0, 1.0, 2.0]] * 477 + [[float("nan"), 0.0, 0.0]]
    assert isinstance(parse_landmarks(json.dumps(pts).encode()), Err)
    pts2 = [[0.0, 1.0, 2.0]] * 477 + [[float("inf"), 0.0, 0.0]]
    assert isinstance(parse_landmarks(json.dumps(pts2).encode()), Err)


def test_empty_and_non_bytes_are_err() -> None:
    empty = parse_landmarks(b"")
    assert isinstance(empty, Err)
    assert isinstance(empty.error, EmptyPayload)
    assert isinstance(parse_landmarks("x"), Err)
    assert isinstance(parse_landmarks(None), Err)
    assert isinstance(parse_landmarks(123), Err)
