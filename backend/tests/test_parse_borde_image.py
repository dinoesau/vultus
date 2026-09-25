"""Borde imagen: bytes crudos a ImageBytes una vez, sin R2 real.

Literales jpeg/png minimos contra vacio, gigante, magia rota y tipos
no-bytes. Mapeo a 400 via domain_to_status.
"""

from __future__ import annotations

from backend.domain import (
    Err,
    InvalidImage,
    Ok,
    domain_to_status,
    parse_image_bytes,
)

JPEG_MIN = bytes([0xFF, 0xD8, 0xFF, 0x00]) + bytes(60)
PNG_MIN = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes(56)


def test_empty_oversized_and_broken_magic_are_err() -> None:
    assert isinstance(parse_image_bytes(b""), Err)
    assert isinstance(parse_image_bytes(bytes([0xFF, 0xD8, 0xFF]) + bytes(8 * 1024 * 1024)), Err)
    assert isinstance(parse_image_bytes(bytes([1, 2, 3, 4])), Err)
    assert isinstance(parse_image_bytes(b"no-json"), Err)


def test_jpeg_and_png_min_accept_and_roundtrip() -> None:
    jpeg = parse_image_bytes(JPEG_MIN)
    assert isinstance(jpeg, Ok)
    assert jpeg.value.as_bytes() == JPEG_MIN
    assert len(jpeg.value) == len(JPEG_MIN)
    png = parse_image_bytes(PNG_MIN)
    assert isinstance(png, Ok)
    assert png.value.as_bytes() == PNG_MIN


def test_bytearray_and_memoryview_accepted() -> None:
    assert isinstance(parse_image_bytes(bytearray(JPEG_MIN)), Ok)
    assert isinstance(parse_image_bytes(memoryview(JPEG_MIN)), Ok)


def test_non_bytes_types_are_err() -> None:
    assert isinstance(parse_image_bytes("x"), Err)
    assert isinstance(parse_image_bytes(None), Err)
    assert isinstance(parse_image_bytes(123), Err)
    assert isinstance(parse_image_bytes(["x"]), Err)


def test_invalid_image_maps_to_400() -> None:
    bad = parse_image_bytes(b"")
    assert isinstance(bad, Err)
    assert domain_to_status(bad.error) == 400
    assert isinstance(bad.error, InvalidImage)
