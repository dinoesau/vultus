"""Dominio: comportamiento por interface publica, goldens literales a mano."""

from __future__ import annotations

import json
import uuid

from backend.domain import (
    LANDMARKS_LEN,
    MAX_IMAGE_BYTES,
    UV_LEN,
    Err,
    Ok,
    Stage,
    default_ttl,
    new_job_id,
    parse_base_url,
    parse_complete_uv,
    parse_flaw_uv,
    parse_gnm_mesh,
    parse_heatmap,
    parse_image_bytes,
    parse_job_id,
    parse_landmarks,
    parse_progress,
    parse_r2_key,
    parse_stage,
    parse_ttl_secs,
    zero_progress,
)


def _jpeg_min() -> bytes:
    return bytes([0xFF, 0xD8, 0xFF, 0x00]) + bytes(60)


def _png_min() -> bytes:
    return bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes(56)


def test_rejects_empty_huge_and_broken_magic_accepts_jpeg_and_png() -> None:
    assert isinstance(parse_image_bytes(b""), Err)
    assert isinstance(parse_image_bytes(bytes([0xFF, 0xD8, 0xFF]) + bytes(MAX_IMAGE_BYTES)), Err)
    assert isinstance(parse_image_bytes(bytes([1, 2, 3, 4])), Err)
    jpeg = parse_image_bytes(_jpeg_min())
    assert isinstance(jpeg, Ok)
    assert jpeg.value.as_bytes() == _jpeg_min()
    png = parse_image_bytes(_png_min())
    assert isinstance(png, Ok)
    assert png.value.as_bytes() == _png_min()


def test_ttl_only_1_to_3600_default_60() -> None:
    assert default_ttl().value() == 60
    assert isinstance(parse_ttl_secs(1), Ok)
    assert isinstance(parse_ttl_secs(3600), Ok)
    assert isinstance(parse_ttl_secs(0), Err)
    assert isinstance(parse_ttl_secs(3601), Err)
    assert default_ttl().reaper_interval_secs() == 30
    assert default_ttl().purge_after_secs() == 120


def test_progress_only_0_to_1_finite() -> None:
    assert zero_progress().value() == 0.0
    assert isinstance(parse_progress(0.0), Ok)
    assert isinstance(parse_progress(0.4), Ok)
    assert isinstance(parse_progress(1.0), Ok)
    assert isinstance(parse_progress(-0.1), Err)
    assert isinstance(parse_progress(1.1), Err)
    assert isinstance(parse_progress(float("nan")), Err)
    assert isinstance(parse_progress(float("inf")), Err)
    assert isinstance(parse_progress("0.5"), Err)


def test_r2key_rejects_empty_and_dotdot() -> None:
    assert isinstance(parse_r2_key("jobs/1/a"), Ok)
    assert isinstance(parse_r2_key(""), Err)
    assert isinstance(parse_r2_key("  "), Err)
    assert isinstance(parse_r2_key("a/../b"), Err)
    trimmed = parse_r2_key("  jobs/1/a  ")
    assert isinstance(trimmed, Ok)
    assert trimmed.value.as_str() == "jobs/1/a"


def test_jobid_trims_uuid() -> None:
    fresh = str(uuid.uuid4())
    parsed = parse_job_id(f"  {fresh}  ")
    assert isinstance(parsed, Ok)
    assert parsed.value.as_str() == fresh
    assert isinstance(parse_job_id("not-a-uuid"), Err)
    assert isinstance(parse_job_id(""), Err)
    assert isinstance(parse_job_id(123), Err)
    assert isinstance(new_job_id().as_str(), str)


def test_stage_parses_known_and_rejects_unknown() -> None:
    assert isinstance(parse_stage("queued"), Ok)
    result = parse_stage("queued")
    assert isinstance(result, Ok)
    assert result.value == Stage.QUEUED
    assert isinstance(parse_stage("nope"), Err)


def test_landmarks_accepts_478_and_rejects_stub() -> None:
    pts = [[0.0, 1.0, 2.0]] * LANDMARKS_LEN
    raw = json.dumps(pts).encode("utf-8")
    assert isinstance(parse_landmarks(raw), Ok)
    assert isinstance(parse_landmarks(b'{"todo":"landmarks"}'), Err)
    assert isinstance(parse_landmarks(b""), Err)
    assert isinstance(parse_landmarks(bytes([1, 2, 3])), Err)


def test_uv_lengths_exact_canonical() -> None:
    assert isinstance(parse_complete_uv(bytes(UV_LEN)), Ok)
    assert isinstance(parse_complete_uv(bytes(UV_LEN - 1)), Err)
    assert isinstance(parse_flaw_uv(bytes(UV_LEN)), Ok)
    assert isinstance(parse_heatmap(bytes(UV_LEN)), Ok)
    assert isinstance(parse_heatmap(b'{"todo":"x"}'), Err)


def test_base_url_trims_slash_and_rejects_scheme() -> None:
    ok_result = parse_base_url("https://ml.internal:8081/")
    assert isinstance(ok_result, Ok)
    assert ok_result.value.as_str() == "https://ml.internal:8081"
    assert isinstance(parse_base_url("ml.internal:8081"), Err)
    assert isinstance(parse_base_url("  "), Err)


def test_gnm_mesh_rejects_truncated_and_bad_magic() -> None:
    assert isinstance(parse_gnm_mesh(b""), Err)
    assert isinstance(parse_gnm_mesh(bytes([1, 2, 3])), Err)
