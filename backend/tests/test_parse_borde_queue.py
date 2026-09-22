"""Borde cola: envelope canonico, flip legacy a/b a Err, traversal y dead-letter.

Slice vertical sobre parse_queue_envelope (dominio) mas process_message
(runner): literales conocidos, sin R2 ni HTTP reales. El path malo de
process_message no toca red, solo dead-letter en memoria.
"""

from __future__ import annotations

import json
import uuid

import pytest

from backend.domain import (
    Err,
    InvalidR2Key,
    MlFailed,
    NotFound,
    Ok,
    domain_to_status,
    parse_job_id,
    parse_queue_envelope,
    parse_r2_key,
)
from backend.local_runner import (
    _DEAD_LETTERS,
    _METRICS,
    UNKNOWN_JOB_ID,
    _fetch_blob,
    process_message,
)

JOB = "123e4567-e89b-12d3-a456-426614174000"
KEY_A = "jobs/abc/image_a.png"
KEY_B = "jobs/abc/image_b.png"


def _envelope(job: object = JOB, a: object = KEY_A, b: object = KEY_B) -> dict[str, object]:
    return {"job_id": job, "r2_keys": {"image_a": a, "image_b": b}}


def test_canonical_dict_yields_job() -> None:
    result = parse_queue_envelope(_envelope())
    assert isinstance(result, Ok)
    assert result.value.job_id.as_str() == JOB
    assert result.value.r2_a.as_str() == KEY_A
    assert result.value.r2_b.as_str() == KEY_B


def test_body_and_message_json_string_plus_camelcase() -> None:
    body = json.dumps(_envelope())
    assert isinstance(parse_queue_envelope({"body": body}), Ok)
    assert isinstance(parse_queue_envelope({"message": body}), Ok)
    camel = {"jobId": JOB, "r2Keys": {"image_a": KEY_A, "image_b": KEY_B}}
    assert isinstance(parse_queue_envelope(camel), Ok)
    assert isinstance(parse_queue_envelope({"body": None, "message": body}), Ok)


def test_body_none_strict_falsy_dict_does_not_fall_through() -> None:
    valid = json.dumps(_envelope())
    assert isinstance(parse_queue_envelope({"body": {}, "message": valid}), Err)


def test_legacy_ab_flipped_to_err() -> None:
    legacy = {"job_id": JOB, "r2_keys": {"a": KEY_A, "b": KEY_B}}
    assert isinstance(parse_queue_envelope(legacy), Err)


def test_mixed_canonical_legacy_is_err() -> None:
    mixed = {"job_id": JOB, "r2_keys": {"image_a": KEY_A, "a": KEY_A, "b": KEY_B}}
    assert isinstance(parse_queue_envelope(mixed), Err)
    mixed2 = {"job_id": JOB, "r2_keys": {"image_a": KEY_A, "image_b": 123, "a": KEY_A, "b": KEY_B}}
    assert isinstance(parse_queue_envelope(mixed2), Err)


def test_canonical_plus_legacy_extras_is_err() -> None:
    smuggled = {
        "job_id": JOB,
        "r2_keys": {"image_a": KEY_A, "image_b": KEY_B, "a": KEY_A, "b": KEY_B},
    }
    result = parse_queue_envelope(smuggled)
    assert isinstance(result, Err)
    assert isinstance(result.error, InvalidR2Key)


def test_canonical_plus_unknown_extra_ignored_for_forward_compat() -> None:
    extra = {
        "job_id": JOB,
        "r2_keys": {"image_a": KEY_A, "image_b": KEY_B, "role": "admin"},
    }
    result = parse_queue_envelope(extra)
    assert isinstance(result, Ok)
    assert result.value.r2_a.as_str() == KEY_A
    assert result.value.r2_b.as_str() == KEY_B


def test_bad_uuid_missing_keys_nondict_are_err() -> None:
    assert isinstance(parse_queue_envelope(_envelope(job="not-a-uuid")), Err)
    assert isinstance(parse_queue_envelope({"job_id": JOB}), Err)
    assert isinstance(parse_queue_envelope({"job_id": JOB, "r2_keys": {}}), Err)
    assert isinstance(parse_queue_envelope({"job_id": JOB, "r2_keys": "x"}), Err)
    assert isinstance(parse_queue_envelope({"body": "no-json{"}), Err)
    assert isinstance(parse_queue_envelope({"body": "[1,2]"}), Err)
    assert isinstance(parse_queue_envelope([1, 2, 3]), Err)
    assert isinstance(parse_queue_envelope(None), Err)


@pytest.mark.parametrize(
    "bad",
    [
        "a/../b",
        "..",
        "a/..",
        "%2e%2e/a",
        "%2E%2E/a",
        "%252e%252e/a",
        "%252E%252E/a",
        "a\\b",
        "/leading",
        "",
        "   ",
        "a\x00b",
        "a\x1fb",
        "x" * 1025,
    ],
)
def test_traversal_and_shape_edges_are_err(bad: str) -> None:
    assert isinstance(parse_queue_envelope(_envelope(a=bad)), Err)
    assert isinstance(parse_queue_envelope(_envelope(b=bad)), Err)
    assert isinstance(parse_r2_key(bad), Err)


def test_whitespace_trim_and_unicode_accepted() -> None:
    result = parse_queue_envelope(_envelope(a=f"  {KEY_A}  ", b=KEY_B))
    assert isinstance(result, Ok)
    assert result.value.r2_a.as_str() == KEY_A
    uni = "jobs/café/imagen.png"
    assert isinstance(parse_r2_key(uni), Ok)


def test_r2key_stores_trimmed_exact_key() -> None:
    parsed = parse_r2_key(f"  {KEY_A}  ")
    assert isinstance(parsed, Ok)
    assert parsed.value.as_str() == KEY_A


def test_process_message_bad_yields_unknown() -> None:
    outcome = process_message({"bad": "payload"})
    assert outcome.job_id == UNKNOWN_JOB_ID
    assert outcome.job_id == "unknown"
    assert outcome.action == "skipped-bad-message"


def test_dead_letter_bounded_and_unknown_stable() -> None:
    _DEAD_LETTERS.clear()
    start = _METRICS.get("queue_skip", 0)
    for i in range(105):
        outcome = process_message({"bad": i})
        assert outcome.job_id == UNKNOWN_JOB_ID
    assert len(_DEAD_LETTERS) == 100
    assert _METRICS.get("queue_skip", 0) - start == 105
    _DEAD_LETTERS.clear()


def test_fresh_job_id_parses() -> None:
    fresh = str(uuid.uuid4())
    assert isinstance(parse_queue_envelope(_envelope(job=fresh)), Ok)


def _jpeg_bytes() -> bytes:
    return bytes([0xFF, 0xD8, 0xFF, 0x00]) + bytes(60)


def _proven_fetch_args(key: str):  # type: ignore[no-untyped-def]
    job = parse_job_id(JOB)
    parsed_key = parse_r2_key(key)
    assert isinstance(job, Ok)
    assert isinstance(parsed_key, Ok)
    return job.value, parsed_key.value


def test_fetch_suffix_mismatch_returns_err_without_http(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import backend.local_runner as _runner

    calls: list[tuple[str, str]] = []

    def _fake_http(method: str, url: str, body: bytes | None = None, content_type: str = "application/json") -> tuple[int, bytes]:
        calls.append((method, url))
        return 200, _jpeg_bytes()

    monkeypatch.setattr(_runner, "_http", _fake_http)
    job, key_a = _proven_fetch_args("jobs/abc/blob/a")
    result = _fetch_blob(job, key_a, "b")  # type: ignore[arg-type]
    assert isinstance(result, Err)
    assert isinstance(result.error, InvalidR2Key)
    assert domain_to_status(result.error) == 400
    job2, key_b = _proven_fetch_args("jobs/abc/blob/b")
    result2 = _fetch_blob(job2, key_b, "a")  # type: ignore[arg-type]
    assert isinstance(result2, Err)
    assert isinstance(result2.error, InvalidR2Key)
    assert calls == []


def test_fetch_invalid_slot_returns_err_without_http(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import backend.local_runner as _runner

    calls: list[tuple[str, str]] = []

    def _fake_http(method: str, url: str, body: bytes | None = None, content_type: str = "application/json") -> tuple[int, bytes]:
        calls.append((method, url))
        return 200, _jpeg_bytes()

    monkeypatch.setattr(_runner, "_http", _fake_http)
    job, key = _proven_fetch_args("jobs/abc/image.png")
    result = _fetch_blob(job, key, "c")  # type: ignore[arg-type]
    assert isinstance(result, Err)
    assert isinstance(result.error, InvalidR2Key)
    assert calls == []


def test_fetch_suffix_absent_uses_explicit_slot(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import backend.local_runner as _runner

    seen: list[str] = []

    def _fake_http(method: str, url: str, body: bytes | None = None, content_type: str = "application/json") -> tuple[int, bytes]:
        seen.append(url)
        return 200, _jpeg_bytes()

    monkeypatch.setattr(_runner, "_http", _fake_http)
    job, key = _proven_fetch_args("jobs/abc/image.png")
    result_a = _fetch_blob(job, key, "a")
    assert isinstance(result_a, Ok)
    assert result_a.value.as_bytes() == _jpeg_bytes()
    assert seen[-1].endswith(f"{JOB}/a")
    result_b = _fetch_blob(job, key, "b")
    assert isinstance(result_b, Ok)
    assert seen[-1].endswith(f"{JOB}/b")


def test_fetch_uppercase_and_trailing_slash_passthrough(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import backend.local_runner as _runner

    seen: list[str] = []

    def _fake_http(method: str, url: str, body: bytes | None = None, content_type: str = "application/json") -> tuple[int, bytes]:
        seen.append(url)
        return 200, _jpeg_bytes()

    monkeypatch.setattr(_runner, "_http", _fake_http)
    for raw_key, slot, suffix in [
        ("jobs/abc/blob/A", "a", "a"),
        ("jobs/abc/blob/B", "b", "b"),
        ("jobs/abc/image.png/", "a", "a"),
    ]:
        job, key = _proven_fetch_args(raw_key)
        result = _fetch_blob(job, key, slot)  # type: ignore[arg-type]
        assert isinstance(result, Ok)
        assert seen[-1].endswith(f"{JOB}/{suffix}")


def test_fetch_distinct_keys_distinct_behavior(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import backend.local_runner as _runner

    calls: list[str] = []

    def _fake_http(method: str, url: str, body: bytes | None = None, content_type: str = "application/json") -> tuple[int, bytes]:
        calls.append(url)
        return 200, _jpeg_bytes()

    monkeypatch.setattr(_runner, "_http", _fake_http)
    job, key_ok = _proven_fetch_args("jobs/abc/blob/a")
    assert isinstance(_fetch_blob(job, key_ok, "a"), Ok)
    n_calls = len(calls)
    _, key_bad = _proven_fetch_args("jobs/abc/blob/b")
    result = _fetch_blob(job, key_bad, "a")
    assert isinstance(result, Err)
    assert isinstance(result.error, InvalidR2Key)
    assert len(calls) == n_calls


def test_fetch_404_maps_notfound_vs_599_maps_mlfaild(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import backend.local_runner as _runner

    job, key = _proven_fetch_args("jobs/abc/image.png")
    monkeypatch.setattr(_runner, "_http", lambda _m, _u, _b=None, _c="application/json": (404, b""))
    missing = _fetch_blob(job, key, "a")
    assert isinstance(missing, Err)
    assert isinstance(missing.error, NotFound)
    assert domain_to_status(missing.error) == 404

    monkeypatch.setattr(_runner, "_http", lambda _m, _u, _b=None, _c="application/json": (599, b""))
    failed = _fetch_blob(job, key, "a")
    assert isinstance(failed, Err)
    assert isinstance(failed.error, MlFailed)
    assert domain_to_status(failed.error) == 500
