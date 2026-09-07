"""Seam 1 HTTP: servidor real sin mocks, goldens literales."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.store import MemoryQueue


def _png() -> bytes:
    return bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes(56)


def _client() -> TestClient:
    return TestClient(create_app(queue=MemoryQueue(), sidecar=None))


def test_post_compare_202_queued_luego_get_status() -> None:
    client = _client()
    resp = client.post(
        "/v1/compare",
        files={"image_a": ("a.png", _png(), "image/png"), "image_b": ("b.png", _png(), "image/png")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    status = client.get(f"/v1/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] in ("queued", "processing")


def test_400_en_imagen_faltante_o_invalida_y_uuid_roto() -> None:
    client = _client()
    missing = client.post("/v1/compare", files={"image_a": ("a.png", _png(), "image/png")})
    assert missing.status_code == 400
    bad = client.post(
        "/v1/compare",
        files={"image_a": ("a.bin", b"\x01\x02\x03", "application/octet-stream"), "image_b": ("b.bin", b"\x04\x05\x06", "application/octet-stream")},
    )
    assert bad.status_code == 400
    assert client.get("/v1/jobs/not-a-uuid").status_code == 400


def test_404_en_desconocido_y_409_en_result_pre_done() -> None:
    client = _client()
    unknown = "11111111-1111-4111-8111-111111111111"
    assert client.get(f"/v1/jobs/{unknown}").status_code == 404
    assert client.get(f"/v1/jobs/{unknown}/result").status_code == 404
    resp = client.post(
        "/v1/compare",
        files={"image_a": ("a.png", _png(), "image/png"), "image_b": ("b.png", _png(), "image/png")},
    )
    job_id = resp.json()["job_id"]
    early = client.get(f"/v1/jobs/{job_id}/result")
    assert early.status_code == 409


def test_health_reporta_cola_real_y_ttl() -> None:
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["queue"] == "ok"
    assert body["ttl_secs"] == 60


def test_ws_emite_snapshot_queued_y_cierra_en_terminal() -> None:
    client = _client()
    resp = client.post(
        "/v1/compare",
        files={"image_a": ("a.png", _png(), "image/png"), "image_b": ("b.png", _png(), "image/png")},
    )
    job_id = resp.json()["job_id"]
    with client.websocket_connect(f"/v1/jobs/{job_id}/events") as ws:
        event = ws.receive_json()
        assert event["job_id"] == job_id
        assert event["status"] in ("queued", "processing")
        assert event["stage"] in ("queued", "landmarks", "flame", "freeuv", "bake", "done")


def test_ws_handshake_falla_en_desconocido() -> None:
    import pytest
    from starlette.websockets import WebSocketDisconnect

    client = _client()
    unknown = "11111111-1111-4111-8111-111111111111"
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(f"/v1/jobs/{unknown}/events"):
        raise AssertionError("unknown job must not upgrade to WS")
