#!/usr/bin/env bash
set -euo pipefail
# Smoke Fase 0: API + frontend deben responder tras `docker compose up`.
# Cubre E2E upload 2 PNG: POST /v1/compare -> 202 {job_id} -> GET /v1/jobs/{id} -> queued/processing.
API=${API_URL:-http://localhost:8000}
FRONT=${FRONT_URL:-http://localhost:4321}
curl -f "$API/health" | grep -q '"status":"ok"'
echo "api health ok"
curl -f "$FRONT" | grep -q "Vultus"
echo "frontend ok"
# Job dummy con 2 PNG minimos (magic + filler), luego verifica status.
python3 - <<'PY'
import json
import os
import urllib.request
import uuid

png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes(56)
boundary = "----vultus" + uuid.uuid4().hex

def part(name, filename, data):
    return (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n"
    ).encode() + data + b"\r\n"

body = part("image_a", "a.png", png) + part("image_b", "b.png", png) + f"--{boundary}--\r\n".encode()
api = os.environ.get("API_URL", "http://localhost:8000")
req = urllib.request.Request(
    f"{api}/v1/compare",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
with urllib.request.urlopen(req) as r:
    assert r.status == 202, r.status
    payload = json.loads(r.read().decode())
    job_id = payload["job_id"]
    assert payload.get("status") == "queued", payload
    print(f"compare 202 ok: {job_id}")

with urllib.request.urlopen(f"{api}/v1/jobs/{job_id}") as r:
    assert r.status == 200, r.status
    payload = json.loads(r.read().decode())
    assert payload["job_id"] == job_id, payload
    assert payload["status"] in ("queued", "processing"), payload
    print(f"status ok: {payload['status']}")
PY
echo "smoke Fase 0 ok"
