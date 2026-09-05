#!/usr/bin/env bash
set -euo pipefail
# Smoke Fase 1 (canonico UV): API + frontend + pipeline done + zip canonico.
# Patron: scripts/smoke-fase0.sh. Rapido por defecto (<70s); expiracion real
# solo con SMOKE_TTL_TEST=1 (65s, afirma expired + result 404).
# Expiracion logica sin espera larga esta cubierta por tests ManualClock TTL1.
API=${API_URL:-http://localhost:8000}
FRONT=${FRONT_URL:-http://localhost:4321}

HEALTH=$(curl -f "$API/health")
echo "$HEALTH" | grep -q '"status":"ok"' || { echo "health status != ok: $HEALTH"; exit 1; }
echo "$HEALTH" | grep -q '"sidecar":"\(ok\|disabled\)"' || { echo "sidecar no sano: $HEALTH"; exit 1; }
echo "api health ok (sidecar ok/disabled)"

curl -f "$FRONT" | grep -q "Vultus"
echo "frontend ok"

export API_URL="$API"
python3 - <<'PY'
import json
import os
import time
import urllib.request
import urllib.error
import uuid
import zipfile

PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
png = PNG_MAGIC + bytes(56)
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

# Poll hasta done: timeout 70s cubre total 60s + cold 2s, intervalo 2s.
deadline = time.time() + 70
status = None
while time.time() < deadline:
    with urllib.request.urlopen(f"{api}/v1/jobs/{job_id}") as r:
        assert r.status == 200, r.status
        payload = json.loads(r.read().decode())
        status = payload["status"]
    if status == "done":
        print("status done ok")
        break
    if status in ("failed", "expired"):
        raise SystemExit(f"job termino prematuro en {status}")
    assert status in ("queued", "processing"), payload
    time.sleep(2)
else:
    raise SystemExit(f"timeout esperando done, ultimo status={status}")
assert status == "done", status

# Result zip canonico: 200 + application/zip + 3 PNG con magic.
def fetch_result():
    with urllib.request.urlopen(f"{api}/v1/jobs/{job_id}/result") as r:
        assert r.status == 200, r.status
        ctype = r.headers.get("Content-Type", "")
        assert "application/zip" in ctype, ctype
        return r.read()

data = fetch_result()
with open("/tmp/result.zip", "wb") as f:
    f.write(data)
with zipfile.ZipFile("/tmp/result.zip") as z:
    names = set(z.namelist())
    assert names == {"uv_a.png", "uv_b.png", "heatmap.png"}, names
    for n in ("uv_a.png", "uv_b.png", "heatmap.png"):
        blob = z.read(n)
        assert blob[:8] == PNG_MAGIC, f"{n} sin magic PNG"
print("result zip canonico ok (3 PNG con magic)")

# Segunda descarga tambien 200: multiple descargas permitidas.
data2 = fetch_result()
assert len(data2) > 0
print("segunda descarga ok")

# /status sigue done dentro de ventana (no expired prematuro).
with urllib.request.urlopen(f"{api}/v1/jobs/{job_id}") as r:
    payload = json.loads(r.read().decode())
    assert payload["status"] == "done", payload
print("status aun done en ventana ok")

# Test opt-in de expiracion real: espera TTL 60s + margen y afirma expired + 404.
if os.environ.get("SMOKE_TTL_TEST") == "1":
    print("SMOKE_TTL_TEST=1: esperando 65s para expiracion real...")
    time.sleep(65)
    with urllib.request.urlopen(f"{api}/v1/jobs/{job_id}") as r:
        payload = json.loads(r.read().decode())
        assert payload["status"] == "expired", payload
    print("expired tras TTL ok")
    try:
        urllib.request.urlopen(f"{api}/v1/jobs/{job_id}/result")
        raise SystemExit("result debio dar 404 tras expired")
    except urllib.error.HTTPError as e:
        assert e.code == 404, e.code
        print("result 404 tras expired ok")
else:
    print("expiracion cubierta por tests ManualClock TTL1 (SMOKE_TTL_TEST=1 para real)")

print(f"JOB_ID={job_id}")
PY
echo "smoke Fase 1 ok"
