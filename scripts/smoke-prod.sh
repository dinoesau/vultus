#!/usr/bin/env bash
set -euo pipefail
# Smoke prod/preview: gateway vivo sin mocks.
# Uso: API_URL=https://api.vultus.esau.com.mx bash scripts/smoke-prod.sh
# Cubre Seam 1: health gateway worker + compare 202 + status + negativos 400/404.
API=${API_URL:-http://localhost:8000}
echo "smoke prod contra $API"
curl -sf "$API/health" | grep -q '"gateway":"worker"'
echo "health ok"
curl -sf "$API/health" | grep -q '"ttl_secs":60'
echo "ttl ok"
python3 - <<'PY'
import json, os, urllib.request, urllib.error
api = os.environ.get("API_URL", "http://localhost:8000")
UA = {"User-Agent": "Mozilla/5.0 (compatible; VultusSmoke/1.0)"}
png = bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A])+bytes(56)
b = "----vultusprod"
def part(n,fn,d):
    return (f"--{b}\r\nContent-Disposition: form-data; name=\"{n}\"; filename=\"{fn}\"\r\nContent-Type: image/png\r\n\r\n").encode()+d+b"\r\n"
def call(req):
    try:
        return urllib.request.urlopen(req, timeout=15)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} en {req.full_url}, body: {e.read().decode(errors='replace')[:500]}")
        raise
body = part("image_a","a.png",png)+part("image_b","b.png",png)+f"--{b}--\r\n".encode()
req = urllib.request.Request(f"{api}/v1/compare", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}", **UA}, method="POST")
with call(req) as r:
    assert r.status == 202, r.status
    p = json.loads(r.read().decode())
    assert p.get("status") == "queued", p
    jid = p["job_id"]
    print(f"compare 202 ok: {jid}")
with call(urllib.request.Request(f"{api}/v1/jobs/{jid}", headers=UA)) as r:
    assert r.status == 200, r.status
    p = json.loads(r.read().decode())
    assert p["status"] in ("queued","processing","done","failed","expired"), p
    print(f"status ok: {p['status']}")
for path, want in (("not-a-uuid",400), ("11111111-1111-4111-8111-111111111111",404)):
    try:
        call(urllib.request.Request(f"{api}/v1/jobs/{path}", headers=UA))
        raise SystemExit(f"{path} -> unexpected 2xx")
    except urllib.error.HTTPError as e:
        assert e.code == want, f"{path} -> {e.code}, want {want}"
        print(f"{path} -> {e.code} ok")
PY
echo "smoke prod ok"
