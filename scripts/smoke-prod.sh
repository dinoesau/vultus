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
import json, os, urllib.request
api = os.environ.get("API_URL", "http://localhost:8000")
png = bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A])+bytes(56)
b = "----vultusprod"
def part(n,fn,d):
    return (f"--{b}\r\nContent-Disposition: form-data; name=\"{n}\"; filename=\"{fn}\"\r\nContent-Type: image/png\r\n\r\n").encode()+d+b"\r\n"
body = part("image_a","a.png",png)+part("image_b","b.png",png)+f"--{b}--\r\n".encode()
req = urllib.request.Request(f"{api}/v1/compare", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"}, method="POST")
with urllib.request.urlopen(req, timeout=15) as r:
    assert r.status == 202, r.status
    p = json.loads(r.read().decode())
    assert p.get("status") == "queued", p
    jid = p["job_id"]
    print(f"compare 202 ok: {jid}")
with urllib.request.urlopen(f"{api}/v1/jobs/{jid}", timeout=15) as r:
    assert r.status == 200, r.status
    p = json.loads(r.read().decode())
    assert p["status"] in ("queued","processing","done","failed","expired"), p
    print(f"status ok: {p['status']}")
for path, want in (("not-a-uuid",400), ("11111111-1111-4111-8111-111111111111",404)):
    try:
        urllib.request.urlopen(f"{api}/v1/jobs/{path}", timeout=15)
        raise SystemExit(f"{path} -> unexpected 2xx")
    except Exception as e:
        code = getattr(e,"code",None)
        assert code == want, f"{path} -> {code}, want {want}"
        print(f"{path} -> {code} ok")
PY
echo "smoke prod ok"
