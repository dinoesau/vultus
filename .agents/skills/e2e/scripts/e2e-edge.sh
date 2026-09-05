#!/usr/bin/env bash
set -euo pipefail
# E2E edge: wrangler dev efimero mas matriz curl. Corre desde la raiz del repo.
# Uso: .agents/skills/e2e/scripts/e2e-edge.sh [port]
PORT="${1:-8788}"
npx wrangler dev --port "$PORT" > /tmp/wrangler-dev.log 2>&1 &
echo $! > /tmp/wrangler.pid
for _ in $(seq 1 25); do curl -sf "http://localhost:$PORT/health" && break; sleep 2; done
echo "---HEALTH---"
curl -s "http://localhost:$PORT/health"; echo
echo "---COMPARE+STATUS---"
PORT="$PORT" python3 - <<'PY'
import json, os, uuid, urllib.request
port = os.environ.get("PORT", "8788")
png = bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A])+bytes(56)
b="----vultus"+uuid.uuid4().hex
def part(n,fn,d):
    return (f"--{b}\r\nContent-Disposition: form-data; name=\"{n}\"; filename=\"{fn}\"\r\nContent-Type: image/png\r\n\r\n").encode()+d+b"\r\n"
body=part("image_a","a.png",png)+part("image_b","b.png",png)+f"--{b}--\r\n".encode()
req=urllib.request.Request(f"http://localhost:{port}/v1/compare",data=body,headers={"Content-Type":f"multipart/form-data; boundary={b}"},method="POST")
with urllib.request.urlopen(req) as r:
    p=json.loads(r.read().decode()); print("compare:",r.status,p); jid=p["job_id"]
with urllib.request.urlopen(f"http://localhost:{port}/v1/jobs/{jid}") as r:
    print("status:",r.status,json.loads(r.read().decode()))
for path in ("not-a-uuid", "11111111-1111-4111-8111-111111111111"):
    try:
        urllib.request.urlopen(f"http://localhost:{port}/v1/jobs/{path}")
        print(path, "-> unexpected 2xx")
    except Exception as e:
        print(path, "->", getattr(e,"code",e))
PY
kill "$(cat /tmp/wrangler.pid)" 2>/dev/null || true; echo "wrangler detenido"
tail -20 /tmp/wrangler-dev.log
