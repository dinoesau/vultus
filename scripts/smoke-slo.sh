#!/usr/bin/env bash
set -euo pipefail
# SLO warm deploy-real-models: 1 warmup excluido + N corridas de smoke-fase1.sh
# con fotos LFW reales, reporta p95 compare-a-done (SLO menor a 20s, solo warm).
# Uso: API_URL=http://localhost:8001 FRONT_URL=http://localhost:4321 \
#   SMOKE_SLO_RUNS=10 LFW_A=.../0001.jpg LFW_B=.../0002.jpg bash scripts/smoke-slo.sh
# Requiere API Rust con ML_SIDECAR_URL al endpoint Modal (ver fixtures/README.md).
RUNS=${SMOKE_SLO_RUNS:-10}
LFW_A=${LFW_A:-/Users/esau.martinez/Code/datasets/lfw/George_W_Bush/George_W_Bush_0001.jpg}
LFW_B=${LFW_B:-/Users/esau.martinez/Code/datasets/lfw/George_W_Bush/George_W_Bush_0002.jpg}
API=${API_URL:-http://localhost:8000}

run_once() {
  python3 - "$API" "$LFW_A" "$LFW_B" <<'PY'
import json, sys, time, urllib.request, uuid
api, pa, pb = sys.argv[1], sys.argv[2], sys.argv[3]
a, b = open(pa, "rb").read(), open(pb, "rb").read()
bd = "----vultus" + uuid.uuid4().hex
def part(n, fn, d):
    return (f"--{bd}\r\nContent-Disposition: form-data; name=\"{n}\"; filename=\"{fn}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + d + b"\r\n"
body = part("image_a", "a.jpg", a) + part("image_b", "b.jpg", b) + f"--{bd}--\r\n".encode()
t0 = time.time()
req = urllib.request.Request(f"{api}/v1/compare", data=body, headers={"Content-Type": f"multipart/form-data; boundary={bd}"}, method="POST")
with urllib.request.urlopen(req) as r:
    jid = json.loads(r.read().decode())["job_id"]
deadline = time.time() + 100
while time.time() < deadline:
    time.sleep(2)
    with urllib.request.urlopen(f"{api}/v1/jobs/{jid}") as r:
        s = json.loads(r.read().decode())["status"]
    if s == "done":
        print("%.1f" % (time.time() - t0))
        raise SystemExit(0)
    if s in ("failed", "expired"):
        raise SystemExit(f"job {s}")
raise SystemExit("timeout esperando done")
PY
}

echo "warmup (excluido: 3 exitos seguidos, hasta 12 intentos en frio)..."
consec=0
for w in $(seq 1 12); do
  if run_once > /dev/null; then consec=$((consec+1)); echo "warmup ok ($consec/3)"; else consec=0; echo "warmup intento $w en frio, racha a cero..."; fi
  if [ "$consec" -ge 3 ]; then break; fi
  sleep 5
done
[ "$consec" -ge 3 ] || { echo "warmup sin estabilizar"; exit 1; }
: > /tmp/slo-times.txt
for i in $(seq 1 "$RUNS"); do
  d=$(run_once)
  echo "$d" >> /tmp/slo-times.txt
  echo "run $i: ${d}s"
done
python3 - <<'PY'
import statistics
ds = sorted(float(x) for x in open("/tmp/slo-times.txt") if x.strip())
n = len(ds)
p95 = ds[min(n - 1, int(n * 0.95))]
print(f"runs={n} min={ds[0]:.1f} p50={statistics.median(ds):.1f} p95={p95:.1f} max={ds[-1]:.1f}")
assert p95 < 20, f"SLO incumplido: p95 {p95:.1f}s >= 20s"
print("SLO warm ok (p95 < 20s)")
PY
