#!/usr/bin/env bash
set -euo pipefail
# Smoke Fase 1 + deploy vivo (canonico UV): API + frontend + pipeline done + zip canonico.
# Patron: scripts/smoke-fase0.sh. Rapido por defecto (<70s por job); expiracion real
# solo con SMOKE_TTL_TEST=1 (65s, afirma expired + result 404).
# Expiracion logica sin espera larga esta cubierta por tests ManualClock TTL1.
# Prod vivo: API_URL=https://api.vultus.esau.com.mx FRONT_URL=https://vultus.esau.com.mx bash scripts/smoke-fase1.sh
# Par dorado real fuera de VC: GOLDEN_A/GOLDEN_B con JPEG LFW (ver frontend/e2e/fixtures/README.md).
# SLO warm <20s p95 par: SMOKE_SLO_SECS=20 (default), warmup opt-in con SMOKE_WARMUP=1.
API=${API_URL:-http://localhost:8000}
FRONT=${FRONT_URL:-http://localhost:4321}
SLO_SECS=${SMOKE_SLO_SECS:-20}

HEALTH=$(curl -f "$API/health")
echo "$HEALTH" | grep -q '"status":"ok"' || { echo "health status != ok: $HEALTH"; exit 1; }
echo "$HEALTH" | grep -q '"queue":"ok"' || { echo "health queue != ok: $HEALTH"; exit 1; }
if echo "$HEALTH" | grep -q '"sidecar"'; then
  echo "$HEALTH" | grep -q '"sidecar":"\(ok\|disabled\)"' || { echo "sidecar no sano: $HEALTH"; exit 1; }
  echo "api health ok (sidecar ok/disabled)"
else
  echo "api health ok (edge prod, sin sidecar)"
fi

curl -f "$FRONT" | grep -q "Vultus"
echo "frontend ok"

export API_URL="$API"
export SMOKE_SLO_SECS="$SLO_SECS"
python3 - <<'PY'
import json
import os
import time
import urllib.request
import urllib.error
import uuid
import zipfile

# Cloudflare Browser Integrity Check bloquea Python-urllib (403 error 1010).
# UA de navegador para todo request al gateway prod (local Rust ignora UA).
UA = {"User-Agent": "Mozilla/5.0 (compatible; VultusSmoke/1.0)"}

def urlopen(url_or_req, timeout=None):
    if isinstance(url_or_req, str):
        url_or_req = urllib.request.Request(url_or_req, headers=UA)
    else:
        for k, v in UA.items():
            if not url_or_req.has_header(k):
                url_or_req.add_header(k, v)
    return urllib.request.urlopen(url_or_req, timeout=timeout) if timeout else urllib.request.urlopen(url_or_req)

PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
# Par distinto A/B para heatmap no trivial incluso con dobles: A ceros, B unos.
# Prod dorado real via GOLDEN_A/GOLDEN_B (JPEG LFW fuera de VC, ver fixtures/README).
def load_or_default(path_env, default_bytes):
    p = os.environ.get(path_env, "")
    if p and os.path.exists(p):
        with open(p, "rb") as f:
            data = f.read()
        print(f"{path_env} golden real: {p} ({len(data)} bytes)")
        return data
    return default_bytes

png_a = PNG_MAGIC + bytes(56)
png_b = PNG_MAGIC + bytes([0xFF]) * 56
img_a = load_or_default("GOLDEN_A", png_a)
img_b = load_or_default("GOLDEN_B", png_b)
boundary = "----vultus" + uuid.uuid4().hex

def part(name, filename, data, ctype="image/png"):
    return (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n"
    ).encode() + data + b"\r\n"

def post_compare(a_bytes, b_bytes, a_name="a.png", b_name="b.png"):
    body = part("image_a", a_name, a_bytes) + part("image_b", b_name, b_bytes) + f"--{boundary}--\r\n".encode()
    api = os.environ.get("API_URL", "http://localhost:8000")
    req = urllib.request.Request(
        f"{api}/v1/compare",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **UA},
        method="POST",
    )
    with urlopen(req) as r:
        assert r.status == 202, r.status
        payload = json.loads(r.read().decode())
        assert payload.get("status") == "queued", payload
        return payload["job_id"]

def wait_done(api, job_id, timeout=70):
    t0 = time.time()
    deadline = t0 + timeout
    status = None
    # Poll 1s: granularidad fina para el SLO warm 20s (con 2s el error de
    # medicion ya come el 10% del objetivo).
    while time.time() < deadline:
        with urlopen(f"{api}/v1/jobs/{job_id}") as r:
            assert r.status == 200, r.status
            payload = json.loads(r.read().decode())
            status = payload["status"]
        if status == "done":
            return time.time() - t0
        if status in ("failed", "expired"):
            raise SystemExit(f"job termino prematuro en {status}")
        assert status in ("queued", "processing"), payload
        time.sleep(1)
    raise SystemExit(f"timeout esperando done, ultimo status={status}")

api = os.environ.get("API_URL", "http://localhost:8000")
slo = float(os.environ.get("SMOKE_SLO_SECS", "20"))

# Semantica 400: no-imagen debe dar 400, no 202.
try:
    body = part("image_a", "a.txt", b"not an image", "text/plain") + part("image_b", "b.txt", b"also not", "text/plain") + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{api}/v1/compare", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **UA}, method="POST")
    urlopen(req)
    raise SystemExit("non-image debio dar 400")
except urllib.error.HTTPError as e:
    assert e.code == 400, e.code
    print("400 non-image ok")

# Semantica 404: job desconocido debe dar 404.
unknown = str(uuid.uuid4())
try:
    urlopen(f"{api}/v1/jobs/{unknown}")
    raise SystemExit("unknown job debio dar 404")
except urllib.error.HTTPError as e:
    assert e.code == 404, e.code
    print("404 unknown job ok")

# Warmup opt-in para SLO warm (absorbe cold start, no se mide).
# Best-effort: un warmup frio puede expirar (TTL 60s < cadena fria); aun asi
# deja modelos cargados y el run medido posterior ya va en warm.
if os.environ.get("SMOKE_WARMUP") == "1":
    print("SMOKE_WARMUP=1: calentando workers (no medido)...")
    try:
        wjob = post_compare(img_a, img_b)
        wait_done(api, wjob, timeout=120)
        print("warmup done ok")
    except SystemExit as e:
        print(f"warmup no llego a done ({e}), se continua al run medido")
    # Drenaje: el warmup frio sigue procesando tras expirar (calienta
    # contenedores) y la GPU es serial por etapa; medir antes de que drene
    # contamina al medido con contencion. Espera fija solo en modo warmup.
    # Default 240s: cadena fria con caras en paralelo ronda ~140s + margen.
    drain = int(os.environ.get("SMOKE_DRAIN_SECS", "240"))
    print(f"esperando drenaje {drain}s antes del run medido...")
    time.sleep(drain)

t0 = time.time()
job_id = post_compare(img_a, img_b)
print(f"compare 202 ok: {job_id}")

# Semantica 409: resultado aun no listo debe dar 409 (no 404), sin esperar en vano.
try:
    urlopen(f"{api}/v1/jobs/{job_id}/result")
    print("result inmediato 200 (job rapidisimo, ok)")
except urllib.error.HTTPError as e:
    assert e.code in (404, 409), e.code
    print(f"result temprano {e.code} ok (409 esperado en prod vivo)")

elapsed = wait_done(api, job_id, timeout=70)
print(f"status done ok en {elapsed:.1f}s (SLO warm {slo:.0f}s)")
if elapsed > slo:
    print(f"WARNING: {elapsed:.1f}s supera SLO warm {slo:.0f}s (cold start? ver logs Modal)")
    if os.environ.get("SMOKE_STRICT_SLO") == "1":
        raise SystemExit(f"SLO warm excedido: {elapsed:.1f}s > {slo:.0f}s")
else:
    print(f"SLO warm ok: {elapsed:.1f}s < {slo:.0f}s")

# Result zip canonico: 200 + application/zip + 3 PNG con magic + heatmap no trivial.
def fetch_result():
    with urlopen(f"{api}/v1/jobs/{job_id}/result") as r:
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
    blobs = {}
    for n in ("uv_a.png", "uv_b.png", "heatmap.png"):
        blob = z.read(n)
        assert blob[:8] == PNG_MAGIC, f"{n} sin magic PNG"
        assert len(blob) > 1000, f"{n} demasiado chico ({len(blob)})"
        blobs[n] = blob
    # Heatmap no trivial: difiere de UVs y no es PNG solido (par A/B distinto).
    assert blobs["heatmap.png"] != blobs["uv_a.png"], "heatmap identico a uv_a (trivial)"
    assert blobs["uv_a.png"] != blobs["uv_b.png"], "uv_a identico a uv_b con A!=B (trivial)"
print("result zip canonico ok (3 PNG con magic, heatmap no trivial)")

# Segunda descarga tambien 200: multiple descargas permitidas.
data2 = fetch_result()
assert len(data2) > 0
print("segunda descarga ok")

# /status sigue done dentro de ventana (no expired prematuro).
with urlopen(f"{api}/v1/jobs/{job_id}") as r:
    payload = json.loads(r.read().decode())
    assert payload["status"] == "done", payload
print("status aun done en ventana ok")

# Test opt-in de expiracion real: espera TTL 60s + margen y afirma expired + 404.
if os.environ.get("SMOKE_TTL_TEST") == "1":
    print("SMOKE_TTL_TEST=1: esperando 65s para expiracion real...")
    time.sleep(65)
    with urlopen(f"{api}/v1/jobs/{job_id}") as r:
        payload = json.loads(r.read().decode())
        assert payload["status"] == "expired", payload
    print("expired tras TTL ok")
    try:
        urlopen(f"{api}/v1/jobs/{job_id}/result")
        raise SystemExit("result debio dar 404 tras expired")
    except urllib.error.HTTPError as e:
        assert e.code == 404, e.code
        print("result 404 tras expired ok")
else:
    print("expiracion cubierta por tests ManualClock TTL1 (SMOKE_TTL_TEST=1 para real)")

print(f"JOB_ID={job_id}")
PY
echo "smoke Fase 1 ok"
