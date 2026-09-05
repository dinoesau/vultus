---
name: e2e
description: 'Run the full end-to-end verification for Vultus: Docker stack health, smoke script, Playwright, and edge wrangler dev with real curl. Writes Given-When-Then acceptance criteria plus a 12-factor audit before running. Self-updates when new features or surfaces need testing. Use when user says E2E, end-to-end, smoke, or verifica de punta a punta.'
license: MIT
allowed-tools: Bash
---

# E2E - Verificacion de punta a punta

## Overview

E2E real, no `config` ni `dry-run`.
Orden fijo: stack Docker -> smoke -> Playwright -> edge `wrangler dev` + curl.
Antes de correr, escribe criterios Given-When-Then y auditoria 12-Factor.
Si algo no esta cubierto, primero actualiza esta skill (ver Self-update), luego prueba.

## Preconditions

- `docker compose up -d --build` ya corriendo (lo hace el usuario o un paso previo).
- `frontend/node_modules` puede faltar: hacer `npm install` + `npx playwright install chromium` una vez.

## Step 0 - Self-update protocol (obligatorio)

Esta skill se queda obsoleta en cada fase. Antes de correr:

1. Lista superficies nuevas sin escenario: endpoints (`GET /result`, `/ml/*` reales), WS streaming, visores UV/3D, PDF, rate-limit, auth, colas reales.
2. Si hay alguna sin `E2E-N` en `Coverage registry`, edita este `SKILL.md`: añade el escenario GWT, el comando curl/playwright exacto y el Then esperado.
3. Sigue con el run ya con la skill actualizada. La actualizacion va en el mismo commit o uno previo `test(e2e): extiende cobertura a X`.

Nunca digas "no cubierto, lo salto". Lo cubres o lo registras como deuda explicita en el veredicto.

## Step 1 - Criterios Given-When-Then

Escribe estos (o sus sucesores) en el chat antes de correr:

- **E2E-1 Stack**: Given compose up, When `curl :8000/health + :4321 + :8081/docs`, Then `api ok/60`, `frontend Vultus`, `sidecar /ml/*`.
- **E2E-2 Job dummy**: Given stack sano, When `POST /v1/compare` 2 PNG + `GET /v1/jobs/{id}` + `WS /events`, Then `202 queued`, `200 queued|processing`, WS snapshot `{queued, 0.0}`.
- **E2E-3 Errores**: When imagen invalida / incompleto / uuid roto / desconocido, Then `400/400/400/404` sin encolar.
- **E2E-4 Stateless**: Given `TtlSecs=60`, When pasa `TTL` y `2xTTL`, Then `expired` luego `NotFound`, `job_dir` inexistente, sin redis.
- **E2E-5 Edge**: Given `wrangler dev`, When `compare + status + bad uuid + unknown`, Then `202 + 200 queued + 400 + 404` desde el DO (nunca `queued` fantasma).

## Step 2 - Auditoria 12-Factor (resumen, no saltar)

I codebase, II deps pineadas (`Cargo.lock`, `requirements.txt`, `package-lock.json`), III config por env + `.env.example` sin secretos, IV backing via bindings/adapter, V build/run separados, VI stateless + `tmpfs`, VII ports `8000/8081/4321`, VIII concurrencia tokio/Modal, IX desechable (healthcheck + graceful shutdown + reaper), X paridad local/prod (fallbacks edge solo-dev), XI logs stdout con `job_id` sin bytes, XII admin en `scripts/`.

## Step 3 - Run sequence

```bash
docker compose ps
curl -s http://localhost:8000/health; echo
curl -s http://localhost:4321 | head -c 300; echo
curl -s http://localhost:8081/docs | head -c 200; echo
bash scripts/smoke-fase0.sh
```

```bash
# frontend/
npm install
npx playwright install chromium
npm run test:e2e
```

```bash
npx wrangler dev --port 8788 > /tmp/wrangler-dev.log 2>&1 &
echo $! > /tmp/wrangler.pid
for i in $(seq 1 25); do curl -sf http://localhost:8788/health && break; sleep 2; done
curl -s http://localhost:8788/health; echo
python3 - <<'PY'
import json, uuid, urllib.request
png = bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A])+bytes(56)
b="----vultus"+uuid.uuid4().hex
def part(n,fn,d):
    return (f"--{b}\r\nContent-Disposition: form-data; name=\"{n}\"; filename=\"{fn}\"\r\nContent-Type: image/png\r\n\r\n").encode()+d+b"\r\n"
body=part("image_a","a.png",png)+part("image_b","b.png",png)+f"--{b}--\r\n".encode()
req=urllib.request.Request("http://localhost:8788/v1/compare",data=body,headers={"Content-Type":f"multipart/form-data; boundary={b}"},method="POST")
with urllib.request.urlopen(req) as r:
    p=json.loads(r.read().decode()); print("compare:",r.status,p); jid=p["job_id"]
with urllib.request.urlopen(f"http://localhost:8788/v1/jobs/{jid}") as r:
    print("status:",r.status,json.loads(r.read().decode()))
PY
kill $(cat /tmp/wrangler.pid) 2>/dev/null; echo "wrangler detenido"
```

## Step 4 - Verdict + cleanup

- Reporta por escenario: verde/rojo con evidencia (status code + payload).
- Limpieza: `rm -rf frontend/test-results`, matar wrangler, `git status --porcelain`.
- Si `npm install` genero `package-lock.json` nuevo, commitealo: `chore(frontend): fija lockfile tras E2E verde`.
- Nunca commitees `.env`, `*.pem/key`, ni `test-results/`.

## Coverage registry (extender en cada fase)

- Fase 0: E2E-1..E2E-5 arriba. Sin consumidor real `Queued->Done` (deuda conocida, Fase 1).
- Fase 1: añadir `POST /ml/*` reales, `GET /result` zip, visor UV + heatmap.
- Fase 2+: añadir visor 3D, metricas PDF, rate-limit, Turnstile/WAF.
