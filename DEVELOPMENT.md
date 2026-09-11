# DEVELOPMENT - Guía de Desarrollo

> Estado objetivo sin Rust: un solo compose sin toolchain Rust y un solo gateway.
> Comandos nuevos: `pip install -r backend/requirements-api.txt`, `mypy --strict backend/domain.py backend/gnm.py backend/pipeline_local.py backend/local_runner.py`,
> `ruff check backend/`, `pytest backend/tests -q`, `npx vitest run edge/contract.test.ts`,
> `npx vitest run --config vitest.pool.config.ts` (desde `frontend/`).

## 1. Requisitos

Sin toolchain Rust. Solo Python + Node + Docker.
Instala `Docker` 24+ y `Docker Compose` v2.
Para GPU instala `nvidia-container-toolkit` y verifica con `nvidia-smi`.
Node 22+ para frontend Astro (Astro 7 exige 20.19+ o 22.12+).
Python 3.12+ para runner local y sidecar ML (`backend/local_runner.py`, `backend/modal_app.py`, `Dockerfile.runner`, `Dockerfile.gpu`).
Node 22+ para el gateway (Worker `edge/worker.ts` + entrada dev `edge/worker.dev.ts`).
Historico: antes API Python FastAPI y Rust Axum (ver ADRs), borrados en plan-single-gateway-ts.

## 2. Setup Python

```bash
pip install -r backend/requirements-api.txt
mypy --strict --explicit-package-bases --namespace-packages backend/domain.py backend/gnm.py backend/pipeline_local.py backend/local_runner.py
ruff check backend/
pytest backend/tests -q
```

Gateway en `http://localhost:8000` (`/health` con `gateway:"worker"`, `POST /v1/compare` -> `202 {job_id, status:"queued"}`, `GET /v1/jobs/{id}` -> `200 {job_id, status}`).
Mismo handler que prod, servido local via `wrangler dev -c wrangler.dev.toml`.
Sidecar ML local en `:8081` vía `modal_app.sidecar`.

```bash
python3 -c "import sys; sys.path.insert(0,'.'); import backend.modal_app; import uvicorn; uvicorn.run(backend.modal_app.sidecar, port=8081)"
```

`ML_SIDECAR_URL=http://localhost:8081` lo consume `MlSidecarClient(BaseUrl)`.
`BaseUrl` exige `http(s)://` y recorta `/` final.

## 3. Estructura backend

```
backend/
├── requirements-api.txt   # deps compute/ML local (httpx/Pillow/numpy + pytest/mypy/ruff)
├── Dockerfile.wrangler    # gateway worker local (misma entrada dev que prod)
├── Dockerfile.runner      # runner local (webhook + pipeline contra sidecar)
├── Dockerfile.gpu         # sidecar Python ML (torch/diffusers)
├── domain.py              # tipos probados ML + Result + errores (sin mitad gateway)
├── gnm.py                 # template + heatmap + PNG + zip CPU compartido
├── gnm_head.py            # cabeza GNM real 17821/35324/253/68 + 5 islas reales
├── gnm_fit.py             # fit real ridge identidad + camara, expresion neutra
├── gnm_texture.py         # proyeccion foto resize 512 + warp + inpaint solo ocluidas
├── gnm_assemble.py        # 5 islas + PBR + GLB personalizado + zip 7 nombres
├── pipeline_local.py      # orquestador local con sink (report/complete/fail) + timeouts
├── local_runner.py        # webhook de queue, blobs por rutas dev, sink HTTP
├── modal_app.py           # sidecar ML: MediaPipe/fit/textura + POST /ml/*
└── tests/                 # test_domain/pipeline/gnm(_fit|_texture|_assemble)/modal_compat
```

```
edge/
├── contract.ts            # fuente unica del contrato HTTP (brands, Result, hitos)
├── worker.ts              # gateway prod (fino: valida + R2 + queue + DO)
├── worker.dev.ts          # entrada dev: envuelve prod + 2 rutas dev + consumer
├── progress-do.ts         # Durable Object de progreso (TTL + WS vivo)
├── worker.http.test.ts    # suite pool: port 1:1 del contrato HTTP en runtime
└── contract.test.ts       # unit del contrato
```

`domain.py` expone `ImageBytes`, `JobId`, `Progress`, `Stage`,
`Landmarks` 478, `GnmCoeffs` 253, `CameraParams` 12, `UvRegion` 1-5, `FitResult`,
`CompleteUv`/`Heatmap` (`UV_LEN`), `BaseUrl`, `CompareResult`.
El ciclo de vida del job (queue, status, TTL) vive solo en `edge/`; Python no lo duplica.
Ningún otro módulo importa `torch/diffusers/mediapipe` salvo `modal_app.py`.

## 4. Docker (solo dev local, no CI ni prod)

Docker es solo para paridad local.
No se usa en CI ni en prod.
Cloudflare (Workers + Pages) y Modal no ejecutan imágenes Docker.
Modal solo reusa `backend/Dockerfile.gpu` como receta de build vía `modal.Image.from_dockerfile`.

### 4.1 Full stack local

```bash
docker compose up --build
```

Levanta `api` (worker runtime, misma entrada que prod) en 8000, `runner` en 8001, `ml-sidecar` en 8081 y `frontend` en 4321.
`/tmp` está montado como `tmpfs` en runner y sidecar para stateless.
En prod este stack se reemplaza por `Cloudflare Workers + Queues + R2 + Modal` con el mismo handler.
El estado del worker dev persiste en el volumen `wrangler-state` (efimero en tests).

### 4.2 Workers GPU local

```bash
docker compose --profile gpu up --build
```

Usa `Dockerfile.gpu` con `nvidia/cuda:12.6-runtime`.
Verifica `docker exec vultus-worker-gpu nvidia-smi`.
En prod los workers GPU corren en `Modal` (`modal deploy backend/modal_app.py`) con `T4 16GB`, `cold start 1-2s`, `$30/mes free`.

### 4.3 Workers GPU en Modal (prod)

```bash
modal deploy backend/modal_app.py   # despliega MediaPipe/fit/textura/assemble en Modal
modal app logs vultus-workers        # logs GPU
```

Modal escala `0 -> 100` GPUs, paga por segundo. Ver `ARCHITECTURE.md` ADR-004.

### 4.4 Edge en Cloudflare (prod)

```bash
npx wrangler dev -c wrangler.dev.toml  # Workers API + Queues + R2 + DO local
npx --yes wrangler@4 deploy --env production  # Worker prod (CD lo hace solo)
API_URL=https://api.vultus.esau.com.mx bash scripts/smoke-prod.sh
```

CD en `.github/workflows/cd.yml`: PR `main` -> `production` y el merge despliega Worker `--env production` + Pages `vultus`, humo edge, luego `modal deploy`, humo final.
`main` es integracion y solo corre CI.
Preview usa `--env preview` con bucket y queue aislados, sin dominio custom.
Config en `wrangler.toml`. Queues `10k ops/día free`, R2 `10GB free`, Pages free.

### 4.5 Rebuild rápido local

```bash
docker compose build api
docker compose up -d api
```

## 5. Frontend Astro

```bash
cd frontend
npm install
npm run dev
```

Frontend en `http://localhost:4321`.
Build con `npm run build` y preview con `npm run preview`.
Islas React en `src/components`.

## 6. Queues y workers

El contrato es el sink estrecho en `backend/pipeline_local.py` (`report`, `complete`, `fail`):

- **Local/dev/test:** el runner (`backend/local_runner.py`) recibe el mensaje dev por webhook, trae los blobs por las rutas dev del gateway, corre `run_pair` contra el sidecar y reporta por HTTP. Los tests inyectan el sink en memoria.
- **Prod:** los workers GPU consumen vía `HTTP Pull Consumer`, reportan progreso al mismo seam `POST /v1/jobs/{id}/progress` y escriben `result.zip` a R2. El progreso va por `Durable Objects WS` con `Stage` enum.

El pipeline no conoce la infra; solo el sink (`InMemorySink` vs `HttpProgressSink`) decide.

## 7. Testing

### 7.1 Backend

```bash
pytest backend/tests -q
pytest backend/tests/test_pipeline.py -q
pytest backend/tests/test_domain.py -q
pytest backend/tests/test_gnm.py -q
```

```bash
cd frontend
npx vitest run ../edge/contract.test.ts --root ..
npx vitest run --config vitest.pool.config.ts
```

La suite es una sola: contrato TS (fuente unica) + pool HTTP en runtime worker + ML/compute Python.
Seam 1 con suite pool real (`202 {job_id, status queued}`, `400` imagen / faltante / uuid, `404` desconocido, `409` pre-done, `health` con `gateway:"worker"`) mas WS real (snapshot `queued`, handshake falla en desconocido) y negativo del backdoor dev (`404` con vars prod).
Seam 2 con sink en memoria (`report` ordenado `fit/texture/assemble`, `complete` guarda, `fail` marca).
Seam 3 con golden `UV_LEN = 786432` (`[10,200] vs [4,210] -> [6,10]`, `GLB magic` personalizado) + `Landmarks` 478 rechaza stubs.
Goldens literales a mano para `Progress`, `JobId`, heatmap, albedo y coefs.
No mockees el pipeline interno.
Valor esperado es literal golden, no recomputado.
Regenerar bin: `python3 scripts/extract_gnm_template.py --check` (sin `--check` escribe el bin).
Gate real: `python3 scripts/e2e-gnm-real.py` (LFW Bush misma/distinta con margen).
Goldens LFW congelados por sha256.
CHECK 5 fija la orientacion V: ancla nariz (dist < 60) + evidencia >= 0.15.
Diag de camara: `python3 scripts/render_diag.py --photo <jpg> --out <png>`.
No commitear JPEGs LFW.

### 7.2 Frontend E2E

```bash
cd frontend
npm run test:e2e
```

Playwright contra `http://localhost:4321` con API real vía `docker compose`.
`e2e/compare.spec.ts` sube 2 PNG mínimos vía `setInputFiles` y espera `job ... queued`.
E2E stack en CI: `docker compose up -d` + `bash scripts/smoke-fase0.sh` (health + frontend + `POST 202` + `GET status queued`).

### 7.3 Stateless check

```bash
pytest backend/tests/test_pipeline.py -q
```

Verifica `tmpfs` vacío tras cada par (`job_dir` no existe) y TTL canónico en el DO (`ttl_secs == 60` en `/health`).

## 8. GPU sin hardware local

Si no tienes GPU local, corre `pytest backend/tests -q` (CPU puro con dobles deterministas).
Inyecta `MlSidecarClient::new(BaseUrl::parse("http://localhost:8081"))` fake que retorna `CompleteUv` golden sin cargar `torch`.
En CI los workers GPU corren solo en runner con GPU o se skippean.
En prod usa `Modal` para fit/textura GPU sin hardware local y consume tus `$30/mes free` (~50h T4).

## 9. Lint y formato

```bash
mypy --strict --explicit-package-bases --namespace-packages backend/domain.py backend/gnm.py backend/pipeline_local.py backend/local_runner.py backend/gnm_fit.py backend/gnm_texture.py backend/gnm_assemble.py
ruff check backend/
pytest backend/tests -q
```

Sidecar Python con `ruff check backend/modal_app.py` si lo tocas.
Frontend con `npm run lint` y `npm run format`.

## 10. Flujo de trabajo

Crea rama desde `main`.
Implementa un vertical slice `1 test RED -> 1 implementación GREEN` por seam.
No mezcles refactor en el loop.
Commitea `Cargo.toml` y `Cargo.lock` juntos.
Abre PR y verifica `docker compose up` + `pytest backend/tests -q` pasan E2E.

## 11. Troubleshooting

`docker build` falla: reintenta `docker compose build api runner` (gateway worker + runner Python).
`redis connection refused`: doc vieja, ya no aplica. Nunca hubo `Redis`: el estado vive en el DO/R2 (prod) o emulado (dev). Verifica `/health` y `ttl_secs`.
`wrangler deploy` falla (prod): verifica `wrangler.toml` bindings de Queues/R2 y `CLOUDFLARE_API_TOKEN`.
`modal deploy` falla: verifica `modal token` y `modal_app.py` image con `nvidia/cuda:12.6-runtime`.
`CUDA out of memory` (local o Modal): baja `concurrency_limit` a 1 en `texture_worker` / `fit_worker` (`modal_app.py`).
`Ml::Decode` en `landmarks/fit/texture`: verifica `FitRequest`/`TextureRequest` y `UV_LEN`.
`WS no conecta`: verifica `VITE_API_URL` en `frontend/.env` y `Durable Objects` binding en `wrangler.toml` (prod) o `wrangler.dev.toml` (local).
`Queues 128KB exceeded`: no encoles bytes, el worker solo manda `{job_id, r2_keys jobs/{id}/a|b}`.
`Pool suite aislada falla en WS`: corre con la config commiteada (`isolatedStorage:false`), el DO retiene storage con timers vivos.
