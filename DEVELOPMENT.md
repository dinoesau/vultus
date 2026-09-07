# DEVELOPMENT - Guía de Desarrollo

> Estado objetivo sin Rust: un solo compose sin toolchain Rust.
> Comandos nuevos: `pip install -r backend/requirements-api.txt`, `mypy --strict backend/`,
> `ruff check backend/`, `pytest backend/tests -q`, `npx vitest run edge/contract.test.ts`.

## 1. Requisitos

Sin toolchain Rust. Solo Python + Node + Docker.
Instala `Docker` 24+ y `Docker Compose` v2.
Para GPU instala `nvidia-container-toolkit` y verifica con `nvidia-smi`.
Node 22+ para frontend Astro (Astro 7 exige 20.19+ o 22.12+).
Python 3.12+ para API local y sidecar ML (`backend/app.py`, `backend/modal_app.py`, `Dockerfile.api`, `Dockerfile.gpu`).
Historico: antes Rust Axum (ver ADR), borrado en plan-remove-rust-stack.

## 2. Setup Python

```bash
pip install -r backend/requirements-api.txt
mypy --strict --explicit-package-bases --namespace-packages backend/domain.py backend/store.py backend/gnm.py backend/pipeline_local.py backend/app.py
ruff check backend/
pytest backend/tests -q
```

API en `http://localhost:8000` (`/health`, `POST /v1/compare` -> `202 {job_id, status:"queued"}`, `GET /v1/jobs/{id}` -> `200 {job_id, status}`).
Sidecar ML local en `:8081` vía `modal_app.sidecar`.

```bash
python3 -c "import sys; sys.path.insert(0,'.'); import backend.modal_app; import uvicorn; uvicorn.run(backend.modal_app.sidecar, port=8081)"
```

`ML_SIDECAR_URL=http://localhost:8081` lo consume `MlSidecarClient(BaseUrl)`.
`BaseUrl` exige `http(s)://` y recorta `/` final.

## 3. Estructura backend

```
backend/
├── requirements-api.txt   # deps API local (fastapi/uvicorn/httpx/pytest/mypy/ruff/Pillow)
├── Dockerfile.api         # imagen Python vultus-api (ML_SIDECAR_URL)
├── Dockerfile.gpu         # sidecar Python ML (torch/diffusers)
├── domain.py              # tipos probados + Result + errores (dueno local)
├── store.py               # cola en memoria TTL60 + reloj inyectable
├── gnm.py                 # bake + heatmap + GLB + zip CPU compartido
├── pipeline_local.py      # orquestador local paralelo + timeouts
├── app.py                 # API FastAPI (Seam 1)
├── modal_app.py           # sidecar ML: MediaPipe/FLAME/FreeUV + POST /ml/*
└── tests/                 # test_domain/store/api/pipeline/gnm (29 tests)
```

`domain.py` expone `ImageBytes`, `JobId`, `JobStatus`, `Progress`, `Stage`, `TtlSecs`,
`R2Key`/`R2Keys`, `EnqueueCommand`, `EnqueuedJob`, `Landmarks` 478,
`FlawUv`/`CompleteUv`/`Heatmap` (`UV_LEN`), `BaseUrl`, `FlamePayload`, `CompareResult`.
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

Levanta `api` en 8000, `ml-sidecar` en 8081 y `frontend` en 4321 (sin `redis`; `Store` en memoria).
`/tmp` está montado como `tmpfs` para stateless.
En prod este stack se reemplaza por `Cloudflare Workers + Queues + R2 + Modal`. El `Store` en `backend/store.py` (`MemoryQueue` local / `R2PointerQueue` prod) es idéntico, solo cambia el adapter (`EnqueuedJob.is_r2_pointer()`).

### 4.2 Workers GPU local

```bash
docker compose --profile gpu up --build
```

Usa `Dockerfile.gpu` con `nvidia/cuda:12.6-runtime`.
Verifica `docker exec vultus-worker-gpu nvidia-smi`.
En prod los workers GPU corren en `Modal` (`modal deploy backend/modal_app.py`) con `T4 16GB`, `cold start 1-2s`, `$30/mes free`.

### 4.3 Workers GPU en Modal (prod)

```bash
modal deploy backend/modal_app.py   # despliega MediaPipe/FLAME/FreeUV/GNM en Modal
modal app logs vultus-workers        # logs GPU
```

Modal escala `0 -> 100` GPUs, paga por segundo. Ver `ARCHITECTURE.md` ADR-004.

### 4.4 Edge en Cloudflare (prod)

```bash
npx wrangler dev     # Workers API + Queues + R2 + Durable Objects local
npx wrangler deploy  # Pages (Astro) + Workers prod
```

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

El contrato es `Store` compartido en `backend/store.py` (`dict` tras `RLock`):

- **Local/dev/test:** `MemoryQueue` (`enqueue(EnqueueCommand) -> EnqueuedJob{r2_keys None}`, `stored_lens` guarda `(len_a, len_b)`, `progress -> (Progress::zero(), Stage::Queued)`, `set_progress(Progress, Stage)` pasa a `Processing`) y `R2PointerQueue` (mismo `Store`, retorna `Some(R2Keys jobs/{id}/a|b)` para paridad prod, probada en `test_r2_pointer_queue_serves_same_seam`).
- **Prod:** `Cloudflare Queues + R2` vía `wrangler.toml`. El Worker encola `{job_id, r2_keys}` (Queues <128KB, bytes en R2, `R2Key` sin `..` max 1024). Modal consume vía `HTTP Pull Consumer` (`modal_app.py`: `mediapipe_worker`, `flame_worker`, `freeuv_worker`, `queue_pull_consumer`; `gnm_bake_worker` deprecated). Progreso vía `Durable Objects WS` con `Stage` enum.

El código de negocio no conoce la infra; solo el adapter (`MemoryQueue` vs `R2PointerQueue`) decide.
`AppState::new(impl Queue)` inyecta cualquiera tras `Arc<dyn Queue>`.

## 7. Testing

### 7.1 Backend

```bash
cd backend
pytest backend/tests -q
pytest backend/tests/test_api.py -q
pytest backend/tests/test_domain.py -q
pytest backend/tests/test_gnm.py -q
```

56 tests en verde (`16 api: 2 config + 11 seam1 + 3 ws, 37 core: 32 unit + 5 edge_parity, 3 workers_cpu`).
Seam 1 con `TestClient` real (`202 {job_id, status queued}`, `GET` queued, paridad `R2PointerQueue`, `400` imagen / faltante / uuid, `404` desconocido, `409` pre-done) mas WS real con cliente websocket (snapshot `queued`, `processing/flame` tras `set_progress`, handshake falla en desconocido).
Seam 2 con `MemoryQueue` / `R2PointerQueue` (`stored_lens`, `progress`, `NotFound`).
Seam 3 con golden `UV_LEN = 786432` (`black_heatmap`, `[10,200] vs [4,210] -> [6,10]`, `wrong_uv_length_rejected_at_parse`) + `Landmarks` 478 rechaza stubs.
Goldens literales a mano para `Progress`, `TtlSecs 1..=3600`, `R2Key`, heatmap y bake.
No mockees `fit_flame` interno.
Valor esperado es literal golden, no recomputado.

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
pytest backend/tests/test_store.py -q
```

Verifica `stored_lens == (64,64)` en fixture, `NotFound` en job desconocido y `tmpfs` vacío.
TTL canónico `TtlSecs::default() == 60` (`parse(0)` y `parse(3601)` fallan).

## 8. GPU sin hardware local

Si no tienes GPU local, corre `pytest backend/tests -q` (CPU puro con dobles deterministas).
Inyecta `MlSidecarClient::new(BaseUrl::parse("http://localhost:8081"))` fake que retorna `CompleteUv` golden sin cargar `torch`.
En CI los workers GPU corren solo en runner con GPU o se skippean.
En prod usa `Modal`: `modal run backend/modal_app.py::test_vultus --gpu T4` ejecuta FreeUV real sin GPU local y consume tus `$30/mes free` (~50h T4).

## 9. Lint y formato

```bash
cd backend
mypy --strict --explicit-package-bases --namespace-packages backend/domain.py backend/store.py backend/gnm.py backend/pipeline_local.py backend/app.py
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

`docker build` falla: reintenta `docker compose build api` (imagen Python, sin `target/` Rust).
`redis connection refused`: doc vieja, ya no aplica. El código usa `MemoryQueue` / `R2PointerQueue` en memoria (`Store`), sin `Redis`. Verifica `stored_lens` y `TtlSecs`.
`wrangler deploy` falla (prod): verifica `wrangler.toml` bindings de Queues/R2 y `CLOUDFLARE_API_TOKEN`.
`modal deploy` falla: verifica `modal token` y `modal_app.py` image con `nvidia/cuda:12.6-runtime`.
`CUDA out of memory` (local o Modal): baja `concurrency_limit` a 1 en `freeuv_worker` / `flame_worker` (`modal_app.py`).
`Ml::Decode` en `landmarks/flame/freeuv`: el sidecar aún retorna stubs `{"todo":...}` (Fase 1 pendiente), verifica `FlamePayload` y `UV_LEN`.
`WS no conecta`: verifica `VITE_API_URL` en `frontend/.env` y `Durable Objects` binding en `wrangler.toml` (prod).
`Queues 128KB exceeded`: no encoles bytes, usa `EnqueueCommand` + `R2Keys jobs/{id}/a|b` vía `R2PointerQueue`.
