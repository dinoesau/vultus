# DEVELOPMENT - Guía de Desarrollo

## 1. Requisitos

Instala Rust estable (`rustup`, `cargo build`, `cargo test`, `cargo clippy`, `cargo fmt`).
Instala `Docker` 24+ y `Docker Compose` v2.
Para GPU instala `nvidia-container-toolkit` y verifica con `nvidia-smi`.
Node 20+ para frontend Astro.
Python 3.12+ solo para sidecar ML (`backend/modal_app.py`, `Dockerfile.gpu`).
Workspace deps: `anyhow` (errores en `main`), `nutype` (`TtlSecs`), `proptest` (dev).

## 2. Setup híbrido

```bash
cd backend
cargo build
cargo test
cargo run -p vultus-api
```

API en `http://localhost:8000` (`/health`, `POST /v1/compare` -> `202 {job_id, status:"queued"}`, `GET /v1/jobs/{id}` -> `200 {job_id, status}`).
`main` retorna `anyhow::Result` con `context` en `bind` y `serve`.
Sidecar ML local (stubs `{"todo":...}` hasta Fase 1, `gnm_bake_worker` deprecated):

```bash
pip install fastapi uvicorn
python3 -c "import sys; sys.path.insert(0,'backend'); import modal_app; import uvicorn; uvicorn.run(modal_app.sidecar, port=8081)"
```

`ML_SIDECAR_URL=http://localhost:8081` lo consume `MlSidecarClient::new(BaseUrl::parse(url))`.
`BaseUrl` exige `http(s)://` y recorta `/` final.
Nunca uses `pip` para la API: la API es Rust (`cargo`).

## 3. Estructura backend

```
backend/
├── Cargo.toml               # workspace Rust (api, core, workers_cpu) + anyhow/nutype/proptest/utoipa
├── Cargo.lock               # versionado para builds reproducibles
├── Dockerfile               # binario Rust vultus-api (ML_SIDECAR_URL)
├── Dockerfile.gpu           # sidecar Python ML (torch/diffusers)
├── modal_app.py             # sidecar ML: MediaPipe/FLAME/FreeUV + POST /ml/* (stubs Fase 1, gnm deprecated)
├── crates/
│   ├── api/                 # Seam 1 Axum (AppError 400/404/500, AppState Arc<dyn Queue>) + tests/seam1.rs (8 tests)
│   ├── core/                # assert + error taxonómico + job tipado + ml tipado + queue dual (deep)
│   └── workers_cpu/         # bake + heatmap infallibles (deep, CPU puro, sin dep image)
```

`crates/core` expone `ImageBytes` + `ImageBytesRef`, `JobId`, `JobStatus::as_str`, `Progress::zero`, `Stage`, `TtlSecs`, `Job<State>`, `R2Key` / `R2Keys`, `EnqueueCommand`, `EnqueuedJob`, `Landmarks` 478, `FlawUv` / `CompleteUv` / `Heatmap` (`UV_LEN`), `BaseUrl`, `FlamePayload`, `MlSidecarClient`, `Queue` + `MemoryQueue` + `R2PointerQueue`.
Ningún otro módulo importa `torch/diffusers/mediapipe`.

## 4. Docker (dev local)

### 4.1 Full stack local

```bash
docker compose up --build
```

Levanta `api` en 8000, `frontend` en 4321 y `redis` en 6379.
`/tmp` está montado como `tmpfs` para stateless.
En prod este stack se reemplaza por `Cloudflare Workers + Queues + R2 + Modal`. El trait `vultus-core::Queue` (`MemoryQueue` local / `R2PointerQueue` prod) es idéntico, solo cambia el adapter (`EnqueuedJob::is_r2_pointer()`).

### 4.2 Workers GPU local

```bash
docker compose --profile gpu up --build
```

Usa `Dockerfile.gpu` con `nvidia/cuda:12.2-runtime`.
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

El contrato es trait `vultus_core::Queue` con `Store` compartido (`HashMap<JobId, MemoryEntry>` tras `tokio::RwLock`):

- **Local/dev/test:** `MemoryQueue` (`enqueue(EnqueueCommand) -> EnqueuedJob{r2_keys None}`, `stored_lens` guarda `(len_a, len_b)`, `progress -> (Progress::zero(), Stage::Queued)`, `set_progress(Progress, Stage)` pasa a `Processing`) y `R2PointerQueue` (mismo `Store`, retorna `Some(R2Keys jobs/{id}/a|b)` para paridad prod, probada en `test_r2_pointer_queue_serves_same_seam`).
- **Prod:** `Cloudflare Queues + R2` vía `wrangler.toml`. El Worker encola `{job_id, r2_keys}` (Queues <128KB, bytes en R2, `R2Key` sin `..` max 1024). Modal consume vía `HTTP Pull Consumer` (`modal_app.py`: `mediapipe_worker`, `flame_worker`, `freeuv_worker`, `queue_pull_consumer`; `gnm_bake_worker` deprecated). Progreso vía `Durable Objects WS` con `Stage` enum.

El código de negocio no conoce la infra; solo el adapter (`MemoryQueue` vs `R2PointerQueue`) decide.
`AppState::new(impl Queue)` inyecta cualquiera tras `Arc<dyn Queue>`.

## 7. Testing

### 7.1 Backend

```bash
cd backend
cargo test
cargo test -p vultus-api --test seam1
cargo test -p vultus-core job:: -- --nocapture
cargo test -p vultus-workers-cpu
```

36 tests en verde (`8 seam1 + 25 core + 3 workers_cpu`).
Seam 1 con `axum-test::TestServer` (`202 {job_id, status queued}`, `GET` queued, paridad `R2PointerQueue`, `400` imagen / faltante / uuid, `404` desconocido).
Seam 2 con `MemoryQueue` / `R2PointerQueue` (`stored_lens`, `progress`, `NotFound`).
Seam 3 con golden `UV_LEN = 786432` (`black_heatmap`, `[10,200] vs [4,210] -> [6,10]`, `wrong_uv_length_rejected_at_parse`) + `Landmarks` 478 rechaza stubs.
`proptest` para `parse_never_panics`, `Progress`, `TtlSecs 1..=3600`, `R2Key`.
No mockees `fit_flame` interno.
Valor esperado es literal golden, no recomputado.

### 7.2 Frontend E2E

```bash
cd frontend
npm run test:e2e
```

Playwright contra `http://localhost:4321` con API real vía `docker compose`.

### 7.3 Stateless check

```bash
cargo test stored_lens
cargo test unknown_job_is_not_found
```

Verifica `stored_lens == (64,64)` en fixture, `NotFound` en job desconocido y `tmpfs` vacío.
TTL canónico `TtlSecs::default() == 60` (`parse(0)` y `parse(3601)` fallan).

## 8. GPU sin hardware local

Si no tienes GPU local, corre `cargo test` (CPU puro, sidecar con stubs `{"todo":...}` rechazados por `Landmarks` / `UV_LEN` hasta Fase 1).
Inyecta `MlSidecarClient::new(BaseUrl::parse("http://localhost:8081"))` fake que retorna `CompleteUv` golden sin cargar `torch`.
En CI los workers GPU corren solo en runner con GPU o se skippean.
En prod usa `Modal`: `modal run backend/modal_app.py::test_vultus --gpu T4` ejecuta FreeUV real sin GPU local y consume tus `$30/mes free` (~50h T4).

## 9. Lint y formato

```bash
cd backend
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

Sidecar Python con `ruff check backend/modal_app.py` si lo tocas.
Frontend con `npm run lint` y `npm run format`.

## 10. Flujo de trabajo

Crea rama desde `main`.
Implementa un vertical slice `1 test RED -> 1 implementación GREEN` por seam.
No mezcles refactor en el loop.
Commitea `Cargo.toml` y `Cargo.lock` juntos.
Abre PR y verifica `docker compose up` + `cargo test` pasan E2E.

## 11. Troubleshooting

`cargo build` falla: borra `target/` y reintenta `cargo build`.
`redis connection refused` (local legacy): el código actual usa `MemoryQueue` / `R2PointerQueue` en memoria, verifica `Store` y `stored_lens`, no Redis.
`wrangler deploy` falla (prod): verifica `wrangler.toml` bindings de Queues/R2 y `CLOUDFLARE_API_TOKEN`.
`modal deploy` falla: verifica `modal token` y `modal_app.py` image con `nvidia/cuda:12.2-runtime`.
`CUDA out of memory` (local o Modal): baja `concurrency_limit` a 1 en `freeuv_worker` / `flame_worker` (`modal_app.py`).
`Ml::Decode` en `landmarks/flame/freeuv`: el sidecar aún retorna stubs `{"todo":...}` (Fase 1 pendiente), verifica `FlamePayload` y `UV_LEN`.
`WS no conecta`: verifica `VITE_API_URL` en `frontend/.env` y `Durable Objects` binding en `wrangler.toml` (prod).
`Queues 128KB exceeded`: no encoles bytes, usa `EnqueueCommand` + `R2Keys jobs/{id}/a|b` vía `R2PointerQueue`.
