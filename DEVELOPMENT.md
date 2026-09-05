# DEVELOPMENT - Guía de Desarrollo

## 1. Requisitos

Instala `uv` 0.4+ con `curl -LsSf https://astral.sh/uv/install.sh | sh`.
Instala `Docker` 24+ y `Docker Compose` v2.
Para GPU instala `nvidia-container-toolkit` y verifica con `nvidia-smi`.
Node 20+ para frontend Astro.

## 2. Setup híbrido

```bash
cd backend
cargo build
cargo test
cargo run -p vultus-api
```

API en `http://localhost:8000` (`/health`, `POST /v1/compare`).
Sidecar ML local (sin GPU real, stubs hasta Fase 1):

```bash
pip install fastapi uvicorn
python3 -c "import sys; sys.path.insert(0,'backend'); import modal_app; import uvicorn; uvicorn.run(modal_app.sidecar, port=8081)"
```

`ML_SIDECAR_URL=http://localhost:8081` lo consume `MlSidecarClient`.
Nunca uses `pip` para la API: la API es Rust.

## 3. Estructura backend

```
backend/
├── Cargo.toml               # workspace Rust (api, core, workers_cpu)
├── Cargo.lock               # versionado para builds reproducibles
├── Dockerfile               # binario Rust vultus-api
├── Dockerfile.gpu           # sidecar Python ML (torch/diffusers)
├── modal_app.py             # sidecar ML: MediaPipe/FLAME/FreeUV + POST /ml/*
├── crates/
│   ├── api/                 # Seam 1 Axum (shallow)
│   ├── core/                # queue trait + MlSidecarClient + tipos (deep)
│   └── workers_cpu/         # bake + heatmap + report (deep, CPU puro)
└── tests/                   # Seam 1 en crates/api/tests, Seam 3 golden Fase 1
```

`app/models` son adaptadores a `mediapipe`, `torch`, `diffusers`, `gnm`.
Ningún otro módulo importa esas libs.

## 4. Docker (dev local)

### 4.1 Full stack local

```bash
docker compose up --build
```

Levanta `api` en 8000, `frontend` en 4321 y `redis` en 6379.
`/tmp` está montado como `tmpfs` para stateless.
En prod este stack se reemplaza por `Cloudflare Workers + Queues + R2 + Modal`. El contrato `core.queue` es idéntico, solo cambia el adapter.

### 4.2 Workers GPU local

```bash
docker compose --profile gpu up --build
```

Usa `Dockerfile.gpu` con `nvidia/cuda:12.2-runtime`.
Verifica `docker exec facium-worker-gpu nvidia-smi`.
En prod los workers GPU corren en `Modal` (`modal deploy backend/modal_app.py`) con `T4 16GB`, `cold start 1-2s`, `$30/mes free`.

### 4.3 Workers GPU en Modal (prod)

```bash
modal deploy backend/modal_app.py   # despliega MediaPipe/FLAME/FreeUV/GNM en Modal
modal app logs facium-workers        # logs GPU
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

El contrato es `app.core.queue` con adapter dual:

- **Local/dev/test:** `Redis + ARQ` con `uv run arq app.core.queue.WorkerSettings`. El API encola con `await queue.enqueue_job("compare_job", job_id, image_a, image_b)`. Progreso con `ctx["job"].progress`. En test usa `fakeredis`.
- **Prod:** `Cloudflare Queues + R2` vía `wrangler.toml`. El Worker encola `{job_id, r2_keys}` (Queues <128KB, bytes en R2). Modal consume vía `HTTP Pull Consumer` (`modal_app.py`). Progreso vía `Durable Objects WS`.

El código de negocio no conoce la infra; solo `core.queue` decide por env `QUEUE_DRIVER=redis|cloudflare`.

## 7. Testing

### 7.1 Backend

```bash
cd backend
uv run pytest -k seam1
uv run pytest -k seam3
uv run pytest --cov=app
```

Seam 1 con `httpx.AsyncClient`.
Seam 2 con `fakeredis`.
Seam 3 con golden images en `tests/fixtures`.
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
uv run pytest -k test_compare_does_not_persist
```

Verifica `redis.exists == 0` tras 65s y `tmpfs` vacío.

## 8. GPU sin hardware local

Si no tienes GPU local, corre tests de Seam 3 con mocks de boundary.
Inyecta `uv_client` fake que retorna `GOLDEN_UV` sin cargar `torch`.
En CI los workers GPU corren solo en runner con GPU o se skippean con `pytest -k "not gpu"`.
En prod usa `Modal`: `modal run backend/modal_app.py::test_facium --gpu T4` ejecuta FreeUV real sin GPU local y consume tus `$30/mes free` (~50h T4).

## 9. Lint y formato

```bash
cd backend
uv run ruff check app
uv run ruff format app
uv run mypy app
```

Frontend con `npm run lint` y `npm run format`.

## 10. Flujo de trabajo

Crea rama desde `main`.
Implementa un vertical slice `1 test RED -> 1 implementación GREEN` por seam.
No mezcles refactor en el loop.
Commitea `pyproject.toml` y `uv.lock` juntos.
Abre PR y verifica `docker compose up` pasa E2E.

## 11. Troubleshooting

`uv sync` falla: borra `.venv` y reintenta `uv sync --frozen`.
`redis connection refused` (local): verifica `docker compose ps` y `redis` healthy.
`wrangler deploy` falla (prod): verifica `wrangler.toml` bindings de Queues/R2 y `CLOUDFLARE_API_TOKEN`.
`modal deploy` falla: verifica `modal token` y `modal_app.py` image con `nvidia/cuda:12.2-runtime`.
`CUDA out of memory` (local o Modal): baja `concurrency` de worker-gpu a 1 en `WorkerSettings` / `@modal.function(gpu="T4", concurrency_limit=1)`.
`WS no conecta`: verifica `VITE_API_URL` en `frontend/.env` y `Durable Objects` binding en `wrangler.toml` (prod).
`Queues 128KB exceeded`: no encoles bytes, usa patrón `R2 pointer` via `core.queue` adapter.
