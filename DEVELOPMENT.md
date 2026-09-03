# DEVELOPMENT - Guía de Desarrollo

## 1. Requisitos

Instala `uv` 0.4+ con `curl -LsSf https://astral.sh/uv/install.sh | sh`.
Instala `Docker` 24+ y `Docker Compose` v2.
Para GPU instala `nvidia-container-toolkit` y verifica con `nvidia-smi`.
Node 20+ para frontend Astro.

## 2. Setup con uv

```bash
cd backend
uv sync --frozen
uv run pytest
```

`uv sync` crea `.venv` y instala desde `pyproject.toml` y `uv.lock`.
Nunca uses `pip` directo.
Para añadir dependencia usa `uv add fastapi` y commitea `pyproject.toml` y `uv.lock`.

## 3. Estructura backend

```
backend/
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── Dockerfile.gpu
├── app/
│   ├── main.py
│   ├── api/
│   ├── workers/
│   ├── models/
│   └── core/
└── tests/
    ├── api/
    └── workers/
```

`app/models` son adaptadores a `mediapipe`, `torch`, `diffusers`, `gnm`.
Ningún otro módulo importa esas libs.

## 4. Docker

### 4.1 Full stack

```bash
docker compose up --build
```

Levanta `api` en 8000, `frontend` en 4321 y `redis` en 6379.
`/tmp` está montado como `tmpfs` para stateless.

### 4.2 Workers GPU

```bash
docker compose --profile gpu up --build
```

Usa `Dockerfile.gpu` con `nvidia/cuda:12.2-runtime`.
Verifica `docker exec facium-worker-gpu nvidia-smi`.

### 4.3 Rebuild rápido

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

ARQ corre con `uv run arq app.core.queue.WorkerSettings`.
El API encola con `await queue.enqueue_job("compare_job", job_id, image_a, image_b)`.
Progreso se publica con `ctx["job"].progress`.
En dev usa Redis de `docker compose`, en test usa `fakeredis`.

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
`redis connection refused`: verifica `docker compose ps` y `redis` healthy.
`CUDA out of memory`: baja `concurrency` de worker-gpu a 1 en `WorkerSettings`.
`WS no conecta`: verifica `VITE_API_URL` en `frontend/.env`.
