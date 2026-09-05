# Vultus - Comparador Visual Forense en Espacio Canónico

Vultus normaliza 2 caras a un espacio UV canónico y permite comparación pixel a pixel invariante a pose y expresión.
El sistema es stateless por diseño.
No persistimos imágenes ni resultados tras la entrega.

> **Infra elegida: Cloudflare + Modal.** Edge en Cloudflare (Pages + Workers + Queues + R2) y GPU serverless en Modal. Coste fijo ~$5/mes + GPU por segundo ($30/mes free en Modal).

## Quick Start

### Requisitos

Necesitas `uv` 0.4+, `Docker` 24+ y `Docker Compose` v2.
Para workers GPU necesitas `nvidia-container-toolkit`.
Sin GPU puedes correr solo validación y tests de Seam 1 y 2.

### Backend híbrido (Rust API + Python ML)

```bash
cd backend
cargo test
cargo run -p vultus-api
```

El API queda en `http://localhost:8000`.
Sidecar ML local en `:8081` vía `modal_app.sidecar` (stubs `{"todo":...}` hasta Fase 1, `gnm_bake_worker` deprecated a Rust).
`MlSidecarClient::new(BaseUrl)` tipa `landmarks -> Landmarks`, `flame -> FlawUv`, `freeuv -> CompleteUv`.

### Full stack con Docker (dev local)

```bash
docker compose up --build
```

Servicios: `api` en 8000, `frontend` Astro en 4321, `redis` en 6379.
Para workers GPU usa `docker compose --profile gpu up`.
En local/test se usa `MemoryQueue` (`r2_keys None`) y `R2PointerQueue` (`Some jobs/{id}/a|b`) tras el mismo trait `Queue`. En producción se usa `Cloudflare Queues + R2 + Modal` vía el mismo contrato.

### Frontend Astro

```bash
cd frontend
npm install
npm run dev
```

Frontend en `http://localhost:4321`.
En producción el frontend se despliega en `Cloudflare Pages` (static).

### Infra Cloudflare + Modal (prod)

```bash
# Edge: Cloudflare
npx wrangler deploy          # Workers API + Queues + R2 (ver docs/infra-cloudflare.md)
# GPU: Modal
modal deploy backend/modal_app.py  # Workers GPU FreeUV/FLAME con $30/mes free
```

Ver `ROADMAP.md` sección 7 y `ARCHITECTURE.md` ADR-004 para el diseño híbrido.

## Estructura

```
vultus/
├── README.md
├── ROADMAP.md
├── CONTEXT.md
├── PIPELINE.md
├── ARCHITECTURE.md
├── DEVELOPMENT.md
├── wrangler.toml              # gateway edge fino (prod)
├── backend/
│   ├── Cargo.toml             # workspace Rust + anyhow/nutype/proptest
│   ├── modal_app.py           # sidecar Python ML (MediaPipe/FLAME/FreeUV) + POST /ml/* (stubs Fase 1)
│   ├── crates/
│   │   ├── api/               # Seam 1 Axum (AppError, Arc<dyn Queue>, tests/seam1.rs 8 tests)
│   │   ├── core/              # assert + error + job tipado + ml tipado + queue dual (deep)
│   │   └── workers_cpu/       # bake + heatmap infallibles (deep CPU, sin dep image)
└── frontend/
    ├── astro.config.mjs       # -> Cloudflare Pages en prod
    └── src/
```

## Flujo

Usuario sube 2 jpgs en Astro (Cloudflare Pages).
Worker Cloudflare valida y sube a R2, encola puntero en Cloudflare Queues.
Workers GPU en Modal (`MediaPipe -> FLAME -> FreeUV -> GNM Bake`) consumen vía HTTP Pull Consumer en paralelo por cara.
Resultado vuelve a R2 con TTL 60s (lifecycle) y se entrega como zip vía `StreamingResponse` desde el Worker.
Tras 60s todo se borra de R2 + Queues y `/tmp` del contenedor Modal.
En dev local el flujo es idéntico pero con `Redis ARQ` en vez de `Queues + R2` (adapter).

## Documentación

- `ROADMAP.md` - fases 0 a 5 y plan de entrega.
- `CONTEXT.md` - vocabulario de dominio, tipos opacos y seams TDD.
- `PIPELINE.md` - flujo completo, secuencia de modelos y contratos tipados.
- `ARCHITECTURE.md` - módulos, seams y decisiones de diseño (ADR-001 histórico, ADR-005 híbrido, ADR-006 typestate).
- `DEVELOPMENT.md` - guía de desarrollo con `cargo` y Docker.

## API

`POST /v1/compare` multipart con `image_a` y `image_b` retorna `202 {job_id, status:"queued"}`.
`GET /v1/jobs/{id}` retorna `200 {job_id, status: queued|processing|done|failed|expired}`.
Errores `{"detail":...}`: `400` imagen / faltante / uuid / `R2Key` / `BaseUrl`, `404` job desconocido, `500` queue / ML / invariante.
`WS /v1/jobs/{id}/events` emite `(Progress 0.0-1.0, Stage queued|landmarks|flame|freeuv|bake|done)`.
`GET /v1/jobs/{id}/result` retorna zip con `uv_a.png, uv_b.png, heatmap.png, mesh.glb, report.pdf`.
Ver `ARCHITECTURE.md` para contratos completos.

## Stateless

No hay Postgres ni S3 persistente.
En prod: todo vive en `R2` 60s (lifecycle) + `Cloudflare Queues` 24h retención + `/tmp` tmpfs en Modal.
En dev: `Redis 60s` + `tmpfs` para paridad local.
Logs no contienen bytes de imagen.
Ver `PIPELINE.md` sección 5.8 y `ARCHITECTURE.md` ADR-004 para verificación.

## Testing

```bash
cd backend
cargo test
```

36 tests en verde (`8 seam1 + 25 core + 3 workers_cpu`).
Tests Seam 1 en `backend/crates/api/tests/seam1.rs` con `axum-test::TestServer` y paridad `MemoryQueue` / `R2PointerQueue`.
Dominio en `crates/core` con `proptest` + golden `UV_LEN`.
CPU en `crates/workers_cpu` con golden heatmap.
Frontend E2E con `npm run test:e2e` en `frontend/e2e`.

## Licencia

Apache 2.0 para código propio.
Modelos con licencias de sus repos originales.
