# Vultus — Comparador visual forense en espacio canónico

> Gateway único TS (`edge/`) + compute Python (`backend/`) + frontend Astro (`frontend/`).
> `CONTRACT_VERSION = 3` · zip-8 sin heatmap · FLAME 5023 · TTL 60 · stateless.

![contract](https://img.shields.io/badge/contract-v3%20zip--8-blue) ![mesh](https://img.shields.io/badge/mesh-FLAME%205023-green) ![ttl](https://img.shields.io/badge/TTL-60s-orange) ![infra](https://img.shields.io/badge/infra-Cloudflare%20%2B%20Modal-lightgrey)

Vultus normaliza 2 caras a un espacio UV canónico y permite comparación pixel a pixel invariante a pose y expresión. Stateless por diseño: no persisten imágenes ni resultados tras la entrega.

> **Infra elegida: Cloudflare + Modal.** Edge en Cloudflare (Pages + Workers + Queues + R2 + Durable Objects) y GPU serverless en Modal. Coste fijo ~$5/mes + GPU por segundo ($30/mes free en Modal). Ver `ARCHITECTURE.md` ADR-004.

## Quick start

### Requisitos

Docker 24+ y Compose v2, Python 3.12+ (3.10 es el piso Modal), Node 22+. Para GPU local `nvidia-container-toolkit`. Sin GPU corren validación y tests de Seam 1 y 2. Sin toolchain Rust.

### Backend Python (compute + ML local)

```bash
pip install -r backend/requirements-api.txt
pytest backend/tests -q
```

Gateway local en `http://localhost:8000` vía `npx wrangler dev -c wrangler.dev.toml` (`/health` con `gateway:"worker"`). Sidecar ML en `:8081`:

```bash
python3 -c "import sys; sys.path.insert(0,'.'); import backend.modal_app; import uvicorn; uvicorn.run(backend.modal_app.sidecar, port=8081)"
```

`ML_SIDECAR_URL=http://localhost:8081` lo consume `MlSidecarClient(BaseUrl)`: `landmarks → Landmarks(478)`, `fit → FitResult(253 + cámara)`, `texture → RenderedImage(UV_LEN)`.

### Full stack con Docker (dev local)

```bash
docker compose up --build
```

| Servicio | Puerto | Qué es |
|---|---|---|
| `api` | 8000 | Worker gateway (misma entrada que prod) |
| `runner` | 8001 | webhook queue + sink HTTP (`tmpfs /tmp`) |
| `ml-sidecar` | 8081 | MediaPipe / fit / textura (`tmpfs /tmp`, bloque GPU comentado por defecto) |
| `frontend` | 4321 | Astro static (`VITE_API_URL=http://localhost:8000`) |

Sin Redis: R2/Queue/DO emulados en dev, `Queues + R2 + Modal` en prod, mismo contrato (`{job_id, r2_keys}`).

### Frontend Astro

```bash
cd frontend && npm install && npm run dev   # :4321
```

Prod: static en Cloudflare Pages. Visores: UV canónico + 3D estudio blanco (ver `DESIGN.md`).

### Prod (solo release, el CD lo hace)

```bash
npx --yes wrangler@4 deploy --env production   # edge (ver wrangler.toml)
modal deploy backend/modal_app.py              # GPU (nunca modal serve: roba la queue prod)
```

`main` = integración (solo CI). Push a `production` = edge preview. Tag `v*.*.*` = edge prod + Pages + Modal + Release.

## Estructura

```
vultus/
├── README.md / ARCHITECTURE.md / CONTEXT.md / DESIGN.md /
│   DEVELOPMENT.md / PIPELINE.md / ROADMAP.md
├── wrangler.toml / wrangler.dev.toml / docker-compose.yml / pyproject.toml
├── edge/
│   ├── contract.ts      # FUENTE ÚNICA HTTP (brands, Result, ZIP_MANIFEST zip-8, CONTRACT_VERSION=3)
│   ├── worker.ts        # gateway prod (valida + R2 + queue + DO)
│   ├── worker.dev.ts    # dev: prod + GET /dev/blobs + PUT /dev/results + consumer webhook
│   ├── progress-do.ts   # ProgressDO (TTL 60 + purga 2xTTL + WS vivo)
│   └── worker.http.test.ts / contract.test.ts   # pool runtime real + unit
├── backend/
│   ├── domain.py        # tipos probados + Result + errores (espejo local del contrato)
│   ├── pipeline_local.py# orquestador run_pair + wire v2 + sink report/complete/fail
│   ├── local_runner.py  # webhook :8001 + HttpProgressSink
│   ├── gnm_assemble.py  # FLAME 5023 + piel/ojos + PBR + GLB + zip-8
│   ├── gnm.py           # uv_to_png + build_result_zip CPU
│   ├── flame_fit.py / flame_texture.py / deep3d.py / dpr.py / texgan.py
│   ├── modal_app.py     # sidecar /ml/* + workers GPU + pull consumer + v3_figure_worker
│   ├── v3_*.py          # track denso paralelo (20481/1024, sin tocar forense)
│   └── tests/           # 32 módulos (domain, pipeline, gnm*, modal_compat, v3_*_real, bordes)
└── frontend/src/
    ├── pages/index.astro                 # form A/B + flujo POST→WS→GET
    └── components/compare-client.ts      # API_PATHS + RESULT_FILES zip-8 + CONTRACT_VERSION=3
        three-rig.ts / ThreeViewer.astro / UvViewers.astro
```

> No existen `backend/store.py` ni `backend/app.py`: el ciclo vive solo en `edge/`, Python no lo duplica.

## Flujo

```mermaid
graph LR
    U[2 fotos] --> W[Worker POST /v1/compare]
    W --> Q[Queues {job_id, r2_keys}]
    Q --> M[Modal: MediaPipe → fit DECA → textura → assemble]
    M --> R[R2 result.zip 60s]
    R --> D[StreamingResponse zip-8 + visor estudio]
    D -. olvido .-> X[tmpfs wipe + DEL + purga]
```

Usuario sube 2 jpg/png en Pages → Worker valida (8 MiB, magic JPEG/PNG) y encola puntero → GPU procesa en paralelo por cara → `result.zip` con TTL 60 → Worker lo sirve en streaming → todo se borra (R2 lifecycle + purga DO 2xTTL + `unlink /tmp`).

## Documentación

| Doc | Qué responde |
|---|---|
| `ARCHITECTURE.md` | forma del sistema: módulos, seams, ADRs 001-012 |
| `CONTEXT.md` | vocabulario: tipos, reglas, dueños, errores |
| `DESIGN.md` | sistema visual real: colores, tipo, estudio blanco, estados |
| `DEVELOPMENT.md` | guía dev: setup, Docker, edge/Modal, tests, troubleshooting |
| `PIPELINE.md` | flujo: diagramas, etapas, wire v2, pesos y gates |
| `ROADMAP.md` | fases 0-5 + track v3, deploy, riesgos, próximos pasos |

## API

| Ruta | Respuesta |
|---|---|
| `GET /health` | `{status:"ok", gateway:"worker", ttl_secs:60, contract_version:3}` |
| `POST /v1/compare` | `202 {job_id, status:"queued"}` · `400` validación · `500` bindings |
| `GET /v1/jobs/{id}` | `200 {job_id, status}` · `400` uuid roto · `404` desconocido |
| `GET /v1/jobs/{id}/result` | zip-8 en streaming · `404/409` reintentable · `404` tras purga |
| `POST /v1/jobs/{id}/progress` | sink prod (workers GPU) |
| `WS /v1/jobs/{id}/events` | `queued → processing → done\|failed\|expired`, hitos `0.40/0.75/0.95/1.0` |
| `GET /v3/jobs/{id}/figure` | `figure-v3.zip` (track denso, retención días) |

Zip-8 en orden canónico: `uv_a/b.png · mesh_a/b.glb · pbr_a/b.png · render_a/b.png`. Sin heatmap (ADR-008). Errores `{"detail":…}`. Frontend valida `contract_version` y muestra `actualiza` ante mismatch.

## Stateless

Sin Postgres ni S3 persistente. Prod: R2 60 s (lifecycle `jobs/`) + Queues retención 24 h con TTL lógico 60 vía alarm + `/tmp` tmpfs en Modal. Dev: DO TTL 60 + purga 2xTTL + tmpfs. v3 (única excepción): retención de días en cola separada. Logs con `job_id`, sin bytes. Ver `PIPELINE.md §5.8` y ADR-003.

## Testing

```bash
pytest backend/tests -q
cd frontend && npx vitest run ../edge/contract.test.ts --root ..
cd frontend && npx vitest run --config vitest.pool.config.ts
cd frontend && npm run test:e2e
```

Seam 1: pool en runtime real (`202`, `400`, `404`, `409`, `health`, WS, backdoor dev). Seam 2: sink en memoria. Seam 3: golden `UV_LEN = 786432` (fit v2 + GLB FLAME + cero sentinel + ojos 1092). Gates reales: `scripts/e2e-flame-real.py` (margen + SSIM con pesos) y `scripts/modal-weights-sync.sh --check` (puente + cutover + backup). Pesos y LFW nunca se commitean.

## Licencia

Apache 2.0 para código propio. Modelos con licencias de sus repos originales.
