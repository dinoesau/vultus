# Vultus - Comparador Visual Forense en Espacio Canónico

Vultus normaliza 2 caras a un espacio UV canónico y permite comparación pixel a pixel invariante a pose y expresión.
El sistema es stateless por diseño.
No persistimos imágenes ni resultados tras la entrega.

## Quick Start

### Requisitos

Necesitas `uv` 0.4+, `Docker` 24+ y `Docker Compose` v2.
Para workers GPU necesitas `nvidia-container-toolkit`.
Sin GPU puedes correr solo validación y tests de Seam 1 y 2.

### Backend con uv

```bash
cd backend
uv sync --frozen
uv run pytest
uv run fastapi dev app/main.py
```

El API queda en `http://localhost:8000`.
Docs en `http://localhost:8000/docs`.

### Full stack con Docker

```bash
docker compose up --build
```

Servicios: `api` en 8000, `frontend` Astro en 4321, `redis` en 6379.
Para workers GPU usa `docker compose --profile gpu up`.

### Frontend Astro

```bash
cd frontend
npm install
npm run dev
```

Frontend en `http://localhost:4321`.

## Estructura

```
facium/
├── README.md
├── ROADMAP.md
├── CONTEXT.md
├── PIPELINE.md
├── ARCHITECTURE.md
├── DEVELOPMENT.md
├── docker-compose.yml
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   └── tests/
└── frontend/
    ├── astro.config.mjs
    └── src/
```

## Flujo

Usuario sube 2 jpgs en Astro.
FastAPI valida y encola en Redis ARQ.
Workers `MediaPipe -> FLAME -> FreeUV -> GNM Bake` procesan en paralelo por cara.
Resultado vuelve a Redis con TTL 60s y se entrega como zip vía `StreamingResponse`.
Tras 60s todo se borra de Redis y `/tmp`.

## Documentación

- `ROADMAP.md` - fases 0 a 5 y plan de entrega.
- `CONTEXT.md` - vocabulario de dominio y seams TDD.
- `PIPELINE.md` - flujo completo, secuencia de modelos y contratos.
- `ARCHITECTURE.md` - módulos, seams y decisiones de diseño.
- `DEVELOPMENT.md` - guía de desarrollo con `uv` y Docker.

## API

`POST /v1/compare` multipart con `image_a` y `image_b` retorna `202 {job_id}`.
`WS /v1/jobs/{id}/events` emite `progress` 0.0 a 1.0 por etapa.
`GET /v1/jobs/{id}/result` retorna zip con `uv_a.png, uv_b.png, heatmap.png, mesh.glb, report.pdf`.
Ver `ARCHITECTURE.md` para contratos completos.

## Stateless

No hay Postgres ni S3.
Todo vive en Redis 60s y `/tmp` en tmpfs.
Logs no contienen bytes de imagen.
Ver `PIPELINE.md` sección 5.8 para verificación.

## Testing

```bash
cd backend
uv run pytest -k seam1
uv run pytest -k seam3
```

Tests viven en `backend/tests/api` para Seam 1 y `backend/tests/workers` para Seam 3.
Frontend E2E con `npm run test:e2e` en `frontend/e2e`.

## Licencia

Apache 2.0 para código propio.
Modelos con licencias de sus repos originales.
