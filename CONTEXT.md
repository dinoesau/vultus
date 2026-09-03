# CONTEXT - Vultus Vocabulario de Dominio

Este documento define el lenguaje ubicuo del proyecto.
Todo código, tests y ADRs deben usar estos términos.
Evita sinónimos para el mismo concepto.

## Entidades principales

- **compare**: acto de comparar 2 caras para análisis visual forense.
No es identificación biométrica.
Es apoyo visual.

- **job**: trabajo asíncrono en queue con TTL de 60 segundos.
Tiene `job_id` uuid y estados `queued`, `processing`, `done`, `failed`, `expired`.

- **image**: foto de entrada en `bytes` JPEG o PNG.
Debe contener una sola cara frontal o semi-frontal.

- **landmarks**: 478 puntos 2D/3D detectados por MediaPipe.
Subset forense de 68 puntos usado para métricas.

- **mesh**: malla 3D de cabeza humana.
Puede ser `FLAME` para extracción o `GNM` para render.

- **uv**: textura canónica desplegada de 512x512.
Espacio donde ocurre la comparación.
Proveniente de FreeUV.

- **flaw-uv**: UV incompleta con oclusiones antes de inpainting.

- **complete-uv**: UV completa tras diffusion inpainting.

- **heatmap**: imagen `|UV_A - UV_B|` por región.
Visualiza diferencias de textura.

- **bake**: transferencia baricéntrica de textura `BFM -> GNM`.
Convierte UV de topología BFM a UV de GNM sin reentrenar.

- **stateless**: propiedad de no persistir nada tras entrega.
Local: Redis expira a 60s y `/tmp` se limpia. Prod: R2 `lifecycle 60s` + Queues 24h retención (TTL lógico 60s) y `/tmp` tmpfs en Modal.

- **report**: PDF con imágenes originales, UVs, heatmap y tabla de distancias antropométricas.
Incluye disclaimer de no identificación automática.

## Verbos

- **enqueue**: poner un job en la queue (Redis ARQ local / Cloudflare Queues + R2 prod via `core.queue` adapter).

- **consume**: worker toma un job de la queue (ARQ `consume` local / HTTP Pull Consumer desde Modal en prod).

- **unwrap**: proyectar textura de mesh a UV.

- **inpaint**: completar UV incompleta con diffusion.

- **normalize**: llevar cara a pose y expresión neutra canónica.

## Métricas

- **interpupilar**: distancia entre pupilas en UV canónico.
Usada como normalizador para otras distancias.

- **progress**: valor 0.0 a 1.0 emitido por worker vía WS.

## Boundaries

- **Seam 1 API**: `POST /v1/compare`, `GET /v1/jobs/{id}`, `WS /v1/jobs/{id}/events` (FastAPI local / Cloudflare Workers + Durable Objects prod).

- **Seam 2 Queue**: contrato `enqueue` y `consume` agnóstico a infra vía `core.queue` adapter. Local: `Redis ARQ`. Prod: `Cloudflare Queues + R2` (patrón `R2 pointer` por límite 128KB) + `HTTP Pull Consumer` en Modal.

- **Seam 3 Worker**: contrato `image bytes -> {uv bytes, mesh bytes, landmarks}` (local Docker o Modal GPU containers).

Fuera de seams: `fit_flame`, `bake`, `project_uv`.
No se testean directo.

## Convenciones de tests

Nombre de test describe WHAT no HOW.
Ejemplo bueno: `test_frontal_face_produces_512_uv`.
Ejemplo malo: `test_worker_calls_freeuv`.
Valor esperado viene de literal golden verificado manualmente, no de recomputar con misma función.
