# CONTEXT - Vultus Vocabulario de Dominio

> Estado objetivo sin Rust ni API Python: vocabulario TS + Python ML.
> Python: value objects frozen en `backend/domain.py` (solo lado ML/compute) con `Result`, errores estratificados.
> TS: gateway unico en `edge/` (contrato + worker + DO) como unica fuente del ciclo de vida HTTP.
> Comandos nuevos: `pip install -r backend/requirements-api.txt`, `pytest backend/tests -q`, pool suite desde `frontend/`.

Este documento define el lenguaje ubicuo del proyecto.
Todo código, tests y ADRs deben usar estos términos.
Evita sinónimos para el mismo concepto.

## Entidades principales

- **compare**: acto de comparar 2 caras para análisis visual forense.
No es identificación biométrica.
Es apoyo visual.

- **job**: trabajo asíncrono en queue con TTL de 60 segundos.
Tiene `job_id` branded (`parseJobId` con `trim`, error `InvalidJobId`) y estados `queued`, `processing`, `done`, `failed`, `expired` (`parseJobStatus`, terminales en `TERMINAL_STATUSES`).
El ciclo vive en el DO (`ProgressDO`): `init`, `progress`, `status`, alarmas TTL que marcan `expired` y purgan a 2xTTL.
TTL es `TtlSecs` (`1..=3600`, default `60`, clamp total `parseTtlSecs`).

- **image**: foto de entrada en `bytes` JPEG o PNG.
Debe contener una sola cara frontal o semi-frontal.
Tipo `ImageBytes` (`parse` en borde, max `MAX_IMAGE_BYTES = 8MB`, magic JPEG `FF D8 FF` / PNG `89 50 4E 47 0D 0A 1A 0A`, errores `ImageError::SizeOutOfRange | UnsupportedFormat`).
Vista prestada zero-cost `ImageBytesRef::parse(&[u8])` con misma prueba y promoción única `to_owned_image`.

- **landmarks**: 478 puntos 2D/3D detectados por MediaPipe.
Subset forense de 68 puntos usado para métricas.
Tipo `Landmarks::parse(Vec<u8>)` exige JSON `[[x,y,z], ...]` con `LANDMARKS_LEN = 478` puntos finitos.
Rechaza stubs `{"todo":...}` y bytes aleatorios con `Ml::Decode`.
Producido solo por `MlSidecarClient::landmarks(&JobId, &ImageBytes) -> Landmarks`.

- **mesh**: malla 3D de cabeza humana.
Puede ser `FLAME` para extracción o `GNM` para render.

- **uv**: textura canónica desplegada de 512x512.
Espacio donde ocurre la comparación.
Proveniente de FreeUV.
Dims canónicas `UV_WIDTH = 512`, `UV_HEIGHT = 512`, `UV_CHANNELS = 3`, `UV_LEN = 786432`.

- **flaw-uv**: UV incompleta con oclusiones antes de inpainting.
Tipo `FlawUv::parse` exige exactamente `UV_LEN` bytes, si no `Ml::Decode`.
Producida por `MlSidecarClient::flame(&JobId, &ImageBytes, &Landmarks) -> FlawUv` vía `FlamePayload` (`u32 BE len + landmarks_json + image_bytes`).

- **complete-uv**: UV completa tras diffusion inpainting.
Tipo `CompleteUv::parse` exige exactamente `UV_LEN` bytes.
Producida por `MlSidecarClient::freeuv(&JobId, &FlawUv) -> CompleteUv`.

- **heatmap**: imagen `|UV_A - UV_B|` por región.
Visualiza diferencias de textura.
Tipo `Heatmap::parse` exige `UV_LEN` bytes.
Producida solo por `compute_heatmap(&CompleteUv, &CompleteUv) -> Heatmap` (infallible, longitudes ya probadas).

- **bake**: transferencia baricéntrica de textura `BFM -> GNM`.
Convierte UV de topología BFM a UV de GNM sin reentrenar.
Firma `bake_bfm_to_gnm(&FlawUv) -> CompleteUv` (infallible, copia preserva `UV_LEN`; matriz real precomputada llega en Fase 2).

- **stateless**: propiedad de no persistir nada tras entrega.
Local: DO TTL 60s con purga a 2xTTL y `/tmp` tmpfs en runner. Prod: R2 `lifecycle 60s` + Queues 24h retención (TTL lógico 60s) y `/tmp` tmpfs en Modal.

- **r2key**: clave `jobs/{id}/a|b` no vacía, sin `..`.
Solo `Some` en prod (patrón `R2 pointer` por límite 128KB de Queues); en dev el worker la escribe al R2 emulado.

- **enqueue-command**: par de imágenes ya probadas en el borde del worker.
Nunca bytes sueltos cruzando el seam HTTP.

- **enqueued-job**: recibo `{job_id, r2_keys}` en la queue.
El consumer dev lo reenvia al webhook del runner; en prod lo consume el `HTTP Pull Consumer` de Modal.

- **report**: PDF con imágenes originales, UVs, heatmap y tabla de distancias antropométricas.
Incluye disclaimer de no identificación automática.

## Verbos

- **enqueue**: poner un job en la queue vía el worker (`POST /v1/compare` -> `{job_id, r2_keys}`).
Local el consumer dev lo reenvia al runner; prod lo consume Modal.

- **consume**: worker toma un job de la queue (local `Store` en memoria / HTTP Pull Consumer desde Modal en prod).
Estado vía `status(&JobId)`, `progress(&JobId) -> (Progress, Stage)`, `set_progress(&JobId, Progress, Stage)`.

- **stage**: enum ordenado `Stage::{Queued, Landmarks, Flame, Freeuv, Bake, Done}` con `as_str` / `Display`.
Prohibido `&str` suelto en `Queue::set_progress`.

- **base-url**: `BaseUrl::parse(&str)` exige `http(s)://`, recorta `/` final (`BadScheme | Empty`).
`MlSidecarClient::new(BaseUrl)` une con `join("/ml/...")` sin doble slash.

- **flame-payload**: `FlamePayload::encode(&Landmarks, &ImageBytes) -> Vec<u8>` y `decode(Vec<u8>) -> (Landmarks, ImageBytes)` con formato `u32 BE len + landmarks_json + image_bytes`.
Paridad Rust-Python en un solo módulo.

- **unwrap**: proyectar textura de mesh a UV.

- **inpaint**: completar UV incompleta con diffusion.

- **normalize**: llevar cara a pose y expresión neutra canónica.

## Métricas

- **interpupilar**: distancia entre pupilas en UV canónico.
Usada como normalizador para otras distancias.

- **progress**: valor `Progress::parse(f32)` en `0.0..=1.0` no-NaN (`InvalidProgress`), `Progress::zero()`, `value()`.
Emitido por el pipeline vía `sink.report` con `Stage`; hitos `0.15/0.40/0.75/0.95/1.0` en `edge/contract.ts`.
Mapeo HTTP: `400` validación, `404` desconocido, `500` infra (`{"detail":...}`).

## Errores

- `DomainError` es taxonomía ML/compute: `InvalidImage(ImageError)`, `InvalidJobId`, `InvalidProgress`, `InvalidBaseUrl(BaseUrlError)`, `EmptyPayload`, `Ml(MlError::{Transport, BadStatus, Decode, Empty})`, `NotFound`, `Invariant`.
Helpers `domain_to_status` / `domain_to_message`.
`ImageError::{SizeOutOfRange, UnsupportedFormat}`, `BaseUrlError::{BadScheme, Empty}`.
El runner sirve `:8001` con `http.server` stdlib; el gateway sirve `:8000` vía worker runtime.
Nunca `unwrap` en request path; multipart inválido es `400`.

## Boundaries

- **Seam 1 API**: `POST /v1/compare`, `GET /v1/jobs/{id}`, `WS /v1/jobs/{id}/events` (misma entrada Worker en prod y dev).
`GET` con uuid inválido es `400`, job desconocido es `404`.

- **Seam 2 Sink**: contrato `report(Progress, Stage)`, `complete(CompareResult)`, `fail()` en `backend/pipeline_local.py`.
Local: `HttpProgressSink` en el runner (mismo seam HTTP que prod) e `InMemorySink` en tests.
Prod: workers GPU al mismo seam HTTP + R2 directo.

- **Seam 3 Worker**: contrato tipado `&ImageBytes -> Landmarks -> FlawUv -> CompleteUv -> Heatmap` (local CPU vía `MlSidecarClient` + `BaseUrl` + `FlamePayload`, prod GPU vía Modal).
UVs exigen `UV_LEN`, `Landmarks` exige 478 JSON.

Fuera de seams: `fit_flame`, `bake`, `project_uv`.
No se testean directo.

## Convenciones de tests

Nombre de test describe WHAT no HOW.
Ejemplo bueno: `test_frontal_face_produces_512_uv`.
Ejemplo malo: `test_worker_calls_freeuv`.
Valor esperado viene de literal golden verificado manualmente, no de recomputar con misma función.
Golden UV es cabeza literal (`[10, 200]` vs `[4, 210]` -> `[6, 10]`).
Goldens literales para `Progress`, `JobId`, JPEG/PNG con filler.
Seam 1 tiene 6 tests pool en runtime (`edge/worker.http.test.ts`: snapshot `queued`, handshake falla en desconocido, backdoor dev `404` con vars prod).
