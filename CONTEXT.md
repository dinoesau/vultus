# CONTEXT - Vultus Vocabulario de Dominio

Este documento define el lenguaje ubicuo del proyecto.
Todo código, tests y ADRs deben usar estos términos.
Evita sinónimos para el mismo concepto.

## Entidades principales

- **compare**: acto de comparar 2 caras para análisis visual forense.
No es identificación biométrica.
Es apoyo visual.

- **job**: trabajo asíncrono en queue con TTL de 60 segundos.
Tiene `job_id` branded (`JobId::new` / `JobId::parse` con `trim`, error `InvalidJobId`) y estados `queued`, `processing`, `done`, `failed`, `expired` (`JobStatus::as_str` / `Display`).
El ciclo de vida tipado es `Job<Queued> -> Job<Processing> -> Job<Done|Failed|Expired>` (`start`, `set_progress`, `complete` / `fail` / `expire`).
Transiciones ilegales no compilan.
TTL es `TtlSecs` (`nutype` `1..=3600`, default `60`).

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
Local: `Store` TTL 60s (`TtlSecs`) con reaper que purga a 2xTTL y `/tmp` se limpia. Prod: R2 `lifecycle 60s` + Queues 24h retención (TTL lógico 60s) y `/tmp` tmpfs en Modal.

- **r2key**: clave `R2Key::parse(String)` no vacía, `trim`, max 1024 chars, sin `..` (`InvalidR2Key`).
Par `R2Keys::new(R2Key, R2Key)` con campos privados y accesores `image_a()` / `image_b()`.
Solo `Some` en prod (patrón `R2 pointer` por límite 128KB de Queues).

- **enqueue-command**: par de imágenes ya probadas `EnqueueCommand::new(ImageBytes, ImageBytes)`.
Evita soltar `Vec<u8>` en el adapter y hace el seam testeable (`into_pair`, `stored_lens`).

- **enqueued-job**: recibo `EnqueuedJob::new(JobId, Option<R2Keys>)` con campos privados.
Accesores `job_id()`, `r2_keys()`, `is_r2_pointer()`.
`None` en local (`MemoryQueue`), `Some(jobs/{id}/a|b)` en prod (`R2PointerQueue`).

- **report**: PDF con imágenes originales, UVs, heatmap y tabla de distancias antropométricas.
Incluye disclaimer de no identificación automática.

## Verbos

- **enqueue**: poner un job en la queue vía `Queue::enqueue(EnqueueCommand) -> EnqueueCommand`.
Local `MemoryQueue` guarda longitudes (`stored_lens`).
Prod `R2PointerQueue` retorna `Some(R2Keys)`.

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
Emitido por worker vía `Queue::set_progress` con `Stage`.
Mapeo HTTP: dominio `InvalidImage | InvalidJobId | InvalidProgress | InvalidR2Key | InvalidBaseUrl | Empty -> 400`, `NotFound -> 404`, `Queue | Ml | Invariant -> 500` (`AppError` en `vultus-api`, cuerpo `{"detail":...}`).

## Errores

- `CoreError` es taxonomía `Clone + PartialEq + Eq`: `InvalidImage(ImageError)`, `InvalidJobId`, `InvalidProgress`, `InvalidR2Key`, `InvalidBaseUrl(BaseUrlError)`, `Empty`, `Queue(QueueError::Backend)`, `Ml(MlError::{Transport, BadStatus, Decode, Empty})`, `NotFound(String)`, `Invariant(&'static str)`.
Helpers `not_found`, `queue_backend`, `ml_transport`.
`ImageError::{SizeOutOfRange, UnsupportedFormat}`, `BaseUrlError::{BadScheme, Empty}`.
`main` usa `anyhow::Context` en `bind :8000` y `serve`.
Nunca `unwrap` en request path; multipart inválido es `AppError::BadRequest`.

## Boundaries

- **Seam 1 API**: `POST /v1/compare`, `GET /v1/jobs/{id}`, `WS /v1/jobs/{id}/events` (Axum local / Cloudflare Workers + Durable Objects prod).
`AppState(Arc<dyn Queue>)` genérico vía `AppState::new(impl Queue)`, respuestas tipadas `CompareResponse{job_id, status:"queued"}` (`202`) y `JobResponse{job_id, status: JobStatus::as_str}` (`200`).
`GET` con uuid inválido es `400`, job desconocido es `404`.

- **Seam 2 Queue**: contrato `enqueue(EnqueueCommand)`, `status`, `progress`, `set_progress(Progress, Stage)`, `stored_lens` agnóstico a infra vía `vultus-core::Queue`.
Local: `MemoryQueue` (bytes directos, `r2_keys None`).
Prod: `R2PointerQueue` (patrón `R2 pointer` por límite 128KB, `r2_keys Some(jobs/{id}/a|b)`) + `HTTP Pull Consumer` en Modal.
Estado compartido `Store { HashMap<JobId, MemoryEntry> }` tras ambos adapters.
Paridad probada: `test_r2_pointer_queue_serves_same_seam`.

- **Seam 3 Worker**: contrato tipado `&ImageBytes -> Landmarks -> FlawUv -> CompleteUv -> Heatmap` (local CPU Rust o Modal GPU containers vía `MlSidecarClient` + `BaseUrl` + `FlamePayload`).
UVs exigen `UV_LEN`, `Landmarks` exige 478 JSON.

Fuera de seams: `fit_flame`, `bake`, `project_uv`.
No se testean directo.

## Convenciones de tests

Nombre de test describe WHAT no HOW.
Ejemplo bueno: `test_frontal_face_produces_512_uv`.
Ejemplo malo: `test_worker_calls_freeuv`.
Valor esperado viene de literal golden verificado manualmente, no de recomputar con misma función.
Golden UV es `vec![fill; UV_LEN]` con cabeza literal (`[10, 200]` vs `[4, 210]` -> `[6, 10]`).
`proptest` para `parse_never_panics`, rangos `Progress` / `TtlSecs`, `R2Key` trim / `..`, JPEG/PNG con filler.
Seam 1 tiene 11 tests `axum-test::TestServer` + 2 config + 3 WS (`tests/ws_events.rs` con `tokio-tungstenite`: snapshot `queued`, `processing/flame` tras `set_progress`, handshake falla en desconocido).
