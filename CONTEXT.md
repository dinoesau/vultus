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

- **mesh**: malla 3D de cabeza humana personalizada por fit GNM.
Template real `17821` verts / `35324` tris (`TEMPLATE_VERTS/TRIS`).
Cada cara deforma el template con identidad `253` + camara `12`.
Expresion es neutra fija.
`383` expresivos quedan fuera de alcance.

- **uv**: textura canónica desplegada de 512x512.
Espacio donde ocurre la comparación.
Proveniente de la textura GNM (proyeccion foto + warp + inpaint ocluido).
Dims canónicas `UV_WIDTH = 512`, `UV_HEIGHT = 512`, `UV_CHANNELS = 3`, `UV_LEN = 786432`.

- **fit-result**: seam fit->texture `{ coeffs, camera }` ya probados.
Tipos `GnmCoeffs` (`parse` exige 253 floats finitos, error `InvalidCoeffs`) y
`CameraParams` (matriz 3x4 aplanada, 12 finitos, error `InvalidCamera`).
Producido por `MlSidecarClient::fit(&JobId, &ImageBytes, &Landmarks) -> FitResult`
vía `FitRequest` (`u32 BE len + landmarks_json + image_bytes`) sobre `POST /ml/fit`;
respuesta `253 f32 LE + 12 f32 LE` (fallo ruidoso `FitFailed`).
Fit real es ridge identidad + camara con expresion neutra.
Seam `fit_gnm` sin cambios.
Dobles sha256 solo como fallback local sin pesos.

- **complete-uv**: albedo tras bake real 1024 reducido a 512.
Tipo `CompleteUv::parse` exige exactamente `UV_LEN` bytes.
Producida por `MlSidecarClient::texture(&JobId, &ImageBytes, &FitResult, &Landmarks) -> CompleteUv`
sobre `POST /ml/texture`.
El bake (`backend/gnm_texture.py`, `ATLAS_SIZE = 1024`) rasteriza `triangle_uvs`
por baricentricas, proyecta cada texel con `gnm_fit.project` y muestrea la foto
solo si pasa visibilidad triple (facing + `GRAZING_COS_MIN = 0.3` + z-buffer
con `win_tri`); lo no visible queda en gris honesto `NO_DATA = (128,128,128)`,
sin relleno ni inpaint.
Sin pesos no hay retroproyeccion: gris completo determinista.
El contrato 512 no cambia; 1024 vive en el bake y en `atlas_png` opcional del GLB.

- **heatmap**: imagen `|UV_A - UV_B|` por región.
Visualiza diferencias de textura.
Tipo `Heatmap::parse` exige `UV_LEN` bytes.
Producida solo por `compute_heatmap(&CompleteUv, &CompleteUv) -> Heatmap` (infallible, longitudes ya probadas).

- **assemble**: ensamblaje CPU de 5 islas GNM + PBR + GLB personalizado.
Islas reales `UvRegion` 1-5 (`skin/left_eye/right_eye/teeth/tongue`, layout v2).
PBR se deriva del albedo.
Firma `build_personalized_glb(&FitResult, &CompleteUv) -> GnmMesh` y
`build_full_zip` con 7 nombres del manifiesto (`edge/contract.ts` fuente unica).

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

- **stage**: enum ordenado `Stage::{Queued, Fit, Texture, Assemble, Done}` con `as_str`.
Prohibido `&str` suelto en `Queue::set_progress`.

- **base-url**: `BaseUrl::parse(&str)` exige `http(s)://`, recorta `/` final (`BadScheme | Empty`).
`MlSidecarClient::new(BaseUrl)` une con `join("/ml/...")` sin doble slash.

- **fit-request**: `encode_fit_request(&ImageBytes, &Landmarks) -> Vec<u8>` y `decode_fit_request(Vec<u8>) -> (Landmarks, ImageBytes)` con formato `u32 BE len + landmarks_json + image_bytes`.
Respuesta fit: `253 f32 LE + 12 f32 LE`.
Request texture: `u32 BE len(fit_request) + fit_request + fit_result(1060)`.
Contrato wire espejado en `pipeline_local.py` y `modal_app.py` (un solo contrato).

- **unwrap**: proyectar textura de mesh a UV.

- **inpaint**: completar solo texeles ocluidos (mascara de landmarks).

- **normalize**: llevar cara a pose y expresión neutra canónica.

## Métricas

- **interpupilar**: distancia entre pupilas en UV canónico.
Usada como normalizador para otras distancias.

- **progress**: valor `Progress::parse(f32)` en `0.0..=1.0` no-NaN (`InvalidProgress`), `Progress::zero()`, `value()`.
Emitido por el pipeline vía `sink.report` con `Stage`; hitos `0.40/0.75/0.95/1.0` en `edge/contract.ts`.
Mapeo HTTP: `400` validación, `404` desconocido, `500` infra (`{"detail":...}`).

## Errores

- `DomainError` es taxonomía ML/compute: `InvalidImage(ImageError)`, `InvalidJobId`, `InvalidProgress`, `InvalidBaseUrl(BaseUrlError)`, `InvalidCoeffs`, `InvalidCamera`, `EmptyPayload`, `Ml(MlError::{Transport, BadStatus, Decode, Empty})`, `FitFailed(MlError)`, `NotFound`, `Invariant`.
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

- **Seam 3 Worker**: contrato tipado `&ImageBytes + &Landmarks -> FitResult -> CompleteUv -> Heatmap` (fit y textura vía `MlSidecarClient` + `BaseUrl` sobre `POST /ml/fit|texture`, assemble CPU local, prod GPU vía Modal).
UVs exigen `UV_LEN`, `Landmarks` exige 478 JSON, coefs exigen 253 finitos.

Fuera de seams: `coeff_distance`, `project_uv`, islas y PBR internos.
No se testean directo.

## Convenciones de tests

Nombre de test describe WHAT no HOW.
Ejemplo bueno: `test_frontal_face_produces_512_uv`.
Ejemplo malo: `test_worker_calls_texture`.
Valor esperado viene de literal golden verificado manualmente, no de recomputar con misma función.
Golden UV es cabeza literal (`[10, 200]` vs `[4, 210]` -> `[6, 10]`).
Goldens literales para `Progress`, `JobId`, JPEG/PNG con filler.
Seam 1 tiene 6 tests pool en runtime (`edge/worker.http.test.ts`: snapshot `queued`, handshake falla en desconocido, backdoor dev `404` con vars prod).
