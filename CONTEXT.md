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
TTL es TtlSecs (1..=3600) via parseTtlSecs -> Result InvalidTtlSecs; shell default 60 con log, DO 400.
TTL estricto gobierna solo alarmas DO; R2 1d (jobs/ expire 1 dia, red de seguridad) + Queue sin TTL ni cancel (orphan-Queue aceptado solo post-enqueue, observado en consumer log); TTL != 60 es split-brain conocido, no usar distinto de 60 en prod.
backend/local_runner.py:_env es helper generico string->string, no parsea TTL (igual backend/modal_app.py:_env); no hay TtlSecs en backend/domain.py y no se añade en este slice.
Log invalido throttled por instancia (mismo job mismo valor) via `lastLogged` por raw en `load()` revalidation only; `/init` and compare log every invalid unthrottled by design, health silent.

- **image**: foto de entrada en `bytes` JPEG o PNG.
Debe contener una sola cara frontal.
Semi-frontal o 3/4 degrada por diseño (afín frontal por bbox, facing cull frontal, PnP descartado).
No se promete render limpio de lados ocultos.
Tipo `ImageBytes` (`parse` en borde, max `MAX_IMAGE_BYTES = 8MB`, magic JPEG `FF D8 FF` / PNG `89 50 4E 47 0D 0A 1A 0A`, errores `ImageError::SizeOutOfRange | UnsupportedFormat`).
Vista prestada zero-cost `ImageBytesRef::parse(&[u8])` con misma prueba y promoción única `to_owned_image`.

- **landmarks**: 478 puntos 2D/3D detectados por MediaPipe.
Subset forense de 68 puntos usado para métricas.
Tipo `Landmarks::parse(Vec<u8>)` exige JSON `[[x,y,z], ...]` con `LANDMARKS_LEN = 478` puntos finitos.
Rechaza stubs `{"todo":...}` y bytes aleatorios con `Ml::Decode`.
Producido solo por `MlSidecarClient::landmarks(&JobId, &ImageBytes) -> Landmarks`.

- **mesh**: malla 3D de cabeza humana personalizada por fit DECA sobre template FLAME.
Template FLAME `VERT_COUNT = 5023` verts; ojos `[EYE_VERT_START:EYE_VERT_END) = [3931:5023)` (1092 verts) con material propio.
Cuello abierto por diseño (`NECK_BOUNDARY_MODE == "open"`, ADR-009/ADR-010, sin hombros).
Cada cara deforma el template con identidad antropometrica + detalle del puente.
Expresion es neutra fija.
2 primitivas PBR real (`SkinPBR` + `EyePBR`, sin emisivo) con atributo `NORMAL` suave promediado por vértice, offsets `byteOffset % 4 == 0`.
Loader GNM `17821/35324` conservado solo hasta el cutover (ADR-008).

- **uv**: textura canónica desplegada de 512x512.
Espacio donde ocurre la comparación.
Proveniente de la completion FLAME (foto-derivada, piel total).
Dims canónicas `UV_WIDTH = 512`, `UV_HEIGHT = 512`, `UV_CHANNELS = 3`, `UV_LEN = 786432`.

- **fit-result**: seam fit->texture `{ coeffs, camera }` ya probados.
Tipos `GnmCoeffs` (`parse` exige 253 floats finitos, error `InvalidCoeffs`) y
`CameraParams` (matriz 3x4 aplanada, 12 finitos, error `InvalidCamera`).
Producido por `MlSidecarClient::fit(&JobId, &ImageBytes, &Landmarks) -> FitResult`
vía `FitRequest` v2 (`VERSION u8 + u32 BE len + landmarks_json + image_bytes`) sobre `POST /ml/fit`;
respuesta `253 f32 LE + 12 f32 LE` (fallo ruidoso `FitFailed`).
Wire v1 o version desconocida es `VersionMismatch` (400); consumo exacto sin sobrantes.
Fit real es DECA feed-forward determinista (deadline 10s, `Result` total).
Dobles sha256 solo como fallback local sin pesos (sin margen de identidad; el margen estricto `d(A,A)=0 < d(A,B) < d(A,C)` solo se exige con pesos reales).

- **rendered-image**: albedo tras completion FFHQ-UV 1024 con piel total.
Tipo `RenderedImage::parse` exige exactamente `UV_LEN` bytes.
Producida por `MlSidecarClient::texture(&JobId, &ImageBytes, &FitResult, &Landmarks) -> RenderedImage`
sobre `POST /ml/texture`.
El bake (`backend/flame_texture.py`, seed pineado + flags deterministicos torch) deriva cada texel de la foto;
cero pixeles `SKIN_SENTINEL = (255,0,255)` en mascara skin, `evidence >= 0.99`, bytes identicos x2 en mismo digest.
Sin pesos el bake falla ruidoso (`MlFailed`); el gris honesto legacy (`NO_DATA`, `build_albedo`) no tiene caller productivo.
El contrato 512 no cambia; 1024 vive en el bake.
`CompleteUv` sobrevive solo como parsing legacy del wire; el seam produce `RenderedImage`.
Disclosure pericial (ADR-008): la completion es visual, distinta de evidencia.

- **eye-texture**: textura propia de ojos, separada de piel.
Tipo `EyeTexture::parse` exige `UV_LEN` bytes.
Vive embebida en el GLB (primitiva `EyePBR`); el zip lleva `pbr_a/b.png` como duplicado documentado de piel
(placeholder hasta mapas roughness/metalness reales; ojos nunca mezclados con piel en malla ni material).

- **zip-6**: bundle canonico de 6 piezas en orden `ZIP_NAMES`: `uv_a.png, uv_b.png, mesh_a.glb, mesh_b.glb, pbr_a.png, pbr_b.png`.
Sin `heatmap.png` (ADR-008). `edge/contract.ts` es fuente unica; `CONTRACT_VERSION = 2` viaja en `/health`.

- **assemble**: ensamblaje CPU FLAME + PBR + GLB personalizado + zip-6.
Mascara skin excluye `[3931:5023)`; tris a caballo son `Err` (nunca drop silencioso); padding a 4 por seccion.
Firma `build_personalized_glb(fit, albedo) -> GnmMesh` y
`build_result_zip` con 6 nombres del manifiesto (`edge/contract.ts` fuente unica).
Stubs identidad (`project_texture`/`warp_with_landmarks`/`inpaint_occluded`) borrados; sus callers migrados.

- **stateless**: propiedad de no persistir nada tras entrega.
Local: DO TTL 60s con purga a 2xTTL y `/tmp` tmpfs en runner. Prod: R2 `lifecycle 1d (jobs/ expire 1 dia, red de seguridad)` + Queues 24h retención (TTL lógico 60s) y `/tmp` tmpfs en Modal.

- **r2key**: clave `jobs/{id}/a|b` no vacía, sin `..`.
Solo `Some` en prod (patrón `R2 pointer` por límite 128KB de Queues); en dev el worker la escribe al R2 emulado.

- **enqueue-command**: par de imágenes ya probadas en el borde del worker.
Nunca bytes sueltos cruzando el seam HTTP.

- **enqueued-job**: recibo `{job_id, r2_keys}` en la queue.
El consumer dev lo reenvia al webhook del runner; en prod lo consume el `HTTP Pull Consumer` de Modal.

- **report**: bundle zip-6 (UVs + meshes + PBR) para visor estudio blanco + tabla de distancias antropométricas.
Incluye disclaimer de no identificación automática y disclosure de completion visual (ADR-008).

## Verbos

- **enqueue**: poner un job en la queue vía el worker (`POST /v1/compare` -> `{job_id, r2_keys}`).
Local el consumer dev lo reenvia al runner; prod lo consume Modal.

- **consume**: worker toma un job de la queue (local `Store` en memoria / HTTP Pull Consumer desde Modal en prod).
Estado vía `status(&JobId)`, `progress(&JobId) -> (Progress, Stage)`, `set_progress(&JobId, Progress, Stage)`.

- **stage**: enum ordenado `Stage::{Queued, Fit, Texture, Assemble, Done}` con `as_str`.
Prohibido `&str` suelto en `Queue::set_progress`.

- **base-url**: `BaseUrl::parse(&str)` exige `http(s)://`, recorta `/` final (`BadScheme | Empty`).
`MlSidecarClient::new(BaseUrl)` une con `join("/ml/...")` sin doble slash.

- **fit-request**: `encode_fit_request(&ImageBytes, &Landmarks) -> Vec<u8>` y `decode_fit_request(Vec<u8>) -> (Landmarks, ImageBytes)` con formato v2 `VERSION u8 (0x02) + u32 BE len + landmarks_json + image_bytes` (`CODEC_VERSION_LEN = 1`, `VERSION_V1 = 0x01`, `VERSION_V2 = 0x02`, match exhaustivo con guards nombrados).
Respuesta fit: `253 f32 LE + 12 f32 LE`.
Request texture: `VERSION u8 + u32 BE len(fit_request) + fit_request + fit_result(1060)`, consumo exacto.
Contrato wire espejado en `pipeline_local.py` y `modal_app.py` (un solo contrato).
v1 en vuelo durante deploy es `VersionMismatch` 400 por diseno (flag-day con drain TTL60/visibility180, sin dual-read).

- **unwrap**: proyectar textura de mesh a UV (legacy GNM; el path productivo es completion FFHQ-UV).

- **inpaint**: legacy GNM (stubs identidad borrados en Wave 4). La oclusion hoy se resuelve con completion foto-derivada, no con relleno.

- **normalize**: llevar cara a pose y expresión neutra canónica.

## Métricas

- **interpupilar**: distancia entre pupilas en UV canónico.
Usada como normalizador para otras distancias.

- **progress**: valor `Progress::parse(f32)` en `0.0..=1.0` no-NaN (`InvalidProgress`), `Progress::zero()`, `value()`.
Emitido por el pipeline vía `sink.report` con `Stage`; hitos `0.40/0.75/0.95/1.0` en `edge/contract.ts`.
Mapeo HTTP: `400` validación, `404` desconocido, `500` infra (`{"detail":...}`).

## Errores

- `DomainError` es taxonomía ML/compute: `InvalidImage(ImageError)`, `InvalidJobId`, `InvalidProgress`, `InvalidBaseUrl(BaseUrlError)`, `InvalidCoeffs`, `InvalidCamera`, `EmptyPayload`, `Ml(MlError::{Transport, BadStatus, Decode, Empty})`, `FitFailed(MlError)`, `NotFound`, `VersionMismatch`, `Invariant`.
Helpers `domain_to_status` / `domain_to_message` (`VersionMismatch` -> 400 con mensaje `actualiza`).
`ImageError::{SizeOutOfRange, UnsupportedFormat}`, `BaseUrlError::{BadScheme, Empty}`.
El runner sirve `:8001` con `http.server` stdlib; el gateway sirve `:8000` vía worker runtime.
Nunca `unwrap` en request path; multipart inválido es `400`.

## Boundaries

- **Seam 1 API**: `POST /v1/compare`, `GET /v1/jobs/{id}`, `WS /v1/jobs/{id}/events` (misma entrada Worker en prod y dev).
`GET` con uuid inválido es `400`, job desconocido es `404`.

- **Seam 2 Sink**: contrato `report(Progress, Stage)`, `complete(CompareResult)`, `fail()` en `backend/pipeline_local.py`.
Local: `HttpProgressSink` en el runner (mismo seam HTTP que prod) e `InMemorySink` en tests.
Prod: workers GPU al mismo seam HTTP + R2 directo.

- **Seam 3 Worker**: contrato tipado `&ImageBytes + &Landmarks -> FitResult -> RenderedImage + EyeTexture` (fit DECA y textura FFHQ-UV vía `MlSidecarClient` + `BaseUrl` sobre `POST /ml/fit|texture` con wire v2, assemble CPU FLAME local, prod GPU vía Modal).
UVs exigen `UV_LEN`, `Landmarks` exige 478 JSON, coefs exigen 253 finitos.

Fuera de seams: particion piel/ojo, `displaced_positions` y PBR internos.
No se testean directo.

## Convenciones de tests

Nombre de test describe WHAT no HOW.
Ejemplo bueno: `test_frontal_face_produces_512_uv`.
Ejemplo malo: `test_worker_calls_texture`.
Valor esperado viene de literal golden verificado manualmente, no de recomputar con misma función.
Goldens literales para `Progress`, `JobId`, zip-6 sin heatmap, cero `SKIN_SENTINEL` + `evidence >= 0.99`, eye slice 1092, GLB magic FLAME.
Seam 1 tiene 14 tests pool en runtime (`edge/worker.http.test.ts`: 6 base + 8 TTL; base: snapshot `queued`, `400` imagen/uuid, `404` desconocido + `409` pre-done, health `ttl_secs`, ws snapshot + handshake `404`, backdoor dev `404` con vars prod; TTL: resolve default 60, health-ttl_error compare logs, health silent + ttl_error sin log, DO-400-behavior `400` sin store/alarma incl absent ttl_secs (null query) is InvalidTtlSecs 400 by design, DO-400-fake-storage unit sin setAlarm, parity-60 `60` ambos lados, compare-502-cleanup `!ok` 502 sin R2/Queue, storage-corrupto-throttled revalida 60 load() revalidation only throttled por instancia (mismo job mismo valor) + load() absent (undefined key) is silent 60 y alarma 60s/120s, queue-send-throw-orphan).
