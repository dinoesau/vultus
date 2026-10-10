# CONTEXT — Vocabulario de dominio

> **Fuente de verdad:** `edge/contract.ts` (contrato HTTP) · `backend/domain.py` (invariantes ML/compute).
> Todo código, test y ADR usa estos términos. Sin sinónimos.

![contract-v3](https://img.shields.io/badge/contract-v3-blue) ![zip-8](https://img.shields.io/badge/zip--8-canónico-green) ![ttl-60](https://img.shields.io/badge/TTL-60s-orange)

## Cómo leer este doc

| Columna | Significado |
|---|---|
| **Tipo** | Dónde vive el smart constructor (`parse* → Result`, nunca lanza) |
| **Regla** | Invariante que el borde exige; el core la asume |
| **Dueño** | Módulo que puede cambiar la regla |

## Entidades

### `compare` — acto de comparar 2 caras

Apoyo visual forense. **No es identificación biométrica.** Incluye disclaimer de no identificación + disclosure de completion visual (ADR-008: la piel completada alucina plausibilidad, no es medición).

### `job` — trabajo asíncrono con TTL 60

| Aspecto | Regla (código) |
|---|---|
| `job_id` | `parseJobId` con `trim`, uuid v1-5 (`contract.ts:212-217`, `domain.py:474`) · error `InvalidJobId → 400` |
| `status` | `queued, processing, done, failed, expired` (`contract.ts:35`) · terminales `done/failed/expired` (`contract.ts:32`) |
| `stage` | `queued, fit, texture, assemble, done` (`contract.ts:23-29` espejo `domain.py:184-189`) · prohibido `&str` suelto |
| `progress` | `parseProgress` en `0.0..=1.0` finito (`contract.ts:287`) · hitos `0.40/0.75/0.95/1.0` (`contract.ts:41-44`) |
| `ttl` | `TtlSecs 1..=3600`, default 60 (`contract.ts:254-276`) · shell defaultea con log, DO responde `400` |
| ciclo | Vive en `ProgressDO`: `init → progress → status`, alarmas TTL que marcan `expired` y purgan a 2×TTL (`progress-do.ts:232-250`) |

TTL ≠ 60 es split-brain conocido: no usar distinto de 60 en prod. Log inválido throttled por instancia; `/init` y compare loguean cada inválido, health es silencioso.

### `image` — foto de entrada

`ImageBytes::parse` en el borde: máx `MAX_IMAGE_BYTES = 8 MiB` (`contract.ts:18`, `domain.py:37`), magic JPEG `FF D8 FF` / PNG `89 50 4E 47 0D 0A 1A 0A`. Errores `SizeOutOfRange | UnsupportedFormat → 400`. Una sola cara frontal; 3/4 degrada por diseño (ADR-010). Faltante o multipart roto = `400` sin encolar.

### `landmarks` — 478 puntos MediaPipe

`Landmarks::parse(Vec<u8>)` exige JSON `[[x,y,z],…]` con `LANDMARKS_LEN = 478` finitos (`domain.py:44`). Rechaza stubs `{"todo"…}` y bytes aleatorios con `Ml::Decode`. Solo lo produce `MlSidecarClient::landmarks → POST /ml/landmarks` con `X-Job-Id`.

### `mesh` — cabeza FLAME personalizada

| Constante | Valor | Dónde |
|---|---|---|
| `VERT_COUNT` | `5023` | `gnm_assemble.py:41` |
| ojos | `[3931:5023)` = 1092 verts, material propio | `gnm_assemble.py:42-44` |
| tris | `9976` (piel `8000` + ojos resto) | `gnm_assemble.py:46-48` |
| cuello | abierto (`NECK_BOUNDARY_MODE="open"`, sin hombros) | `gnm_assemble.py:430`, ADR-010 |
| material | 2 primitivas PBR real (`SkinPBR + EyePBR`), sin emisivo, `NORMAL` suave, `byteOffset % 4 == 0` | `gnm_assemble.py:542` |

Cada cara deforma el template con identidad + detalle del puente. Expresión neutra fija. Loader GNM `17821/35324` conservado solo hasta el cutover (ADR-008).

### `uv` — textura canónica 512

Espacio donde ocurre la comparación. `UV_WIDTH = 512`, `UV_HEIGHT = 512`, `UV_CHANNELS = 3`, `UV_LEN = 786432` (`domain.py:45-48`, `contract.ts` vía bytes). Proveniente de completion foto-derivada, piel total. El contrato 512 no cambia; 1024 vive solo en el bake/v3.

### `fit-result` — seam fit→texture

`GnmCoeffs` (253 floats finitos, `InvalidCoeffs`) + `CameraParams` (3×4 aplanada, 12 finitos, `InvalidCamera`). `encode_fit_request` v2 (`VERSION u8 0x02 + u32 BE len + landmarks_json + image`) sobre `POST /ml/fit`; respuesta `253 f32 LE + 12 f32 LE = 1060 bytes`. Wire v1 o versión desconocida = `VersionMismatch → 400` (flag-day con drain TTL60/visibility180, sin dual-read). Fit real DECA determinista ×2, deadline 10 s, `Result` total. Dobles sha256 solo como fallback local sin pesos; margen `d(A,A)=0 < d(A,B) < d(A,C)` solo exigible con pesos reales.

### `rendered-image` — albedo con piel total

`RenderedImage::parse` exige exactamente `UV_LEN`. Producida por `POST /ml/texture` (wire v2: `VERSION + u32 BE len(fit_request) + fit_request + fit_result(1060)`, consumo exacto). Bake (`flame_texture.py`, seed pineado + torch determinista): cada texel deriva de la foto; cero `SKIN_SENTINEL = (255,0,255)`, `evidence ≥ 0.99` (`EVIDENCE_MIN`, `flame_texture.py:111`), bytes idénticos ×2. Sin pesos falla ruidoso (`MlFailed`); gris legacy sin caller. `CompleteUv` sobrevive solo como parsing legacy del wire.

### `eye-texture` — ojos separados

`EyeTexture::parse` exige `UV_LEN`. Vive embebida en el GLB (`EyePBR`); el zip lleva `pbr_a/b.png` como duplicado documentado de piel (placeholder hasta roughness/metalness reales). Ojos nunca mezclados con piel en malla ni material.

### `zip-8` — bundle canónico v3

Orden canónico (`domain.py:74-83` espejo `contract.ts:64-73` espejo `compare-client.ts:19-28`):

```
uv_a.png · uv_b.png · mesh_a.glb · mesh_b.glb
pbr_a.png · pbr_b.png · render_a.png · render_b.png
```

Sin `heatmap.png` (ADR-008). Renders frontales 512 al final (ADR-012). `CONTRACT_VERSION = 3` viaja en `/health` (`contract.ts:143`). Mismatch = `VersionMismatch actualiza`, sin dual-read.

### `figure-v3` — bundle denso paralelo

`V3_ZIP_NAMES` (`v3_contract.py:49-56` espejo `contract.ts:98-105`): `mesh_dense.glb`, `albedo_1024.png`, `relight_{neutral,key,fill}.png` → realmente `neutral/key/fill/rim` (6 piezas). Nombres disjuntos del zip-8: prohibido mezclar. `V3_VERT_COUNT = 20481`, `V3_TRI_COUNT = 40832`, atlas 1024, `TV ≤ 2.0`, retención lógica 7 días (`V3_RETENTION_DAYS`, cola separada `v3-queue`), timeouts 600/900. Excepción aparte: `figure-v3.zip` vive hoy bajo `jobs/`, así que la red `expire-jobs` 1 día aplica como piso mínimo hasta prefijo propio. Servido por `GET /v3/jobs/{id}/figure` como `figure-v3.zip`.

### `assemble` — ensamblaje CPU

`build_personalized_glb(fit, albedo) → GnmMesh` + `build_result_zip` con los 8 nombres del manifiesto. Máscara skin excluye `[3931:5023)`; tris a caballo son `Err` (nunca drop silencioso); padding a 4 por sección.

### `stateless` — no persistir tras entrega

Local: DO TTL 60 + purga 2×TTL + `/tmp` tmpfs. Prod: cron 1/min borra `result.zip` en `expired`/purga (#134) + lifecycle red 1 día (`expire-jobs` sobre `jobs/`, mínimo R2) + Queues retención 24 h (TTL lógico 60) + `/tmp` tmpfs en Modal. v3: retención lógica de días en cola separada (única excepción; red 1 día como piso hasta prefijo propio).

### `r2key` / `enqueue-command` / `enqueued-job`

`r2key`: `jobs/{id}/a|b`, no vacía, sin `..`, máx 1024. Solo `Some` en prod (patrón R2 pointer). `enqueue-command`: par de imágenes ya probadas en el borde — nunca bytes sueltos. `enqueued-job`: recibo `{job_id, r2_keys}`; consumer dev → webhook runner; prod → `HTTP Pull Consumer` Modal.

### `report` — lo que ve el perito

Bundle zip-8 (UVs + meshes + PBR + renders) para visor estudio blanco. Sin heatmap, sin slider de opacidad. Incluye disclaimer + disclosure ADR-008.

## Verbos

| Verbo | Regla |
|---|---|
| `enqueue` | `POST /v1/compare → {job_id, r2_keys}`; local → webhook runner, prod → Modal |
| `consume` | `status / progress → (Progress, Stage) / set_progress` con `Stage` enum |
| `stage` | `queued → fit → texture → assemble → done`, `as_str` para logs |
| `base-url` | `BaseUrl::parse` exige `http(s)://`, recorta `/`; `MlSidecarClient::new(BaseUrl)` une con `join("/ml/…")` |
| `fit-request` | `encode/decode_fit_request` v2 espejado en `pipeline_local.py` y `modal_app.py` (un solo contrato) |
| `unwrap` | proyectar textura a UV (legacy GNM; productivo = completion FFHQ-UV) |
| `inpaint` | legacy (stubs identidad borrados Wave 4); hoy completion foto-derivada |
| `normalize` | llevar cara a pose/expresión neutra canónica |

## Métricas

`interpupilar` (normalizador en UV canónico) · `progress` (hitos arriba) · HTTP `400` validación / `404` desconocido / `500` infra con `{"detail"…}`.

## Errores

Taxonomía ML/compute (`domain.py`): `InvalidImage(ImageError)`, `InvalidJobId`, `InvalidProgress`, `InvalidBaseUrl`, `InvalidCoeffs`, `InvalidCamera`, `EmptyPayload`, `Ml::{Transport,BadStatus,Decode,Empty}`, `FitFailed`, `NotFound`, `VersionMismatch`, `Invariant`. Helpers `domain_to_status / domain_to_message` (`VersionMismatch → 400 actualiza`). Runner `:8001` stdlib; gateway `:8000` worker runtime. Nunca `unwrap` en request path.

## Boundaries (resumen)

- **Seam 1 API:** `POST /v1/compare`, `GET /v1/jobs/{id}[ /result]`, `POST …/progress`, `WS …/events`, `GET /v3/jobs/{id}/figure`, `GET /health`. `GET` uuid inválido `400`, desconocido `404`.
- **Seam 2 Sink:** `report/complete/fail` (`pipeline_local.py:82`). Local `HttpProgressSink` + `InMemorySink`; prod mismo seam HTTP + R2.
- **Seam 3 Worker:** `&ImageBytes + &Landmarks → FitResult → RenderedImage + EyeTexture` (wire v2, assemble CPU local, GPU Modal).

Fuera de seams: partición piel/ojo, `displaced_positions`, PBR internos. No se testean directo.

## Convenciones de tests

Nombre = WHAT no HOW. Bueno: `test_frontal_face_produces_512_uv`. Malo: `test_worker_calls_texture`. Esperado = literal golden verificado a mano, no recomputado. Goldens: `Progress`, `JobId`, zip-8 sin heatmap, cero sentinel + `evidence ≥ 0.99`, eye slice 1092, `GLB magic` FLAME. Seam 1: suite pool en runtime (`worker.http.test.ts`) + WS real + negativo backdoor dev con vars prod.
