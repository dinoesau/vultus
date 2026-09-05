# PIPELINE - Flujo Completo Vultus

## 1. Resumen

Este documento describe el flujo end-to-end desde que el usuario sube 2 caras hasta que descarga el resultado.
El pipeline es asíncrono, stateless y sin persistencia.
En prod cada etapa es un worker en `Modal` que consume de `Cloudflare Queues` vía `HTTP Pull Consumer`; en dev local/test consume de `MemoryQueue` o `R2PointerQueue` vía el mismo trait `vultus-core::Queue` (`Store` compartido).

## 2. Diagrama de pipeline

> Infra prod: Cloudflare Pages + Workers + Queues + R2 + Durable Objects + Modal. Dev local: `Store` en memoria (`MemoryQueue` / `R2PointerQueue`) equivalente vía adapter, sin `Redis`.

```mermaid
graph TD
    A[Cliente Astro - Cloudflare Pages Upload 2 jpgs] --> B[Cloudflare Worker POST /v1/compare]
    B --> C{Validacion + R2 PutObject}
    C -->|ok| D["Enqueue {job_id, r2_keys} a Cloudflare Queues"]
    C -->|fail| E[400 Bad Request]
    D --> F[Modal Worker 1 - MediaPipe 478 landmarks CPU]
    F --> G[Modal Worker 2 - FLAME Fitting GPU]
    G --> H[Modal Worker 3 - FreeUV UV completo GPU]
    H --> I[Modal Worker 4 - GNM Bake + Heatmap + Report]
    I --> J[Result bytes en R2 TTL 60s lifecycle]
    J --> K[Cloudflare Worker StreamingResponse zip]
    K --> L[Cliente descarga - UV_A UV_B heatmap mesh PDF]
    J -. lifecycle 60s .-> M[Olvido total - tmpfs wipe + R2 DEL + Queue 24h]
    D -. progress .-> N[Durable Objects WS /v1/jobs/id/events]
    N --> A
    D -. local dev .-> D2[Store en memoria local (MemoryQueue)]
```

## 3. Secuencia de modelos

Esta sección muestra como se encadenan los 4 modelos y que dato produce cada uno.

```mermaid
graph LR
    I["Imagen 512x512"] --> M["MediaPipe<br/>Tasks Vision<br/>CPU 25ms"]
    M -->|"478 landmarks 3D"| F["FLAME Fitting<br/>3DDFA_V3 / DECA<br/>GPU 300ms"]
    F -->|"mesh 5023 verts + flaw-uv 512"| U["FreeUV<br/>SD1.5 + CLIP<br/>GPU 10s"]
    U -->|"complete-uv 512"| G["GNM Bake<br/>BFM->GNM<br/>CPU 150ms"]
    G -->|"mesh GNM 10k verts + textura horneada"| H["Heatmap + Report<br/>CPU 200ms"]
    H --> O["Salida: uv_a, uv_b, heatmap, mesh.glb, report.pdf"]

    style M fill:#e3f2fd
    style F fill:#fff3e0
    style U fill:#fce4ec
    style G fill:#e8f5e9
    style H fill:#f3e5f5
```

Dependencias por modelo:

- **MediaPipe** es entrada.
No depende de nadie.
Salida `landmarks 478` alimenta a FLAME.

- **FLAME Fitting** depende de `image + landmarks`.
Salida `mesh FLAME + flaw-uv` con huecos por oclusión.
Sin landmarks no puede estimar pose.

- **FreeUV** depende de `flaw-uv + image`.
Salida `complete-uv` sin huecos y con iluminación normalizada.
Es el cuello de botella y corre 2 veces en paralelo, una por cara.

- **GNM Bake** depende de `complete-uv` en topología BFM.
Hace transferencia baricéntrica a topología GNM.
Salida `mesh GNM` texturizado.

- **Heatmap + Report** depende de `complete-uv A y B` ya en mismo espacio canónico.
Calcula `|UV_A - UV_B|` y distancias antropométricas.

```mermaid
sequenceDiagram
    participant I as Imagen
    participant MP as MediaPipe
    participant FL as FLAME
    participant FU as FreeUV
    participant GNM as GNM Bake
    participant HR as Heatmap/Report

    I->>MP: bytes jpg
    MP-->>FL: landmarks 478
    I->>FL: bytes jpg
    FL->>FL: fit shape+expression+pose
    FL-->>FU: flaw-uv 512 + mesh
    I->>FU: bytes jpg
    FU->>FU: inpaint SD1.5 + CLIP
    FU-->>GNM: complete-uv 512 BFM
    GNM->>GNM: bake BFM->GNM
    GNM-->>HR: uv_baked
    HR->>HR: diff + metricas
    HR-->>I: output bundle
```

Paralelización:

- Cara A y cara B se procesan en paralelo en `MediaPipe -> FLAME -> FreeUV`.
- Cada cara usa su propio chain.
- `GNM Bake` y `Heatmap` esperan a que ambas ramas terminen y hacen join.

## 4. Secuencia detallada end-to-end

```mermaid
sequenceDiagram
    participant FE as Astro Frontend (Pages)
    participant CF as Cloudflare Worker API
    participant R2 as R2 Bucket
    participant Q as Cloudflare Queues
    participant MO as Modal Workers
    participant W1 as Worker MediaPipe
    participant W2 as Worker FLAME
    participant W3 as Worker FreeUV
    participant W4 as Worker GNM/Report
    participant DO as Durable Objects WS

    FE->>CF: POST /v1/compare multipart 2 images
    CF->>CF: Validar tipo, tamaño, una cara por imagen
    CF->>R2: PutObject r2_keys (images, local: stored_lens en memoria)
    CF->>Q: enqueue compare_job job_id=uuid r2_keys (local: MemoryQueue en memoria)
    CF-->>FE: 202 Accepted {job_id, status: queued}
    FE->>DO: WS /v1/jobs/{id}/events subscribe
    Q->>MO: HTTP Pull Consumer job_id
    MO->>W1: consume job_id + R2 GetObject
    W1->>DO: progress 0.15 landmarks done
    W1->>W2: landmarks + images
    W2->>DO: progress 0.40 FLAME mesh done
    W2->>W3: FLAME mesh + flaw-uv
    W3->>DO: progress 0.75 UV complete done
    W3->>W4: UV_A UV_B
    W4->>DO: progress 0.95 heatmap + bake done
    W4->>R2: PutObject result.zip keep 60s (local Fase 0: sin result, solo status en Store)
    R2->>CF: result ready
    CF->>DO: progress 1.0 done
    DO-->>FE: WS event done
    FE->>CF: GET /v1/jobs/{id}/result
    CF->>R2: fetch result bytes (local Fase 0: n/a, solo Store status/progress)
    CF-->>FE: 200 StreamingResponse zip
    R2->>R2: lifecycle 60s DEL (local: EXPIRE 60s)
    W1->>W1: unlink /tmp/job_id/* (Modal tmpfs)
```

## 5. Etapas

### 5.1 Entrada - POST /v1/compare

El cliente envía `multipart/form-data` con `image_a` y `image_b`.
Cada imagen debe ser JPEG o PNG menor a 8MB (`ImageBytes::parse` + `ImageBytesRef`, errores `SizeOutOfRange | UnsupportedFormat`).
Faltante o multipart roto es `400 {"detail":...}` vía `AppError::BadRequest`.
Imagen inválida es `400` vía `AppError::Domain(InvalidImage)` sin encolar.
Si pasa, construye `EnqueueCommand::new(a, b)`, encola en `Queue` y retorna `202 {job_id, status:"queued"}` (`CompareResponse`).
`GET /v1/jobs/{id}` valida `JobId::parse(trim)` y retorna `200 {job_id, status: JobStatus::as_str}`; uuid roto es `400`, desconocido es `404`. En edge (ADR-007) el `GET` lee el `ProgressDO /status` como fuente de verdad, no dummy.

### 5.2 Queue - Cloudflare Queues + R2 (prod) / MemoryQueue + R2PointerQueue (local/test)

El contrato es `Queue::{enqueue(EnqueueCommand), status, progress, set_progress(Progress, Stage), stored_lens}` agnóstico a la infra con `Store` compartido.

- **Prod (Cloudflare):** El Worker hace `R2 PutObject` con `image_a/b` y serializa solo `compare_job(job_id, r2_keys jobs/{id}/a|b)` a `Cloudflare Queues` (límite 128KB/mensaje, no caben 2x8MB). `Cloudflare Queues` cobra `10k ops/día free` (write/read/delete = 3 ops por job -> ~3.333 jobs/día free), retención `24h` en free pero `TTL lógico 60s` (`TtlSecs` default 60) vía `Durable Object alarm` + `R2 lifecycle 60s`. Modal consume vía `HTTP Pull Consumer`. El progreso va por `Durable Objects WS` con `Stage::{queued, landmarks, flame, freeuv, bake, done}`.
- **Local/dev/test:** `MemoryQueue` (adapter en memoria, `r2_keys None`, guarda `stored_lens` para probar flujo) y `R2PointerQueue` (simula prod con `Some(R2Keys)`, misma `Store`). Progreso vía `set_progress(Progress::zero() -> parse(0.0..=1.0), Stage)`; `status` pasa a `Processing` al avanzar.

No hay Postgres ni MinIO persistente. En prod el egress de R2 es free.

### 5.3 Worker 1 - MediaPipe 478 landmarks

Input: `&ImageBytes`.
Output: `Landmarks` (JSON `[[x,y,z],...]` 478 finitos, `LANDMARKS_LEN`).
Firma `MlSidecarClient::landmarks(&JobId, &ImageBytes) -> Landmarks` (`POST /ml/landmarks` con `X-Job-Id`, `BaseUrl::join`).
Runtime CPU, 20-30ms por cara.
Si el sidecar retorna stub `{"todo":...}` o largo wrong, `Landmarks::parse` falla con `Ml::Decode`.
Escribe landmarks a `/tmp/{job_id}/landmarks.json` en tmpfs.

### 5.4 Worker 2 - FLAME Fitting

Input: `&ImageBytes + &Landmarks`.
Output: `FlawUv` (`UV_LEN = 786432`).
Firma `MlSidecarClient::flame(&JobId, &ImageBytes, &Landmarks) -> FlawUv` con `FlamePayload::encode/decode` (`u32 BE len + landmarks_json + image_bytes`) sobre `POST /ml/flame`.
Runtime GPU con `3DDFA_V3` o `DECA`.
Estima `shape, expression, pose` y proyecta a `mesh` con `UV` de BFM.
Genera `flaw-uv` de 512x512 con huecos por oclusión.
Tiempo 200-400ms por cara en T4.

### 5.5 Worker 3 - FreeUV

Input: `&FlawUv` (`UV_LEN`).
Output: `CompleteUv` (`UV_LEN`).
Firma `MlSidecarClient::freeuv(&JobId, &FlawUv) -> CompleteUv` sobre `POST /ml/freeuv`.
Runtime GPU con `SD1.5 + CLIP`.
Hace inpainting de regiones ocluidas y normaliza iluminación.
Tiempo 8-12s por cara en T4.
Es la etapa más costosa.
`BaseUrl::parse` exige `http(s)://` y recorta `/`; payload vacío es `Ml::Empty`, status no-2xx es `Ml::BadStatus`, truncate es `Ml::Decode`.

### 5.6 Worker 4 - GNM Bake, Heatmap y Report

Input: `&CompleteUv` de ambas caras en topología BFM (`UV_LEN` ya probado).
Pasos: bake infallible `bake_bfm_to_gnm(&FlawUv) -> CompleteUv` (identidad Fase 0, matriz precomputada en Fase 2), `compute_heatmap(&CompleteUv, &CompleteUv) -> Heatmap` (`|a-b|` por byte, `expect` seguro porque ambas prueban `UV_LEN`), cálculo de distancias antropométricas normalizadas por interpupilar en UV canónico, generación de `report.pdf` con imágenes originales, UVs, heatmap y tabla de métricas con disclaimer.
Output: `uv_a.png, uv_b.png, heatmap.png, mesh_gnm.glb, report.pdf`.
Runtime CPU/GPU 300-500ms.
Todo se escribe a `/tmp/{job_id}/` y se retorna como dict de bytes.
Sin dep `image`; tipos `FlawUv` / `CompleteUv` / `Heatmap` cruzan el seam.

### 5.7 Entrega - GET /v1/jobs/{id}/result

El frontend pide el resultado tras recibir `WS done` (Durable Objects en prod).
En prod (Fase 1+) el Worker hace `R2 GetObject(job_id/result.zip)` y en local el API leerá del `Store`; en Fase 0 no hay `GET /result`, solo `GET status` + `WS events`. Cuando exista, armará un `StreamingResponse` con `Content-Type: application/zip` y `Content-Disposition: attachment`.
El zip contiene `uv_a.png, uv_b.png, heatmap.png, mesh_gnm.glb, report.pdf` en memoria, sin escribir a disco.
Tras el stream, en prod `R2 lifecycle 60s` borra solo y en local el reaper purga el `Store` a 2xTTL si aún existe.
El frontend crea `URL.createObjectURL` para descarga y ofrece re-descarga local desde memoria sin volver al servidor.

### 5.8 Limpieza stateless

Cada worker (local Docker o Modal container) hace `unlink` de `/tmp/{job_id}/*` al terminar, éxito o fallo.
Local: `Store` TTL 60s (`TtlSecs`) + reaper que purga a 2xTTL automático. Prod: `R2 lifecycle 60s` + `Queue retención 24h` pero `TTL lógico 60s` vía `Durable Object alarm`.
Logs no contienen bytes de imagen, solo `job_id` y `duration_ms`.
Verificación TDD Fase 0: `test_job_expires_after_ttl_and_lens_gone` + `test_expired_job_is_purged_after_double_ttl` comprueban `status Expired` y `stored_lens NotFound`, y `test_cleanup_removes_dir` comprueba `tmpfs` vacío. Sin `redis.exists`.

## 6. Contratos de datos

Job enqueue: `EnqueueCommand::new(ImageBytes, ImageBytes)` en local; `EnqueuedJob{job_id, Some(R2Keys jobs/{id}/a|b)}` en prod (adapter traduce). Límite Queues 128KB obliga a `R2 pointer` en prod.
`stored_lens(job_id) -> (len_a, len_b)` prueba que los bytes fluyen.
Worker return tipado: `Landmarks -> FlawUv -> CompleteUv -> Heatmap` (cada `parse` exige forma, `Ml::Decode` si no).
`FlamePayload` es `u32 BE len + landmarks_json + image_bytes` para paridad Rust-Python.
Progress events WS: `{job_id, progress: Progress 0.0-1.0, stage: Stage queued|landmarks|flame|freeuv|bake|done}` vía `Store` local o `Durable Objects` prod.
Error HTTP: `400` validación (`InvalidImage | InvalidJobId | InvalidProgress | InvalidR2Key | InvalidBaseUrl | Empty` + multipart), `404` (`NotFound`), `500` (`Queue::Backend | Ml::{Transport,BadStatus,Decode,Empty} | Invariant`) con `{"detail":...}`.

## 7. Manejo de errores

Imagen inválida: `400 {"detail":...}` inmediato sin encolar (`InvalidImage`).
Faltante / multipart roto: `400` (`BadRequest`).
UUID roto: `400` (`InvalidJobId` con `trim`).
Job desconocido: `404` (`NotFound`).
No face / UV wrong / stub: `Ml::Decode` (500 infra, solo desde sidecar, nunca cliente directo).
Sidecar caído / status no-2xx / vacío: `Ml::{Transport, BadStatus, Empty}` (500 `internal error` al cliente, detalle en logs con `job_id`).
Invariante rota (`TtlSecs`, `assert_ok`): `Invariant` (500, pagina al dev).
Timeout: `job_timeout=30s` por etapa, `max 60s` total (`TtlSecs` default 60).
Cliente cierra pestaña: `Store` expira solo, sin leak.

## 8. Observabilidad

Métricas por job: `duration_ms` por etapa, `vram_mb`, `queue_lag_ms` (local `Store` / `Cloudflare Queues lag + Modal GPU util` prod).
Logs estructurados con `job_id` sin datos biométricos (Workers Logs 3 días free, Modal logs).
`main` con `anyhow::Context` en `bind` y `serve`; `tracing_subscriber::EnvFilter`.
Health: `GET /health` verifica `queue ping` (local `Store` / Queues health prod) y `gpu available` (local `nvidia-smi` / Modal `torch.cuda.is_available`). En prod `Cloudflare Analytics` + `OpenTelemetry` + `Sentry` si se configura.

## 9. Escalado

Local: Workers CPU y GPU escalan independiente vía `docker compose --scale`, `Store` tras `tokio::RwLock<HashMap<JobId,_>>`.
Prod: `Cloudflare Workers` escala a 0 automático, `Modal` escala GPU `0 -> 100` (`Starter 10 GPU concurrency free`, `Team 50`), `1-2s` cold start, R2/Queues sin gestión. `FreeUV concurrency=1` por GPU para no OOM sigue vigente en Modal.
Sin storage persistente, no hay cuello de botella de I/O.
Cache opcional efímera `hash(image) -> UV` en R2 con TTL 60s (`TtlSecs`) si se quiere evitar recomputar misma cara en ventana corta, desactivada por defecto por stateless estricto.
