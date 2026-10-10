# DEVELOPMENT — Guía de desarrollo

> Fuente de verdad: `docker-compose.yml` (servicios) · `pyproject.toml` (mypy/ruff) · `.github/workflows/ci.yml` (gates) · `edge/*` + `backend/*` (código).
> Sin toolchain Rust. Solo Python + Node + Docker. Gateway único TS.

![stack](https://img.shields.io/badge/stack-python%20%2B%20node%20%2B%20docker-blue) ![mypy](https://img.shields.io/badge/mypy-strict-green) ![ruff](https://img.shields.io/badge/ruff-py310-green)

## 1. Requisitos

| Herramienta | Versión | Para qué |
|---|---|---|
| Docker + Compose v2 | 24+ | stack local (`api/runner/ml-sidecar/frontend`) |
| Python | 3.12+ local · 3.10 en Modal (piso real, `domain.py:20-26`) | runner, orquestador, assemble, tests |
| Node | 22+ | gateway Worker + frontend Astro 7 |
| `nvidia-container-toolkit` | solo GPU local | `ml-sidecar` con runtime nvidia (bloque `deploy` comentado por defecto) |

Histórico: API Python FastAPI y Rust Axum borrados (ver ADRs). No instalar Rust.

## 2. Setup Python

```bash
pip install -r backend/requirements-api.txt
mypy --strict --explicit-package-bases --namespace-packages \
  backend/domain.py backend/flame_fit.py backend/flame_texture.py \
  backend/gnm.py backend/pipeline_local.py backend/local_runner.py backend/gnm_assemble.py
ruff check backend/
pytest backend/tests -q
```

- `requirements-api.txt` (6 líneas, sin pinear): `httpx, Pillow, numpy, pytest, mypy, ruff` — compute/ML local + dev.
- `requirements.txt` (35 líneas): GPU Modal (`fastapi`, `uvicorn`, `httpx==0.28.1`, `numpy==1.26.4`, `mediapipe==0.10.14`, torch vía cu126).
- `requirements.lock` (2030 líneas, con hashes, `torch==2.13.0`) + `requirements-constraints.txt` (anchor para regen con `uv 0.12.19`).
- `pyproject.toml`: `mypy strict = true`, `ruff target py310` (el piso Modal; sin esto ruff marca `UP036` en el bloque compat), `extend-select ["SLF"]`, ignores para `modal_app.py` (`BLE001,F401,TRY004,SLF001` — baseline preexistente) y `tests/*` (`SLF001` — white-box por diseño).

Gateway local en `http://localhost:8000` (`/health` con `gateway:"worker"`, `POST /v1/compare → 202`, `GET /v1/jobs/{id} → 200`). Misma entrada que prod, servida vía `wrangler dev -c wrangler.dev.toml`. Sidecar ML local en `:8081`:

```bash
python3 -c "import sys; sys.path.insert(0,'.'); import backend.modal_app; import uvicorn; uvicorn.run(backend.modal_app.sidecar, port=8081)"
```

`ML_SIDECAR_URL=http://localhost:8081` lo consume `MlSidecarClient(BaseUrl)` (`http(s)://`, recorta `/`).

## 3. Estructura real

```
backend/
├── domain.py            # tipos probados + Result + errores (dueño local; contrato en edge/contract.ts)
├── pipeline_local.py    # orquestador run_pair + wire v2 + MlSidecarClient + sink report/complete/fail
├── local_runner.py      # webhook :8001 + HttpProgressSink + DeadLetter + serve_forever
├── gnm_assemble.py      # FLAME 5023 + piel/ojos + PBR + GLB + NECK open (VERT_COUNT=5023)
├── gnm.py               # uv_to_png + build_result_zip zip-8 CPU (sin heatmap)
├── gnm_fit.py / gnm_head.py / gnm_texture.py  # legacy hasta cutover, sin caller productivo
├── flame_fit.py         # fit DECA feed-forward determinista (doble + real)
├── flame_texture.py     # completion FFHQ-UV sin gris (doble + real, EVIDENCE_MIN=0.99)
├── deep3d.py            # ReconNet ResNet50 + encode_fit253 + reconstruct_dense (solo decisión)
├── dpr.py               # LightNet SH9 (DPR_SH=9, 250 keys) + fallback gray-world
├── texgan.py            # completion neuronal enmascarada (init w_avg)
├── face_parsing.py / render_preview.py / shell_secrets.py
├── modal_app.py         # sidecar /ml/* + workers + queue_pull_consumer + v3_figure_worker (1668 líneas)
├── v3_contract.py / v3_dense.py / v3_texture.py / v3_pose.py /
│   v3_light.py / v3_bundle.py / v3_worker.py   # track denso paralelo, sin tocar forense
└── tests/               # 32 módulos: domain, pipeline, gnm*, modal_compat, env_drift,
                         # bridge_parity, deploy_pins, texture_confidence, v3_*_real, bordes parse_*

edge/
├── contract.ts          # FUENTE ÚNICA HTTP (brands, Result, hitos, ZIP_MANIFEST zip-8, V3_*, CONTRACT_VERSION=3)
├── worker.ts            # gateway prod (7 rutas + OPTIONS 204 + CORS)
├── worker.dev.ts        # entrada dev: envuelve prod + GET /dev/blobs + PUT /dev/results + consumer webhook
├── progress-do.ts       # ProgressDO (init/progress/status + WS 500ms×120 + alarm 2 fases)
├── assert.ts / globals.d.ts / tsconfig.json
├── contract.test.ts / contract.v3.test.ts       # unit contrato
└── worker.http.test.ts  # suite pool runtime real

frontend/src/
├── pages/index.astro              # form A/B + script POST→WS→GET (522 líneas)
└── components/
    ├── compare-client.ts          # API_PATHS, RESULT_FILES zip-8, CONTRACT_VERSION=3, tryExtractViewerBlobs
    ├── three-rig.ts               # constantes estudio blanco (sin import three)
    ├── three-rig.test.ts
    ├── ThreeViewer.astro          # 2 canvas + sondas + RoomEnvironment
    └── UvViewers.astro            # progreso + UV A/B + download
```

> No existen `backend/store.py` ni `backend/app.py` (docs viejos los citaban): el ciclo vive solo en `edge/`, Python no lo duplica.

## 4. Docker — solo dev local, nunca CI ni prod

Cloudflare y Modal no ejecutan estas imágenes; Modal solo reusa `Dockerfile.gpu` como receta vía `from_dockerfile`.

| Servicio | Puerto | Imagen | Env clave |
|---|---|---|---|
| `api` | 8000 | `Dockerfile.wrangler` | `R2_TTL_SECONDS=60`, `RUNNER_WEBHOOK_URL=http://runner:8001/hooks/queue` |
| `runner` | 8001 | `Dockerfile.runner` | `GATEWAY_URL=http://api:8000`, `ML_SIDECAR_URL=http://ml-sidecar:8081`, `tmpfs /tmp` |
| `ml-sidecar` | 8081 | `Dockerfile.gpu` | `ML_PORT=8081`, `tmpfs /tmp`, bloque GPU comentado |
| `frontend` | 4321 | `frontend/Dockerfile` | `VITE_API_URL=http://localhost:8000` |

```bash
docker compose up --build            # full stack CPU-verde
docker compose build api && docker compose up -d api   # rebuild rápido
# GPU local: descomenta deploy en docker-compose.yml:79-85 y re-up (T4, concurrency=1 en modal_app.py)
```

Volumen `wrangler-state` (efímero en tests). Healthchecks: `api /health`, `runner /health`, `ml-sidecar /docs`.

### Edge en Cloudflare (prod)

```bash
npx wrangler dev -c wrangler.dev.toml                 # local: R2/Queue/DO emulados
npx --yes wrangler@4 deploy --env production          # prod (el CD lo hace solo)
API_URL=https://api.vultus.esau.com.mx bash scripts/smoke-prod.sh
```

`wrangler.toml` (prod): `main=edge/worker.ts`, queue `vultus-jobs`, R2 `vultus-jobs`, DO `ProgressDO`, `R2_TTL_SECONDS=60`, `ALLOW_DEV_ROUTES=0`, ruta `api.vultus.esau.com.mx`. `wrangler.dev.toml`: `main=edge/worker.dev.ts`, R2 `vultus-jobs-dev`, consumer `max_batch_size=1`, `ALLOW_DEV_ROUTES=1`, `RUNNER_WEBHOOK_URL=http://runner:8001/hooks/queue`.

### GPU en Modal (prod)

```bash
modal deploy backend/modal_app.py    # MediaPipe/fit/textura/assemble + consumer
modal app logs vultus-workers
```

Escala `0→100`, T4 16 GB, cold 1-2 s, `$30/mes free`. Receta pineada: base por digest + lock con hashes + `pytorch3d@978cd99` desde source. Deploy real con build log solo en release tag. **Nunca `modal serve`** (roba queue prod por el schedule 5 s; un solo environment `main`, sin staging Modal).

## 5. Frontend Astro

```bash
cd frontend && npm install && npm run dev      # :4321
npm run build && npm run preview               # static (Pages prod)
npm run test:e2e                               # Playwright :4321 + API real
```

Astro 7 `output: static`, `site: https://vultus.esau.com.mx`. Deps: `astro ^7`, `jszip ^3.10`, `three 0.160.1` pineado exacto; dev `vitest ^2`, `@cloudflare/vitest-pool-workers`, `@playwright/test`, `typescript ^5.6`.

## 6. Queues y workers

Contrato = sink estrecho `report/complete/fail` (`pipeline_local.py:82`). El pipeline no conoce la infra; solo el sink decide (`InMemorySink` vs `HttpProgressSink`). Presupuesto dentro de TTL (`PipelineConfig`): `landmarks 5 s + fit 10 s + texture 30 s ≤ total 60 s`; `MlSidecarClient` timeout 35 s; caras A/B en paralelo `ThreadPoolExecutor(2)`; `deadline = start + total` → `fail total timeout`.

## 7. Testing

```bash
pytest backend/tests -q
pytest backend/tests/test_pipeline.py backend/tests/test_domain.py backend/tests/test_gnm.py -q
cd frontend && npx vitest run ../edge/contract.test.ts --root ..
cd frontend && npx vitest run --config vitest.pool.config.ts
```

| Capa | Qué exige |
|---|---|
| Seam 1 (pool runtime) | `202 queued`, `400` imagen/faltante/uuid, `404` desconocido, `409` pre-done, `health gateway:"worker"`, WS snapshot + handshake `404`, backdoor dev `404` con vars prod |
| Seam 2 (sink memoria) | `report` ordenado `fit/texture/assemble`, `complete` guarda, `fail` marca |
| Seam 3 (golden `UV_LEN=786432`) | fit v2 + `GLB magic` FLAME 5023 + `SkinPBR/EyePBR` sin emisivo, eye slice 1092, cero sentinel, `Landmarks` 478 rechaza stubs |
| Goldens | literales a mano (`Progress`, `JobId`, zip-8, sentinel/evidence, coefs). Esperado nunca recomputado. No mockees el pipeline interno. |

Gates reales (pesos nunca commiteados, LFW congelado por sha256, sin JPEGs en repo):

```bash
DECA_DIR=weights/deca FLAME_ASSETS_DIR=weights/flame FFHQ_UV_DIR=weights/ffhq-uv \
TOPO_DIR=weights/topo_assets VULTUS_REAL_ML=1 python3 scripts/e2e-flame-real.py
# margen d(A,A)=0<d(A,B)<d(A,C) + SSIM misma>distinta + landmarks reales (en dobles falla por diseño)

MODAL_VOLUME=... R2_BUCKET=... bash scripts/modal-weights-sync.sh --check   # puente + cutover + backup
python3 scripts/render_diag.py --photo <jpg> --out <png>                     # diag foto+UV+GLB
python3 scripts/extract_flame_template.py --check                            # template real (falla loud sin pkl+OBJ)
cd frontend && npm run test:e2e                                              # Playwright: 2 PNG → queued → done, UV+3D visibles
bash scripts/smoke-fase0.sh                                                  # health + frontend + POST 202 + GET queued
```

## 8. Pesos y bridge (resumen operativo)

Volumen Modal `vultus-weights` (`/weights/{flame,deca,ffhq-uv,gnm,mediapipe}`), `SYNC_VERSION=4`. Canónicos: `face_landmarker.task`, `deca_model.tar`, `flame2023_Open.pkl`, `FLAME_w_HIFI3D_UV.obj + eye_ball_tex.png`, `texgan_ffhq_uv.pth`, `epoch_latest.pth`, `unwrap_1024_info.mat + hifi3dpp_mean_face.obj`, `flame_template.bin + flame_hifi_transfer.npz + flame68_embed.npz`. Envs `WEIGHTS_ROOT|WEIGHTS_DIR`, `TEXGAN/DEEP3D/TOPO/DPR_DIR`, `VULTUS_REAL_ML`, `MODAL_VOLUME`, `FLAME_CUTOVER=1`, `BACKUP_TAG`. Orden: `puente-up → cutover → backup (gnm→gnm.bak-<tag>, manifiesto sha256) → rm por archivo (V1 sin -r)`. Detalle en `PIPELINE.md §7`.

## 9. Lint y formato

```bash
mypy --strict --explicit-package-bases --namespace-packages backend/domain.py backend/flame_fit.py backend/flame_texture.py backend/gnm.py backend/pipeline_local.py backend/local_runner.py backend/gnm_assemble.py
ruff check backend/
ruff check backend/modal_app.py   # si lo tocas (baseline con ignores, ver §2)
cd frontend && npm run lint && npm run format
```

## 10. Flujo de trabajo

Rama desde `main` → vertical slice `1 test RED → 1 implementación GREEN` por seam → sin refactors mezclados → PR con `docker compose up` + `pytest backend/tests -q` + pool suite en verde. `main` = integración (solo CI). `production` + tag `v*.*.*` = release (CD despliega edge `--env preview` en push a production, y edge prod + Pages + Modal + Release solo en tag). Detalle en `PIPELINE.md §8`.

## 11. Troubleshooting

| Síntoma | Causa real / fix |
|---|---|
| `docker build` falla | `docker compose build api runner` (gateway + runner) y reintenta |
| `redis connection refused` | doc vieja: nunca hubo Redis; estado en DO/R2 (prod) o emulado (dev); verifica `/health` + `ttl_secs` |
| `wrangler deploy` falla | bindings Queues/R2 en `wrangler.toml` + `CLOUDFLARE_API_TOKEN` |
| `modal deploy` falla | `modal token` + receta pineada (digest + lock + pytorch3d SHA; nvdiffrast nunca `==` PyPI); build real en tag, nunca `serve` |
| `CUDA OOM` | `concurrency_limit → 1` en `texture_worker/fit_worker` (`modal_app.py`) |
| `Ml::Decode` | `FitRequest/TextureRequest` v2 + `UV_LEN`; v1 = `VersionMismatch 400` por diseño |
| `WS no conecta` | `VITE_API_URL` + binding DO (`wrangler.toml` prod / `wrangler.dev.toml` local) |
| `Queues 128KB exceeded` | nunca encoles bytes: solo `{job_id, r2_keys jobs/{id}/a\|b}` |
| pool WS flaky | corre con `vitest.pool.config.ts` commiteada (`isolatedStorage:false`; DO retiene storage con timers vivos) |
