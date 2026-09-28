# AGENTS.md - Vultus

Stateless forensic face comparator. No persistence beyond 60s. No Rust toolchain.

## Ownership

- `edge/` is the only HTTP gateway. `contract.ts` is source of truth for the contract. `worker.ts` is prod (thin: validate + R2 PutObject + enqueue). `worker.dev.ts` wraps prod and adds exactly 2 dev routes (`GET /dev/blobs`, `PUT /dev/results`, only with `ALLOW_DEV_ROUTES=1`). `progress-do.ts` owns job lifecycle (TTL 60, purge at 2xTTL).
- `backend/` is Python only: `local_runner.py` (thin webhook + HTTP sink), `pipeline_local.py` (orchestrator, `report`/`complete`/`fail` sink), `flame_fit.py` (DECA feed-forward fit, `Result` total) + `flame_texture.py` (FFHQ-UV completion sin gris, `Result` total), `gnm_assemble.py` + `gnm.py` (FLAME assemble + PBR + zip-6, sin heatmap), `modal_app.py` (ML sidecar `POST /ml/landmarks|fit|texture`, only place importing `torch/mediapipe`).
- `frontend/` is Astro 4 + React islands, talks only via Seam 1. Deploys to Cloudflare Pages.
- Never duplicate job lifecycle in Python. Never put vision logic in `edge/`.
- Docs: `ARCHITECTURE.md` (seams/ADRs), `PIPELINE.md` (flow), `CONTEXT.md` (ubiquitous language), `DEVELOPMENT.md` (setup).

## Commands (exact)

```bash
pip install -r backend/requirements-api.txt
mypy --strict --explicit-package-bases --namespace-packages backend/domain.py backend/flame_fit.py backend/flame_texture.py backend/gnm.py backend/pipeline_local.py backend/local_runner.py backend/gnm_assemble.py
ruff check backend/
pytest backend/tests -q
pytest backend/tests/test_pipeline.py -q   # focused: sink + tmpfs cleanup
pytest backend/tests/test_domain.py -q     # focused: value objects
```

```bash
cd frontend
npx vitest run ../edge/contract.test.ts --root ..  # TS contract unit, must run from frontend/
npx vitest run --config vitest.pool.config.ts      # gateway pool suite in worker runtime, must run from frontend/
npm run build
npm run test:e2e   # Playwright vs :4321 + real API via compose
```

```bash
docker compose up --build                        # local parity: api:8000 runner:8001 ml-sidecar:8081 frontend:4321
npx wrangler dev -c wrangler.dev.toml            # same gateway handler, emulated R2/Queue/DO on :8000
npx --yes wrangler@4 deploy --dry-run            # edge check, no deploy
```

Order that matters: `mypy -> ruff -> pytest -> vitest contract -> vitest pool` (same as `ci.yml` fast job).

## Gotchas (verified)

- `wrangler.dev.toml` is never deployed. Prod uses `wrangler.toml` with `ALLOW_DEV_ROUTES=0`. Backdoor dev routes return 404 in prod (covered by test).
- Pool config requires `isolatedStorage:false, singleWorker:true` (already in `frontend/vitest.pool.config.ts`). Do not "fix" it. Use fresh `job_id`s per test; the DO retains storage.
- Queue pattern is R2 pointer only: enqueue `{job_id, r2_keys jobs/{id}/a|b}`, never bytes (Queues 128KB limit). `R2Key` rejects empty and `..`.
- Wire formats are load-bearing: fit request v2 `VERSION u8 + u32 BE len + landmarks_json + image_bytes`; fit response `253 f32 LE + 12 f32 LE`; texture request `VERSION u8 + u32 BE len + fit_request + fit_result(1060)`; v1 is `VersionMismatch` 400. Mirrored in `pipeline_local.py` and `modal_app.py`.
- Canonical constants: `UV_LEN = 512*512*3 = 786432`, `LANDMARKS_LEN = 478`, `GnmCoeffs = 253`, `CameraParams = 12`, `TtlSecs` default 60, zip-6 `ZIP_NAMES`, `CONTRACT_VERSION = 2` in `/health`. `BaseUrl` requires `http(s)://`, strips trailing `/`.
- `pyproject.toml` ignores `BLE001,F401,TRY004,SLF001` for `backend/modal_app.py` + `SLF001` for `backend/tests/*` (pre-existing, white-box tests). New code stays strict (`ruff select SLF` marks `._value` outside the defining module).
- No `Redis`/`Postgres` anywhere. `redis connection refused` in old docs is stale; check `/health` + `ttl_secs` instead.
- Docker is local parity only, not CI/prod. Prod is `wrangler deploy --env production` + `modal deploy backend/modal_app.py`. GPU OOM: set `concurrency_limit=1` on `texture_worker` in `modal_app.py`.
- Assets: `python3 scripts/extract_gnm_template.py --check` verifies `backend/assets/gnm_template.bin`; without `--check` it rewrites the bin. Real-ML gate: `python3 scripts/e2e-flame-real.py` (needs `DECA_DIR` + `FLAME_ASSETS_DIR` + `FFHQ_UV_DIR` + `VULTUS_REAL_ML=1` + `LANDMARKS_REAL=1` for strict margin/SSIM; doubles intentionally FAIL margin/SSIM). Bridge gate: `MODAL_VOLUME=... R2_BUCKET=... bash scripts/modal-weights-sync.sh --check` (bridge 5/5 required; cutover/backup/rm need `FLAME_CUTOVER=1` + `BACKUP_TAG`; `volume rm gnm` only after code cutover + verified backup, never before). Camera diag: `python3 scripts/render_diag.py --photo <jpg> --out <png>`. Never commit LFW JPEGs.
- Modal has a single environment (`main`) and one scheduled consumer (`queue_pull_consumer` every 5s): `modal serve` would steal prod queue jobs, and `modal deploy` updates prod directly. There is no Modal staging; the release tag flow in `cd.yml` is the deploy path. Never `serve`/`deploy` from a dev branch without owner sign-off.
- Frontend E2E needs compose up first: `docker compose up -d` + `bash scripts/smoke-fase0.sh`.

## Branches / releases

- `main` is integration (CI only). `production` branch is staging (worker preview, no GPU). Release is tag `v*.*.*` on production tip: CD deploys edge, smokes, then Modal, then final smoke, then GitHub Release. `**.md`/`docs/**` changes skip CI/CD.
- Work on branches from `main`, one vertical slice per seam (`RED 1 test -> GREEN 1 impl`), no mixing refactors.
