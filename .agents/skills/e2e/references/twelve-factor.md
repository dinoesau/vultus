# Twelve-Factor - Checklist de auditoria

Escribe esta auditoria en el chat antes de correr. Marca cada factor con cumple, parcial o no cumple, mas evidencia `fichero:linea`.

- **I Codebase**: un repo, un gateway (`edge/worker.ts` + `edge/worker.dev.ts`), deploys via CI (`wrangler.toml`, `wrangler.dev.toml`, `frontend/package.json`).
- **II Deps**: todo pineado (`backend/requirements-api.txt`, `frontend/package-lock.json`, `wrangler@4`). Sin lock nuevo sin commitear es parcial.
- **III Config**: todo por env (`PORT`, `R2_TTL_SECONDS`, `RUNNER_WEBHOOK_URL`, `GATEWAY_URL`, `ML_SIDECAR_URL`, `VITE_API_URL`), `.env.example` sin secretos, `wrangler.dev.toml [vars]` con defaults.
- **IV Backing**: `R2/Queues/DO` como recursos atados via `wrangler.toml` (prod) y `wrangler.dev.toml` (local emulado). Sin adapters Python.
- **V Build/Run**: `Dockerfile.wrangler` + `Dockerfile.runner` multi-uso, `compose up --build` separa build de run, CI construye las imagenes.
- **VI Procesos**: stateless, TTL logico 60s en `ProgressDO` (`edge/progress-do.ts`) mas `tmpfs /tmp` en runner, nada en disco.
- **VII Port binding**: `8000/8001/8081/4321` auto-contenidos, bind `0.0.0.0` en compose y `wrangler dev --ip`.
- **VIII Concurrencia**: un job por webhook en el runner, `Modal concurrency` en GPU, `compose --scale` en CPU.
- **IX Desechable**: `healthcheck` en compose y Dockerfiles, alarmas TTL del DO como reaper, consumer con `retry` ante webhook caido.
- **X Paridad**: total (mismo handler en prod y local via `edge/worker.dev.ts`). Rutas dev solo con `ALLOW_DEV_ROUTES=1`, nunca en el bundle prod.
- **XI Logs**: a stdout, `job_id` sin bytes, sin ficheros.
- **XII Admin**: `scripts/smoke-fase0.sh`, `pytest backend/tests`, pool suite `vitest.pool.config.ts`, sin migraciones en Fase 0.

Deuda anotada explicitamente en el veredicto, nunca en silencio.
