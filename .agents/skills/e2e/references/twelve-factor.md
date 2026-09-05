# Twelve-Factor - Checklist de auditoria

Escribe esta auditoria en el chat antes de correr. Marca cada factor con cumple, parcial o no cumple, mas evidencia `fichero:linea`.

- **I Codebase**: un repo, deploys via CI (`backend/Cargo.toml`, `wrangler.toml`, `frontend/astro.config.mjs`).
- **II Deps**: todo pineado (`backend/Cargo.lock`, `backend/requirements.txt`, `frontend/package-lock.json`). Sin lock nuevo sin commitear es parcial.
- **III Config**: todo por env (`PORT`, `QUEUE_DRIVER`, `R2_TTL_SECONDS`, `ML_SIDECAR_URL`, `VITE_API_URL`), `.env.example` sin secretos, `config.rs` con defaults.
- **IV Backing**: `R2/Queues` como recursos atados via `wrangler.toml`, local via adapter `MemoryQueue/R2PointerQueue` en `queue.rs`.
- **V Build/Run**: `Dockerfile` multi-stage, `compose up --build` separa build de run, CI construye las imagenes.
- **VI Procesos**: stateless, `Store` con `TTL 60` mas `tmpfs /tmp`, nada en disco.
- **VII Port binding**: `8000/8081/4321` auto-contenidos, bind `0.0.0.0` en `config.rs`.
- **VIII Concurrencia**: `tokio`, `Modal concurrency_limit`, `compose --scale`.
- **IX Desechable**: `graceful shutdown` en `main.rs`, `healthcheck` en compose y Dockerfiles, reaper `TTL/2`.
- **X Paridad**: alta (`edge_parity.rs`, `Dockerfile.gpu` reusado en `modal_app.py`). Fallbacks edge marcados solo-dev, nunca en prod.
- **XI Logs**: `tracing` a stdout, `job_id` sin bytes, sin ficheros.
- **XII Admin**: `scripts/smoke-fase0.sh`, `cargo test`, sin migraciones en Fase 0.

Deuda anotada explicitamente en el veredicto, nunca en silencio.
