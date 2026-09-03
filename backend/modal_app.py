"""
Modal GPU workers para Vultus - facium.

Prod: Cloudflare Queues (HTTP Pull Consumer) -> Modal -> R2
Local: Redis ARQ -> Docker workers (paridad via core.queue adapter)

Starter plan: $30/mes free (~50h T4 = ~9.300 compares). Cold start 1-2s.
Deploy: modal deploy backend/modal_app.py
Logs:   modal app logs facium-workers
"""

import modal

app = modal.App("facium-workers")

# Imagen base GPU con torch + diffusers + mediapipe
# Reusa Dockerfile.gpu local para paridad
image = (
    modal.Image.from_dockerfile("backend/Dockerfile.gpu")
    .pip_install("boto3", "httpx")  # R2 + Queues HTTP Pull
)

# Volume para cachear pesos FreeUV / FLAME / GNM (evita re-descarga en cold start)
weights = modal.Volume.from_name("facium-weights", create_if_missing=True)

# Secrets: Cloudflare R2 + Queues creds
# modal secret create facium-cloudflare CLOUDFLARE_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=16384,
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("facium-cloudflare")],
    concurrency_limit=1,  # FreeUV OOM si >1 por GPU
    timeout=60,
    min_containers=0,
    max_containers=10,  # Starter free: 10 GPU concurrency
)
def freeuv_worker(job_id: str, r2_keys: dict):
    """
    Worker 3 - FreeUV SD1.5 inpainting.
    En prod consume desde Cloudflare Queues via HTTP Pull Consumer (ver pull_consumer.py).
    Lee flaw-uv + image desde R2, escribe complete-uv a R2.
    En local este mismo código corre en Docker sin Modal.
    """
    # TODO: import models.freeuv, descargar pesos a /weights si no existen
    # r2 = boto3.client("s3", endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com")
    # flaw_uv = r2.get_object(Bucket="facium-jobs", Key=r2_keys["flaw_uv"])["Body"].read()
    # complete_uv = models.freeuv.inpaint(flaw_uv)
    # r2.put_object(Bucket="facium-jobs", Key=f"{job_id}/uv.png", Body=complete_uv)
    pass


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=32768,
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("facium-cloudflare")],
    concurrency_limit=1,
    timeout=60,
)
def flame_worker(job_id: str, r2_keys: dict):
    """Worker 2 - FLAME Fitting 3DDFA_V3/DECA."""
    pass


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    secrets=[modal.Secret.from_name("facium-cloudflare")],
    concurrency_limit=4,
    timeout=30,
)
def mediapipe_worker(job_id: str, r2_keys: dict):
    """Worker 1 - MediaPipe 478 landmarks CPU (puede correr también en Cloudflare Workers si se portara a WASM)."""
    pass


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    secrets=[modal.Secret.from_name("facium-cloudflare")],
    timeout=30,
)
def gnm_bake_worker(job_id: str, r2_keys: dict):
    """Worker 4 - GNM Bake + Heatmap + Report. Lee complete-uv de R2, escribe result.zip a R2 con lifecycle 60s."""
    pass


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    secrets=[modal.Secret.from_name("facium-cloudflare")],
    schedule=modal.Period(seconds=5),
)
def queue_pull_consumer():
    """
    HTTP Pull Consumer para Cloudflare Queues.
    Polls Queues REST API, despacha a workers Modal via .spawn().
    Alternativa a Queues consumer binding (que solo funciona dentro de Workers).
    Ver https://developers.cloudflare.com/queues/configuration/pull-consumers/
    """
    # TODO: implementar con httpx + Cloudflare Queues REST API
    # messages = httpx.get(f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/queues/{QUEUE_ID}/messages/pull")
    # for msg in messages: freeuv_worker.spawn(msg["job_id"], msg["r2_keys"])
    pass
