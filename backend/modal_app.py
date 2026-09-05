"""
Modal GPU workers para Vultus (híbrido Rust + Python).

Arquitectura híbrida (ADR-005):
- Rust (Axum) es dueño de Seam 1 API + Seam 2 queue + Worker 4 CPU
  (GNM bake, heatmap, report). Ver backend/crates/.
- Python aquí es solo sidecar ML GPU: MediaPipe / FLAME / FreeUV.
  Rust nunca importa torch/diffusers/mediapipe; los consume vía HTTP:
  `MlSidecarClient { landmarks, flame, freeuv }` -> `POST /ml/*`.

Prod: Cloudflare Queues (HTTP Pull Consumer) -> Modal -> R2
Local sin Modal: `python -m modal_app --serve :8081` expone el mismo
contrato /ml/* y el binario Rust lo consume vía ML_SIDECAR_URL.

Starter plan: $30/mes free (~50h T4 = ~9.300 compares). Cold start 1-2s.
Deploy: modal deploy backend/modal_app.py
Logs:   modal app logs vultus-workers
"""

import modal

app = modal.App("vultus-workers")

# Imagen base GPU con torch + diffusers + mediapipe
# Reusa Dockerfile.gpu local para paridad
image = (
    modal.Image.from_dockerfile("backend/Dockerfile.gpu")
    .pip_install("boto3", "httpx")  # R2 + Queues HTTP Pull
)

# Volume para cachear pesos FreeUV / FLAME / GNM (evita re-descarga en cold start)
weights = modal.Volume.from_name("vultus-weights", create_if_missing=True)

# Secrets: Cloudflare R2 + Queues creds
# modal secret create vultus-cloudflare CLOUDFLARE_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=16384,
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("vultus-cloudflare")],
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
    # flaw_uv = r2.get_object(Bucket="vultus-jobs", Key=r2_keys["flaw_uv"])["Body"].read()
    # complete_uv = models.freeuv.inpaint(flaw_uv)
    # r2.put_object(Bucket="vultus-jobs", Key=f"{job_id}/uv.png", Body=complete_uv)
    pass


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=32768,
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("vultus-cloudflare")],
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
    secrets=[modal.Secret.from_name("vultus-cloudflare")],
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
    secrets=[modal.Secret.from_name("vultus-cloudflare")],
    timeout=30,
)
def gnm_bake_worker(job_id: str, r2_keys: dict):
    """Worker 4 - DEPRECATED en híbrido: vive en Rust `vultus-workers-cpu`.

    Se mantiene el stub para no romper deploys antiguos.
    No añadir lógica aquí: `compute_heatmap` + `bake_bfm_to_gnm` en Rust.
    """
    raise NotImplementedError("moved to Rust vultus-workers-cpu")


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    secrets=[modal.Secret.from_name("vultus-cloudflare")],
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


# --- Sidecar HTTP consumido por Rust (MlSidecarClient) ---
# Mismo contrato en Modal (@app.function con web_endpoint) y en local
# (`python modal_app.py --serve`). Rust envía bytes, recibe bytes.
# Nunca se expone fuera del VPC/prod interno; sin auth externa.

try:
    from fastapi import FastAPI, Request, Response

    sidecar = FastAPI(title="vultus-ml-sidecar")

    @sidecar.post("/ml/landmarks")
    async def http_landmarks(request: Request):
        body = await request.body()
        # TODO Fase 1: models.mediapipe.landmarks(body) -> 478x3 json bytes
        return Response(content=b'{"todo":"landmarks"}', media_type="application/octet-stream")

    @sidecar.post("/ml/flame")
    async def http_flame(request: Request):
        await request.body()
        # TODO Fase 1: models.flame.fit(image, landmarks) -> flaw-uv bytes
        return Response(content=b'{"todo":"flaw-uv"}', media_type="application/octet-stream")

    @sidecar.post("/ml/freeuv")
    async def http_freeuv(request: Request):
        await request.body()
        # TODO Fase 1: models.freeuv.inpaint(flaw_uv) -> complete-uv bytes
        return Response(content=b'{"todo":"complete-uv"}', media_type="application/octet-stream")

    @app.function(image=image, cpu=1, memory=1024)
    @modal.fastapi_endpoint(method="POST")
    async def ml_endpoint(request: Request):
        path = request.url.path
        body = await request.body()
        job_id = request.headers.get("X-Job-Id", "unknown")
        if path.endswith("/landmarks"):
            return Response(content=b'{"todo":"landmarks"}', media_type="application/octet-stream")
        if path.endswith("/flame"):
            return Response(content=b'{"todo":"flaw-uv"}', media_type="application/octet-stream")
        if path.endswith("/freeuv"):
            # En prod delega a freeuv_worker.spawn(job_id, ...) y lee R2.
            return Response(content=b'{"todo":"complete-uv"}', media_type="application/octet-stream")
        return Response(content=b'{"error":"unknown ml route"}', status_code=404)
except ImportError:  # Modal image sin fastapi en tests unitarios
    sidecar = None  # type: ignore
