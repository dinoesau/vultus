"""Chequeo efimero Fase 5 en GPU Modal (app aparte, sin tocar prod).

Mide: forward texgan 1024 + ajuste 25 pasos en T4 (presupuesto
TEXTURE_TIMEOUT_SECS=30), forward Deep3D 224, DPR 512, determinismo x2.
Solo lectura del Volume. Nunca `modal serve` / `modal deploy`.
Uso: `modal run scripts/modal_check_fase5.py`
"""

import time

import modal

app = modal.App("vultus-fase5-check")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04@sha256:e4a0337cd453e253ede68d4fe564941bc30336143c770eaad3c9b0db506c3bce",
        add_python="3.10",
    )
    .pip_install("torch==2.13.0", "torchvision==0.28.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy==1.26.4", "scipy==1.11.4", "Pillow")
    .add_local_python_source("backend")
)
weights = modal.Volume.from_name("vultus-weights", create_if_missing=False)


@app.function(image=image, gpu="T4", volumes={"/weights": weights}, timeout=600)
def check() -> dict:
    import numpy as np
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out: dict = {"cuda": bool(torch.cuda.is_available())}
    t0 = time.perf_counter()
    from backend.texgan import fit_steps, load_decoder

    dec = load_decoder("/weights/checkpoints/texgan_model/texgan_ffhq_uv.pth")
    out["texgan_keys_ok"] = dec.missing_keys() == [] and dec.unexpected_keys() == []
    net = dec._net.cuda().eval()  # noqa: SLF001 - chequeo efimero, acceso interno documentado
    w = dec.w_avg_batch().cuda()
    torch.cuda.synchronize()
    s = time.perf_counter()
    with torch.no_grad():
        img = net.synthesis(w, noise_mode="const")
    torch.cuda.synchronize()
    out["synth1024_ms"] = round((time.perf_counter() - s) * 1000, 1)
    s = time.perf_counter()
    wfit = w.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([wfit], lr=0.1)
    tgt = torch.rand(1, 3, 512, 512, device="cuda")
    msk = torch.ones(1, 3, 512, 512, device="cuda")
    for _ in range(fit_steps()):
        opt.zero_grad(set_to_none=True)
        o = net.synthesis(wfit, noise_mode="const")
        o = ((o + 1.0) * 0.5).clamp(0, 1)
        o = torch.nn.functional.interpolate(o, size=(512, 512), mode="bilinear", align_corners=False)
        loss = (((o - tgt) * msk) ** 2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    out["fit25_ms"] = round((time.perf_counter() - s) * 1000, 1)
    with torch.no_grad():
        a = net.synthesis(dec.w_avg_batch().cuda(), noise_mode="const")
        b = net.synthesis(dec.w_avg_batch().cuda(), noise_mode="const")
    out["synth_deterministic"] = bool(torch.equal(a, b))
    tgt = torch.rand(1, 3, 512, 512, device="cuda")
    msk = torch.ones(1, 3, 512, 512, device="cuda")
    w1 = dec.fit_latent(tgt, msk, steps=5)
    w2 = dec.fit_latent(tgt, msk, steps=5)
    out["fit_runs"] = bool(w1.shape == (1, 18, 512) and w2.shape == (1, 18, 512))
    out["fit_deterministic"] = bool(torch.equal(w1, w2))

    from backend.deep3d import load_recon

    s = time.perf_counter()
    recon = load_recon("/weights/checkpoints/deep3d_model/epoch_latest.pth")
    x = torch.rand(1, 3, 224, 224, device="cuda")
    with torch.no_grad():
        c = recon.forward_coeffs(x.cuda() if hasattr(x, "cuda") else x)
    torch.cuda.synchronize()
    out["deep3d_ms"] = round((time.perf_counter() - s) * 1000, 1)
    out["deep3d_shape"] = list(c.shape)

    from backend.dpr import load_light_net

    s = time.perf_counter()
    lnet = load_light_net("/weights/checkpoints/dpr_model/trained_model_03.t7")
    L = np.random.RandomState(0).rand(512, 512)
    sh = lnet.estimate(L)
    torch.cuda.synchronize()
    out["dpr_ms"] = round((time.perf_counter() - s) * 1000, 1)
    out["dpr_sh0"] = round(float(sh[0]), 4) if sh is not None else None
    out["total_s"] = round(time.perf_counter() - t0, 1)
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(check.remote(), indent=1))
