"""Completion neuronal texgan (Fase 1, issue 94).

Port de inferencia del decoder `TextureGAN` de FFHQ-UV (`RGB_Fitting`,
solo forward, sin el RGB fitting iterativo de minutos): el mismo input
da los mismos bytes. Sin torch top-level (lazy como `face_parsing.py`);
CI sin torch sigue en verde con los tests puros.

Via feed-forward dentro de TTL 60:
1. Verifica el puente (`texgan_ffhq_uv.pth` en candidatos `TEXGAN_DIR`).
   Sin el la completion cae a piel media foto-derivada (via actual),
   documentado en stats (`texgan=0.0`), nunca magenta silencioso.
2. Inicializa el latente en `w_avg` del checkpoint (clave
   `mapping.w_avg`, sin sidecar `.pt` ni 100k muestras) y lo ajusta con
   Adam de presupuesto fijo (`TEXGAN_FIT_STEPS`) sobre MSE enmascarado
   en UV 512: solo texeles validos de la foto; `noise_mode='const'`
   (buffers del checkpoint, determinista).
3. La salida neuronal se usa SOLO en texeles no validos: lo valido
   sigue muestreando la foto (fidelidad primero).

Arquitectura: `SynthesisNetwork` 1024 skip (`w_dim=512`, `num_ws=18`,
`channel_base=32768`, `channel_max=512`) + `MappingNetwork` (solo por
sus keys `mapping.*` en la carga estricta). Operadores puros torch con
el mismo math upstream (modconv fusionada, upfirdn2d referencia,
bias_act lrelu alpha=0.2 gain=sqrt(2)): sin plugins CUDA. Carga
`strict` con 0 faltantes/inesperados como `face_parsing.py`.

Config solo por env (`TEXGAN_DIR`; mas `WEIGHTS_ROOT`/`WEIGHTS_DIR` ya
existentes). Entradas bytes probados, salidas bytes. Sin logging.
"""

from __future__ import annotations

# mypy: allow-untyped-defs, allow-untyped-calls
# Decoder texgan vendored: interior torch sin stubs en CI (torch es
# lazy/opcional, idiom face_parsing.py). La API publica si va tipada.
import math
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

# --- Constantes del decoder (espejo de network/texgan.py + stylegan2) ---
TEXGAN_W_DIM = 512
TEXGAN_NUM_WS = 18
TEXGAN_UV_NATIVE = 1024
TEXGAN_IMG_CHANNELS = 3
TEXGAN_CHANNEL_BASE = 32768
TEXGAN_CHANNEL_MAX = 512
# Presupuesto del ajuste de latente (forward+backward por paso en T4).
# Forense 512: 25 pasos caben en TEXTURE_TIMEOUT_SECS=30 con margen
# (T4 ~0.6s/paso medido via duration_ms + nvidia-smi dmon pico VRAM ~6GB
# con 1 input por GPU y max_containers=2; CPU local ~4s/paso: los tests
# usan 2 vs 6 pasos para verificar descenso sin pagar 25 en CI).
# V3 1024 largo: V3_TEXGAN_FIT_STEPS=300 en backend/v3_contract.py con
# V3_TEXTURE_TIMEOUT_SECS=900 (worker de minutos, sin TTL 60).
# E2E rapido usa V3_E2E_TEXGAN_STEPS=2 solo como smoke; prod usa el largo.
TEXGAN_FIT_STEPS = 25
TEXGAN_FIT_STEPS_MIN = 1
TEXGAN_FIT_STEPS_MAX = 100
TEXGAN_FIT_LR = 0.1
TEXGAN_FIT_REG = 0.01
TEXGAN_PTH_NAME = "texgan_ffhq_uv.pth"

_BLOCK_RESOLUTIONS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
_MAPPING_LAYERS = 8


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def texgan_dir() -> str:
    """Checkpoint TexGAN desde env. Vacio = ausente."""
    return _env("TEXGAN_DIR")


def fit_steps() -> int:
    """Presupuesto de pasos Adam (env con default acotado)."""
    try:
        raw = int(_env("TEXGAN_FIT_STEPS") or str(TEXGAN_FIT_STEPS))
    except ValueError:
        return TEXGAN_FIT_STEPS
    return max(TEXGAN_FIT_STEPS_MIN, min(TEXGAN_FIT_STEPS_MAX, raw))


def _candidate_dirs() -> list[str]:
    cands: list[str] = []
    for d in (texgan_dir(), _env("FFHQ_UV_DIR")):
        if d and d not in cands:
            cands.append(d)
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        fallback = os.path.join(root, "checkpoints", "texgan_model")
        if fallback not in cands:
            cands.append(fallback)
    return cands


def find_pth() -> str | None:
    """Ruta del checkpoint en candidatos, o None si ausente."""
    for d in _candidate_dirs():
        cand = os.path.join(d, TEXGAN_PTH_NAME)
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
        except OSError:
            continue
    return None


def weights_present() -> bool:
    """True solo con el pth no vacio en candidatos. Total, nunca raise."""
    return find_pth() is not None


def torch_available() -> bool:
    """True si torch importa. Total, nunca raise."""
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def texgan_available() -> bool:
    """True si hay backend (torch) y pesos. Gate de la via neuronal."""
    return torch_available() and weights_present()


def expected_state_keys() -> frozenset[str]:
    """Set exacto de keys del checkpoint (puro python, sin torch).

    Espejo de la jerarquia: `mapping.fc0..7.{weight,bias}` + `w_avg`
    (17) + `synthesis.b4.{const,conv1.*,torgb.*}` + `resample_filter`
    (13) + bloques b8..b1024 `{conv0.*,conv1.*,torgb.*,resample_filter}`
    (19 x 8). Total 182, verificado contra el pth real.
    """
    keys: set[str] = set()
    for idx in range(_MAPPING_LAYERS):
        keys.add(f"mapping.fc{idx}.weight")
        keys.add(f"mapping.fc{idx}.bias")
    keys.add("mapping.w_avg")
    for res in _BLOCK_RESOLUTIONS:
        keys.add(f"synthesis.b{res}.resample_filter")
        if res == 4:
            keys.add("synthesis.b4.const")
        layers = ("conv1",) if res == 4 else ("conv0", "conv1")
        for layer in layers:
            keys.add(f"synthesis.b{res}.{layer}.weight")
            keys.add(f"synthesis.b{res}.{layer}.affine.weight")
            keys.add(f"synthesis.b{res}.{layer}.affine.bias")
            keys.add(f"synthesis.b{res}.{layer}.noise_const")
            keys.add(f"synthesis.b{res}.{layer}.noise_strength")
            keys.add(f"synthesis.b{res}.{layer}.bias")
            keys.add(f"synthesis.b{res}.{layer}.resample_filter")
        keys.add(f"synthesis.b{res}.torgb.weight")
        keys.add(f"synthesis.b{res}.torgb.affine.weight")
        keys.add(f"synthesis.b{res}.torgb.affine.bias")
        keys.add(f"synthesis.b{res}.torgb.bias")
    return frozenset(keys)


def check_state_keys(keys: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(faltantes, inesperadas) contra el set esperado. Puro python."""
    have = frozenset(keys)
    want = expected_state_keys()
    return (want - have, have - want)


def blend_completion(sampled: bytes, synth: bytes, valid: bytes) -> bytes:
    """Fidelidad primero: texel valido de la foto, resto del decoder.

    `sampled`/`synth`: RGB plano 512x512; `valid`: 1 byte por texel
    (0/1). Total: ValueError si largos no cuadran.
    """
    expect_rgb = 512 * 512 * 3
    expect_mask = 512 * 512
    if len(sampled) != expect_rgb or len(synth) != expect_rgb or len(valid) != expect_mask:
        raise ValueError(
            f"texgan blend lengths invalid: sampled={len(sampled)} synth={len(synth)} valid={len(valid)}"
        )
    a = np.frombuffer(sampled, dtype=np.uint8).reshape(512, 512, 3)
    b = np.frombuffer(synth, dtype=np.uint8).reshape(512, 512, 3)
    m = np.frombuffer(valid, dtype=np.uint8).reshape(512, 512, 1) != 0
    return np.where(m, a, b).astype(np.uint8).tobytes()


def _build_net():
    """SynthesisNetwork + Mapping con keys exactas upstream.

    Todo torch es lazy aqui dentro (idiom `face_parsing.py`): sin torch
    el modulo importa igual. Operadores con el math exacto del
    upstream (derivaciones en docstrings internos): modconv solo via
    fusionada eval-fp32, upfirdn2d referencia, bias_act lrelu
    alpha=0.2 gain=sqrt(2). Lanza RuntimeError con causa sin torch.
    """
    import math

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(f"torch missing for texgan decoder: {exc}") from exc
    import torch.nn.functional as _F  # type: ignore[import-not-found]

    _LRELU_ALPHA = 0.2
    _LRELU_GAIN = math.sqrt(2.0)

    def _setup_filter():
        # setup_filter([1,3,3,1]): outer no separable + normaliza (suma 64).
        f = torch.tensor([1.0, 3.0, 3.0, 1.0], dtype=torch.float32)
        f = f[:, None] * f[None, :]
        return f / f.sum()

    def _bias_act_lrelu(x, bias, gain):
        # bias_act referencia: bias + leaky_relu(alpha) * gain; sin clamp.
        if bias is not None:
            x = x + bias.reshape(1, -1, 1, 1).to(x.dtype)
        return _F.leaky_relu(x, _LRELU_ALPHA) * gain

    def _fused_conv(x, weight, styles):
        # Caso upstream up=1, padding=1, kernel=3 (conv1 de cada bloque).
        b = x.shape[0]
        out_ch, in_ch, kh, _kw = weight.shape
        w = weight.unsqueeze(0) * styles.reshape(b, 1, -1, 1, 1)
        d = (w.square().sum(dim=(2, 3, 4)) + 1e-8).rsqrt()
        w = w * d.reshape(b, -1, 1, 1, 1)
        xx = x.reshape(1, -1, *x.shape[2:])
        ww = w.reshape(-1, in_ch, kh, kh)
        y = _F.conv2d(xx, ww.to(xx.dtype), padding=1, groups=b)
        return y.reshape(b, out_ch, y.shape[2], y.shape[3])

    def _fused_upconv(x, weight, styles, filt):
        # Caso upstream up=2, padding=1, kernel=3, f 4-tap (conv0).
        # Equivale al path "upsampling => transpose strided conv" de
        # conv2d_resample: transpose stride=2 padding=[0,0] groups=B,
        # luego FIR con gain=up^2=4 y padding [1,1,1,1] (sustitucion de
        # px0=3,px1=2,py0=3,py1=2, pxt=pyt=0 para f 4-tap y kh=3).
        b, in_ch, _h, _w = x.shape
        out_ch = weight.shape[0]
        kh = weight.shape[2]
        w = weight.unsqueeze(0) * styles.reshape(b, 1, -1, 1, 1)
        d = (w.square().sum(dim=(2, 3, 4)) + 1e-8).rsqrt()
        w = w * d.reshape(b, -1, 1, 1, 1)
        ww = w.transpose(1, 2).reshape(b * in_ch, out_ch, kh, kh)
        xx = x.reshape(1, b * in_ch, x.shape[2], x.shape[3])
        y = _F.conv_transpose2d(xx, ww.to(xx.dtype), stride=2, padding=0, groups=b)
        ff = (filt * 4.0).to(y.dtype)
        ff = ff.flip([0, 1])[None, None].repeat(y.shape[1], 1, 1, 1)
        y = _F.pad(y, [1, 1, 1, 1])
        y = _F.conv2d(y, ff, groups=y.shape[1])
        return y.reshape(b, out_ch, y.shape[2], y.shape[3])

    def _upsample_img(img, filt):
        # upsample2d(img, f) con up=2: padding derivado [2,1,2,1]
        # (padx0+(fw+up-1)//2 etc.), gain=up^2=4.
        n, c, h, w = img.shape
        up = 2
        x = img.reshape(n, c, h, 1, w, 1)
        x = _F.pad(x, [0, up - 1, 0, 0, 0, up - 1])
        x = x.reshape(n, c, h * up, w * up)
        ff = (filt * 4.0).to(x.dtype)
        ff = ff.flip([0, 1])[None, None].repeat(c, 1, 1, 1)
        x = _F.pad(x, [2, 1, 2, 1])
        return _F.conv2d(x, ff, groups=c)

    class _FC(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self, in_f, out_f, bias_init=0.0, lr_mult=1.0):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(out_f, in_f) / lr_mult)
            self.bias = torch.nn.Parameter(torch.full((out_f,), float(bias_init)))
            self._gain = lr_mult / math.sqrt(in_f)
            self._bias_gain = lr_mult

        def forward(self, x):
            # Mapping upstream: linear + bias (activation lineal).
            w = self.weight.to(x.dtype) * self._gain
            b = self.bias.to(x.dtype) * self._bias_gain
            return torch.addmm(b.unsqueeze(0), x, w.t())

    class _SynthLayer(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self, in_ch, out_ch, res, up):
            super().__init__()
            self.resolution = res
            self.up = up
            self.register_buffer("resample_filter", _setup_filter())
            self.affine = _FC(TEXGAN_W_DIM, in_ch, bias_init=1.0)
            self.weight = torch.nn.Parameter(torch.randn(out_ch, in_ch, 3, 3))
            self.register_buffer("noise_const", torch.randn(res, res))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
            self.bias = torch.nn.Parameter(torch.zeros(out_ch))

        def forward(self, x, w):
            styles = self.affine(w)
            if self.up > 1:
                y = _fused_upconv(x, self.weight, styles, self.resample_filter)
            else:
                y = _fused_conv(x, self.weight, styles)
            const: Any = self.noise_const
            strength: Any = self.noise_strength
            noise = const * strength
            y = y.add_(noise.to(y.dtype))
            return _bias_act_lrelu(y, self.bias, _LRELU_GAIN)

    class _ToRGB(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self, in_ch):
            super().__init__()
            self.affine = _FC(TEXGAN_W_DIM, in_ch, bias_init=1.0)
            self.weight = torch.nn.Parameter(torch.randn(TEXGAN_IMG_CHANNELS, in_ch, 1, 1))
            self.bias = torch.nn.Parameter(torch.zeros(TEXGAN_IMG_CHANNELS))
            self._gain = 1.0 / math.sqrt(in_ch)

        def forward(self, x, w):
            # ToRGB upstream: 1x1 mod sin demod + bias lineal (sin act).
            styles = self.affine(w) * self._gain
            b = x.shape[0]
            ww = self.weight.unsqueeze(0) * styles.reshape(b, 1, -1, 1, 1)
            xx = x.reshape(1, -1, *x.shape[2:])
            www = ww.reshape(-1, x.shape[1], 1, 1)
            y = _F.conv2d(xx, www.to(xx.dtype), groups=b)
            y = y.reshape(b, -1, y.shape[2], y.shape[3])
            return y + self.bias.reshape(1, -1, 1, 1).to(y.dtype)

    class _Block(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self, in_ch, out_ch, res):
            super().__init__()
            self.in_channels = in_ch
            self.resolution = res
            self.register_buffer("resample_filter", _setup_filter())
            self.num_conv = 0
            self.num_torgb = 0
            if in_ch == 0:
                self.const = torch.nn.Parameter(torch.randn(out_ch, res, res))
            else:
                self.conv0 = _SynthLayer(in_ch, out_ch, res, 2)
                self.num_conv += 1
            self.conv1 = _SynthLayer(out_ch, out_ch, res, 1)
            self.num_conv += 1
            self.torgb = _ToRGB(out_ch)
            self.num_torgb += 1

        def forward(self, x, img, ws):
            # Orden upstream skip: conv(s) con sus w, torgb con la suya.
            parts = ws.unbind(dim=1)
            it = iter(parts)
            if self.in_channels == 0:
                const = self.const.to(torch.float32).unsqueeze(0).repeat(ws.shape[0], 1, 1, 1)
                x = self.conv1(const, next(it))
            else:
                x = self.conv0(x.to(torch.float32), next(it))
                x = self.conv1(x, next(it))
            if img is not None:
                img = _upsample_img(img, self.resample_filter)
            y = self.torgb(x, next(it)).to(torch.float32)
            img = y if img is None else img.add_(y)
            return x, img

    class _Synthesis(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self):
            super().__init__()
            self.w_dim = TEXGAN_W_DIM
            self.img_resolution = TEXGAN_UV_NATIVE
            channels = {res: min(TEXGAN_CHANNEL_BASE // res, TEXGAN_CHANNEL_MAX) for res in _BLOCK_RESOLUTIONS}
            self.num_ws = 0
            for res in _BLOCK_RESOLUTIONS:
                in_ch = channels[res // 2] if res > 4 else 0
                block = _Block(in_ch, channels[res], res)
                setattr(self, f"b{res}", block)
                self.num_ws += block.num_conv
                if res == TEXGAN_UV_NATIVE:
                    self.num_ws += block.num_torgb

        def forward(self, ws, noise_mode="const"):
            assert noise_mode == "const"
            ws = ws.to(torch.float32)
            idx = 0
            x = None
            img = None
            for res in _BLOCK_RESOLUTIONS:
                block = getattr(self, f"b{res}")
                take = block.num_conv + block.num_torgb
                cur = ws.narrow(1, idx, take)
                idx += block.num_conv
                x, img = block(x, img, cur)
            return img

    class _Mapping(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self):
            super().__init__()
            for i in range(_MAPPING_LAYERS):
                setattr(self, f"fc{i}", _FC(TEXGAN_W_DIM, TEXGAN_W_DIM, lr_mult=0.01))
            self.register_buffer("w_avg", torch.zeros(TEXGAN_W_DIM))

    class _Generator(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self):
            super().__init__()
            self.synthesis = _Synthesis()
            self.mapping = _Mapping()

    return _Generator()


class TexGanDecoder:
    """Decoder texgan con torch lazy. Construir via `load_decoder`.

    `missing_keys()`/`unexpected_keys()` vacias <=> carga estricta
    verificada (0 faltantes como parsing). `synth_uv_map` es el
    `TextureGAN.synth_uv_map` upstream: W directo, `noise_mode='const'`,
    salida 0-1.
    """

    def __init__(self, net: Any, missing: list[str], unexpected: list[str]) -> None:
        self._net = net
        self._missing = list(missing)
        self._unexpected = list(unexpected)
        self._device = "cpu"

    def _ensure_device(self, device: Any) -> None:

        dev = str(device)
        if dev != self._device:
            self._net.to(dev)
            self._device = dev

    def missing_keys(self) -> list[str]:
        return list(self._missing)

    def unexpected_keys(self) -> list[str]:
        return list(self._unexpected)

    def w_avg_batch(self, batch: int = 1) -> Any:
        import torch

        w_avg = self._net.mapping.w_avg.to(torch.float32)
        return w_avg.reshape(1, 1, TEXGAN_W_DIM).repeat(batch, TEXGAN_NUM_WS, 1)

    def synth_uv_map(self, w: Any) -> Any:
        import torch

        self._ensure_device(w.device)
        self._net.eval()
        with torch.no_grad():
            img = self._net.synthesis(w.to(torch.float32), noise_mode="const")
        return ((img + 1.0) * 0.5).clamp(0.0, 1.0)

    def fit_latent(self, target_01: Any, valid_01: Any, steps: int, lr: float = TEXGAN_FIT_LR, reg: float = TEXGAN_FIT_REG) -> Any:
        """Ajusta w desde w_avg con Adam de presupuesto fijo.

        Loss: MSE enmascarado a 512 + reg||w-w_avg||^2. Determinista
        con flags de `ensure_deterministic_texture` (init fijo, const
        noise, sin muestreo). Retorna w ajustada (detach).
        """
        import torch

        net = self._net
        net.eval()
        w0 = self.w_avg_batch()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ensure_device(dev)
        w0 = w0.to(dev)
        w = w0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        tgt = target_01.to(torch.float32).to(dev)
        mask = valid_01.to(torch.float32).to(dev)
        denom = mask.sum().clamp_min(1.0)
        for _ in range(max(1, steps)):
            opt.zero_grad(set_to_none=True)
            img1024 = net.synthesis(w, noise_mode="const")
            img01 = ((img1024 + 1.0) * 0.5).clamp(0.0, 1.0)
            img512 = torch.nn.functional.interpolate(img01, size=(512, 512), mode="bilinear", align_corners=False)
            diff = (img512 - tgt) * mask
            loss = (diff * diff).sum() / denom + reg * ((w - w0) * (w - w0)).mean()
            loss.backward()
            opt.step()
        return w.detach()


_decoder_cache: dict[str, TexGanDecoder] = {}


def load_decoder(pth_path: str) -> TexGanDecoder:
    """Carga estricta del checkpoint (0 faltantes/inesperados o raise).

    Cacheada por ruta. RuntimeError con causa si falta torch, el archivo
    o hay mismatch de keys (nunca carga parcial silenciosa).
    """
    cached = _decoder_cache.get(pth_path)
    if cached is not None:
        return cached
    import torch

    if not os.path.isfile(pth_path):
        raise RuntimeError(f"texgan checkpoint missing: {pth_path}")
    net = _build_net()
    try:
        state = torch.load(pth_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"texgan checkpoint unreadable: {pth_path}") from exc
    if not isinstance(state, dict):
        raise TypeError(f"texgan checkpoint unexpected type: {type(state).__name__}")
    loaded = net.load_state_dict(state, strict=False)
    missing = sorted(str(k) for k in loaded.missing_keys)
    unexpected = sorted(str(k) for k in loaded.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"texgan checkpoint mismatch: missing={missing} unexpected={unexpected}")
    net.eval()
    dec = TexGanDecoder(net, missing, unexpected)
    _decoder_cache[pth_path] = dec
    return dec


def synth_to_bytes(synth_01: NDArray[np.float64]) -> bytes:
    """Sintesis 0-1 float64 512 -> RGB plano uint8."""
    arr = np.clip(synth_01, 0.0, 1.0)
    return np.rint(arr * 255.0).astype(np.uint8).tobytes()


def match_color_to_sampled(
    synth_255: NDArray[np.float64], sampled_255: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Iguala tono del decoder al de la foto (espejo de `match_color` upstream).

    Ajuste afin por canal sobre texeles validos: el latente ajustado por
    MSE puede derivar en tono global (p.ej. verdoso bajo target DPR);
    sin esto la completion canta contra lo muestreado. Puro numpy.
    Total: retorna synth intacta si no hay validos o varianza nula.
    """
    try:
        if not bool(valid.any()):
            return synth_255
        out = np.asarray(synth_255, dtype=np.float64).copy()
        ref = np.asarray(sampled_255, dtype=np.float64)
        for c in range(3):
            sv = ref[valid, c]
            gv = out[valid, c]
            mu_s, mu_g = float(sv.mean()), float(gv.mean())
            sd_s, sd_g = float(sv.std()), float(gv.std())
            if sd_g < 1e-9 or not (math.isfinite(mu_s) and math.isfinite(mu_g)):
                continue
            out[..., c] = (out[..., c] - mu_g) * (sd_s / sd_g) + mu_s
        if not bool(np.isfinite(out).all()):
            return synth_255
        return np.clip(out, 0.0, 255.0)
    except (IndexError, ValueError, TypeError):
        return synth_255


def neural_completion(
    sampled_255: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> NDArray[np.float64] | None:
    """Completion neuronal texgan sobre texeles no validos.

    Ajusta el latente desde `w_avg` con Adam de presupuesto fijo sobre
    MSE enmascarado y sintetiza el UV 1024 -> 512 BILINEAR. Retorna
    float64 512x512x3 en 0-255, o None si no hay backend (torch) o
    pesos: el caller cae a piel media foto-derivada (via documentada
    en stats). Total: nunca lanza.
    """
    try:
        if not texgan_available():
            return None
        import torch
        from PIL import Image

        pth = find_pth()
        if pth is None:
            return None
        dec = load_decoder(pth)
        tgt = torch.from_numpy(np.ascontiguousarray(sampled_255 / 255.0)).permute(2, 0, 1).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(1, 3, 1, 1)
        w = dec.fit_latent(tgt, mask, steps=fit_steps())
        img01 = dec.synth_uv_map(w)
        arr = img01.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        small = Image.fromarray(np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
            (512, 512), Image.Resampling.BILINEAR
        )
        synth = np.asarray(small, dtype=np.float64)
        return match_color_to_sampled(synth, np.asarray(sampled_255, dtype=np.float64), np.asarray(valid, dtype=bool))
    except Exception:  # noqa: BLE001 - sin backend/pesos o fallo numerico: via piel media
        return None
