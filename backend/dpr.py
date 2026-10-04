"""Luz DPR SH9 (Fase 3, issue 94).

Port de inferencia del `HourglassNet` gris de DPR
(`defineHourglass_512_gray_skip.py`, solo forward, sin torch
top-level como `face_parsing.py`). El `.t7` es un state_dict PyTorch
(250 tensores, verificado); carga `strict` con 0 faltantes.

Uso: `estimate_sh` predice 9 SH grises del canal L 512 en una pasada
(dummy de target_light a ceros: solo afecta la imagen relit, nunca la
luz predicha). El bake divide el sombreado relativo (tono preservado):
`albedo = L * mean(shading)/shading`. Sin torch/pesos el bake cae a
gray-world (via documentada en stats `dpr`).

Modulos con keys exactas upstream: `pre_conv/bn`, `light.*` (top +
`HG0.middle` compartido), bloques HG3..HG0 (`upper`/`low1` con affine,
`low2`/`upper` InstanceNorm sin affine), colas `conv_1..3`/`output`.
`BasicBlock` con `batchNorm_type=1` no registra affine (sin keys, como
el upstream).

Config solo por env (`DPR_DIR`; mas `WEIGHTS_ROOT`/`WEIGHTS_DIR` ya
existentes). Sin logging.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

# --- Constantes DPR (espejo del HourglassNet gris) ---
DPR_SH = 9
DPR_INPUT_SIZE = 512
DPR_LIGHT_IN = 27
DPR_LIGHT_MID = 128
EXPECTED_DPR_KEYS = 250
DPR_T7_NAME = "trained_model_03.t7"

_BN_NAMES = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def dpr_dir() -> str:
    """Checkpoint DPR desde env. Vacio = ausente."""
    return _env("DPR_DIR")


def _candidate_dirs() -> list[str]:
    cands: list[str] = []
    if dpr_dir() and dpr_dir() not in cands:
        cands.append(dpr_dir())
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        fallback = os.path.join(root, "checkpoints", "dpr_model")
        if fallback not in cands:
            cands.append(fallback)
    return cands


def find_t7() -> str | None:
    """Ruta del checkpoint en candidatos, o None si ausente."""
    for d in _candidate_dirs():
        cand = os.path.join(d, DPR_T7_NAME)
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
        except OSError:
            continue
    return None


def weights_present() -> bool:
    """True solo con el t7 no vacio. Total, nunca raise."""
    return find_t7() is not None


def torch_available() -> bool:
    """True si torch importa. Total, nunca raise."""
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def dpr_available() -> bool:
    """Gate de la via DPR: torch + t7."""
    return torch_available() and weights_present()


def _basic_keys(affine: bool) -> list[str]:
    keys = ["conv1.weight", "conv2.weight"]
    if affine:
        for bn in ("bn1", "bn2"):
            keys.extend(f"{bn}.{n}" for n in _BN_NAMES)
    keys.append("shortcuts.weight")
    return keys


def expected_dpr_keys() -> frozenset[str]:
    """Set exacto de keys del state_dict (puro python, sin torch).

    Espejo de la jerarquia con el modulo `light` compartido (top-level
    + `HG0.middle`): 250 keys verificadas contra el t7 real.
    """
    keys: set[str] = set()
    keys.add("pre_conv.weight")
    keys.add("pre_conv.bias")
    keys.update(f"pre_bn.{n}" for n in _BN_NAMES)
    light = [
        "predict_FC1.weight",
        "predict_relu1.weight",
        "predict_FC2.weight",
        "post_FC1.weight",
        "post_relu1.weight",
        "post_FC2.weight",
    ]
    keys.update(f"light.{k}" for k in light)
    tails = ["conv_1", "conv_2", "conv_3"]
    for t in tails:
        keys.add(f"{t}.weight")
        keys.add(f"{t}.bias")
    for b in ("bn_1", "bn_2", "bn_3"):
        keys.update(f"{b}.{n}" for n in _BN_NAMES)
    keys.add("output.weight")
    keys.add("output.bias")

    def _own(prefix: str) -> None:
        for leaf in _basic_keys(affine=False):
            keys.add(f"{prefix}.upper.{leaf}")
        for leaf in _basic_keys(affine=True):
            keys.add(f"{prefix}.low1.{leaf}")
        for leaf in _basic_keys(affine=False):
            keys.add(f"{prefix}.low2.{leaf}")

    def _chain(prefix: str, depth: int) -> None:
        # Cadena middle anidada de `depth` bloques + light al fondo.
        _own(prefix)
        if depth == 0:
            keys.update(f"{prefix}.middle.{k}" for k in light)
        else:
            _chain(f"{prefix}.middle", depth - 1)

    _chain("HG0", 0)
    _chain("HG1", 1)
    _chain("HG2", 2)
    _chain("HG3", 3)
    return frozenset(keys)


def check_dpr_keys(keys: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(faltantes, inesperadas) contra el set esperado. Puro python."""
    have = frozenset(keys)
    want = expected_dpr_keys()
    return (want - have, have - want)


def sh_basis(normals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Base SH Sloan (9) sobre normales unitarias [...,3] -> [...,9].

    Puro numpy. Convencion del bake (verificada contra DPR: el lado
    brillante de la foto coincide con mayor sombreado).
    """
    n = np.asarray(normals, dtype=np.float64)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack(
        [
            np.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ],
        axis=-1,
    )


def _build_net():
    """HourglassNet gris con keys exactas upstream. Torch lazy dentro.

    Sin torch el modulo importa igual (idiom `face_parsing.py`).
    Lanza RuntimeError con causa sin torch.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(f"torch missing for dpr light net: {exc}") from exc
    import torch.nn.functional as _F

    class _Basic(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self, inp, outp, affine):
            super().__init__()
            self.inplanes = inp
            self.outplanes = outp
            self.conv1 = nn.Conv2d(inp, outp, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(outp, outp, 3, padding=1, bias=False)
            norm = nn.BatchNorm2d if affine else nn.InstanceNorm2d
            self.bn1 = norm(outp, affine=affine)
            self.bn2 = norm(outp, affine=affine)
            self.shortcuts = nn.Conv2d(inp, outp, 1, bias=False)

        def forward(self, x):
            out = _F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + (self.shortcuts(x) if self.inplanes != self.outplanes else x)
            return _F.relu(out)

    class _HG(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self, inp, mid, middle):
            super().__init__()
            self.skipLayer = True
            self.upper = _Basic(inp, inp, False)
            self.downSample = nn.MaxPool2d(2, 2)
            self.upSample = nn.Upsample(scale_factor=2, mode="nearest")
            self.low1 = _Basic(inp, mid, True)
            self.middle = middle
            self.low2 = _Basic(mid, inp, False)

        def forward(self, x, light, count, skip_count):
            mid_out, pred_light = self.middle(self.low1(self.downSample(x)), light, count + 1, skip_count)
            out = self.upSample(self.low2(mid_out))
            if count >= skip_count and self.skipLayer:
                out = out + self.upper(x)
            return out, pred_light

    class _Light(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self):
            super().__init__()
            self.predict_FC1 = nn.Conv2d(DPR_LIGHT_IN, DPR_LIGHT_MID, 1, bias=False)
            self.predict_relu1 = nn.PReLU()
            self.predict_FC2 = nn.Conv2d(DPR_LIGHT_MID, DPR_SH, 1, bias=False)
            self.post_FC1 = nn.Conv2d(DPR_SH, DPR_LIGHT_MID, 1, bias=False)
            self.post_relu1 = nn.PReLU()
            self.post_FC2 = nn.Conv2d(DPR_LIGHT_MID, DPR_LIGHT_IN, 1, bias=False)
            self.post_relu2 = nn.ReLU()

        def forward(self, inner, target, count, skip_count):
            x = inner[:, 0:DPR_LIGHT_IN]
            feat = x.mean(dim=(2, 3), keepdim=True)
            light = self.predict_FC2(self.predict_relu1(self.predict_FC1(feat)))
            up = self.post_relu2(self.post_FC2(self.post_relu1(self.post_FC1(target))))
            inner[:, 0:DPR_LIGHT_IN] = up.repeat(1, 1, *x.shape[2:])
            return inner, light

    # Jerarquia EXACTA del checkpoint (250 keys, tensores duplicados
    # identicos = modulos compartidos en entrenamiento): cadena anidada
    # con el `light` compartido (top-level + HG0.middle). El forward de
    # inferencia usa la cadena HG3 (la mas profunda).
    light = _Light()

    class _Net(nn.Module):  # type: ignore[misc]  # torch ausente en CI
        def __init__(self):
            super().__init__()
            self.pre_conv = nn.Conv2d(1, 16, 5, padding=2)
            self.pre_bn = nn.BatchNorm2d(16)
            self.light = light
            hg0 = _HG(64, 155, light)
            self.HG0 = hg0
            hg1 = _HG(32, 64, hg0)
            self.HG1 = hg1
            hg2 = _HG(16, 32, hg1)
            self.HG2 = hg2
            self.HG3 = _HG(16, 16, hg2)
            self.conv_1 = nn.Conv2d(16, 16, 3, padding=1)
            self.bn_1 = nn.BatchNorm2d(16)
            self.conv_2 = nn.Conv2d(16, 16, 1)
            self.bn_2 = nn.BatchNorm2d(16)
            self.conv_3 = nn.Conv2d(16, 16, 1)
            self.bn_3 = nn.BatchNorm2d(16)
            self.output = nn.Conv2d(16, 1, 1)

        def forward(self, x, target_light, skip_count=0):
            feat = _F.relu(self.pre_bn(self.pre_conv(x)))
            feat, out_light = self.HG3(feat, target_light, 0, skip_count)
            feat = _F.relu(self.bn_1(self.conv_1(feat)))
            feat = _F.relu(self.bn_2(self.conv_2(feat)))
            feat = _F.relu(self.bn_3(self.conv_3(feat)))
            return torch.sigmoid(self.output(feat)), out_light

    return _Net(), light


class LightNet:
    """HourglassNet gris con torch lazy. Via `load_light_net`."""

    def __init__(self, net: Any, missing: list[str], unexpected: list[str]) -> None:
        self._net = net
        self._missing = list(missing)
        self._unexpected = list(unexpected)
        self._device = "cpu"

    def missing_keys(self) -> list[str]:
        return list(self._missing)

    def unexpected_keys(self) -> list[str]:
        return list(self._unexpected)

    def _ensure_device(self, device: Any) -> None:
        dev = str(device)
        if dev != self._device:
            self._net.to(dev)
            self._device = dev

    def estimate(self, L01: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Estima 9 SH grises del canal L 512 en 0-1. Total: None si falla.

        Target dummy a ceros (solo afecta la imagen relit, nunca la luz).
        Determinista en eval (sin muestreo).
        """
        try:
            import torch  # type: ignore[import-not-found]

            arr = np.clip(np.asarray(L01, dtype=np.float64), 0.0, 1.0)
            if arr.shape != (DPR_INPUT_SIZE, DPR_INPUT_SIZE):
                return None
            inner = self._net
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._ensure_device(dev)
            inner.eval()
            with torch.no_grad():
                x = torch.from_numpy(np.ascontiguousarray(arr)).reshape(1, 1, 512, 512).to(torch.float32).to(dev)
                dummy = torch.zeros(1, DPR_SH, 1, 1, device=dev)
                _img, light = inner(x, dummy, 0)
                if light is None:
                    return None
                out = light.detach().cpu().numpy().reshape(-1)
            if out.shape != (DPR_SH,) or not bool(np.isfinite(out).all()):
                return None
            return np.asarray(out, dtype=np.float64)
        except Exception:  # noqa: BLE001 - SH invalida: via gray-world
            return None


_net_cache: dict[str, LightNet] = {}


def load_light_net(t7_path: str) -> LightNet:
    """Carga estricta del t7 (0 faltantes/inesperados o raise).

    Cacheada por ruta. RuntimeError con causa si falta torch, el archivo
    o hay mismatch (nunca carga parcial silenciosa).
    """
    cached = _net_cache.get(t7_path)
    if cached is not None:
        return cached
    import torch  # type: ignore[import-not-found]

    if not os.path.isfile(t7_path):
        raise RuntimeError(f"dpr checkpoint missing: {t7_path}")
    net, _light = _build_net()
    try:
        state = torch.load(t7_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"dpr checkpoint unreadable: {t7_path}") from exc
    if not isinstance(state, dict):
        raise TypeError(f"dpr checkpoint unexpected type: {type(state).__name__}")
    loaded = net.load_state_dict(state, strict=False)
    missing = sorted(str(k) for k in loaded.missing_keys)
    unexpected = sorted(str(k) for k in loaded.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"dpr checkpoint mismatch: missing={missing} unexpected={unexpected}")
    net.eval()
    wrapper = LightNet(net, missing, unexpected)
    _net_cache[t7_path] = wrapper
    return wrapper


def estimate_sh(net: LightNet, L01: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Estima 9 SH grises del canal L 512 en 0-1. Total: None si falla."""
    try:
        return net.estimate(L01)
    except Exception:  # noqa: BLE001 - estimacion caida: via gray-world
        return None
