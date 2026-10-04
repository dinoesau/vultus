"""Parsing facial BiSeNet (CelebAMask-HQ 19 clases) para mascara de piel.

Refina el unwrap: solo texeles cuya muestra foto cae en piel (cara, sin
microfonos, pelo, fondo o ropa) muestrean la foto; el resto va a completion
de piel media. Sin parsing (sin torch o sin pesos) el unwrap procede sin
mascara, documentado y determinista por env, nunca Err por este refinamiento.

Arquitectura vendored de face-parsing.PyTorch (zllrunning, MIT):
BiSeNet con backbone Resnet18, N_CLASSES=19. Pesos
`checkpoints/parsing_model/79999_iter.pth` (strict, 0 faltantes). Sin
descargas en runtime: init Kaiming local y luego load strict del pth
(el Resnet18 original baja resnet18-5c106cde.pth en __init__, aqui no).

Config solo por env (`PARSING_DIR`; mas `WEIGHTS_ROOT`/`WEIGHTS_DIR` ya
existentes). Sin torch top-level: todo import es lazy dentro del singleton
y sin torch este modulo es inerte (el caller decide el fallback).
Entradas ya probadas (ndarray foto), salidas probadas (bool HxW).
"""

from __future__ import annotations

# mypy: allow-untyped-defs, allow-untyped-calls
# Vendored BiSeNet: clases torch sin stubs en CI (torch es lazy/opcional).
# La API publica del modulo si va tipada (abajo); el interior de la red
# queda sin chequeo estricto por depender de torch dinamico.
import os
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Puente parsing: checkpoint unico (mirror de README_ckp_topo de FFHQ-UV).
PARSING_PTH_NAME = "79999_iter.pth"

# BiSeNet CelebAMask-HQ: 19 clases fijas del pth (strict, 0 faltantes).
N_CLASSES = 19
# Entrada de la red: fully-conv, 473 es el tamano de referencia del repo.
PARSE_SIZE = 473
# Normalizacion ImageNet del repo face-parsing.PyTorch.
_PARSE_MEAN = (0.485, 0.456, 0.406)
_PARSE_STD = (0.229, 0.224, 0.225)

# Etiquetas CelebAMask-HQ que cuentan como piel para el atlas:
# 1 skin, 2 nose, 3-5 ojos, 6-7 brows, 8-9 ears, 10 mouth, 11-12 lips,
# 17 neck. Fuera: 0 background, 13 hair, 14 hat, 15-16 accesorios, 18 cloth.
SKIN_LABELS = frozenset((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17))

_PARSE_LOCK = threading.Lock()
_PARSE_NET = None


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def parsing_dir() -> str:
    """Checkpoint parsing desde env. Vacio = ausente."""
    return _env("PARSING_DIR")


def _candidate_dirs() -> list[str]:
    cands: list[str] = []
    direct = parsing_dir()
    if direct:
        cands.append(direct)
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR")
    if root:
        fallback = os.path.join(root, "checkpoints", "parsing_model")
        if fallback not in cands:
            cands.append(fallback)
    return cands


def _find_pth() -> str | None:
    for d in _candidate_dirs():
        cand = os.path.join(d, PARSING_PTH_NAME)
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
        except OSError:
            continue
    return None


def weights_present() -> bool:
    """True solo con el pth de parsing no vacio."""
    return _find_pth() is not None


def _build_net():
    """BiSeNet(19) vendored sin descargas: init local + load strict fuera.

    Todo torch es lazy aqui dentro para no acoplar el import del modulo.
    Lanza RuntimeError con causa si torch o los pesos faltan.
    """
    try:
        import torch  # type: ignore[import-not-found]
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(f"torch missing for face parsing: {exc}") from exc
    import torch.nn.functional as _F  # type: ignore[import-not-found]

    def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> Any:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class _BasicBlock(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self, in_chan: int, out_chan: int, stride: int = 1):
            super().__init__()
            self.conv1 = _conv3x3(in_chan, out_chan, stride)
            self.bn1 = nn.BatchNorm2d(out_chan)
            self.conv2 = _conv3x3(out_chan, out_chan)
            self.bn2 = nn.BatchNorm2d(out_chan)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = None
            if in_chan != out_chan or stride != 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_chan),
                )

        def forward(self, x):
            residual = self.conv1(x)
            residual = _F.relu(self.bn1(residual))
            residual = self.conv2(residual)
            residual = self.bn2(residual)
            shortcut = x if self.downsample is None else self.downsample(x)
            return self.relu(shortcut + residual)

    def _make_layer(in_chan: int, out_chan: int, bnum: int, stride: int = 1) -> Any:
        layers = [_BasicBlock(in_chan, out_chan, stride=stride)]
        for _ in range(bnum - 1):
            layers.append(_BasicBlock(out_chan, out_chan, stride=1))
        return nn.Sequential(*layers)

    class _Resnet18(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = _make_layer(64, 64, 2, stride=1)
            self.layer2 = _make_layer(64, 128, 2, stride=2)
            self.layer3 = _make_layer(128, 256, 2, stride=2)
            self.layer4 = _make_layer(256, 512, 2, stride=2)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=1)

        def forward(self, x):
            x = self.conv1(x)
            x = _F.relu(self.bn1(x))
            x = self.maxpool(x)
            x = self.layer1(x)
            feat8 = self.layer2(x)
            feat16 = self.layer3(feat8)
            feat32 = self.layer4(feat16)
            return feat8, feat16, feat32

    class _ConvBNReLU(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self, in_chan: int, out_chan: int, ks: int = 3, stride: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=ks // 2, bias=False)
            self.bn = nn.BatchNorm2d(out_chan)
            self.relu = nn.ReLU(inplace=True)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=1)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class _BiSeNetOutput(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self, in_chan: int, mid_chan: int, n_classes: int):
            super().__init__()
            self.conv = _ConvBNReLU(in_chan, mid_chan, ks=3, stride=1)
            self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

        def forward(self, x):
            return self.conv_out(self.conv(x))

    class _AttentionRefinementModule(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self, in_chan: int, out_chan: int):
            super().__init__()
            self.conv = _ConvBNReLU(in_chan, out_chan, ks=3, stride=1)
            self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
            self.bn_atten = nn.BatchNorm2d(out_chan)
            self.sigmoid_atten = nn.Sigmoid()

        def forward(self, x):
            feat = self.conv(x)
            atten = _F.adaptive_avg_pool2d(feat, (1, 1))
            atten = self.conv_atten(atten)
            atten = self.bn_atten(atten)
            atten = self.sigmoid_atten(atten)
            return feat * atten

    class _ContextPath(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self) -> None:
            super().__init__()
            self.resnet = _Resnet18()
            self.arm16 = _AttentionRefinementModule(256, 128)
            self.arm32 = _AttentionRefinementModule(512, 128)
            self.conv_head32 = _ConvBNReLU(128, 128, ks=3, stride=1)
            self.conv_head16 = _ConvBNReLU(128, 128, ks=3, stride=1)
            self.conv_avg = _ConvBNReLU(512, 128, ks=1, stride=1)

        def forward(self, x):
            h, w = x.size()[2:]
            feat8, feat16, feat32 = self.resnet(x)
            avg = _F.adaptive_avg_pool2d(feat32, (1, 1))
            avg = self.conv_avg(avg)
            avg_up = _F.interpolate(avg, (feat32.size()[2], feat32.size()[3]), mode="nearest")
            feat32_arm = self.arm32(feat32)
            feat32_sum = feat32_arm + avg_up
            feat32_up = _F.interpolate(feat32_sum, (feat16.size()[2], feat16.size()[3]), mode="nearest")
            feat32_up = self.conv_head32(feat32_up)
            feat16_arm = self.arm16(feat16)
            feat16_sum = feat16_arm + feat32_up
            feat16_up = _F.interpolate(feat16_sum, (feat8.size()[2], feat8.size()[3]), mode="nearest")
            feat16_up = self.conv_head16(feat16_up)
            return feat8, feat16_up, _F.interpolate(feat32_up, (h // 8, w // 8), mode="nearest")

    class _FeatureFusionModule(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self, in_chan: int, out_chan: int):
            super().__init__()
            self.convblk = _ConvBNReLU(in_chan, out_chan, ks=1, stride=1)
            self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, fsp, fcp):
            fcat = torch.cat([fsp, fcp], dim=1)
            feat = self.convblk(fcat)
            atten = _F.adaptive_avg_pool2d(feat, (1, 1))
            atten = self.conv1(atten)
            atten = self.relu(atten)
            atten = self.conv2(atten)
            atten = self.sigmoid(atten)
            return feat * atten + feat

    class _BiSeNet(nn.Module):  # type: ignore[misc]  # torch ausente en CI: nn.Module es Any
        def __init__(self, n_classes: int):
            super().__init__()
            self.cp = _ContextPath()
            self.ffm = _FeatureFusionModule(256, 256)
            self.conv_out = _BiSeNetOutput(256, 256, n_classes)
            self.conv_out16 = _BiSeNetOutput(128, 64, n_classes)
            self.conv_out32 = _BiSeNetOutput(128, 64, n_classes)

        def forward(self, x):
            h, w = x.size()[2:]
            feat_res8, feat_cp8, feat_cp16 = self.cp(x)
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            out = self.conv_out(feat_fuse)
            out16 = self.conv_out16(feat_cp8)
            out32 = self.conv_out32(feat_cp16)
            out = _F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            out16 = _F.interpolate(out16, (h, w), mode="bilinear", align_corners=True)
            out32 = _F.interpolate(out32, (h, w), mode="bilinear", align_corners=True)
            return out, out16, out32

    net = _BiSeNet(N_CLASSES)
    pth = _find_pth()
    if pth is None:
        raise RuntimeError(f"parsing weights missing: {PARSING_PTH_NAME}")
    state = torch.load(pth, map_location="cpu", weights_only=True)
    info = net.load_state_dict(state, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"parsing weights mismatch: missing={info.missing_keys[:3]} unexpected={info.unexpected_keys[:3]}")
    net.eval()
    return net


def _net():
    """Singleton BiSeNet eval. Lanza RuntimeError con causa si no disponible."""
    global _PARSE_NET
    if _PARSE_NET is not None:
        return _PARSE_NET
    with _PARSE_LOCK:
        if _PARSE_NET is not None:
            return _PARSE_NET
        _PARSE_NET = _build_net()
        return _PARSE_NET


def face_skin_mask(photo: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mascara de piel HxW sobre foto RGB float64. Total: lanza RuntimeError
    si torch o pesos faltan (el caller decide el fallback documentado)."""
    import torch

    if photo.ndim != 3 or photo.shape[2] != 3:
        raise ValueError("photo must be HxWx3")
    height, width = int(photo.shape[0]), int(photo.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("degenerate photo dimensions")
    from PIL import Image

    small = Image.fromarray(np.clip(photo, 0.0, 255.0).astype(np.uint8)).resize(
        (PARSE_SIZE, PARSE_SIZE), Image.Resampling.BILINEAR
    )
    arr = np.asarray(small, dtype=np.float64) / 255.0
    arr = (arr - np.asarray(_PARSE_MEAN, dtype=np.float64)) / np.asarray(_PARSE_STD, dtype=np.float64)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        logits, _, _ = _net()(tensor)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    skin_small = np.isin(pred, list(SKIN_LABELS))
    mask_img = Image.fromarray(skin_small).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask_img, dtype=bool)
