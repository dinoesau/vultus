"""Forma real Deep3D-HiFi3D++ (Fase 2, issue 94).

Port de inferencia del `ReconNetWrapper` de FFHQ-UV (`network/
recon_deep3d.py` + `network/resnet/backbone.py` ResNet50 V1.5 sin fc
final + 7 cabezas 1x1 -> 1049 coefs), solo forward, sin torch
top-level (lazy como `face_parsing.py`). Bit-identico al backbone
torchvision cargado con los mismos pesos (verificado offline).

Layout de salida 1049 (espejo de `hifi3dpp.py split_coeff`):
`id[0:532] exp[532:577] tex[577:1016] angle[1016:1019] gamma[1019:1046]
trans[1046:1049]`. La moneda del contrato sigue siendo el fit 253:
`[id200 exp45 angle3 txtytz spare2]` (200+45+3+3+2=253).

Desviacion documentada del upstream: el crop de entrada usa el bbox
de landmarks 478 (+margen) en vez del alineado 68lm+MTCNN (el .pb TF1
no corre en este stack); sigue siendo inferencia real foto->coefs de
una pasada dentro del deadline de fit.

`displaced_positions` (gnm_assemble) reconstruye con la base HiFi3D++
(`idBase`/`exBase` del `.mat`) y transfiere 20481->5023 via el asset
`flame_hifi_transfer.npz` (IDW k=3 precomputado offline, solo numpy
en runtime: sin scipy en el path caliente). Sin scipy/mat el
desplazamiento cae al legado geometrico (dobles/CI intactos).

Config solo por env (`DEEP3D_DIR`, `TOPO_DIR`; mas `WEIGHTS_ROOT` /
`WEIGHTS_DIR` ya existentes). Sin logging.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.domain import DomainError, Err, MlDecode, MlFailed, Ok

# --- Layout Deep3D-HiFi3D++ (espejo de hifi3dpp.py + recon_deep3d.py) ---
DEEP3D_ID = 532
DEEP3D_EXP = 45
DEEP3D_TEX = 439
DEEP3D_ANGLE = 3
DEEP3D_GAMMA = 27
DEEP3D_TT = 3
DEEP3D_COEFF = DEEP3D_ID + DEEP3D_EXP + DEEP3D_TEX + DEEP3D_ANGLE + DEEP3D_GAMMA + DEEP3D_TT
# Moneda fit 253: identidad 200 + expresion 45 + pose 3+3 + 2 reserva.
FIT_ID = 200
FIT_EXP = 45
FIT_ANGLE = 3
FIT_TRANS = 3
FIT_SPARE = 2

DEEP3D_EPOCH_NAME = "epoch_latest.pth"
HIFI_MAT_NAME = "hifi3dpp_model_info.mat"
TRANSFER_NAME = "flame_hifi_transfer.npz"

CROP_SIZE = 224
CROP_MARGIN = 0.15

_LAYER_BLOCKS = (("layer1", 3, 64, 1), ("layer2", 4, 128, 2), ("layer3", 6, 256, 2), ("layer4", 3, 512, 2))


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def deep3d_dir() -> str:
    """Checkpoint Deep3D desde env. Vacio = ausente."""
    return _env("DEEP3D_DIR")


def topo_dir() -> str:
    """Assets topologia (.mat HiFi3D++) desde env. Vacio = ausente."""
    return _env("TOPO_DIR")


def _candidate_epoch_dirs() -> list[str]:
    cands: list[str] = []
    if deep3d_dir() and deep3d_dir() not in cands:
        cands.append(deep3d_dir())
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        fallback = os.path.join(root, "checkpoints", "deep3d_model")
        if fallback not in cands:
            cands.append(fallback)
    return cands


def _candidate_topo_dirs() -> list[str]:
    cands: list[str] = []
    for d in (topo_dir(), _env("FFHQ_UV_DIR")):
        if d and d not in cands:
            cands.append(d)
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        fallback = os.path.join(root, "topo_assets")
        if fallback not in cands:
            cands.append(fallback)
    return cands


def find_epoch() -> str | None:
    """Ruta de epoch_latest.pth en candidatos, o None."""
    for d in _candidate_epoch_dirs():
        cand = os.path.join(d, DEEP3D_EPOCH_NAME)
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
        except OSError:
            continue
    return None


def find_mat() -> str | None:
    """Ruta de hifi3dpp_model_info.mat en candidatos, o None."""
    for d in _candidate_topo_dirs():
        cand = os.path.join(d, HIFI_MAT_NAME)
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
        except OSError:
            continue
    return None


def find_transfer() -> str | None:
    """Ruta del asset de transferencia (repo u overrides). Solo numpy."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "assets", TRANSFER_NAME)]
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR")
    if root:
        cands.append(os.path.join(root, "flame", TRANSFER_NAME))
    for cand in cands:
        try:
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
        except OSError:
            continue
    return None


def weights_present() -> bool:
    """True solo con epoch no vacio. Total, nunca raise."""
    return find_epoch() is not None


def torch_available() -> bool:
    """True si torch importa. Total, nunca raise."""
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def scipy_available() -> bool:
    """True si scipy.io importa (para el .mat). Total, nunca raise."""
    try:
        import scipy.io  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def deep3d_available() -> bool:
    """Gate de la via real: torch + epoch + mat (+scipy)."""
    return torch_available() and scipy_available() and weights_present() and find_mat() is not None


def expected_net_recon_keys() -> frozenset[str]:
    """Set exacto de keys de `net_recon` (puro python, sin torch).

    Nombres relativos al backbone (ResNet50 torchvision sin fc: 318,
    sin el prefijo `backbone.`) + `final_layers.*` (7 cabezas 1x1: 14).
    Total 332, verificado contra epoch_latest.
    """
    keys: set[str] = set()
    for name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        keys.add(f"bn1.{name}")
    keys.add("conv1.weight")
    for layer, blocks, planes, _stride in _LAYER_BLOCKS:
        for i in range(blocks):
            p = f"{layer}.{i}"
            for conv, bn in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
                keys.add(f"{p}.{conv}.weight")
                for name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    keys.add(f"{p}.{bn}.{name}")
            if i == 0:
                keys.add(f"{p}.downsample.0.weight")
                for name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    keys.add(f"{p}.downsample.1.{name}")
    for head in range(7):
        stem = f"final_layers.{head}" if head != 1 else "final_layers.1.0"
        keys.add(f"{stem}.weight")
        keys.add(f"{stem}.bias")
    return frozenset(keys)


def check_net_recon_keys(keys: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(faltantes, inesperadas) contra el set esperado. Puro python."""
    have = frozenset(keys)
    want = expected_net_recon_keys()
    return (want - have, have - want)


def split_coeff_vector(vec: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Rebana el vector 1049 en semantica (puro numpy)."""
    flat = np.asarray(vec, dtype=np.float64).reshape(-1)
    if flat.shape[0] != DEEP3D_COEFF:
        raise ValueError(f"deep3d coeff len {flat.shape[0]} != {DEEP3D_COEFF}")
    off = 0
    parts: dict[str, NDArray[np.float64]] = {}
    for name, size in (
        ("id", DEEP3D_ID),
        ("exp", DEEP3D_EXP),
        ("tex", DEEP3D_TEX),
        ("angle", DEEP3D_ANGLE),
        ("gamma", DEEP3D_GAMMA),
        ("trans", DEEP3D_TT),
    ):
        parts[name] = flat[off : off + size]
        off += size
    return parts


def encode_fit253(
    id_coeffs: NDArray[np.float64],
    exp_coeffs: NDArray[np.float64],
    angle: NDArray[np.float64],
    trans: NDArray[np.float64],
) -> tuple[float, ...]:
    """Empaqueta la moneda fit 253: id200 + exp45 + angle3 + trans3 + 2 ceros."""
    idv = np.asarray(id_coeffs, dtype=np.float64).reshape(-1)
    exv = np.asarray(exp_coeffs, dtype=np.float64).reshape(-1)
    anv = np.asarray(angle, dtype=np.float64).reshape(-1)
    trv = np.asarray(trans, dtype=np.float64).reshape(-1)
    if idv.shape[0] < FIT_ID or exv.shape[0] != FIT_EXP or anv.shape[0] != FIT_ANGLE or trv.shape[0] != FIT_TRANS:
        raise ValueError("deep3d fit253 slices invalid")
    return tuple(float(v) for v in (*idv[:FIT_ID], *exv, *anv, *trv, 0.0, 0.0))


def photo_array(image_bytes: Any) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Foto a RGB float64 HWC desde ImageBytes. Total: Err si no decodifica."""
    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(io.BytesIO(image_bytes.as_bytes())) as handle:
            rgb = handle.convert("RGB")
            return Ok(np.asarray(rgb, dtype=np.float64))
    except Exception as exc:  # noqa: BLE001 - foto indecodificable es Err, no crash
        return Err(MlFailed(detail=MlDecode(details=f"deep3d photo decode failed: {exc}")))


def face_crop224(photo: NDArray[np.float64], xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.float32]:
    """Crop de cara por bbox normalizado (+margen) a 224 /255 float32 HWC."""
    from PIL import Image  # type: ignore[import-not-found]

    height, width = int(photo.shape[0]), int(photo.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("deep3d degenerate photo dimensions")
    lx = np.asarray(xs, dtype=np.float64) * float(width)
    ly = np.asarray(ys, dtype=np.float64) * float(height)
    bw = float(lx.max() - lx.min())
    bh = float(ly.max() - ly.min())
    if bw < 1.0 or bh < 1.0:
        raise ValueError("deep3d degenerate face bbox")
    x0 = max(0, int(lx.min() - CROP_MARGIN * bw))
    x1 = min(width, int(lx.max() + CROP_MARGIN * bw) + 1)
    y0 = max(0, int(ly.min() - CROP_MARGIN * bh))
    y1 = min(height, int(ly.max() + CROP_MARGIN * bh) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("deep3d degenerate face crop")
    crop = Image.fromarray(np.clip(photo[y0:y1, x0:x1], 0.0, 255.0).astype(np.uint8))
    small = crop.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BICUBIC)
    return (np.asarray(small, dtype=np.float32) / 255.0).astype(np.float32)


class ReconNet:
    """ReconNetWrapper funcional con torch lazy. Via `load_recon`.

    `missing_keys()`/`unexpected_keys()` vacias <=> carga estricta del
    `net_recon` de epoch_latest (0 faltantes). `forward_coeffs` es el
    forward eval (backbone ResNet50 V1.5 + 7 cabezas), determinista.
    """

    def __init__(self, params: dict[str, Any], missing: list[str], unexpected: list[str]) -> None:
        self._params = dict(params)
        self._missing = list(missing)
        self._unexpected = list(unexpected)

    def missing_keys(self) -> list[str]:
        return list(self._missing)

    def unexpected_keys(self) -> list[str]:
        return list(self._unexpected)

    def forward_coeffs(self, x01: Any) -> Any:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as _F  # type: ignore[import-not-found]

        sd = self._params

        def _bn(t: Any, p: str) -> Any:
            return _F.batch_norm(
                t,
                sd[f"{p}.running_mean"],
                sd[f"{p}.running_var"],
                sd[f"{p}.weight"],
                sd[f"{p}.bias"],
                training=False,
                eps=1e-5,
            )

        def _block(t: Any, p: str, stride: int, down: bool) -> Any:
            # V1.5: conv1 1x1 stride 1, stride en conv2 3x3.
            o = _F.conv2d(t, sd[f"{p}.conv1.weight"], stride=1, padding=0)
            o = _F.relu(_bn(o, f"{p}.bn1"), inplace=True)
            o = _F.conv2d(o, sd[f"{p}.conv2.weight"], stride=stride, padding=1)
            o = _F.relu(_bn(o, f"{p}.bn2"), inplace=True)
            o = _bn(_F.conv2d(o, sd[f"{p}.conv3.weight"]), f"{p}.bn3")
            ident = t
            if down:
                ident = _F.conv2d(t, sd[f"{p}.downsample.0.weight"], stride=stride)
                ident = _bn(ident, f"{p}.downsample.1")
            return _F.relu(o + ident, inplace=True)

        with torch.no_grad():
            x = x01.to(torch.float32)
            x = _F.conv2d(x, sd["conv1.weight"], stride=2, padding=3)
            x = _F.relu(_bn(x, "bn1"), inplace=True)
            x = _F.max_pool2d(x, 3, stride=2, padding=1)
            in_ch = 64
            for layer, blocks, planes, stride in _LAYER_BLOCKS:
                for i in range(blocks):
                    s = stride if i == 0 else 1
                    down = i == 0 and (s != 1 or in_ch != planes * 4)
                    x = _block(x, f"{layer}.{i}", s, down)
                in_ch = planes * 4
            feat = _F.adaptive_avg_pool2d(x, (1, 1))
            outs = []
            for head in range(7):
                stem = f"final_layers.{head}" if head != 1 else "final_layers.1.0"
                o = _F.conv2d(feat, sd[stem + ".weight"]) + sd[stem + ".bias"].reshape(1, -1, 1, 1)
                if head == 1:
                    o = _F.relu(o, inplace=True)
                outs.append(o.flatten())
            return torch.cat(outs).reshape(1, -1)


_recon_cache: dict[str, ReconNet] = {}


def load_recon(epoch_path: str) -> ReconNet:
    """Carga estricta del `net_recon` (0 faltantes/inesperados o raise).

    Cacheada por ruta. RuntimeError con causa si falta torch, el archivo
    o hay mismatch (nunca carga parcial silenciosa). Sin el `net_recon`
    (p.ej. checkpoint de otro modelo) es TypeError/ValueError loud.
    """
    cached = _recon_cache.get(epoch_path)
    if cached is not None:
        return cached
    import torch  # type: ignore[import-not-found]

    if not os.path.isfile(epoch_path):
        raise RuntimeError(f"deep3d checkpoint missing: {epoch_path}")
    try:
        ckpt = torch.load(epoch_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"deep3d checkpoint unreadable: {epoch_path}") from exc
    if not isinstance(ckpt, dict) or "net_recon" not in ckpt:
        raise TypeError("deep3d checkpoint without net_recon")
    state = ckpt["net_recon"]
    if not isinstance(state, dict):
        raise TypeError("deep3d net_recon unexpected type")
    stripped = {str(k)[len("backbone.") :]: v for k, v in state.items() if str(k).startswith("backbone.")}
    heads = {str(k): v for k, v in state.items() if str(k).startswith("final_layers.")}
    missing, unexpected = check_net_recon_keys([*stripped, *heads])
    if missing or unexpected:
        raise RuntimeError(f"deep3d checkpoint mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    recon = ReconNet({**stripped, **heads}, [], [])
    _recon_cache[epoch_path] = recon
    return recon


_basis_cache: dict[str, dict[str, NDArray[np.float64]]] = {}


def load_hifi_basis() -> dict[str, NDArray[np.float64]] | None:
    """Base HiFi3D++ del .mat (idBase200, exBase45, mean20481). Cacheada.

    None sin scipy/mat o ante cualquier fallo (el caller cae al legado).
    float32 del .mat promovido a float64 solo en salida.
    """
    mat = find_mat()
    if mat is None:
        return None
    cached = _basis_cache.get(mat)
    if cached is not None:
        return cached
    try:
        if not scipy_available():
            return None
        from scipy.io import loadmat  # type: ignore[import-not-found]

        m = loadmat(mat)
        mean = np.asarray(m["meanshape"], dtype=np.float64).reshape(-1, 3)
        idb = np.asarray(m["idBase"], dtype=np.float64)
        exb = np.asarray(m["exBase"], dtype=np.float64)
        if mean.shape[0] != 20481 or idb.shape[0] != 61443 or exb.shape[0] != 61443:
            return None
        out = {
            "mean": mean,
            "id200": idb[:, :FIT_ID],
            "exp45": exb[:, :FIT_EXP],
            "id_full": np.asarray(m["idBase"], dtype=np.float32),
            "ex_full": np.asarray(m["exBase"], dtype=np.float32),
            "head_tri": np.asarray(m["head_tri"], dtype=np.int64),
        }
    except Exception:  # noqa: BLE001 - sin scipy/mat: via legado geometrico
        return None
    _basis_cache[mat] = out
    return out


_transfer_cache: dict[str, dict[str, NDArray[np.float64]]] = {}


def load_transfer() -> dict[str, NDArray[np.float64]] | None:
    """Asset IDW 20481->5023 (solo numpy). None si ausente/invalido."""
    path = find_transfer()
    if path is None:
        return None
    cached = _transfer_cache.get(path)
    if cached is not None:
        return cached
    try:
        z = np.load(path)
        idx = np.asarray(z["idx"], dtype=np.int64)
        w = np.asarray(z["w"], dtype=np.float64)
        scale = np.asarray(z["scale"], dtype=np.float64)
        if idx.shape != (5023, 3) or w.shape != (5023, 3) or scale.shape != (3,):
            return None
        if idx.min() < 0 or idx.max() >= 20481:
            return None
        out = {"idx": idx, "w": w, "scale": scale}
    except Exception:  # noqa: BLE001 - asset invalido: via legado geometrico
        return None
    _transfer_cache[path] = out
    return out


def reconstruct_dense(
    id_coeffs: NDArray[np.float64], exp_coeffs: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Reconstruye la malla densa HiFi3D++ (20481v) con la base del .mat.

    `id_shape = mean + idBase@id`, `exp_shape = id + exBase@exp`
    (espejo de `ParametricFaceModel.compute_shape`). Solo para la
    decision de topologia (Fase 4): el contrato sirve FLAME 5023.
    None sin base o ante cualquier fallo. Solo numpy.
    """
    try:
        basis = load_hifi_basis()
        if basis is None or "id_full" not in basis:
            return None
        idv = np.asarray(id_coeffs, dtype=np.float32).reshape(-1)
        exv = np.asarray(exp_coeffs, dtype=np.float32).reshape(-1)
        if idv.shape[0] != DEEP3D_ID or exv.shape[0] != DEEP3D_EXP:
            return None
        mean32 = np.asarray(basis["mean"], dtype=np.float32)
        id_shape = (mean32 + (basis["id_full"] @ idv).reshape(-1, 3)).reshape(-1, 3)
        exp_shape = (id_shape + (basis["ex_full"] @ exv).reshape(-1, 3)).reshape(-1, 3)
        if id_shape.shape != (20481, 3) or exp_shape.shape != (20481, 3):
            return None
        if not bool(np.isfinite(id_shape).all() and np.isfinite(exp_shape).all()):
            return None
        return (np.asarray(id_shape, dtype=np.float64), np.asarray(exp_shape, dtype=np.float64))
    except Exception:  # noqa: BLE001 - base invalida: sin malla densa
        return None


def real_displacement(coeffs253: tuple[float, ...]) -> NDArray[np.float64] | None:
    """Delta 5023 real: base HiFi3D++ reconstruida + transfer IDW.

    Interpreta la moneda fit 253 (id200/exp45). None sin base/transfer
    o ante cualquier fallo: el caller cae al legado. Solo numpy.
    """
    try:
        if len(coeffs253) != 253:
            return None
        basis = load_hifi_basis()
        transfer = load_transfer()
        if basis is None or transfer is None:
            return None
        idv = np.asarray(coeffs253[:FIT_ID], dtype=np.float64)
        exv = np.asarray(coeffs253[FIT_ID : FIT_ID + FIT_EXP], dtype=np.float64)
        dense = basis["id200"] @ idv + basis["exp45"] @ exv
        field = dense.reshape(-1, 3)
        idx = transfer["idx"]
        w = transfer["w"]
        moved = w[:, :, None] * field[idx]
        delta = moved.sum(axis=1) * transfer["scale"][None, :]
        if not bool(np.isfinite(delta).all()):
            return None
        return np.asarray(delta, dtype=np.float64)
    except Exception:  # noqa: BLE001 - desplazamiento invalido: via legado geometrico
        return None
