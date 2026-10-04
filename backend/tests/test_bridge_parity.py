"""Paridad puente P2: BRIDGE_FILES+ALTS del sh == consts flame_fit/flame_texture.

El script admin `scripts/modal-weights-sync.sh` y el codigo Python deben
acordar los mismos archivos canonicos. Si agregas un asset al puente,
actualiza ambos lados o este test falla en CI.
"""

from __future__ import annotations

import os
import re

from backend.deep3d import TRANSFER_NAME
from backend.flame_fit import DECA_TAR_NAME, FLAME_PKL_NAMES
from backend.flame_texture import (
    EYE_MAP_NAME,
    MEAN_FACE_NAME,
    TEXGAN_NAME,
    UNWRAP_MAT_NAME,
    UV_OBJ_NAME,
)
from backend.gnm_assemble import FLAME68_NAME, FLAME_TEMPLATE_NAME

_SYNC_SH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scripts", "modal-weights-sync.sh"
)

# Valores esperados del script (literales independientes del codigo Python).
EXPECTED_BRIDGE_FILES = frozenset(
    {
        "mediapipe/face_landmarker.task",
        "deca/deca_model.tar",
        "ffhq-uv/FLAME_w_HIFI3D_UV.obj",
        "ffhq-uv/eye_ball_tex.png",
        "checkpoints/texgan_model/texgan_ffhq_uv.pth",
        "checkpoints/deep3d_model/epoch_latest.pth",
        "topo_assets/unwrap_1024_info.mat",
        "topo_assets/hifi3dpp_mean_face.obj",
        "flame/flame_template.bin",
        "flame/flame_hifi_transfer.npz",
        "flame/flame68_embed.npz",
    }
)
EXPECTED_FLAME_PKL_ALTS = frozenset(
    {
        "flame/flame2023_Open.pkl",
        "flame/generic_model.pkl",
    }
)


def _sh_var(name: str) -> list[str]:
    with open(os.path.normpath(_SYNC_SH), encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(rf'^{name}="([^"]*)"', text, re.MULTILINE)
    assert match is not None, f"{name} no encontrado en modal-weights-sync.sh"
    return match.group(1).split()


def test_bridge_files_pinned_in_sync_script() -> None:
    assert frozenset(_sh_var("BRIDGE_FILES")) == EXPECTED_BRIDGE_FILES
    assert frozenset(_sh_var("BRIDGE_FLAME_PKL_ALTS")) == EXPECTED_FLAME_PKL_ALTS


def test_bridge_parity_with_python_consts() -> None:
    bridge = _sh_var("BRIDGE_FILES")
    alts = _sh_var("BRIDGE_FLAME_PKL_ALTS")
    assert f"deca/{DECA_TAR_NAME}" in bridge
    assert f"ffhq-uv/{UV_OBJ_NAME}" in bridge
    assert f"ffhq-uv/{EYE_MAP_NAME}" in bridge
    assert f"checkpoints/texgan_model/{TEXGAN_NAME}" in bridge
    assert f"topo_assets/{UNWRAP_MAT_NAME}" in bridge
    assert f"topo_assets/{MEAN_FACE_NAME}" in bridge
    assert f"flame/{FLAME_TEMPLATE_NAME}" in bridge
    assert f"flame/{TRANSFER_NAME}" in bridge
    assert f"flame/{FLAME68_NAME}" in bridge
    assert frozenset(alts) == frozenset(f"flame/{name}" for name in FLAME_PKL_NAMES)
