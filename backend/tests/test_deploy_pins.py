"""Pins reproducibles F2: lock unificado + receta Modal pineada por digest.

El lock es la fuente reproducible para CI (`pip install --require-hashes`,
job lock-check); la receta Modal instala primero el build cu126 de la misma
VERSION y luego requirements.txt (torch-free por diseno). Si cambias torch o
la base CUDA, regenera el lock y re-pinea el digest o estos tests fallan.
"""

from __future__ import annotations

import os
import re

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCK_PATH = os.path.join(_BACKEND_DIR, "requirements.lock")
_MODAL_APP_PATH = os.path.join(_BACKEND_DIR, "modal_app.py")

_WEIGHT_KEYS = (
    "WEIGHTS_ROOT",
    "WEIGHTS_DIR",
    "GNM_ASSETS_DIR",
    "FFHQ_UV_DIR",
    "DECA_DIR",
    "FLAME_ASSETS_DIR",
    "MODAL_VOLUME",
)

_PRINT_CONSTS = (
    "from backend.modal_app import "
    "WEIGHTS_ROOT, WEIGHTS_DIR, GNM_ASSETS_DIR, FFHQ_UV_DIR, "
    "DECA_DIR, FLAME_ASSETS_DIR, MODAL_VOLUME_NAME; "
    "print(WEIGHTS_ROOT); print(WEIGHTS_DIR); print(GNM_ASSETS_DIR); "
    "print(FFHQ_UV_DIR); print(DECA_DIR); print(FLAME_ASSETS_DIR); "
    "print(MODAL_VOLUME_NAME)"
)


def _modal_consts(extra_env: dict[str, str]) -> list[str]:
    """Importa backend.modal_app en un proceso limpio y devuelve las consts.

    Subproceso para no contaminar el env del suite: los defaults se bindean
    en importacion.
    """
    import subprocess
    import sys

    root = os.path.dirname(_BACKEND_DIR)
    env = {k: v for k, v in os.environ.items() if k not in _WEIGHT_KEYS}
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", _PRINT_CONSTS],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"import modal_app fallo: {proc.stderr[-500:]}"
    return proc.stdout.splitlines()


def _lock_torch_version() -> str:
    with open(_LOCK_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("torch=="):
                return line.split("torch==", 1)[1].split()[0].rstrip("\\").strip()
    raise AssertionError("torch no pineado en requirements.lock")


def _lock_torch_hashes() -> int:
    count = 0
    inside = False
    with open(_LOCK_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("torch=="):
                inside = True
                continue
            if inside:
                if line.startswith("    --hash="):
                    count += 1
                else:
                    break
    return count


def test_lock_pins_torch_unified_with_hashes() -> None:
    # El lock pinea el build CPU (`2.13.0+cpu`, diseno PyPI-CPU-en-lock);
    # la variante cu126 vive solo en la receta Modal. Se compara sin sufijo local.
    assert _lock_torch_version().split("+")[0] == "2.13.0"
    assert _lock_torch_hashes() >= 2, "el pin de torch en el lock debe traer hashes"


def test_modal_recipe_pins_cuda_base_by_digest() -> None:
    with open(_MODAL_APP_PATH, encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(
        r"from_registry\(\s*\"nvidia/cuda:12\.6\.0-devel-ubuntu22\.04@sha256:[0-9a-f]{64}\"",
        text,
    )
    assert match is not None, "base CUDA sin pin digest en from_registry"
    assert "Manifest evidence" in text or "manifest-list digest" in text


def test_weights_root_defaults_compatible() -> None:
    """F4 sin cambio por defecto: raiz /weights y derivados historicos."""
    assert _modal_consts({}) == [
        "/weights",
        "/weights",
        "/weights/gnm",
        "/weights/ffhq-uv",
        "/weights/deca",
        "/weights/flame",
        "vultus-weights",
    ]


def test_weights_root_env_overrides_all_defaults() -> None:
    assert _modal_consts({"WEIGHTS_ROOT": "/x"}) == [
        "/x",
        "/x",
        "/x/gnm",
        "/x/ffhq-uv",
        "/x/deca",
        "/x/flame",
        "vultus-weights",
    ]


def test_explicit_dir_env_wins_over_root() -> None:
    assert _modal_consts({"WEIGHTS_ROOT": "/x", "FFHQ_UV_DIR": "/custom/uv"}) == [
        "/x",
        "/x",
        "/x/gnm",
        "/custom/uv",
        "/x/deca",
        "/x/flame",
        "vultus-weights",
    ]


def test_modal_deploy_has_no_raw_weights_literals() -> None:
    """Superficie deploy sin literales /weights: volumes/env van por consts.

    Los comentarios pueden mencionar rutas; solo los dicts `volumes=`/`env=`
    de `app.function` deben usar nombres, nunca literales.
    """
    import ast

    with open(_MODAL_APP_PATH, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename="modal_app.py")
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg not in ("volumes", "env"):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values):
            for part in (key_node, value_node):
                if (
                    isinstance(part, ast.Constant)
                    and isinstance(part.value, str)
                    and part.value.startswith("/weights")
                ):
                    offenders.append(f"{node.arg}={part.value!r} linea={part.lineno}")
                if (
                    isinstance(part, ast.Constant)
                    and isinstance(part.value, str)
                    and part.value.startswith("/")
                    and node.arg == "volumes"
                ):
                    offenders.append(f"{node.arg}={part.value!r} linea={part.lineno}")
    assert not offenders, f"literales de mount en deploy: {offenders}"
