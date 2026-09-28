"""Guardrail 12-factor F4/P2: config por env sin drift con .env.example.

Toda key `*_DIR` / `MODAL_VOLUME` / `*CUTOVER*` / `*BACKUP*` leida via
`_env(...)` u `os.environ.get(...)` en `backend/*.py` debe tener entrada en
`.env.example`. Si agregas una key de config, agregala al example o este
test falla en CI.
"""

from __future__ import annotations

import ast
import os
import re

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_EXAMPLE = os.path.join(os.path.dirname(_BACKEND_DIR), ".env.example")
_DRIFT_PATTERN = re.compile(r"(?:_DIR|MODAL_VOLUME|CUTOVER|BACKUP)")
_KEY_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=")

# Llaves que el plan exige documentadas (literales independientes del codigo).
REQUIRED_KEYS = frozenset(
    {
        "WEIGHTS_ROOT",
        "WEIGHTS_DIR",
        "GNM_ASSETS_DIR",
        "DECA_DIR",
        "FLAME_ASSETS_DIR",
        "FFHQ_UV_DIR",
        "MODAL_VOLUME",
        "FLAME_CUTOVER",
        "BACKUP_TAG",
    }
)


def _example_keys() -> set[str]:
    keys: set[str] = set()
    with open(_ENV_EXAMPLE, encoding="utf-8") as fh:
        for line in fh:
            candidate = line.strip().lstrip("#").strip()
            match = _KEY_ASSIGN.match(candidate)
            if match:
                keys.add(match.group(1))
    return keys


def _call_key_strings(tree: ast.AST) -> set[str]:
    """Llaves string leidas via `_env("K")` u `os.environ.get("K")`."""
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        arg: object = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
        if isinstance(func, ast.Name) and func.id == "_env":
            if isinstance(arg, str):
                keys.add(arg)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance((recv := func.value), ast.Attribute)
            and recv.attr == "environ"
            and isinstance(recv.value, ast.Name)
            and recv.value.id == "os"
            and isinstance(arg, str)
        ):
            keys.add(arg)
    return keys


def _code_keys() -> set[str]:
    keys: set[str] = set()
    for name in sorted(os.listdir(_BACKEND_DIR)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(_BACKEND_DIR, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=name)
        for key in _call_key_strings(tree):
            if _DRIFT_PATTERN.search(key):
                keys.add(key)
    return keys


def test_required_bridge_keys_present_in_env_example() -> None:
    example = _example_keys()
    missing = sorted(REQUIRED_KEYS - example)
    assert not missing, f"keys del puente sin entrada en .env.example: {missing}"


def test_no_drift_between_code_and_env_example() -> None:
    example = _example_keys()
    code = _code_keys()
    assert code, "el scan no encontro keys: el extractor AST esta roto"
    missing = sorted(code - example)
    assert not missing, f"config leida via _env sin entrada en .env.example: {missing}"
