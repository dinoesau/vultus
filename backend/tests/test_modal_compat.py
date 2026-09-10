"""Compatibilidad Modal: `domain.py` + `gnm.py` viajan a la imagen GPU.

La imagen Modal pina Python 3.10 (`add_python="3.10"` en `modal_app.py`).
Estos modulos deben parsear con gramatica 3.10 o el consumer muere con
`SyntaxError`/`ImportError` en bake/heatmap/GLB/zip (prod), aunque todo
pase en local 3.12+. Regresion real vista en publish post-merge.
"""

import ast
import os

_MODAL_SHIPPED = ("domain.py", "gnm.py", "gnm_head.py")


def test_modal_shipped_modules_parse_on_py310_grammar():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(tests_dir)
    for name in _MODAL_SHIPPED:
        rel = os.path.join("backend", name)
        with open(os.path.join(backend_dir, name)) as f:
            src = f.read()
        ast.parse(src, filename=rel, feature_version=(3, 10))


def test_assets_dir_prefers_env_then_weights_then_repo(monkeypatch):
    from backend import gnm

    monkeypatch.delenv("GNM_ASSETS_DIR", raising=False)
    monkeypatch.delenv("WEIGHTS_DIR", raising=False)
    repo = os.path.join(os.path.dirname(gnm.__file__), "assets")
    assert gnm._assets_dir() == repo
    monkeypatch.setenv("WEIGHTS_DIR", "/weights")
    assert gnm._assets_dir() == os.path.join("/weights", "gnm")
    monkeypatch.setenv("GNM_ASSETS_DIR", "/custom")
    assert gnm._assets_dir() == "/custom"
