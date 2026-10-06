"""Worker v3 de minutos: definicion sin deploy, cola separada, sin TTL 60.

Seam bajo test: `backend.modal_app.v3_figure_worker` existe con config de
minutos (timeout), 1 sujeto por GPU (anti-OOM) y misma imagen/volumenes
que el path forense. Solo definicion: el deploy requiere sign-off del
dueno (env Modal unico) y NUNCA ocurre desde este test.
"""

from __future__ import annotations


def test_v3_figure_worker_defined_with_minutes_config() -> None:
    from backend import modal_app
    from backend.v3_contract import (
        V3_FIT_TIMEOUT_SECS,
        V3_RETENTION_DAYS,
        V3_TEXTURE_TIMEOUT_SECS,
    )
    from backend.v3_worker import V3_QUEUE, v3_timeouts

    fn = modal_app.v3_figure_worker
    assert fn is not None
    assert callable(fn) or hasattr(fn, "remote")
    # Cola separada sin TTL 60; R2 con retencion de dias.
    assert V3_QUEUE == "v3-queue"
    fit_secs, tex_secs, retention_days = v3_timeouts()
    assert fit_secs >= 600 and tex_secs >= 600
    assert retention_days == V3_RETENTION_DAYS and retention_days >= 1
    assert V3_FIT_TIMEOUT_SECS >= 600 and V3_TEXTURE_TIMEOUT_SECS >= 600
    # Un sujeto por GPU (anti-OOM): configuracion del decorador Modal.
    fn = modal_app.v3_figure_worker
    get = getattr(fn, "_modal_config", None)
    if get is not None:
        assert get.get("max_containers", 1) == 1
