"""Worker v3 (GPU de minutos, cola separada sin TTL 60, R2 dias).

Nada de vision en `edge/`; nada de `torch` top-level aqui (lazy en los
modulos designados). Cola separada `v3-queue` sin TTL 60; R2 con retencion
de dias (`V3_RETENTION_DAYS`). Timeouts de minutos (`V3_FIT_TIMEOUT_SECS`,
`V3_TEXTURE_TIMEOUT_SECS`).

Ramas efectivas por job con el patron existente (`deep3d=`, `pose=`,
`texgan=`, `dpr=`, `displacement=`); prohibido cualquier fallback sin
linea de log: `format_v3_branches` es la unica via y el caller la loguea
siempre (incluso en fallbacks).
"""

from __future__ import annotations

from backend.v3_contract import (
    V3_FIT_TIMEOUT_SECS,
    V3_RETENTION_DAYS,
    V3_TEXTURE_TIMEOUT_SECS,
)

# Cola separada sin TTL 60; R2 con retencion de dias.
V3_QUEUE = "v3-queue"

# Claves de ramas efectivas (mismo patron que `modal_app.py`).
V3_BRANCH_KEYS = ("deep3d", "pose", "texgan", "dpr", "displacement")


def v3_timeouts() -> tuple[int, int, int]:
    """(fit_secs, texture_secs, retention_days). Sin TTL 60 por diseno."""
    return (int(V3_FIT_TIMEOUT_SECS), int(V3_TEXTURE_TIMEOUT_SECS), int(V3_RETENTION_DAYS))


def format_v3_branches(stats: dict[str, float]) -> str:
    """Linea de log `deep3d= pose= texgan= dpr= displacement=` por job.

    Unica via para reportar ramas: el caller la loguea siempre, incluso
    cuando alguna rama cae a fallback (prohibido fallback silencioso).
    Total: nunca lanza (valores no numericos caen a 0).
    """
    parts: list[str] = []
    for key in V3_BRANCH_KEYS:
        try:
            value = float(stats.get(key, 0.0))
        except (ValueError, TypeError, AttributeError):
            value = 0.0
        parts.append(f"{key}={value:.0f}")
    return " ".join(parts)
