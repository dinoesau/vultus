"""Contrato v3 FFHQ-UV figure (track paralelo, sin tocar la via forense).

La via forense queda congelada: `backend.domain.ZIP_NAMES` (zip-6),
`backend.gnm_assemble.VERT_COUNT = 5023`, TTL 60 (`TOTAL_TIMEOUT_SECS`).
Este modulo es dueno de las constantes v3, en paralelo y sin mezclar:

- `V3_CONTRACT_VERSION = 3` (migracion versionada; forense sigue en 2).
- Malla densa HiFi3D++ `20481v/40832f` (port `ParametricFaceModel`).
- Atlas `1024` (`V3_UV_LEN = 1024*1024*3`), TV maxima 2.0, cero sentinel.
- Bundle v3 nuevo de 6 piezas (malla densa + albedo 1024 + neutral + 3
  relights con esferas como la figura). Nombres disjuntos del zip-6:
  prohibido mezclar.
- Retencion de dias (R2 con retencion de dias) y worker GPU de minutos
  con cola separada sin TTL 60.
- Ramas efectivas con el patron existente (`deep3d=`, `pose=`, `texgan=`,
  `dpr=`, `displacement=`); prohibido fallback sin linea de log (el log
  vive en `modal_app.py`, aqui solo las claves).

Solo constantes y helpers puros. Sin torch, sin I/O, sin logging.
Revoca explicitamente solo para v3: ADR-009 (FLAME 5023), TTL 60 y zip-6.
"""

from __future__ import annotations

# Migracion versionada: forense = 2 (congelado), v3 = 3.
V3_CONTRACT_VERSION = 3

# Topologia densa HiFi3D++ (espejo de `ParametricFaceModel`: mean 20481x3,
# `head_tri` 40832 caras). El contrato forense sigue en 5023/9976.
V3_VERT_COUNT = 20481
V3_TRI_COUNT = 40832

# Atlas v3 1024 (forense: 512). TV maxima y cero sentinel por test RED.
V3_UV_SIZE = 1024
V3_UV_CHANNELS = 3
V3_UV_LEN = V3_UV_SIZE * V3_UV_SIZE * V3_UV_CHANNELS
V3_TV_MAX = 2.0

# Bundle v3 nuevo: malla densa + albedo 1024 + neutral + 3 relights.
# 6 piezas como la figura (neutral texturizada + 3 re-iluminaciones con
# esferas + albedo + malla). Nombres disjuntos del zip-6 forense.
V3_ZIP_MESH = "mesh_dense.glb"
V3_ZIP_ALBEDO = "albedo_1024.png"
V3_ZIP_NEUTRAL = "relight_neutral.png"
V3_ZIP_KEY = "relight_key.png"
V3_ZIP_FILL = "relight_fill.png"
V3_ZIP_RIM = "relight_rim.png"

V3_ZIP_NAMES = (
    V3_ZIP_MESH,
    V3_ZIP_ALBEDO,
    V3_ZIP_NEUTRAL,
    V3_ZIP_KEY,
    V3_ZIP_FILL,
    V3_ZIP_RIM,
)

# Retencion de dias (R2 con retencion de dias) y worker de minutos.
# La via forense sigue en TTL 60; v3 no vive en 60s por diseno.
V3_RETENTION_DAYS = 7
V3_FIT_TIMEOUT_SECS = 600
V3_TEXTURE_TIMEOUT_SECS = 900

# Ajuste TexGAN enmascarado largo (cientos de pasos, init `w_avg`).
V3_TEXGAN_FIT_STEPS = 300

# Gate de entrada v3: resolucion minima FFHQ y umbral de yaw.
V3_MIN_SIDE_PX = 1024
V3_YAW_MAX_DEG = 15.0

# Claves de ramas efectivas (patron existente en `modal_app.py`).
V3_BRANCH_KEYS = ("deep3d", "pose", "texgan", "dpr", "displacement")


def v3_zip_names() -> tuple[str, ...]:
    """Nombres del bundle v3 en orden canonico. Solo via esta funcion."""
    return V3_ZIP_NAMES


def is_v3_zip_name(name: object) -> bool:
    """True si el nombre pertenece al bundle v3 (nunca al zip-6)."""
    return isinstance(name, str) and name in V3_ZIP_NAMES


def is_forensic_zip_name(name: object) -> bool:
    """True si el nombre pertenece al zip-6 forense (nunca a v3)."""
    try:
        from backend.domain import ZIP_NAMES as _FORENSIC
    except ImportError:
        return False
    return isinstance(name, str) and name in tuple(_FORENSIC)
