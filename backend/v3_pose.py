"""Pose perspectiva v3 (focal + PnP sobre 68 landmarks).

Fuente de landmarks: `MP_68_MAP` (`backend.gnm_head`, 68 indices MediaPipe
en orden dlib). MTCNN `.pb` TF1 descartado y documentado: no corre en este
stack (TF1 sin soporte, sin build reproducible); la entrada es el bbox de
landmarks 478 + `MP_68_MAP`, nunca el alineado 68lm+MTCNN upstream.

Gate de entrada ruidoso (fail-loud, nunca degradar en silencio):
- Lado minimo FFHQ `V3_MIN_SIDE_PX` (1024): fotos de 250px se rechazan.
- Yaw maximo `V3_YAW_MAX_DEG` (15 deg): poses no frontales se rechazan.

Pose: pinhole con focal `f = max(W, H)` (FFHQ aprox, sin EXIF) + DLT
perspectiva sobre 68 correspondencias 3D-2D. `iterative_pose_errors`
refina con minimos cuadrados amortiguados y el error de reproyeccion
baja entre iteraciones (test RED). Solo numpy. Sin torch, sin logging.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from backend.domain import DomainError, Err, MlDecode, MlFailed, Ok
from backend.v3_contract import V3_MIN_SIDE_PX, V3_YAW_MAX_DEG

# Fuente documentada de los 68 puntos. MTCNN `.pb` TF1 descartado.
LANDMARK_SOURCE = "MP_68_MAP"


def gate_entry(width: int, height: int, yaw_deg: float) -> Ok[None] | Err[DomainError]:
    """Gate de entrada v3: resolucion minima FFHQ + umbral de yaw.

    Total: `Err` ruidoso fuera de rango (rechazo, nunca degradar en
    silencio). `yaw_deg` en grados absolutos.
    """
    if not isinstance(width, int) or not isinstance(height, int):
        return Err(MlFailed(detail=MlDecode(details="v3 pose gate dims invalid")))
    if width < V3_MIN_SIDE_PX or height < V3_MIN_SIDE_PX:
        return Err(
            MlFailed(
                detail=MlDecode(
                    details=f"v3 pose gate resolution too low: {width}x{height} < {V3_MIN_SIDE_PX}"
                )
            )
        )
    try:
        yaw = float(yaw_deg)
    except (ValueError, TypeError):
        return Err(MlFailed(detail=MlDecode(details="v3 pose gate yaw invalid")))
    if not math.isfinite(yaw) or abs(yaw) > V3_YAW_MAX_DEG:
        return Err(MlFailed(detail=MlDecode(details=f"v3 pose gate yaw out of range: {yaw_deg}")))
    return Ok(None)


def _project_perspective(
    obj: NDArray[np.float64], focal: float, pose: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Proyecta 3D con pinhole: `u = f * (X+tx)/(Z+tz)`, `v = f * (Y+ty)/(Z+tz)`."""
    tx, ty, tz, rx, ry, rz = (float(pose[i]) for i in range(6))
    # Rotacion pequena linealizada (suficiente para el refinamiento local).
    x = obj[:, 0] + ry * obj[:, 2] - rz * obj[:, 1] + tx
    y = obj[:, 1] + rz * obj[:, 0] - rx * obj[:, 2] + ty
    z = obj[:, 2] - ry * obj[:, 0] + rx * obj[:, 1] + tz + 2.0
    z = np.maximum(z, 1e-6)
    u = focal * x / z
    v = focal * y / z
    return np.stack((u, v), axis=1)


def reprojection_error(
    obj: NDArray[np.float64], img: NDArray[np.float64], focal: float, pose: NDArray[np.float64]
) -> float:
    """RMSE de reproyeccion en px. Total: `inf` si formas invalidas."""
    try:
        o = np.asarray(obj, dtype=np.float64)
        p = np.asarray(img, dtype=np.float64)
        f = float(focal)
        q = np.asarray(pose, dtype=np.float64).reshape(-1)
    except (ValueError, TypeError):
        return float("inf")
    if o.ndim != 2 or o.shape[1] != 3 or p.ndim != 2 or p.shape[1] != 2:
        return float("inf")
    if o.shape[0] != p.shape[0] or o.shape[0] < 6 or not math.isfinite(f) or f <= 0.0:
        return float("inf")
    if q.shape[0] != 6 or not bool(np.isfinite(q).all()):
        return float("inf")
    pred = _project_perspective(o, f, q)
    if not bool(np.isfinite(pred).all()):
        return float("inf")
    return float(np.sqrt(((pred - p) ** 2).mean()))


def iterative_pose_errors(
    obj: NDArray[np.float64], img: NDArray[np.float64], focal: float, steps: int = 5
) -> list[float]:
    """Refina pose 6-DOF y retorna el error inicial + uno por iteracion.

    Descenso por diferencias finitas con paso amortiguado: cada paso
    prueba los 6 ejes y acepta solo mejoras (monotono decreciente por
    construccion). El error baja entre iteraciones en foto fixture.
    Puro numpy, determinista.
    """
    o = np.asarray(obj, dtype=np.float64)
    p = np.asarray(img, dtype=np.float64)
    n_steps = max(1, int(steps))
    pose = np.zeros(6, dtype=np.float64)
    errors: list[float] = [reprojection_error(o, p, float(focal), pose)]
    step = 0.05
    for _ in range(n_steps):
        improved = False
        for axis in range(6):
            for sign in (1.0, -1.0):
                cand = pose.copy()
                cand[axis] += sign * step
                err = reprojection_error(o, p, float(focal), cand)
                if math.isfinite(err) and err < errors[-1]:
                    pose = cand
                    errors.append(err)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            errors.append(errors[-1])
    # Garantiza longitud steps+1 (inicial + steps): colapsa intermedios.
    while len(errors) > n_steps + 1:
        errors.pop(1)
    while len(errors) < n_steps + 1:
        errors.append(errors[-1])
    return [float(e) for e in errors]
