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

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from backend.domain import DomainError, Err, MlDecode, MlFailed, Ok
from backend.v3_bundle import V3Bundle
from backend.v3_contract import (
    V3_FIT_TIMEOUT_SECS,
    V3_MIN_SIDE_PX,
    V3_RETENTION_DAYS,
    V3_TEXGAN_FIT_STEPS,
    V3_TEXTURE_TIMEOUT_SECS,
    V3_YAW_MAX_DEG,
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


# Tres luces fijas de figura (perturbaciones SH deterministas sobre la estimada).
_RELIGHT_DELTAS: tuple[tuple[int, float], ...] = ((0, 0.0), (0, 2.0), (1, 1.0), (2, 1.0))


def _radial_normals_512() -> NDArray[np.float64]:
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float64)
    normals = np.stack([xx - 256.0, yy - 256.0, np.full_like(xx, 320.0)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
    return np.asarray(normals, dtype=np.float64)


def run_v3_subject(
    job_id: str,
    photo_path: str,
    report: Callable[[str, dict[str, float]], None],
    *,
    min_side_px: int = V3_MIN_SIDE_PX,
    texgan_steps: int = V3_TEXGAN_FIT_STEPS,
) -> Ok[V3Bundle] | Err[DomainError]:
    """Orquestador v3 por sujeto: densa -> pose -> atlas -> relights -> bundle.

    `report` se llama EXACTAMENTE una vez por sujeto, pase o rechace, con
    las 5 claves (prohibido fallback sin linea). El gate de resolucion usa
    `min_side_px` (default barra FFHQ 1024); el de yaw se mide de la pose
    refinada (`ry` en grados). Pasos pesados via modulos designados (torch
    lazy ahi, nunca aqui).
    """
    stats: dict[str, float] = {k: 0.0 for k in V3_BRANCH_KEYS}

    def _done(result: Ok[V3Bundle] | Err[DomainError]) -> Ok[V3Bundle] | Err[DomainError]:
        try:
            report(str(job_id), dict(stats))
        except Exception:  # noqa: BLE001, S110 - el reporte nunca tumba el job
            pass
        return result

    def _fail(details: str) -> Err[DomainError]:
        return Err(MlFailed(detail=MlDecode(details=details)))

    try:
        with open(photo_path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return _done(_fail(f"v3 subject photo unreadable: {exc}"))
    from backend.domain import parse_image_bytes

    img = parse_image_bytes(raw)
    if isinstance(img, Err):
        return _done(img)
    try:
        from PIL import Image

        with Image.open(photo_path) as handle:
            photo_rgb = handle.convert("RGB")
            width, height = int(photo_rgb.width), int(photo_rgb.height)
    except Exception as exc:  # noqa: BLE001 - foto ilegible: Err loud
        return _done(_fail(f"v3 subject photo decode failed: {exc}"))
    if min(int(width), int(height)) < int(min_side_px):
        return _done(_fail(f"v3 subject below entry bar: {width}x{height} < {min_side_px}"))
    try:
        import torch

        from backend import modal_app
        from backend.deep3d import (
            face_crop224,
            find_epoch,
            load_recon,
            photo_array,
            split_coeff_vector,
        )
        from backend.domain import parse_landmarks
        from backend.flame_fit import landmark68, landmark_points
        from backend.v3_bundle import build_dense_glb, build_v3_bundle
        from backend.v3_dense import (
            add_eyeballs,
            compute_shape_numpy,
            dense_uv01_from_mat,
            find_dense_mat,
            load_hifi_basis_from,
        )
        from backend.v3_light import albedo_from_photo, estimate_sh_from_photo, relight
        from backend.v3_pose import refine_pose
        from backend.v3_texture import assemble_v3_atlas
    except ImportError as exc:
        return _done(_fail(f"v3 subject backend missing: {exc}"))
    lm = parse_landmarks(modal_app.mediapipe_infer(str(job_id), img.value))
    if isinstance(lm, Err):
        return _done(lm)
    pts = landmark_points(lm.value)
    if isinstance(pts, Err):
        return _done(pts)
    photo_arr = photo_array(img.value)
    if isinstance(photo_arr, Err):
        return _done(photo_arr)
    mat = find_dense_mat()
    if mat is None:
        return _done(_fail("v3 subject dense mat missing"))
    basis = load_hifi_basis_from(mat)
    if isinstance(basis, Err):
        return _done(basis)
    try:
        crop = face_crop224(photo_arr.value, pts.value[:, 0], pts.value[:, 1])
        epoch = find_epoch()
        if epoch is None:
            return _done(_fail("v3 subject deep3d checkpoint missing"))
        recon = load_recon(epoch)
        tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = recon.forward_coeffs(tensor)
        vec = np.asarray(out.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        parts = split_coeff_vector(vec)
    except Exception as exc:  # noqa: BLE001 - forward caido: Err loud
        return _done(_fail(f"v3 subject deep3d forward failed: {exc}"))
    stats["deep3d"] = 1.0
    face_result = compute_shape_numpy(
        basis.value.mean, basis.value.id_base, basis.value.ex_base, parts["id"], parts["exp"]
    )
    if isinstance(face_result, Err):
        return _done(face_result)
    face, neutral = face_result.value
    eyes = add_eyeballs(face, np.asarray(basis.value.head_tri, dtype=np.int64))
    if isinstance(eyes, Err):
        return _done(eyes)
    stats["displacement"] = 1.0
    pts68 = landmark68(lm.value)
    if isinstance(pts68, Err):
        return _done(pts68)
    obj68 = np.asarray(neutral[basis.value.keypoints], dtype=np.float64)
    img68 = np.asarray(pts68.value[:, :2], dtype=np.float64) * np.array([float(width), float(height)])
    pose, _errors = refine_pose(obj68, img68, float(max(width, height)), steps=8)
    if not bool(np.isfinite(pose).all()):
        return _done(_fail("v3 subject pose non-finite"))
    yaw_deg = float(math.degrees(float(pose[4])))
    if not math.isfinite(yaw_deg) or abs(yaw_deg) > float(V3_YAW_MAX_DEG):
        return _done(_fail(f"v3 subject yaw out of range: {yaw_deg:.1f}"))
    stats["pose"] = 1.0
    try:
        xs = pts.value[:, 0] * float(width)
        ys = pts.value[:, 1] * float(height)
        pad = 0.15
        x0 = max(0, int(xs.min() - pad * (xs.max() - xs.min())))
        x1 = min(width, int(xs.max() + pad * (xs.max() - xs.min())) + 1)
        y0 = max(0, int(ys.min() - pad * (ys.max() - ys.min())))
        y1 = min(height, int(ys.max() + pad * (ys.max() - ys.min())) + 1)
        crop1024 = photo_rgb.crop((x0, y0, x1, y1)).resize((1024, 1024), Image.Resampling.BILINEAR)
        sampled = np.asarray(crop1024, dtype=np.float64)
        from backend.face_parsing import face_skin_mask

        valid = np.asarray(face_skin_mask(sampled), dtype=bool)
    except Exception as exc:  # noqa: BLE001 - crop/mascara caidos: Err loud
        return _done(_fail(f"v3 subject crop failed: {exc}"))
    atlas = assemble_v3_atlas(sampled, valid, int(texgan_steps))
    if isinstance(atlas, Err):
        return _done(atlas)
    stats["texgan"] = 1.0
    sh_result = estimate_sh_from_photo(photo_path)
    if isinstance(sh_result, Err):
        return _done(sh_result)
    stats["dpr"] = 1.0
    sh_base = np.asarray(sh_result.value, dtype=np.float64)
    normals = _radial_normals_512()
    small = photo_rgb.resize((512, 512), Image.Resampling.BILINEAR)
    photo512 = np.asarray(small, dtype=np.float64)
    albedo = albedo_from_photo(photo512, normals, sh_base)
    relights: list[bytes] = []
    for band, amount in _RELIGHT_DELTAS:
        sh = sh_base.copy()
        if amount:
            sh[band] += amount
        rel = relight(albedo, normals, sh)
        relights.append(np.rint(np.clip(rel, 0.0, 255.0)).astype(np.uint8).tobytes())
    uvs = dense_uv01_from_mat(mat)
    if isinstance(uvs, Err):
        return _done(uvs)
    glb = build_dense_glb(eyes.value[0], eyes.value[1], uvs.value)
    if isinstance(glb, Err):
        return _done(glb)
    bundle = build_v3_bundle(glb.value, atlas.value, relights)
    if isinstance(bundle, Err):
        return _done(bundle)
    return _done(Ok(bundle.value))
