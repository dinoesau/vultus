"""Diag overlay: detected-68 (verde) vs projected-68 (rojo) + error medio en px.

Uso: python3 scripts/render_diag.py --photo <jpg> --out <png>
Imprime en stdout el hallazgo de ejes + mean reprojection error finito.
Sin torch. Solo PIL + numpy.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.domain import Err, parse_image_bytes, parse_landmarks
from backend.gnm_fit import fit_gnm, project
from backend.gnm_head import eval_landmarks68, mediapipe478_to_gnm68_targets

TASK_CANDIDATES = (
    REPO_ROOT / "weights" / "mediapipe" / "face_landmarker.task",
    Path.home() / "Code" / "weights" / "mediapipe" / "face_landmarker.task",
)


def _real_478(rgb: np.ndarray) -> bytes | None:
    task: Path | None = None
    for cand in TASK_CANDIDATES:
        if cand.is_file():
            task = cand
            break
    if task is None:
        return None
    try:
        import mediapipe as mp
    except ImportError:
        return None
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base = mp_python.BaseOptions(model_asset_path=str(task))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with mp_vision.FaceLandmarker.create_from_options(opts) as landmarker:
            result = landmarker.detect(mp_image)
        if not result.face_landmarks or len(result.face_landmarks[0]) != 478:
            return None
        pts = [[float(p.x), float(p.y), float(p.z)] for p in result.face_landmarks[0]]
        return json.dumps(pts).encode("utf-8")
    except Exception:  # noqa: BLE001 - sin MediaPipe no hay diag real, fallback documentado
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = Path(args.photo).read_bytes()
    parsed_img = parse_image_bytes(raw)
    if isinstance(parsed_img, Err):
        print(f"FAIL no parsea imagen: {parsed_img.error}")
        return 1
    image = parsed_img.value
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    rgb = np.asarray(pil)
    h, w = rgb.shape[0], rgb.shape[1]
    raw478 = _real_478(rgb)
    if raw478 is None:
        print("LANDMARKS_SYNTHETIC=1 (sin MediaPipe real; error ilustrativo)")
        pts = []
        for i in range(478):
            x = 0.25 + 0.5 * ((i % 22) / 21.0)
            y = 0.25 + 0.5 * ((i // 22) / 21.0)
            pts.append([x, y, 0.0])
        raw478 = json.dumps(pts).encode("utf-8")
    else:
        print("LANDMARKS_REAL=1")
    parsed_lm = parse_landmarks(raw478)
    if isinstance(parsed_lm, Err):
        print(f"FAIL no parsean landmarks: {parsed_lm.error}")
        return 1
    landmarks = parsed_lm.value
    fit_res = fit_gnm(image, landmarks)
    if isinstance(fit_res, Err):
        print(f"FAIL fit: {fit_res.error}")
        return 1
    fit = fit_res.value
    detected68 = np.asarray(mediapipe478_to_gnm68_targets(raw478), dtype=np.float64)
    mesh68 = np.asarray(eval_landmarks68(fit.coeffs.as_tuple()), dtype=np.float64)
    proj = np.asarray(project(fit.camera, mesh68), dtype=np.float64)
    det_px = detected68 * np.asarray([w, h], dtype=np.float64)
    proj_px = proj * np.asarray([w, h], dtype=np.float64)
    err = np.linalg.norm(det_px - proj_px, axis=1)
    mean_err = float(err.mean())
    print(f"mean_reproj_px_error={mean_err:.2f} w={w} h={h}")
    print(f"camera={list(fit.camera.as_tuple())}")
    print("AXIS_FINDING: project sin flip (y-down MediaPipe); si el overlay rojo "
          "aparece espejado en Y respecto al verde, el bake necesita flip de ejes.")
    out = pil.copy()
    draw = ImageDraw.Draw(out)
    for x, y in det_px:
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(0, 255, 0), width=1)
    for x, y in proj_px:
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 0, 0), width=1)
    out.save(args.out)
    print(f"diag_saved={args.out}")
    if not np.isfinite(mean_err):
        print("FAIL mean error no finito")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
