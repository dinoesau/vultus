"""Diag FLAME: foto + UV atlas + GLB cabeza coherente.

Uso: python3 scripts/render_diag.py --photo <jpg> --out <png>
Imprime verts, evidencia, sentinel y guarda diagnostico lado a lado
(foto crop + uv atlas). Sin torch. Solo PIL + numpy.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.domain import Err, Ok, parse_image_bytes, parse_landmarks
from backend.flame_fit import fit_flame
from backend.flame_texture import bake_flame, count_sentinel, texture_evidence
from backend.gnm import uv_to_png
from backend.gnm_assemble import VERT_COUNT, build_personalized_glb, is_flame_synthetic

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
    raw478 = _real_478(rgb)
    if raw478 is None:
        print("LANDMARKS_SYNTHETIC=1 (sin MediaPipe real; diag ilustrativo)")
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
    assert isinstance(parsed_lm, Ok)
    landmarks = parsed_lm.value
    fit_res = fit_flame(image, landmarks)
    if isinstance(fit_res, Err):
        print(f"FAIL fit: {fit_res.error}")
        return 1
    assert isinstance(fit_res, Ok)
    alb_res = bake_flame(image, fit_res.value, landmarks)
    if isinstance(alb_res, Err):
        print(f"FAIL albedo: {alb_res.error}")
        return 1
    assert isinstance(alb_res, Ok)
    raw_uv = alb_res.value.as_bytes()
    print(f"verts={VERT_COUNT} synthetic={is_flame_synthetic()} evidence={texture_evidence(raw_uv):.4f} sentinel={count_sentinel(raw_uv)}")
    glb_res = build_personalized_glb(fit_res.value, alb_res.value)
    if isinstance(glb_res, Err):
        print(f"FAIL glb: {glb_res.error}")
        return 1
    print(f"glb_len={len(glb_res.value.as_bytes())} magic={glb_res.value.as_bytes()[0:4]!r}")
    uv_png = uv_to_png(alb_res.value)
    uv_img = Image.open(io.BytesIO(uv_png)).convert("RGB")
    thumb = pil.copy()
    thumb.thumbnail((512, 512))
    canvas = Image.new("RGB", (1024, 512), (0, 0, 0))
    canvas.paste(thumb, (0, 0))
    canvas.paste(uv_img.resize((512, 512)), (512, 0))
    canvas.save(args.out)
    print(f"diag_saved={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
