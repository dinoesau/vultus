"""Genera los goldens del E2E viewer-3d (S6): mesh_a.glb + mesh_b.glb + pbr PNGs.

Determinista: mismos marcadores -> mismos bytes (sha256 estables).
Via unica: backend.gnm_assemble.build_personalized_glb (GLB real con
SkinPBR + EyePBR, 5023 verts, PBR 512x512), nunca un GLB minimo falso que
salte GLTFLoader o el WebGLRenderer real.

Uso: python3 frontend/e2e/fixtures/generate_golden_glb.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.domain import (
    FitResult,
    Ok,
    parse_camera_params,
    parse_complete_uv,
    parse_eye_texture,
    parse_gnm_coeffs,
    parse_landmarks,
)
from backend.gnm import uv_to_png
from backend.gnm_assemble import build_personalized_glb

FIXTURES = Path(__file__).resolve().parent

# Marcadores congelados (espejo de test_zip8_por_seam_ensamblaje):
# A = fit 0.1 / albedo 0xA1, B = fit 0.5 / albedo 0xB2.
GOLDEN_MARKS = (("mesh_a", 0.1, 0xA1), ("mesh_b", 0.5, 0xB2))


def _fit(marker: float) -> FitResult:
    lm = parse_landmarks(json.dumps([[0.0, 0.0, 0.0]] * 478).encode("utf-8"))
    assert isinstance(lm, Ok)
    coeffs = parse_gnm_coeffs([marker] * 253)
    camera = parse_camera_params([0.0] * 12)
    assert isinstance(coeffs, Ok)
    assert isinstance(camera, Ok)
    return FitResult(coeffs=coeffs.value, camera=camera.value)


def _eye_raw(marker: int) -> bytes:
    # Patron determinista 512x512 distinto por cara (no blanco fallback).
    return bytes([(marker + i) & 0xFF for i in range(512 * 512) for _ in (0, 1, 2)][: 512 * 512 * 3])


def main() -> None:
    manifest: dict[str, str] = {}
    for name, fit_mark, alb_mark in GOLDEN_MARKS:
        raw_albedo = bytes([alb_mark & 0xFF, (alb_mark * 2) & 0xFF]) + bytes(512 * 512 * 3 - 2)
        albedo = parse_complete_uv(raw_albedo)
        assert isinstance(albedo, Ok)
        eye = parse_eye_texture(_eye_raw(alb_mark))
        assert isinstance(eye, Ok)
        out = build_personalized_glb(_fit(fit_mark), albedo.value, eye_texture=eye.value)
        assert isinstance(out, Ok), f"{name}: build fallo"
        glb = out.value.as_bytes()
        assert glb[0:4] == b"glTF"
        (FIXTURES / f"{name}.glb").write_bytes(glb)
        pbr = uv_to_png(albedo.value)
        assert pbr[0:8] == b"\x89PNG\r\n\x1a\n"
        (FIXTURES / f"pbr_{name[-1]}.png").write_bytes(pbr)
        manifest[f"{name}.glb"] = hashlib.sha256(glb).hexdigest()
        manifest[f"pbr_{name[-1]}.png"] = hashlib.sha256(pbr).hexdigest()
    # Determinismo x2: regenerar A y exigir bytes identicos.
    _, fit_mark, alb_mark = GOLDEN_MARKS[0]
    raw_albedo = bytes([alb_mark & 0xFF, (alb_mark * 2) & 0xFF]) + bytes(512 * 512 * 3 - 2)
    albedo = parse_complete_uv(raw_albedo)
    assert isinstance(albedo, Ok)
    again = build_personalized_glb(_fit(fit_mark), albedo.value,
                                   eye_texture=parse_eye_texture(_eye_raw(alb_mark)).value)  # type: ignore[union-attr]
    assert isinstance(again, Ok)
    assert hashlib.sha256(again.value.as_bytes()).hexdigest() == manifest["mesh_a.glb"]
    print(json.dumps(manifest, indent=2))
    manifest_path = FIXTURES / "golden-sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
