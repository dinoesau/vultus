#!/usr/bin/env python3
"""Genera la LUT BFM->GNM v2 baricentrica offline una vez (gnm-real-mesh).

Cada texel destino 512x512 (262144) mapea al triangulo del template
GRID 64 que contiene su centro (u,v) + 3 pesos baricentricos f32.
El runtime solo hace lookup + sampleo en los 3 vertices, sin torch.

Layout por texel (16 B LE): u32 tri, f32 w0, f32 w1, f32 w2.
Orden de pesos = orden de vertices del triangulo en el template:
  T0=(A,C,B) con (wA,wC,wB), T1=(B,C,D) con (wB,wC,wD).
Ver scripts/extract_face_template.py para A/B/C/D.

Uso:
  python3 scripts/compute_bfm_to_gnm.py            # genera backend/assets/bfm_to_gnm_v2.bin
  python3 scripts/compute_bfm_to_gnm.py --check    # verifica reproducible (hash estable)

La LUT v1 por valor (256 B) queda retirada; no conviven dos caminos.
"""

import hashlib
import os
import struct
import sys

UV = 512
GRID = 64
TEXELS = UV * UV  # 262144
ENTRY_BYTES = 16

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "backend", "assets", "bfm_to_gnm_v2.bin")


def lut_entry(x: int, y: int):
    u = (x + 0.5) / UV
    v = (y + 0.5) / UV
    qx = int(u * GRID)
    qy = int(v * GRID)
    if qx > GRID - 1:
        qx = GRID - 1
    if qy > GRID - 1:
        qy = GRID - 1
    fu = u * GRID - qx
    fv = v * GRID - qy
    quad = qy * GRID + qx
    if fu + fv <= 1.0:
        return quad * 2, (1.0 - fu - fv, fv, fu)
    return quad * 2 + 1, (1.0 - fv, 1.0 - fu, fu + fv - 1.0)


def build_lut() -> bytes:
    buf = bytearray(TEXELS * ENTRY_BYTES)
    for y in range(UV):
        for x in range(UV):
            tri, (w0, w1, w2) = lut_entry(x, y)
            off = (y * UV + x) * ENTRY_BYTES
            struct.pack_into("<I3f", buf, off, tri, w0, w1, w2)
    return bytes(buf)


def main(argv: list) -> int:
    out = DEFAULT_OUT
    check = False
    for a in argv[1:]:
        if a == "--check":
            check = True
        elif not a.startswith("-"):
            out = a
    expected = build_lut()
    digest = hashlib.sha256(expected).hexdigest()
    if check:
        if not os.path.exists(out):
            print(f"missing asset: {out}", file=sys.stderr)
            return 1
        with open(out, "rb") as f:
            actual = f.read()
        if actual != expected:
            print(
                f"LUT drift: {out} sha256={hashlib.sha256(actual).hexdigest()} "
                f"expected sha256={digest}",
                file=sys.stderr,
            )
            return 1
        print(f"LUT v2 ok: {out} texels={TEXELS} sha256={digest}")
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(expected)
    print(f"LUT v2 wrote: {out} len={len(expected)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
