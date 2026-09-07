#!/usr/bin/env python3
"""Extractor offline del template facial GNM real (Paso 1, gnm-real-mesh).

Congela un layout documentado como asset versionado porque el pkl de FLAME
no trae UVs destino utilizables en este repo (ver Paso 0):
- Grid UV 64x64 que cubre [0,1]^2 completo para que cada texel BFM 512
  tenga triangulo destino + pesos baricentricos (LUT v2).
- Superficie con volumen: elipsoide frontal + nariz gaussiana, expresion
  neutra fija. No son ojos/dientes/lengua separados: esos llegan solo como
  textura horneada (limitacion documentada del Paso 0 redefinido).
- Determinista, sin pesos externos, reproducible con --check.

Formato backend/assets/gnm_template.bin (LE):
  u32 verts (=4225), u32 tris (=8192),
  verts*3 f32 positions, verts*2 f32 uvs, tris*3 u32 indices.

Uso:
  python3 scripts/extract_face_template.py            # genera asset
  python3 scripts/extract_face_template.py --check    # verifica reproducible
"""

import hashlib
import math
import os
import struct
import sys

GRID = 64
VERTS = (GRID + 1) * (GRID + 1)  # 4225
TRIS = GRID * GRID * 2  # 8192

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "backend", "assets", "gnm_template.bin")


def build_template():
    """Retorna (positions, uvs, indices) como listas planas de floats/ints."""
    positions = []
    uvs = []
    for iy in range(GRID + 1):
        v = iy / GRID
        for ix in range(GRID + 1):
            u = ix / GRID
            x = (u - 0.5) * 1.64
            y = (0.5 - v) * 2.1
            r2 = (x / 0.82) ** 2 + (y / 1.05) ** 2
            z_base = 0.9 * math.sqrt(max(0.0, 1.0 - r2)) if r2 < 1.0 else 0.0
            nose = 0.28 * math.exp(-(x * x / 0.02 + (y + 0.05) ** 2 / 0.03))
            z = z_base + nose
            positions.append((x, y, z))
            uvs.append((u, v))
    indices = []
    stride = GRID + 1
    for qy in range(GRID):
        for qx in range(GRID):
            a = qy * stride + qx
            b = a + 1
            c = a + stride
            d = c + 1
            indices.append((a, c, b))
            indices.append((b, c, d))
    return positions, uvs, indices


def encode_template(positions, uvs, indices) -> bytes:
    out = struct.pack("<II", len(positions), len(indices) // 3 if isinstance(indices[0], tuple) else len(indices))
    # indices llega como lista de triplas; aplanar
    flat_idx = []
    for t in indices:
        flat_idx.extend(t)
    tris = len(flat_idx) // 3
    out = struct.pack("<II", len(positions), tris)
    for x, y, z in positions:
        out += struct.pack("<3f", x, y, z)
    for u, v in uvs:
        out += struct.pack("<2f", u, v)
    for i in flat_idx:
        out += struct.pack("<I", i)
    return out


def main(argv: list) -> int:
    out = DEFAULT_OUT
    check = False
    for a in argv[1:]:
        if a == "--check":
            check = True
        elif not a.startswith("-"):
            out = a
    positions, uvs, indices = build_template()
    expected = encode_template(positions, uvs, indices)
    digest = hashlib.sha256(expected).hexdigest()
    if check:
        if not os.path.exists(out):
            print(f"missing asset: {out}", file=sys.stderr)
            return 1
        with open(out, "rb") as f:
            actual = f.read()
        if actual != expected:
            print(
                f"template drift: {out} sha256={hashlib.sha256(actual).hexdigest()} "
                f"expected sha256={digest}",
                file=sys.stderr,
            )
            return 1
        print(f"template ok: {out} verts={VERTS} tris={TRIS} sha256={digest}")
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(expected)
    print(f"template wrote: {out} verts={VERTS} tris={TRIS} len={len(expected)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
