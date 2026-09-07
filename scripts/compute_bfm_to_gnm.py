#!/usr/bin/env python3
"""Genera la LUT BFM->GNM offline una vez (Fase 2).

La LUT es una tabla de 256 bytes: remap por valor para el bake CPU puro.
Formula congelada v1: LUT[i] = (i * 5 + 17) % 256.
El runtime Rust la embebe con include_bytes! y solo hace lookup por texel;
el espejo Python la carga desde GNM_ASSETS_DIR con fallback a la formula.

Uso:
  python3 scripts/compute_bfm_to_gnm.py            # genera backend/assets/bfm_to_gnm.bin
  python3 scripts/compute_bfm_to_gnm.py --check    # verifica reproducible (hash estable)

Pesos GNM nunca entran al repo; solo esta LUT pequena (256 B) y el template
implicito en el builder GLB. Versionada por contenido (sha256 impreso).
"""

import hashlib
import os
import sys

LUT_FORMULA_A = 5
LUT_FORMULA_B = 17
LUT_LEN = 256

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "backend", "assets", "bfm_to_gnm.bin")


def build_lut() -> bytes:
    return bytes((i * LUT_FORMULA_A + LUT_FORMULA_B) % 256 for i in range(LUT_LEN))


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
        print(f"LUT ok: {out} sha256={digest}")
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(expected)
    print(f"LUT wrote: {out} len={len(expected)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
