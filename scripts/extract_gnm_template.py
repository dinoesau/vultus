"""Extractor offline del template GNM real (Step 2, plan-gnm-real-fit).

Lee weights/gnm/versions/v3_0/gnm_head.npz y congela
backend/assets/gnm_template.bin con la topologia real:
- positions = template_vertex_positions (17821x3 f32)
- vertex uvs derivadas de quads/quad_uvs con last-wins (17662x4)
- indices = triangles (35324x3 u32)

Formato backend/assets/gnm_template.bin (LE, byte-estable):
  u32 verts (=17821), u32 tris (=35324),
  verts*3 f32 positions, verts*2 f32 uvs, tris*3 u32 indices.

Solo numpy+struct, sin torch. Compatible con Python 3.10.
Sin timestamps: mismo npz de entrada -> mismos bytes.

Uso:
  python3 scripts/extract_gnm_template.py
  python3 scripts/extract_gnm_template.py --output PATH
  python3 scripts/extract_gnm_template.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import tempfile

import numpy as np

VERTS = 17821
TRIS = 35324
QUADS = 17662

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "backend", "assets", "gnm_template.bin")
NPZ_REL = os.path.join("weights", "gnm", "versions", "v3_0", "gnm_head.npz")


def _candidate_npz_paths() -> list:
    cands: list = []
    direct = os.environ.get("GNM_NPZ_PATH", "").strip()
    if direct:
        cands.append(direct)
    assets_dir = os.environ.get("GNM_ASSETS_DIR", "").strip()
    if assets_dir:
        cands.append(os.path.join(assets_dir, "versions", "v3_0", "gnm_head.npz"))
        cands.append(os.path.join(assets_dir, "gnm_head.npz"))
    weights_dir = os.environ.get("WEIGHTS_DIR", "").strip()
    if weights_dir:
        cands.append(os.path.join(weights_dir, "gnm", "versions", "v3_0", "gnm_head.npz"))
        cands.append(os.path.join(weights_dir, "gnm", "gnm_head.npz"))
    cands.append(os.path.join(REPO_ROOT, NPZ_REL))
    cands.append(os.path.join(os.path.dirname(REPO_ROOT), "weights", "gnm", NPZ_REL))
    cands.append(os.path.join("/Users/esau.martinez/Code/weights/gnm", "versions", "v3_0", "gnm_head.npz"))
    cands.append(os.path.join("/Users/esau.martinez/Code/weights/gnm", "gnm_head.npz"))
    seen: list = []
    for c in cands:
        if c and c not in seen:
            seen.append(c)
    return seen


def resolve_npz_path() -> str:
    for cand in _candidate_npz_paths():
        if os.path.isfile(cand):
            return cand
    tried = ", ".join(_candidate_npz_paths())
    raise RuntimeError(f"gnm npz missing (buscado en {tried})")


def load_arrays(npz_path: str) -> tuple:
    data = np.load(npz_path)
    try:
        positions = np.asarray(data["template_vertex_positions"], dtype=np.float32)
        quads = np.asarray(data["quads"], dtype=np.int64)
        quad_uvs = np.asarray(data["quad_uvs"], dtype=np.float32)
        triangles = np.asarray(data["triangles"], dtype=np.int64)
    finally:
        data.close()
    if positions.shape != (VERTS, 3):
        raise RuntimeError(f"positions shape {positions.shape} != ({VERTS}, 3)")
    if quads.shape != (QUADS, 4):
        raise RuntimeError(f"quads shape {quads.shape} != ({QUADS}, 4)")
    if quad_uvs.shape != (QUADS, 4, 2):
        raise RuntimeError(f"quad_uvs shape {quad_uvs.shape} != ({QUADS}, 4, 2)")
    if triangles.shape != (TRIS, 3):
        raise RuntimeError(f"triangles shape {triangles.shape} != ({TRIS}, 3)")
    if not bool(np.all(np.isfinite(positions))):
        raise RuntimeError("posicion no finita en template_vertex_positions")
    if not bool(np.all(np.isfinite(quad_uvs))):
        raise RuntimeError("uv no finita en quad_uvs")
    if int(quads.min()) < 0 or int(quads.max()) >= VERTS:
        raise RuntimeError("quads con indice fuera de rango")
    if int(triangles.min()) < 0 or int(triangles.max()) >= VERTS:
        raise RuntimeError("triangles con indice fuera de rango")
    return positions, quads, quad_uvs, triangles


def build_vertex_uvs(quads: np.ndarray, quad_uvs: np.ndarray) -> np.ndarray:
    """Replica GNM vertex_uvs: last-wins sobre quads aplanados.

    indices = quads.ravel(), uvs = quad_uvs.reshape(-1, 2); ante vertices
    compartidos (seams) gana la ultima ocurrencia, resuelta con unique
    sobre el orden inverso (primera ocurrencia en reverso = ultima real).
    """
    flat_idx = np.asarray(quads.reshape(-1), dtype=np.int64)
    flat_uv = np.asarray(quad_uvs.reshape(-1, 2), dtype=np.float32)
    rev_idx = flat_idx[::-1]
    rev_uv = flat_uv[::-1]
    uniq, first = np.unique(rev_idx, return_index=True)
    vertex_uvs = np.zeros((VERTS, 2), dtype=np.float32)
    vertex_uvs[uniq] = rev_uv[first]
    if len(uniq) != VERTS:
        missing = sorted(set(range(VERTS)) - set(uniq.tolist()))
        raise RuntimeError(f"quads no cubren {len(missing)} verts, ej. {missing[:5]}")
    if not bool(np.all(np.isfinite(vertex_uvs))):
        raise RuntimeError("uv derivada no finita")
    return vertex_uvs


def encode_template(positions: np.ndarray, uvs: np.ndarray, triangles: np.ndarray) -> bytes:
    out = bytearray()
    out += struct.pack("<II", VERTS, TRIS)
    for x, y, z in positions.tolist():
        out += struct.pack("<3f", x, y, z)
    for u, v in uvs.tolist():
        out += struct.pack("<2f", u, v)
    for a, b, c in triangles.tolist():
        out += struct.pack("<III", a, b, c)
    return bytes(out)


def build_bytes(npz_path: str) -> bytes:
    positions, quads, quad_uvs, triangles = load_arrays(npz_path)
    uvs = build_vertex_uvs(quads, quad_uvs)
    return encode_template(positions, uvs, triangles)


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae gnm_template.bin real desde gnm_head.npz")
    parser.add_argument("--check", action="store_true", help="verifica byte-identico sin escribir")
    parser.add_argument("--output", default=DEFAULT_OUT, help="ruta destino del .bin")
    return parser.parse_args(argv[1:])


def main(argv: list) -> int:
    args = parse_args(argv)
    try:
        npz_path = resolve_npz_path()
        expected = build_bytes(npz_path)
    except RuntimeError as exc:
        print(f"extract_gnm_template error: {exc}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(expected).hexdigest()
    if args.check:
        if not os.path.isfile(args.output):
            print(f"missing asset: {args.output}", file=sys.stderr)
            return 1
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            tmp.write(expected)
            tmp.flush()
            with open(tmp.name, "rb") as f:
                staged = f.read()
        with open(args.output, "rb") as f:
            actual = f.read()
        if actual != staged:
            print(
                f"template drift: {args.output} "
                f"sha256={hashlib.sha256(actual).hexdigest()} expected sha256={digest}",
                file=sys.stderr,
            )
            return 1
        print(
            f"template ok byte-identical: {args.output} "
            f"verts={VERTS} tris={TRIS} len={len(expected)} sha256={digest} npz={npz_path}"
        )
        return 0
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(expected)
    print(
        f"template wrote: {args.output} "
        f"verts={VERTS} tris={TRIS} len={len(expected)} sha256={digest} npz={npz_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
