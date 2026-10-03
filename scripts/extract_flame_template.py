"""Extractor offline del template FLAME real (Slice 1 RGB fitting).

Lee flame2023_Open.pkl (o generic_model.pkl) + FLAME_w_HIFI3D_UV.obj y
congela flame_template.bin con topologia canonica:
- 5023 verts, 9976 tris
- positions desde pkl (v_template) + uvs/faces desde OBJ HIFI3D
- split ojos [3931:5023) preservado por el OBJ (cara 0-1, ojos 2-3)

Formato flame_template.bin (LE, byte-estable):
  u32 verts (=5023), u32 tris (=9976),
  verts*3 f32 positions, verts*2 f32 uvs, tris*3 u32 indices.

Solo stdlib+numpy+struct, sin torch. Compatible Python 3.10.
Sin timestamps: mismos inputs -> mismos bytes.

Uso:
  DECA_DIR=/weights/deca FLAME_ASSETS_DIR=/weights/flame FFHQ_UV_DIR=/weights/ffhq-uv \\
    python3 scripts/extract_flame_template.py
  python3 scripts/extract_flame_template.py --check
  python3 scripts/extract_flame_template.py --output PATH
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import struct
import sys
import tempfile

import numpy as np

VERTS = 5023
TRIS = 9976
EYE_START = 3931

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "backend", "assets", "flame_template.bin")


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _candidate_pkl_paths() -> list[str]:
    cands: list[str] = []
    flame_dir = _env("FLAME_ASSETS_DIR")
    if flame_dir:
        cands.append(os.path.join(flame_dir, "flame2023_Open.pkl"))
        cands.append(os.path.join(flame_dir, "generic_model.pkl"))
    weights_dir = _env("WEIGHTS_DIR")
    if weights_dir:
        cands.append(os.path.join(weights_dir, "flame", "flame2023_Open.pkl"))
        cands.append(os.path.join(weights_dir, "flame", "generic_model.pkl"))
    cands.append(os.path.join(REPO_ROOT, "weights", "flame", "flame2023_Open.pkl"))
    seen: list[str] = []
    for c in cands:
        if c and c not in seen:
            seen.append(c)
    return seen


def _candidate_uv_paths() -> list[str]:
    cands: list[str] = []
    uv_dir = _env("FFHQ_UV_DIR")
    if uv_dir:
        cands.append(os.path.join(uv_dir, "FLAME_w_HIFI3D_UV.obj"))
    weights_dir = _env("WEIGHTS_DIR")
    if weights_dir:
        cands.append(os.path.join(weights_dir, "ffhq-uv", "FLAME_w_HIFI3D_UV.obj"))
    seen: list[str] = []
    for c in cands:
        if c and c not in seen:
            seen.append(c)
    return seen


def resolve_pkl_path() -> str:
    for cand in _candidate_pkl_paths():
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(f"flame pkl missing (buscado en {', '.join(_candidate_pkl_paths())})")


def resolve_uv_path() -> str:
    for cand in _candidate_uv_paths():
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(f"FLAME_w_HIFI3D_UV.obj missing (buscado en {', '.join(_candidate_uv_paths())})")


def load_positions(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")
    arr = None
    if isinstance(data, dict):
        for key in ("v_template", "verts", "vertices", "template", "vtemplate"):
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
                break
        if arr is None and "params" in data and isinstance(data["params"], dict):
            for key in ("v_template", "verts"):
                if key in data["params"]:
                    arr = np.asarray(data["params"][key], dtype=np.float32)
                    break
    else:
        for attr in ("v_template", "verts"):
            if hasattr(data, attr):
                arr = np.asarray(getattr(data, attr), dtype=np.float32)
                break
    if arr is None:
        raise RuntimeError(f"pkl sin v_template reconocible: {pkl_path} keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
    if arr.shape != (VERTS, 3):
        raise RuntimeError(f"positions shape {arr.shape} != ({VERTS}, 3)")
    if not bool(np.all(np.isfinite(arr))):
        raise RuntimeError("posicion no finita en pkl")
    return arr


def load_uv_faces(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    """UVs por vertice (last-wins sobre seams) + tris sobre v.

    El OBJ HIFI3D trae 5023 v, 5150 vt (seams ojos/boca) y 9976 f v/vt.
    Las caras v coinciden con la triangulacion del pkl; las vt se reducen
    a una por vertice con last-wins (igual que extract_gnm_template).
    """
    vt: list[tuple[float, float]] = []
    faces_v: list[tuple[int, int, int]] = []
    faces_vt: list[tuple[int, int, int]] = []
    with open(obj_path, encoding="utf-8", errors="strict") as fh:
        for line in fh:
            if line.startswith("vt "):
                parts = line.split()
                vt.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                v_idx: list[int] = []
                t_idx: list[int] = []
                for tok in line.split()[1:]:
                    if "/" in tok:
                        segs = tok.split("/")
                        v_idx.append(int(segs[0]) - 1)
                        t_idx.append(int(segs[1]) - 1 if len(segs) > 1 and segs[1] else -1)
                    else:
                        v_idx.append(int(tok) - 1)
                        t_idx.append(-1)
                if len(v_idx) == 3:
                    faces_v.append((v_idx[0], v_idx[1], v_idx[2]))
                    faces_vt.append((t_idx[0], t_idx[1], t_idx[2]))
                elif len(v_idx) == 4:
                    faces_v.append((v_idx[0], v_idx[1], v_idx[2]))
                    faces_vt.append((t_idx[0], t_idx[1], t_idx[2]))
                    faces_v.append((v_idx[0], v_idx[2], v_idx[3]))
                    faces_vt.append((t_idx[0], t_idx[2], t_idx[3]))
    vt_arr = np.asarray(vt, dtype=np.float32)
    tris = np.asarray(faces_v, dtype=np.int64)
    if tris.shape != (TRIS, 3):
        raise RuntimeError(f"faces shape {tris.shape} != ({TRIS}, 3) en {obj_path}")
    if not bool(np.all(np.isfinite(vt_arr))):
        raise RuntimeError("uv no finita en OBJ")
    if int(tris.min()) < 0 or int(tris.max()) >= VERTS:
        raise RuntimeError("faces con indice fuera de rango")
    uvs = np.zeros((VERTS, 2), dtype=np.float32)
    seen = np.zeros((VERTS,), dtype=bool)
    for (a, b, c), (ta, tb, tc) in zip(reversed(faces_v), reversed(faces_vt)):
        for v, t in ((a, ta), (b, tb), (c, tc)):
            if not seen[v] and 0 <= t < len(vt):
                uvs[v] = vt_arr[t]
                seen[v] = True
    if not bool(seen.all()):
        missing = int((~seen).sum())
        raise RuntimeError(f"vt no cubren {missing} verts en {obj_path}")
    if not bool(np.all(np.isfinite(uvs))):
        raise RuntimeError("uv derivada no finita")
    return uvs, tris


def encode_template(positions: np.ndarray, uvs: np.ndarray, tris: np.ndarray) -> bytes:
    out = bytearray()
    out += struct.pack("<II", VERTS, TRIS)
    for x, y, z in positions.tolist():
        out += struct.pack("<3f", x, y, z)
    for u, v in uvs.tolist():
        out += struct.pack("<2f", u, v)
    for a, b, c in tris.tolist():
        out += struct.pack("<III", int(a), int(b), int(c))
    return bytes(out)


def build_bytes(pkl_path: str, obj_path: str) -> bytes:
    positions = load_positions(pkl_path)
    uvs, tris = load_uv_faces(obj_path)
    return encode_template(positions, uvs, tris)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae flame_template.bin real desde pkl+OBJ")
    parser.add_argument("--check", action="store_true", help="verifica byte-identico sin escribir")
    parser.add_argument("--output", default=DEFAULT_OUT, help="ruta destino del .bin")
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        pkl_path = resolve_pkl_path()
        obj_path = resolve_uv_path()
        expected = build_bytes(pkl_path, obj_path)
    except RuntimeError as exc:
        print(f"extract_flame_template error: {exc}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(expected).hexdigest()
    if args.check:
        if not os.path.isfile(args.output):
            print(f"missing asset: {args.output} (genera con pkl+OBJ reales)", file=sys.stderr)
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
        print(f"template ok byte-identical: {args.output} verts={VERTS} tris={TRIS} len={len(expected)} sha256={digest}")
        return 0
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(expected)
    print(f"template wrote: {args.output} verts={VERTS} tris={TRIS} len={len(expected)} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
