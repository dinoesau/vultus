"""Golden E2E figura v3: foto -> figura en el trio LFW con metricas congeladas.

Cadena por sujeto: run_v3_subject (densa -> pose -> atlas TexGAN largo ->
4 relights DPR -> bundle v3 6 piezas) + margen de identidad sobre la malla
densa neutra + determinismo x2 a nivel bundle.

Deps: stdlib + numpy + Pillow + torch/mediapipe/scipy (pesos reales).
Sin JPEGs commiteados: lee DATASET_DIR o ../datasets/lfw (fuera del repo)
y verifica los tres sha256 congelados (misma fuente que e2e-flame-real).
Imprime cada check con valores; exit 0 solo si todo PASS.

Nota honesta: el trio LFW es 250px < barra FFHQ 1024. El golden verifica
AMBOS: el gate por defecto rechaza ruidoso (con linea de ramas) Y la
cadena completa corre con min_side_px explicito (eval bajo la barra).
TEXGAN_STEPS por env (default 2, rapido; prod usa V3_TEXGAN_FIT_STEPS).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.domain import Err, Ok  # noqa: E402
from backend.v3_bundle import V3Bundle  # noqa: E402
from backend.v3_texture import count_sentinel_1024, uv_total_variation_1024  # noqa: E402
from backend.v3_worker import V3_BRANCH_KEYS, run_v3_subject  # noqa: E402

MIRROR = Path.home() / "Code" / "weights" / "ffhq-uv-hf"

SHA_A = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"
SHA_B = "f04d53698da366ca8562b2d24ad9ed058116621b8fd0d51fcb46a6e5e470e0f3"
SHA_C = "b68ed8d50ba85209d826b962987077bc8e1826f7f2f325469f20738e1bc8bad2"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILURES.append(name)


def dataset_dir() -> Path:
    raw = os.environ.get("DATASET_DIR")
    if raw:
        return Path(raw)
    candidate = REPO_ROOT.parent / "datasets" / "lfw"
    if candidate.is_dir():
        return candidate
    print("[INPUT] FAIL set DATASET_DIR")
    raise SystemExit(1)


def use_mirror_env() -> None:
    os.environ.setdefault("DEEP3D_DIR", str(MIRROR / "checkpoints" / "deep3d_model"))
    os.environ.setdefault("TOPO_DIR", str(MIRROR / "topo_assets"))
    os.environ.setdefault("TEXGAN_DIR", str(MIRROR / "checkpoints" / "texgan_model"))
    os.environ.setdefault("PARSING_DIR", str(MIRROR / "checkpoints" / "parsing_model"))
    os.environ.setdefault("DPR_DIR", str(MIRROR / "checkpoints" / "dpr_model"))
    os.environ.setdefault("WEIGHTS_DIR", str(MIRROR.parent))
    os.environ.pop("VULTUS_REAL_ML", None)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_neutral(path: Path) -> np.ndarray:
    import torch

    from backend import modal_app
    from backend.deep3d import face_crop224, find_epoch, load_recon, photo_array, split_coeff_vector
    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.flame_fit import landmark_points
    from backend.v3_dense import compute_shape_numpy, load_hifi_basis_from

    raw = path.read_bytes()
    img = parse_image_bytes(raw)
    assert isinstance(img, Ok)
    lm = parse_landmarks(modal_app.mediapipe_infer("e2e", img.value))
    assert isinstance(lm, Ok)
    pts = landmark_points(lm.value)
    assert isinstance(pts, Ok)
    photo = photo_array(img.value)
    assert isinstance(photo, Ok)
    import backend.v3_dense as _vd

    mat = _vd.find_dense_mat()
    assert mat is not None
    basis = load_hifi_basis_from(mat)
    assert isinstance(basis, Ok)
    crop = face_crop224(photo.value, pts.value[:, 0], pts.value[:, 1])
    recon = load_recon(find_epoch())
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        vec = np.asarray(recon.forward_coeffs(tensor).detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    parts = split_coeff_vector(vec)
    face_result = compute_shape_numpy(
        basis.value.mean, basis.value.id_base, basis.value.ex_base, parts["id"], parts["exp"]
    )
    assert isinstance(face_result, Ok)
    return np.asarray(face_result.value[1], dtype=np.float64)


def main() -> int:
    use_mirror_env()
    from backend import modal_app

    modal_app.WEIGHTS_DIR = str(MIRROR.parent)
    modal_app._LM = None
    dataset = dataset_dir()
    paths = {
        "A": dataset / "George_W_Bush" / "George_W_Bush_0001.jpg",
        "B": dataset / "George_W_Bush" / "George_W_Bush_0002.jpg",
        "C": dataset / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg",
    }
    shas = {"A": SHA_A, "B": SHA_B, "C": SHA_C}
    for key, path in paths.items():
        check(f"sha-{key}", path.is_file() and sha256_file(path) == shas[key], str(path))
    if FAILURES:
        return 1
    steps = int(os.environ.get("V3_E2E_TEXGAN_STEPS", "2"))
    bundles: dict[str, V3Bundle] = {}
    for key, path in paths.items():
        lines: list[tuple[str, dict[str, float]]] = []

        def _rec(job: str, stats: dict[str, float], _slot: list[tuple[str, dict[str, float]]] = lines) -> None:
            _slot.append((job, stats))

        gate = run_v3_subject(f"e2e-{key}-gate", str(path), _rec)
        check(f"gate-{key}-rejects", isinstance(gate, Err), "250px bajo barra FFHQ")
        check(
            f"gate-{key}-line",
            len(lines) == 1 and all(k in lines[0][1] for k in V3_BRANCH_KEYS),
            "linea con 5 claves en rechazo",
        )
        lines.clear()
        result = run_v3_subject(f"e2e-{key}", str(path), _rec, min_side_px=250, texgan_steps=steps)
        check(f"bundle-{key}", isinstance(result, Ok) and isinstance(result.value, V3Bundle), "6 piezas")
        if not (isinstance(result, Ok) and isinstance(result.value, V3Bundle)):
            continue
        bundles[key] = result.value
        check(f"branch-{key}", len(lines) == 1 and all(k in lines[0][1] for k in V3_BRANCH_KEYS), str(lines[0][1]))
        albedo = result.value.albedo_1024_png
        check(
            f"atlas-{key}",
            len(albedo) == 1024 * 1024 * 3
            and count_sentinel_1024(albedo) == 0
            and uv_total_variation_1024(albedo) <= 2.0,
            f"TV={uv_total_variation_1024(albedo):.3f}",
        )
        relights = [result.value.relight_neutral, result.value.relight_key, result.value.relight_fill]
        arrs = [np.frombuffer(r, dtype=np.uint8).reshape(512, 512, 3) for r in relights]
        diffs = all(float(np.abs(arrs[0].astype(int) - other.astype(int)).max()) > 0 for other in arrs[1:])
        check(f"relights-{key}-differ", diffs, "neutral/key/fill distintos")
        check(f"glb-{key}", result.value.mesh_dense_glb[:4] == b"glTF", f"{len(result.value.mesh_dense_glb)} bytes")
    if len(bundles) != 3:
        return 1
    n_a = dense_neutral(paths["A"])
    n_b = dense_neutral(paths["B"])
    n_c = dense_neutral(paths["C"])
    d_same = float(np.linalg.norm((n_a - n_b).reshape(-1)))
    d_diff = float(np.linalg.norm((n_a - n_c).reshape(-1)))
    check("margin-same-lt-diff", 0.0 < d_same < d_diff, f"same={d_same:.4f} diff={d_diff:.4f}")
    lines2: list[tuple[str, dict[str, float]]] = []

    def _rec2(job: str, stats: dict[str, float]) -> None:
        lines2.append((job, stats))

    rerun = run_v3_subject("e2e-A-rerun", str(paths["A"]), _rec2, min_side_px=250, texgan_steps=steps)
    same_bytes = (
        isinstance(rerun, Ok)
        and isinstance(rerun.value, V3Bundle)
        and rerun.value.mesh_dense_glb == bundles["A"].mesh_dense_glb
        and rerun.value.albedo_1024_png == bundles["A"].albedo_1024_png
    )
    check("determinism-bundle-x2", bool(same_bytes), "GLB + albedo identicos")
    print(f"E2E v3 figure: {'PASS' if not FAILURES else 'FAIL ' + ','.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
