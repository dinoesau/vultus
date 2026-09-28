"""Gate E2E FLAME real (Step 7 plan-flame-deca-render): rojo sobre dobles, verde con forward real.

Pipeline por imagen: fit_flame + bake_flame + build_personalized_glb + zip-6,
sobre 2 fotos LFW misma persona (Bush 0001/0002, hashes congelados) + 1 foto
diferente-persona (Aaron Eckhart 0001, hash congelado).

Deps: stdlib + numpy + Pillow (Pillow declarada en backend/requirements-api.txt
y backend/requirements.txt; solo se usa en el borde para decodificar JPEG a RGB).
Sin torch, sin skimage: SSIM global implementado aqui.
CI-gatable: imprime cada check con valores y sale 0 solo si todo PASS, 1 si
algun FAIL. Sobre los dobles sha256 actuales el gate DEBE salir en rojo
(exit 1): los dobles no discriminan identidad (margen y SSIM fallan).

Dataset portable: DATASET_DIR (env) o candidato repo-relative
(../datasets/lfw junto al repo); sin ninguno se exige set DATASET_DIR.
Los tres sha256 (SHA_A/B/C) se verifican siempre.

Landmarks: intenta MediaPipe real si esta instalado y hay task en
weights/mediapipe/face_landmarker.task o ~/Code/weights/mediapipe/. Si no,
fallback determinista documentado: grilla sintetica de 478 puntos finitos
(imprime LANDMARKS_SYNTHETIC=1). Con VULTUS_REAL_ML=1 el fallback es
check 0-landmarks FAIL con SystemExit 1 (gameable cerrado); sin el flag el
gate sigue y falla por margen/SSIM.

Margen/SSIM solo pueden dar PASS con pipeline real (pesos del puente +
landmarks reales): sobre dobles un orden aparente se fuerza a FAIL.
El determinismo se prueba como propiedad (10x reruns identicos).
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.domain import (
    UV_LEN,
    ZIP_NAMES,
    Err,
    FitResult,
    ImageBytes,
    Landmarks,
    Ok,
    ZipBundle,
    parse_image_bytes,
    parse_landmarks,
    parse_rendered_image,
)
from backend.flame_fit import fit_flame, flame_distance
from backend.flame_fit import weights_present as fit_weights_present
from backend.flame_texture import (
    EVIDENCE_MIN,
    bake_flame,
    count_sentinel,
    max_abs_diff,
    texture_evidence,
)
from backend.flame_texture import weights_present as tex_weights_present
from backend.gnm import build_result_zip, uv_to_png
from backend.gnm_assemble import (
    EYE_COUNT,
    EYE_VERT_END,
    EYE_VERT_START,
    VERT_COUNT,
    build_personalized_glb,
    eye_vertex_indices,
    skin_vertex_indices,
)
from backend.pipeline_local import resolve_pbr_pngs

DATASET_ENV_VAR = "DATASET_DIR"


def dataset_dir() -> Path:
    """DATASET_DIR (env) o candidato repo-relative; sin ninguno exige env."""
    raw = os.environ.get(DATASET_ENV_VAR)
    if raw:
        return Path(raw)
    candidate = REPO_ROOT.parent / "datasets" / "lfw"
    if candidate.is_dir():
        return candidate
    print(f"[INPUT] FAIL set {DATASET_ENV_VAR} (candidato repo-relative ausente: {candidate})")
    raise SystemExit(1)


DATASET = dataset_dir()
PATH_A = DATASET / "George_W_Bush" / "George_W_Bush_0001.jpg"
PATH_B = DATASET / "George_W_Bush" / "George_W_Bush_0002.jpg"
PATH_C = DATASET / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg"

SHA_A = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"
SHA_B = "f04d53698da366ca8562b2d24ad9ed058116621b8fd0d51fcb46a6e5e470e0f3"
SHA_C = "b68ed8d50ba85209d826b962987077bc8e1826f7f2f325469f20738e1bc8bad2"

# Reruns del determinismo de margen (propiedad, no punto unico).
MARGIN_RERUNS = 10

# Geometria FLAME sucesora (no GNM 17821 ni doble viejo 4225).
EXPECTED_VERTS = VERT_COUNT
EXPECTED_EYE = EYE_COUNT
EXPECTED_PRIMS = 2
EXPECTED_MATERIALS = ("SkinPBR", "EyePBR")

LANDMARKS_TOTAL = 478

TASK_CANDIDATES = (
    REPO_ROOT / "weights" / "mediapipe" / "face_landmarker.task",
    Path.home() / "Code" / "weights" / "mediapipe" / "face_landmarker.task",
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_landmarks_bytes() -> bytes:
    # Fallback determinista documentado: grilla 22x22 recortada a 478 puntos
    # en [0, 1] con z=0. Pasa parse_landmarks (478 ternas finitas). Solo se
    # usa cuando no hay MediaPipe real; el gate sigue fallando por margen.
    pts: list[list[float]] = []
    for i in range(LANDMARKS_TOTAL):
        x = 0.25 + 0.5 * ((i % 22) / 21.0)
        y = 0.25 + 0.5 * ((i // 22) / 21.0)
        pts.append([x, y, 0.0])
    return json.dumps(pts).encode("utf-8")


def real_landmarks_bytes(rgb: np.ndarray[Any, Any]) -> bytes | None:
    """Intenta FaceLandmarker real. Devuelve None si no disponible."""
    task_path: Path | None = None
    for cand in TASK_CANDIDATES:
        if cand.is_file():
            task_path = cand
            break
    if task_path is None:
        return None
    try:
        import mediapipe as mp  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
        mp_base = importlib.import_module("mediapipe.tasks.python")
    except ImportError:
        return None
    try:
        base = mp_base.BaseOptions(model_asset_path=str(task_path))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with mp_vision.FaceLandmarker.create_from_options(opts) as landmarker:
            result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        if len(face) != LANDMARKS_TOTAL:
            return None
        pts = [[float(p.x), float(p.y), float(p.z)] for p in face]
        return json.dumps(pts).encode("utf-8")
    except Exception:  # noqa: BLE001 - cualquier fallo de MediaPipe cae al fallback documentado
        return None


def load_landmarks(image_raw: bytes, tag: str) -> tuple[Landmarks, bool]:
    """Devuelve (Landmarks, synthetic_flag). Falla duro si no parsea."""
    from PIL import Image

    try:
        pil = Image.open(io.BytesIO(image_raw)).convert("RGB")
    except Exception as exc:
        print(f"[LANDMARKS {tag}] FAIL no se pudo decodificar imagen: {exc}")
        raise SystemExit(1) from exc
    rgb: np.ndarray[Any, Any] = np.asarray(pil)
    raw = real_landmarks_bytes(rgb)
    if raw is not None:
        print("LANDMARKS_REAL=1")
        parsed = parse_landmarks(raw)
        if isinstance(parsed, Err):
            print(f"[LANDMARKS {tag}] FAIL landmarks reales no parsean: {parsed.error}")
            raise SystemExit(1)
        print(f"[LANDMARKS {tag}] real MediaPipe 478 puntos OK")
        return parsed.value, False
    print("LANDMARKS_SYNTHETIC=1")
    print(
        f"[LANDMARKS {tag}] fallback grilla sintetica determinista "
        f"({LANDMARKS_TOTAL} puntos, documentado; el gate falla por margen, no por esto)"
    )
    parsed = parse_landmarks(synthetic_landmarks_bytes())
    if isinstance(parsed, Err):
        print(f"[LANDMARKS {tag}] FAIL fallback no parsea: {parsed.error}")
        raise SystemExit(1)
    assert isinstance(parsed, Ok)
    return parsed.value, True


def glb_json(glb: bytes) -> dict[str, Any] | None:
    """JSON chunk del GLB o None si magia/header/json invalidos."""
    if len(glb) < 20 or glb[0:4] != b"glTF":
        return None
    json_len = int.from_bytes(glb[12:16], "little")
    try:
        doc: Any = json.loads(glb[20 : 20 + json_len].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    return doc


def glb_verts(doc: dict[str, Any]) -> int | None:
    accessors = doc.get("accessors")
    if isinstance(accessors, list) and accessors:
        first = accessors[0]
        if isinstance(first, dict):
            count = first.get("count")
            if isinstance(count, int) and not isinstance(count, bool):
                return count
    extras = doc.get("extras")
    if isinstance(extras, dict):
        verts = extras.get("verts")
        if isinstance(verts, int) and not isinstance(verts, bool):
            return verts
    return None


def glb_material_names(doc: dict[str, Any]) -> list[str]:
    names: list[str] = []
    mats = doc.get("materials")
    if isinstance(mats, list):
        for m in mats:
            if isinstance(m, dict):
                name = m.get("name")
                if isinstance(name, str):
                    names.append(name)
    return names


def glb_has_emissive(doc: dict[str, Any]) -> bool:
    mats = doc.get("materials")
    if isinstance(mats, list):
        for m in mats:
            if isinstance(m, dict) and ("emissiveFactor" in m or "emissiveTexture" in m):
                return True
    return False


def glb_primitive_count(doc: dict[str, Any]) -> int:
    total = 0
    meshes = doc.get("meshes")
    if isinstance(meshes, list):
        for mesh in meshes:
            if isinstance(mesh, dict):
                prims = mesh.get("primitives")
                if isinstance(prims, list):
                    total += len(prims)
    return total


def glb_offsets_aligned(doc: dict[str, Any]) -> bool:
    views = doc.get("bufferViews")
    if not isinstance(views, list) or not views:
        return False
    for view in views:
        if not isinstance(view, dict):
            return False
        off = view.get("byteOffset")
        if not isinstance(off, int) or isinstance(off, bool) or off % 4 != 0:
            return False
    return True


def glb_image_count(doc: dict[str, Any]) -> int:
    images = doc.get("images")
    if isinstance(images, list):
        return len(images)
    return 0


def ssim_global(a: bytes, b: bytes) -> float:
    """SSIM global (no ventaneado) sobre buffers RGB planos. 1.0 = identicos.

    Self-contained en numpy (sin skimage): media/varianza/covarianza
    globales con constantes C1/C2 estandar. Evidencia de cara real:
    SSIM(misma persona) > SSIM(distinta persona).
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    x: np.ndarray[Any, Any] = np.frombuffer(a, dtype=np.uint8).astype(np.float64)
    y: np.ndarray[Any, Any] = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mx = float(x.mean())
    my = float(y.mean())
    vx = float(((x - mx) ** 2).mean())
    vy = float(((y - my) ** 2).mean())
    cov = float(((x - mx) * (y - my)).mean())
    num = (2.0 * mx * my + c1) * (2.0 * cov + c2)
    den = (mx * mx + my * my + c1) * (vx + vy + c2)
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return num / den


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        results.append((name, passed, detail))
        print(f"[CHECK {name}] {'PASS' if passed else 'FAIL'} {detail}")

    # --- Modo real vs dobles (ruidoso, nunca silencioso) ---
    real_demanded = os.environ.get("VULTUS_REAL_ML") == "1"
    bridge_fit = fit_weights_present()
    bridge_tex = tex_weights_present()
    if real_demanded and not (bridge_fit and bridge_tex):
        check(
            "0-real",
            False,
            "VULTUS_REAL_ML=1 pero puente local ausente "
            f"(fit_weights={bridge_fit} tex_weights={bridge_tex}; DECA_DIR/FFHQ_UV_DIR vacios)",
        )
        print("GATE: FAIL (real exigido sin pesos)")
        return 1
    if bridge_fit or bridge_tex:
        print(f"BRIDGE_LOCAL=1 (fit_weights={bridge_fit} tex_weights={bridge_tex}; el forward real decide)")
    else:
        print("DOUBLES=1 (sin pesos locales; margen/SSIM deben fallar sobre dobles sha256)")

    # --- Prereq: JPEGs existen + sha256 congelados ---
    if DATASET_ENV_VAR in os.environ:
        print(f"[INPUT] dataset desde env {DATASET_ENV_VAR}={DATASET}")
    else:
        print(f"[INPUT] dataset repo-relative={DATASET} (set {DATASET_ENV_VAR} para override)")
    raws: dict[str, bytes] = {}
    for tag, path in (("A", PATH_A), ("B", PATH_B), ("C", PATH_C)):
        if not path.is_file():
            check(f"input-{tag}", False, f"no existe {path}")
            print("GATE: FAIL (faltan inputs LFW)")
            return 1
        raws[tag] = path.read_bytes()
    print(f"[INPUT] A={PATH_A} ({len(raws['A'])} bytes)")
    print(f"[INPUT] B={PATH_B} ({len(raws['B'])} bytes)")
    print(f"[INPUT] C={PATH_C} ({len(raws['C'])} bytes)")

    sha_a = sha256_hex(raws["A"])
    sha_b = sha256_hex(raws["B"])
    sha_c = sha256_hex(raws["C"])
    sha_ok = sha_a == SHA_A and sha_b == SHA_B and sha_c == SHA_C
    check(
        "0-sha256",
        sha_ok,
        f"A={sha_a} (esperado {SHA_A}); B={sha_b} (esperado {SHA_B}); "
        f"C={sha_c} (esperado {SHA_C})",
    )
    if not sha_ok:
        print("GATE: FAIL (sha256 no coincide con hashes congelados)")
        return 1

    # --- Parse imagenes + landmarks ---
    images: dict[str, ImageBytes] = {}
    for tag in ("A", "B", "C"):
        parsed = parse_image_bytes(raws[tag])
        if isinstance(parsed, Err):
            check(f"parse-{tag}", False, f"parse_image_bytes: {parsed.error}")
            print("GATE: FAIL (input no parsea)")
            return 1
        assert isinstance(parsed, Ok)
        images[tag] = parsed.value
    landmarks: dict[str, Landmarks] = {}
    synthetic: dict[str, bool] = {}
    for tag in ("A", "B", "C"):
        lm, syn = load_landmarks(raws[tag], tag)
        landmarks[tag] = lm
        synthetic[tag] = syn
    syn_tags = sorted(t for t in ("A", "B", "C") if synthetic[t])
    if real_demanded and syn_tags:
        check(
            "0-landmarks",
            False,
            f"sinteticas en {syn_tags} con VULTUS_REAL_ML=1 (fallback gameable cerrado)",
        )
        print("GATE: FAIL (real exigido con landmarks sinteticas)")
        return 1
    check(
        "0-landmarks",
        True,
        "reales MediaPipe 478 en A/B/C"
        if not syn_tags
        else f"sinteticas en {syn_tags} (dobles; margen/SSIM deben fallar)",
    )
    real_pipeline = bool(bridge_fit and bridge_tex) and not syn_tags
    print(
        f"PIPELINE_REAL={int(real_pipeline)} "
        f"(puente fit/tex={bridge_fit}/{bridge_tex} sinteticas={syn_tags or 'ninguna'}; "
        "margen/SSIM solo pasan en pipeline real; PBR duplica en ambos modos)"
    )

    # --- Pipeline x3: fit + albedo + glb ---
    fits: dict[str, FitResult] = {}
    albedos: dict[str, bytes] = {}
    glbs: dict[str, bytes] = {}
    for tag in ("A", "B", "C"):
        fit_res = fit_flame(images[tag], landmarks[tag])
        if isinstance(fit_res, Err):
            check(f"fit-{tag}", False, f"fit_flame: {fit_res.error}")
            print("GATE: FAIL (fit; con pesos reales esto es forward caido, sin pesos es doble)")
            return 1
        assert isinstance(fit_res, Ok)
        fits[tag] = fit_res.value
        alb_res = bake_flame(images[tag], fits[tag], landmarks[tag])
        if isinstance(alb_res, Err):
            check(f"albedo-{tag}", False, f"bake_flame: {alb_res.error}")
            print("GATE: FAIL (albedo)")
            return 1
        assert isinstance(alb_res, Ok)
        albedos[tag] = alb_res.value.as_bytes()
        glb_res = build_personalized_glb(fits[tag], alb_res.value)
        if isinstance(glb_res, Err):
            check(f"glb-{tag}", False, f"build_personalized_glb: {glb_res.error}")
            print("GATE: FAIL (glb)")
            return 1
        assert isinstance(glb_res, Ok)
        glbs[tag] = glb_res.value.as_bytes()

    # --- CHECK 1: albedo_len + glb magic + verts FLAME + PBR sin emisivo ---
    lens = {t: len(albedos[t]) for t in ("A", "B", "C")}
    magics = {t: glbs[t][0:4] for t in ("A", "B", "C")}
    docs = {t: glb_json(glbs[t]) for t in ("A", "B", "C")}
    verts: dict[str, int | None] = {}
    prims: dict[str, int] = {}
    mats: dict[str, list[str]] = {}
    emissive: dict[str, bool] = {}
    aligned: dict[str, bool] = {}
    imgcount: dict[str, int] = {}
    for t in ("A", "B", "C"):
        doc = docs[t]
        verts[t] = glb_verts(doc) if doc is not None else None
        prims[t] = glb_primitive_count(doc) if doc is not None else -1
        mats[t] = glb_material_names(doc) if doc is not None else []
        emissive[t] = glb_has_emissive(doc) if doc is not None else True
        aligned[t] = glb_offsets_aligned(doc) if doc is not None else False
        imgcount[t] = glb_image_count(doc) if doc is not None else -1
    c1_ok = (
        all(v == UV_LEN for v in lens.values())
        and all(m == b"glTF" for m in magics.values())
        and all(v == EXPECTED_VERTS for v in verts.values())
        and all(p == EXPECTED_PRIMS for p in prims.values())
        and all(m == list(EXPECTED_MATERIALS) for m in mats.values())
        and not any(emissive.values())
        and all(aligned.values())
        and all(n == 2 for n in imgcount.values())
    )
    check(
        "1-geometria",
        c1_ok,
        f"albedo_len A/B/C={lens['A']}/{lens['B']}/{lens['C']} (esperado {UV_LEN}); "
        f"verts A/B/C={verts['A']}/{verts['B']}/{verts['C']} (esperado {EXPECTED_VERTS}); "
        f"prims A/B/C={prims['A']}/{prims['B']}/{prims['C']} (esperado {EXPECTED_PRIMS}); "
        f"mats={mats['A']} (esperado {list(EXPECTED_MATERIALS)}); "
        f"emissive={emissive['A']}/{emissive['B']}/{emissive['C']} (esperado False); "
        f"offsets%4==0 A/B/C={aligned['A']}/{aligned['B']}/{aligned['C']}; images={imgcount['A']}",
    )

    # --- CHECK 2: cero sentinel + evidencia sobre umbral ---
    sentinels = {t: count_sentinel(albedos[t]) for t in ("A", "B", "C")}
    evidences = {t: texture_evidence(albedos[t]) for t in ("A", "B", "C")}
    c2_ok = all(v == 0 for v in sentinels.values()) and all(
        v >= EVIDENCE_MIN for v in evidences.values()
    )
    check(
        "2-sin-gris",
        c2_ok,
        f"sentinel A/B/C={sentinels['A']}/{sentinels['B']}/{sentinels['C']} (esperado 0); "
        f"evidence A/B/C={evidences['A']:.4f}/{evidences['B']:.4f}/{evidences['C']:.4f} "
        f"(min {EVIDENCE_MIN})",
    )

    # --- CHECK 3: ojos separados [3931:5023) contiguos, piel disjunta ---
    eye_idx = eye_vertex_indices()
    skin_idx = skin_vertex_indices()
    eye_set = set(eye_idx)
    skin_set = set(skin_idx)
    c3_ok = (
        len(eye_idx) == EXPECTED_EYE
        and eye_idx == list(range(EYE_VERT_START, EYE_VERT_END))
        and len(skin_idx) == EYE_VERT_START
        and len(eye_set & skin_set) == 0
        and min(eye_idx) == EYE_VERT_START
        and max(eye_idx) == EYE_VERT_END - 1
    )
    check(
        "3-ojos",
        c3_ok,
        f"eye len={len(eye_idx)} (esperado {EXPECTED_EYE}) rango=[{min(eye_idx)}:{max(eye_idx)}] "
        f"(esperado [{EYE_VERT_START}:{EYE_VERT_END - 1}] contiguo); skin len={len(skin_idx)} "
        f"interseccion={len(eye_set & skin_set)} (esperado 0)",
    )

    # --- CHECK 4: determinismo bytes-identicos-x2 (fit + albedo) ---
    fit_a2 = fit_flame(images["A"], landmarks["A"])
    assert isinstance(fit_a2, Ok)
    d_rerun = flame_distance(fits["A"].coeffs, fit_a2.value.coeffs)
    alb_a2 = bake_flame(images["A"], fits["A"], landmarks["A"])
    assert isinstance(alb_a2, Ok)
    alb_a2_bytes = alb_a2.value.as_bytes()
    rerun_diff = max_abs_diff(albedos["A"], alb_a2_bytes)
    c4_ok = d_rerun == 0.0 and rerun_diff == 0 and albedos["A"] == alb_a2_bytes
    check(
        "4-determinismo",
        c4_ok,
        f"fit rerun distance={d_rerun:.6f} (esperado 0.0); albedo max_abs_diff={rerun_diff} "
        f"(esperado 0); bytes-identicos={albedos['A'] == alb_a2_bytes}",
    )

    # --- CHECK 5: margen estricto d(A,A)==0 < d(A,B) < d(A,C), 10x + solo-real ---
    d_self = flame_distance(fits["A"].coeffs, fit_a2.value.coeffs)
    d_same = flame_distance(fits["A"].coeffs, fits["B"].coeffs)
    d_diff = flame_distance(fits["A"].coeffs, fits["C"].coeffs)
    rerun_ok = True
    for _ in range(MARGIN_RERUNS):
        fr = fit_flame(images["A"], landmarks["A"])
        if isinstance(fr, Err):
            rerun_ok = False
            break
        if flame_distance(fits["A"].coeffs, fr.value.coeffs) != 0.0:
            rerun_ok = False
            break
    order_ok = d_self == 0.0 and d_self < d_same < d_diff
    if real_pipeline:
        c5_ok = rerun_ok and order_ok
        c5_detail = (
            f"d(A,A)={d_self:.6f} d(A,Bmisma)={d_same:.6f} d(A,Cdistinta)={d_diff:.6f} "
            f"(exige 0==d_self<d_same<d_diff estricto; {MARGIN_RERUNS}x reruns identicos={rerun_ok})"
        )
    else:
        c5_ok = False
        c5_detail = (
            f"d(A,A)={d_self:.6f} d(A,Bmisma)={d_same:.6f} d(A,Cdistinta)={d_diff:.6f} "
            f"({MARGIN_RERUNS}x reruns identicos={rerun_ok}; orden aparente={order_ok} "
            "forzado a FAIL: PASS exige pipeline real; dobles sha256 no discriminan)"
        )
    check("5-margen", c5_ok, c5_detail)

    # --- CHECK 6: zip-6 canonico + sync python==TS ---
    alb_a_parsed = parse_rendered_image(albedos["A"])
    assert isinstance(alb_a_parsed, Ok)
    alb_b_parsed = parse_rendered_image(albedos["B"])
    assert isinstance(alb_b_parsed, Ok)
    uv_a_png = uv_to_png(alb_a_parsed.value)
    uv_b_png = uv_to_png(alb_b_parsed.value)
    # PBR skin-duplicate (DAG v10 HIL dueno): mismo helper que
    # pipeline/local_runner/modal. Ambos modos duplican uv como placeholder
    # honesto: sin mapas roughness/metalness reales; los ojos viven en el
    # GLB (2 primitivas SkinPBR+EyePBR). TODO mapas reales -> pbr!=uv.
    pbr_resolved = resolve_pbr_pngs(alb_a_parsed.value, alb_b_parsed.value)
    if isinstance(pbr_resolved, Err):
        check(
            "6-pbr",
            False,
            f"resolve_pbr_pngs Err inesperado en duplicate: {pbr_resolved.error}",
        )
        print("GATE: FAIL (pbr resolve caido)")
        return 1
    assert isinstance(pbr_resolved, Ok)
    pbr_a, pbr_b = pbr_resolved.value
    bundle = ZipBundle(
        uv_a_png=uv_a_png,
        uv_b_png=uv_b_png,
        mesh_a_glb=glbs["A"],
        mesh_b_glb=glbs["B"],
        pbr_a=pbr_a,
        pbr_b=pbr_b,
    )
    pbr_dup = uv_a_png == bundle.pbr_a and uv_b_png == bundle.pbr_b
    c6b_ok = pbr_dup
    c6b_detail = (
        f"PBR_DUPLICATE=1 (placeholder honesto ambos modos: pbr==uv; "
        f"ojos en GLB 2 primitivas; TODO mapas reales; duplicado={pbr_dup})"
    )
    check("6-pbr", c6b_ok, c6b_detail)
    zip_bytes = build_result_zip(bundle)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        compress = [zf.getinfo(n).compress_type for n in names]
    ts_text = (REPO_ROOT / "edge" / "contract.ts").read_text(encoding="utf-8")
    ts_ok = all(f'"{n}"' in ts_text for n in ZIP_NAMES)
    c6_ok = names == list(ZIP_NAMES) and all(c == zipfile.ZIP_STORED for c in compress) and ts_ok
    check(
        "6-zip6",
        c6_ok,
        f"namelist={names} (esperado {list(ZIP_NAMES)}); STORED={all(c == zipfile.ZIP_STORED for c in compress)}; "
        f"sync TS={ts_ok}; zip_len={len(zip_bytes)}",
    )

    # --- CHECK 7: SSIM evidencia cara real (misma > distinta), estable + solo-real ---
    s_self = ssim_global(albedos["A"], albedos["A"])
    s_same = ssim_global(albedos["A"], albedos["B"])
    s_diff = ssim_global(albedos["A"], albedos["C"])
    s_same2 = ssim_global(alb_a2_bytes, albedos["B"])
    s_diff2 = ssim_global(alb_a2_bytes, albedos["C"])
    stable_ok = s_same2 == s_same and s_diff2 == s_diff
    order_ok = s_self >= 0.999 and s_same > s_diff
    if real_pipeline:
        c7_ok = order_ok and stable_ok
        c7_detail = (
            f"SSIM(A,A)={s_self:.6f} (esperado 1.0); SSIM(A,Bmisma)={s_same:.6f} "
            f"SSIM(A,Cdistinta)={s_diff:.6f} (exige misma>distinta; estable rerun={stable_ok})"
        )
    else:
        c7_ok = False
        c7_detail = (
            f"SSIM(A,A)={s_self:.6f}; SSIM(A,Bmisma)={s_same:.6f} "
            f"SSIM(A,Cdistinta)={s_diff:.6f} (orden aparente={order_ok} "
            "estable rerun="
            f"{stable_ok} forzado a FAIL: PASS exige pipeline real; dobles dan ~iguales)"
        )
    check("7-ssim", c7_ok, c7_detail)

    # --- Diagnostico de version de codec (informativo) ---
    print(f"[CODEC] fit v2 VERSION=0x02 + u32 BE; albedos {UV_LEN} bytes; struct pack <{253 + 12}f")

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"GATE: FAIL (rojo) checks fallidos: {', '.join(failed)}")
        print(
            "CAUSA: dobles sha256 detectados (margen/SSIM/PBR sin orden por construccion) "
            f"o forward real caido (fit/albedo Err ruidoso); landmarks sinteticas={syn_tags or 'ninguna'}. "
            "Verde solo en Modal con DECA + FFHQ-UV cableados, landmarks reales y fotos congeladas."
        )
        return 1
    print("GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
