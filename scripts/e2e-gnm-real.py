"""Gate E2E GNM real (Step 3 plan-gnm-real-fit): rojo sobre los dobles actuales.

Pipeline por imagen: fit_gnm + build_albedo + build_personalized_glb +
coeff_distance, sobre 2 fotos LFW misma persona (Bush 0001/0002, hashes
congelados) + 1 foto diferente-persona (Aaron Eckhart 0001).

Solo stdlib + numpy + PIL (sin torch). CI-gatable: imprime cada check con
valores y sale 0 solo si todo PASS, 1 si algun FAIL. Sobre los dobles
sha256 actuales el gate DEBE salir en rojo (exit 1).

Landmarks: intenta MediaPipe real si esta instalado y hay task en
weights/mediapipe/face_landmarker.task o ~/Code/weights/mediapipe/. Si no,
fallback determinista documentado: grilla sintetica de 478 puntos finitos
(no se silencia: imprime LANDMARKS_SYNTHETIC=1). El gate falla por
ruido/textura, no por landmarks.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.domain import (
    LANDMARKS_LEN,
    UV_LEN,
    Err,
    Ok,
    parse_image_bytes,
    parse_landmarks,
)
from backend.gnm_assemble import build_personalized_glb
from backend.gnm_fit import coeff_distance, fit_gnm
from backend.gnm_texture import build_albedo

DATASET = Path("/Users/esau.martinez/Code/datasets/lfw")
PATH_A = DATASET / "George_W_Bush" / "George_W_Bush_0001.jpg"
PATH_B = DATASET / "George_W_Bush" / "George_W_Bush_0002.jpg"
PATH_C = DATASET / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg"

SHA_A = "b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576"
SHA_B = "f04d53698da366ca8562b2d24ad9ed058116621b8fd0d51fcb46a6e5e470e0f3"

EXPECTED_VERTS = 17821
SMOOTH_MAX = 50.0
STD_LO = 15.0
STD_HI = 100.0
CORR_MIN = 0.3

TASK_CANDIDATES = (
    REPO_ROOT / "weights" / "mediapipe" / "face_landmarker.task",
    Path.home() / "Code" / "weights" / "mediapipe" / "face_landmarker.task",
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_landmarks_bytes() -> bytes:
    # Fallback determinista documentado: grilla 22x22 recortada a 478 puntos
    # en [0, 1] con z=0. Pasa parse_landmarks (478 ternas finitas). Solo se
    # usa cuando no hay MediaPipe real; el gate sigue fallando por ruido.
    pts = []
    for i in range(LANDMARKS_LEN):
        x = 0.25 + 0.5 * ((i % 22) / 21.0)
        y = 0.25 + 0.5 * ((i // 22) / 21.0)
        pts.append([x, y, 0.0])
    return json.dumps(pts).encode("utf-8")


def real_landmarks_bytes(rgb: np.ndarray) -> bytes | None:
    """Intenta FaceLandmarker real. Devuelve None si no disponible."""
    task_path: Path | None = None
    for cand in TASK_CANDIDATES:
        if cand.is_file():
            task_path = cand
            break
    if task_path is None:
        return None
    try:
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base = mp_python.BaseOptions(model_asset_path=str(task_path))
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
        if len(face) != LANDMARKS_LEN:
            return None
        pts = [[float(p.x), float(p.y), float(p.z)] for p in face]
        return json.dumps(pts).encode("utf-8")
    except Exception:  # noqa: BLE001 - cualquier fallo de MediaPipe cae al fallback documentado
        return None


def load_landmarks(image_raw: bytes, tag: str) -> tuple[object, bool]:
    """Devuelve (Landmarks, synthetic_flag). Falla duro si no parsea."""
    try:
        pil = Image.open(io.BytesIO(image_raw)).convert("RGB")
    except Exception as exc:
        print(f"[LANDMARKS {tag}] FAIL no se pudo decodificar imagen: {exc}")
        raise SystemExit(1) from exc
    rgb = np.asarray(pil)
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
        f"({LANDMARKS_LEN} puntos, documentado; el gate falla por ruido, no por esto)"
    )
    parsed = parse_landmarks(synthetic_landmarks_bytes())
    if isinstance(parsed, Err):
        print(f"[LANDMARKS {tag}] FAIL fallback no parsea: {parsed.error}")
        raise SystemExit(1)
    assert isinstance(parsed, Ok)
    return parsed.value, True


def glb_verts(glb: bytes) -> int | None:
    """Extrae verts del JSON chunk del GLB (accessors[0].count o extras.verts)."""
    try:
        if len(glb) < 20 or glb[0:4] != b"glTF":
            return None
        json_len = struct.unpack("<I", glb[12:16])[0]
        doc = json.loads(glb[20 : 20 + json_len].decode("utf-8"))
        accessors = doc.get("accessors") or []
        if accessors and isinstance(accessors[0].get("count"), int):
            return int(accessors[0]["count"])
        extras = doc.get("extras") or {}
        if isinstance(extras.get("verts"), int):
            return int(extras["verts"])
        return None
    except (ValueError, KeyError, struct.error, UnicodeDecodeError):
        return None


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(float((x * x).sum()) * float((y * y).sum())))
    if denom == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        results.append((name, passed, detail))
        print(f"[CHECK {name}] {'PASS' if passed else 'FAIL'} {detail}")

    # --- Prereq: JPEGs existen + sha256 congelados ---
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
    sha_ok = sha_a == SHA_A and sha_b == SHA_B
    check(
        "0-sha256",
        sha_ok,
        f"A={sha_a} (esperado {SHA_A}); B={sha_b} (esperado {SHA_B})",
    )
    if not sha_ok:
        print("GATE: FAIL (sha256 no coincide con README congelado)")
        return 1

    # --- Parse imagenes + landmarks ---
    images = {}
    for tag in ("A", "B", "C"):
        parsed = parse_image_bytes(raws[tag])
        if isinstance(parsed, Err):
            check(f"parse-{tag}", False, f"parse_image_bytes: {parsed.error}")
            print("GATE: FAIL (input no parsea)")
            return 1
        images[tag] = parsed.value
    landmarks = {}
    for tag in ("A", "B", "C"):
        landmarks[tag], _ = load_landmarks(raws[tag], tag)

    # --- Pipeline x3: fit + albedo + glb ---
    fits = {}
    albedos: dict[str, bytes] = {}
    glbs: dict[str, bytes | None] = {}
    glb_errors: dict[str, str] = {}
    for tag in ("A", "B", "C"):
        fit_res = fit_gnm(images[tag], landmarks[tag])
        if isinstance(fit_res, Err):
            check(f"fit-{tag}", False, f"fit_gnm: {fit_res.error}")
            print("GATE: FAIL (fit)")
            return 1
        fits[tag] = fit_res.value
        alb_res = build_albedo(images[tag], fits[tag], landmarks[tag])
        if isinstance(alb_res, Err):
            check(f"albedo-{tag}", False, f"build_albedo: {alb_res.error}")
            print("GATE: FAIL (albedo)")
            return 1
        albedos[tag] = alb_res.value.as_bytes()
        glb_res = build_personalized_glb(fits[tag], alb_res.value)
        if isinstance(glb_res, Err):
            # No aborta: el resto de asserts (ruido/textura/margen) se sigue
            # evaluando con fit+albedo. El CHECK 1 reporta este fallo.
            glbs[tag] = None
            glb_errors[tag] = str(glb_res.error)
            print(f"[GLB {tag}] build fallo (se sigue evaluando): {glb_res.error}")
        else:
            glbs[tag] = glb_res.value.as_bytes()

    # --- CHECK 1: albedo_len + glb magic + verts ---
    lens = {t: len(albedos[t]) for t in ("A", "B", "C")}
    magics = {t: (glbs[t][0:4] if glbs[t] is not None else None) for t in ("A", "B", "C")}
    verts = {t: (glb_verts(glbs[t]) if glbs[t] is not None else None) for t in ("A", "B", "C")}
    c1_ok = (
        all(v == UV_LEN for v in lens.values())
        and all(m == b"glTF" for m in magics.values())
        and all(v == EXPECTED_VERTS for v in verts.values())
    )
    glb_note = ""
    if glb_errors:
        glb_note = f" build_error={glb_errors}; "
    check(
        "1-geometria",
        c1_ok,
        f"albedo_len A/B/C={lens['A']}/{lens['B']}/{lens['C']} "
        f"(esperado {UV_LEN}); magic A/B/C="
        f"{magics['A']!r}/{magics['B']!r}/{magics['C']!r}; "
        f"verts A/B/C={verts['A']}/{verts['B']}/{verts['C']} "
        f"(esperado {EXPECTED_VERTS}; doble actual 4225 -> rojo esperado)."
        f"{glb_note}",
    )

    # --- CHECK 2: suavidad espacial mean|albedo[i]-albedo[i+3]| < 50 ---
    smooth: dict[str, float] = {}
    for tag in ("A", "B", "C"):
        arr = np.frombuffer(albedos[tag], dtype=np.uint8).astype(np.float32)
        smooth[tag] = float(np.mean(np.abs(arr[3:] - arr[:-3])))
    c2_ok = all(v < SMOOTH_MAX for v in smooth.values())
    check(
        "2-suavidad",
        c2_ok,
        f"mean|a[i]-a[i+3]| A={smooth['A']:.2f} B={smooth['B']:.2f} "
        f"C={smooth['C']:.2f} (umbral <{SMOOTH_MAX}; ruido sha256 ~80+)",
    )

    # --- CHECK 3: std 15..100 + correlacion con thumbnail > 0.3 ---
    stds: dict[str, float] = {}
    corrs: dict[str, float] = {}
    for tag in ("A", "B", "C"):
        arr = np.frombuffer(albedos[tag], dtype=np.uint8).astype(np.float64)
        stds[tag] = float(arr.std())
        thumb = np.asarray(
            Image.open(io.BytesIO(raws[tag])).convert("RGB").resize((512, 512))
        ).astype(np.float64)
        corrs[tag] = pearson_corr(
            thumb, np.frombuffer(albedos[tag], dtype=np.uint8)
        )
    c3_ok = all(STD_LO <= v <= STD_HI for v in stds.values()) and all(
        v > CORR_MIN for v in corrs.values()
    )
    check(
        "3-foto-vs-ruido",
        c3_ok,
        f"std A={stds['A']:.2f} B={stds['B']:.2f} C={stds['C']:.2f} "
        f"(rango {STD_LO}..{STD_HI}); corr A={corrs['A']:.3f} "
        f"B={corrs['B']:.3f} C={corrs['C']:.3f} (umbral >{CORR_MIN}; "
        f"dobles hash no correlacionan ~0)",
    )

    # --- CHECK 4: margen estricto d(A,A)==0 < d(A,B) < d(A,C) ---
    fit_a2 = fit_gnm(images["A"], landmarks["A"])
    assert isinstance(fit_a2, Ok)
    d_self = coeff_distance(fits["A"].coeffs, fit_a2.value.coeffs)
    d_same = coeff_distance(fits["A"].coeffs, fits["B"].coeffs)
    d_diff = coeff_distance(fits["A"].coeffs, fits["C"].coeffs)
    c4_ok = d_self == 0.0 and d_self < d_same < d_diff
    check(
        "4-margen",
        c4_ok,
        f"d(A,A)={d_self:.6f} d(A,Bmisma)={d_same:.6f} "
        f"d(A,Cdistinta)={d_diff:.6f} "
        f"(exige 0==d_self<d_same<d_diff estricto)",
    )

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"GATE: FAIL (rojo) checks fallidos: {', '.join(failed)}")
        print(
            "CAUSA: dobles sha256 detectados (geometria sin 17821: verts 4225 "
            "o build caido por template, albedo ruido no-suave, corr~0, "
            "margen sin orden)"
        )
        return 1
    print("GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
