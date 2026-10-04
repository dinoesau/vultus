"""Ensamblaje FLAME: malla 5023 + PBR 2 materiales + GLB.

Wave 4 Step 5: piel total sin gris, ojos [3931:5023) con material propio,
PBR real (sin truco emisivo), zip-6 por la unica seam `build_result_zip`
en `backend/gnm.py`. Sin torch, sin FastAPI, sin logging.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import struct

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from backend.domain import (
    UV_HEIGHT,
    UV_WIDTH,
    CompleteUv,
    DomainError,
    Err,
    EyeTexture,
    FitResult,
    GnmMesh,
    ImageBytes,
    MlDecode,
    MlFailed,
    Ok,
    RenderedImage,
    parse_gnm_mesh,
)

# --- Constantes FLAME nombradas (no inventar valores fuera de aqui) ---
# FLAME topologia canonica: 5023 verts; ojos = ultimos 1092 (contiguos).
VERT_COUNT = 5023
EYE_VERT_START = 3931
EYE_VERT_END = 5023
EYE_COUNT = EYE_VERT_END - EYE_VERT_START
SKIN_VERT_COUNT = EYE_VERT_START
FLAME_TRI_COUNT = 9976
FLAME_SKIN_TRIS = 8000
FLAME_EYE_TRIS = FLAME_TRI_COUNT - FLAME_SKIN_TRIS
FLAME_TEMPLATE_NAME = "flame_template.bin"

_flame_cache: (
    tuple[
        list[tuple[float, float, float]],
        list[tuple[float, float]],
        list[tuple[int, int, int]],
    ]
    | None
) = None


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def flame_assets_dir() -> str:
    """Assets FLAME desde env. Vacio = ausente = fixture sintetico."""
    return _env("FLAME_ASSETS_DIR")


def _candidate_flame_paths() -> list[str]:
    cands: list[str] = []
    direct = flame_assets_dir()
    if direct:
        cands.append(os.path.join(direct, FLAME_TEMPLATE_NAME))
    weights_dir = _env("WEIGHTS_DIR") or _env("WEIGHTS_ROOT") or "/weights"
    if weights_dir:
        cands.append(os.path.join(weights_dir, "flame", FLAME_TEMPLATE_NAME))
    here = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.join(here, "assets", FLAME_TEMPLATE_NAME))
    seen: list[str] = []
    for c in cands:
        if c and c not in seen:
            seen.append(c)
    return seen


# Embedding baricentrico oficial MediaPipe->FLAME (105 landmarks FLAME,
# subconjunto dlib-68 util horneado offline a 50 filas; ver asset).
FLAME68_NAME = "flame68_embed.npz"

_embed_cache: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]] | None = None


def _candidate_embed_paths() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, "assets", FLAME68_NAME)]


def load_flame68_embed() -> (
    Ok[tuple[list[int], list[tuple[int, int, int]], list[tuple[float, float, float]]]] | Err[DomainError]
):
    """Embedding dlib->FLAME (50 filas: dlib_idx, tri, pesos). Solo numpy.

    Horneado offline del `mediapipe_landmark_embedding.npz` oficial sobre
    `flame_template.bin` (sha d4140b7b). Total: Err si el asset falta o es
    invalido (el caller cae a similaridad), nunca raise por expected.
    """
    global _embed_cache
    if _embed_cache is not None:
        dlib_idx, tri, w = _embed_cache
        rows: list[int] = [int(i) for i in dlib_idx]
        tris: list[tuple[int, int, int]] = [(int(t[0]), int(t[1]), int(t[2])) for t in tri]
        weights: list[tuple[float, float, float]] = [(float(r[0]), float(r[1]), float(r[2])) for r in w]
        return Ok((rows, tris, weights))
    for cand in _candidate_embed_paths():
        if os.path.isfile(cand):
            try:
                z = np.load(cand)
                dlib_idx = np.asarray(z["dlib_idx"], dtype=np.int64)
                tri = np.asarray(z["tri"], dtype=np.int64)
                w = np.asarray(z["w"], dtype=np.float64)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if dlib_idx.shape != (50,) or tri.shape != (50, 3) or w.shape != (50, 3):
                continue
            if tri.min() < 0 or tri.max() >= VERT_COUNT:
                continue
            if not bool(np.isfinite(w).all()) or not bool(np.allclose(w.sum(axis=1), 1.0)):
                continue
            _embed_cache = (dlib_idx, tri, w)
            rows = [int(i) for i in dlib_idx]
            tris = [(int(t[0]), int(t[1]), int(t[2])) for t in tri]
            weights = [(float(r[0]), float(r[1]), float(r[2])) for r in w]
            typed: tuple[list[int], list[tuple[int, int, int]], list[tuple[float, float, float]]] = (rows, tris, weights)
            return Ok(typed)
    return Err(MlFailed(detail=MlDecode(details="flame68 embed asset missing")))


def flame68_positions(
    positions: list[tuple[float, float, float]],
) -> Ok[tuple[list[int], NDArray[np.float64]]] | Err[DomainError]:
    """Posiciones 3D de los 50 landmarks dlib sobre la malla dada.

    Interpola baricentrica sobre la malla desplazada (personalizada) o el
    template. Retorna (dlib_rows, xyz 50x3). Total: Err si el asset o los
    indices fallan.
    """
    loaded = load_flame68_embed()
    if isinstance(loaded, Err):
        return loaded
    dlib_idx, tri, w = loaded.value
    try:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape != (VERT_COUNT, 3):
            return Err(MlFailed(detail=MlDecode(details="flame68 mesh verts != 5023")))
        out = np.stack(
            [w[k][0] * pos[tri[k][0]] + w[k][1] * pos[tri[k][1]] + w[k][2] * pos[tri[k][2]] for k in range(50)]
        )
    except (IndexError, ValueError, TypeError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"flame68 interp failed: {exc}")))
    return Ok((dlib_idx, np.asarray(out, dtype=np.float64)))


def is_flame_synthetic() -> bool:
    """Flag separado de sha: True si no hay archivo (fixture sintetico, waiver).

    Waiver fixture preservado: la falta local no es Err, es sintetico
    determinista. El sha (`flame_template_sha`) sigue estable en ambos casos;
    este flag dice cual es cual sin mezclar identidad con contenido.
    """
    for cand in _candidate_flame_paths():
        if os.path.isfile(cand):
            return False
    return True


def _pad4_len(n: int) -> int:
    """Longitud alineada a 4 (glTF exige byteOffset%4==0 por bufferView)."""
    return (n + 3) // 4 * 4


def _pad_bytes(buf: bytes) -> bytes:
    """Rellena con ceros hasta multiplo de 4. No-op si ya alineado."""
    target = _pad4_len(len(buf))
    pad = target - len(buf)
    if pad == 0:
        return buf
    return buf + b"\x00" * pad


# Rejilla coherente del fixture: piel lat-long para vecindad por indice.
_SKIN_COLS = 64
_SKIN_ROWS = 62
_EYE_COLS = 26
_EYE_ROWS = 21
_EYE_PER_BALL = _EYE_COLS * _EYE_ROWS


def _synthetic_flame_template() -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float]],
    list[tuple[int, int, int]],
]:
    """Fixture cabeza coherente: elipsoide piel + 2 esferas ojos, tris locales.

    Sustituye la rejilla plana degenerada (area 0, lineas en visor).
    Piel 3931 en lat-long 64x62 recortada, ojos 2x546 en 26x21.
    UVs layout real: piel 0-1, ojos 2-3. Conteos 8000/1976 exactos.
    Determinista, sin I/O, solo para paridad local sin pesos.
    """
    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    for idx in range(SKIN_VERT_COUNT):
        col = idx % _SKIN_COLS
        row = idx // _SKIN_COLS
        theta = 2.0 * math.pi * float(col) / float(_SKIN_COLS)
        phi = math.pi * (float(row) + 0.5) / float(_SKIN_ROWS)
        sin_phi = math.sin(phi)
        x = 0.30 * sin_phi * math.cos(theta)
        y = 0.42 * math.cos(phi) + 0.02
        z = 0.36 * sin_phi * math.sin(theta) + 0.02
        positions.append((x, y, z))
        uvs.append((float(col) / 63.0, float(row) / 61.0))
    eye_centers = ((-0.115, 0.06, 0.30), (0.115, 0.06, 0.30))
    for ball in range(2):
        cx, cy, cz = eye_centers[ball]
        for k in range(_EYE_PER_BALL):
            col = k % _EYE_COLS
            row = k // _EYE_COLS
            theta = 2.0 * math.pi * float(col) / float(_EYE_COLS)
            phi = math.pi * (float(row) + 0.5) / float(_EYE_ROWS)
            sin_phi = math.sin(phi)
            x = cx + 0.055 * sin_phi * math.cos(theta)
            y = cy + 0.055 * math.cos(phi)
            z = cz + 0.055 * sin_phi * math.sin(theta)
            positions.append((x, y, z))
            uvs.append((2.0 + float(col) / 25.0, float(row) / 20.0))
    indices: list[tuple[int, int, int]] = []
    quads: list[tuple[int, int, int, int]] = []
    for r in range(_SKIN_ROWS - 1):
        for c in range(_SKIN_COLS):
            c2 = (c + 1) % _SKIN_COLS
            v0 = r * _SKIN_COLS + c
            v1 = r * _SKIN_COLS + c2
            v2 = (r + 1) * _SKIN_COLS + c
            v3 = (r + 1) * _SKIN_COLS + c2
            if v0 < SKIN_VERT_COUNT and v1 < SKIN_VERT_COUNT and v2 < SKIN_VERT_COUNT and v3 < SKIN_VERT_COUNT:
                quads.append((v0, v1, v2, v3))
    for q in quads:
        indices.append((q[0], q[1], q[2]))
        if len(indices) >= FLAME_SKIN_TRIS:
            break
        indices.append((q[1], q[3], q[2]))
        if len(indices) >= FLAME_SKIN_TRIS:
            break
    qi = 0
    while len(indices) < FLAME_SKIN_TRIS:
        q = quads[qi % len(quads)]
        indices.append((q[0], q[1], q[2]) if len(indices) % 2 == 0 else (q[1], q[3], q[2]))
        qi += 1
    for ball in range(2):
        base = EYE_VERT_START + ball * _EYE_PER_BALL
        stworzone = 0
        for r in range(_EYE_ROWS - 1):
            for c in range(_EYE_COLS):
                if stworzone >= FLAME_EYE_TRIS // 2:
                    break
                c2 = (c + 1) % _EYE_COLS
                v0 = base + r * _EYE_COLS + c
                v1 = base + r * _EYE_COLS + c2
                v2 = base + (r + 1) * _EYE_COLS + c
                v3 = base + (r + 1) * _EYE_COLS + c2
                indices.append((v0, v1, v2))
                stworzone += 1
                if stworzone >= FLAME_EYE_TRIS // 2:
                    break
                indices.append((v1, v3, v2))
                stworzone += 1
            if stworzone >= FLAME_EYE_TRIS // 2:
                break
    return positions, uvs, indices


def _upright_uv(u: float, v: float) -> tuple[float, float]:
    """PNG crudo upright: chin abajo, brow arriba (rango piel 0-1).

    El bin trae v=0 en chin (convención OBJ): sin flip la cara sale
    invertida en el PNG. Solo rama archivo; el fixture sintético ya trae
    v=0 arriba. Ojos (2-3, textura aparte) intactos. GLB y raster usan las
    mismas UVs, asi que malla y atlas siguen consistentes; el flipY del
    visor cancela igual que antes.
    """
    if v <= 1.0:
        return (u, 1.0 - v)
    return (u, v)


def load_flame_template() -> Ok[
    tuple[
        list[tuple[float, float, float]],
        list[tuple[float, float]],
        list[tuple[int, int, int]],
    ]
] | Err[DomainError]:
    """Template sucesor de `gnm_template.bin`: archivo si existe, fixture si no.

    Pre-check solo lectura: `modal volume ls` + pesos locales antes de pedir
    el sha; waiver fixture registrado si Modal inalcanzable (el sintetico es
    determinista y el sha queda registrado en `flame_template_sha`).
    Valida el bin como `gnm.py`: NaN/inf e indices>=verts son Err(MlFailed),
    nunca aceptacion silenciosa ni raise por expected. La falta local es
    waiver sintetico salvo con VULTUS_REAL_ML=1 que falla loud;
    ver `is_flame_synthetic` para distinguirlas.
    """
    global _flame_cache
    if _flame_cache is not None:
        return Ok(_flame_cache)
    for cand in _candidate_flame_paths():
        if os.path.isfile(cand):
            with open(cand, "rb") as f:
                data = f.read()
            if len(data) >= 8:
                verts, tris = struct.unpack_from("<II", data, 0)
                if verts == VERT_COUNT:
                    expect = 8 + verts * 12 + verts * 8 + tris * 12
                    if len(data) == expect:
                        off = 8
                        positions: list[tuple[float, float, float]] = []
                        for _ in range(verts):
                            x, y, z = struct.unpack_from("<3f", data, off)
                            off += 12
                            if not (math.isfinite(float(x)) and math.isfinite(float(y)) and math.isfinite(float(z))):
                                return Err(MlFailed(detail=MlDecode(details="flame template posicion no finita")))
                            positions.append((float(x), float(y), float(z)))
                        uvs: list[tuple[float, float]] = []
                        for _ in range(verts):
                            u, v = struct.unpack_from("<2f", data, off)
                            off += 8
                            if not (math.isfinite(float(u)) and math.isfinite(float(v))):
                                return Err(MlFailed(detail=MlDecode(details="flame template uv no finita")))
                            uvs.append(_upright_uv(float(u), float(v)))
                        indices: list[tuple[int, int, int]] = []
                        for _ in range(tris):
                            a, b, c = struct.unpack_from("<III", data, off)
                            off += 12
                            if int(a) >= verts or int(b) >= verts or int(c) >= verts:
                                return Err(MlFailed(detail=MlDecode(details="flame template indice fuera de rango")))
                            indices.append((int(a), int(b), int(c)))
                        _flame_cache = (positions, uvs, indices)
                        return Ok(_flame_cache)
    if os.environ.get("VULTUS_REAL_ML") == "1":
        return Err(MlFailed(detail=MlDecode(details="real flame template required but flame_template.bin missing")))
    _flame_cache = _synthetic_flame_template()
    return Ok(_flame_cache)


def flame_template_sha() -> str:
    """SHA-256 del template sucesor: archivo si existe, fixture si no."""
    for cand in _candidate_flame_paths():
        if os.path.isfile(cand):
            with open(cand, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
    positions, uvs, indices = _synthetic_flame_template()
    h = hashlib.sha256()
    h.update(struct.pack("<II", VERT_COUNT, len(indices)))
    for x, y, z in positions:
        h.update(struct.pack("<3f", x, y, z))
    for u, v in uvs:
        h.update(struct.pack("<2f", u, v))
    for a, b, c in indices:
        h.update(struct.pack("<III", a, b, c))
    return h.hexdigest()


def is_eye_vertex(idx: int) -> bool:
    """Ojo = [EYE_VERT_START:EYE_VERT_END). Total sobre int."""
    return EYE_VERT_START <= idx < EYE_VERT_END


def eye_vertex_indices() -> list[int]:
    return list(range(EYE_VERT_START, EYE_VERT_END))


def skin_vertex_indices() -> list[int]:
    return list(range(EYE_VERT_START))


# Desplazamiento real (Fase 2): la moneda fit 253 porta identidad HiFi3D++
# (id200/exp45) y el delta se reconstruye con la base + transfer IDW en
# unidades del template. Fail-safe absoluto 3-4x sobre el maximo
# observado (Bush 0.048, Eckhart 0.035): la base PCA es suave por
# construccion, los picos locales los guarda el test de suavidad, no el
# clamp. Sin base/transfer (CI/dobles) rige el legado geometrico 5mm.
_DISPLACE_MAX = 0.15
_LEGACY_DISPLACE_MAX = 0.005


def displaced_positions(fit: FitResult) -> Ok[list[tuple[float, float, float]]] | Err[DomainError]:
    loaded = load_flame_template()
    if isinstance(loaded, Err):
        return loaded
    positions, _, _ = loaded.value
    coeffs = fit.coeffs.as_tuple()
    try:
        from backend.deep3d import real_displacement as _real_delta
    except ImportError:
        _real_delta = None  # type: ignore[assignment]
    delta = None
    if _real_delta is not None:
        try:
            delta = _real_delta(coeffs)
        except Exception:  # noqa: BLE001 - desplazamiento invalido: via legado
            delta = None
    if delta is not None and len(delta) == len(positions):
        return Ok(
            [
                (
                    x + min(max(float(dx), -_DISPLACE_MAX), _DISPLACE_MAX),
                    y + min(max(float(dy), -_DISPLACE_MAX), _DISPLACE_MAX),
                    z + min(max(float(dz), -_DISPLACE_MAX), _DISPLACE_MAX),
                )
                for (x, y, z), (dx, dy, dz) in zip(positions, [tuple(map(float, row)) for row in delta])
            ]
        )
    width = len(coeffs)
    return Ok(
        [
            (x + min(max(coeffs[idx % width] * 0.01, -_LEGACY_DISPLACE_MAX), _LEGACY_DISPLACE_MAX), y, z)
            for idx, (x, y, z) in enumerate(positions)
        ]
    )


def _eye_png() -> bytes:
    img = Image.new("RGB", (32, 32), (240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _smooth_vertex_normals(
    positions: list[tuple[float, float, float]],
    tris: list[tuple[int, int, int]],
) -> list[tuple[float, float, float]]:
    """Normales suaves promediadas por vertice, unitarias y deterministas.

    Promedia las normales de caras adyacentes y normaliza. Vertices
    degenerados (norma ~0 o no finita) caen a (0,0,1) determinista para
    mantener el accessor finito y unitario. Sin NORMAL el visor usa
    sombreado plano facetado; con esta se renderiza cabeza suave.
    """
    pos = np.asarray(positions, dtype=np.float64)
    acc = np.zeros_like(pos)
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a < 0 or b < 0 or c < 0 or a >= len(positions) or b >= len(positions) or c >= len(positions):
            continue
        edge1 = pos[b] - pos[a]
        edge2 = pos[c] - pos[a]
        n = np.cross(edge1, edge2)
        if not bool(np.isfinite(n).all()):
            continue
        acc[a] += n
        acc[b] += n
        acc[c] += n
    out: list[tuple[float, float, float]] = []
    for i in range(len(positions)):
        nx, ny, nz = float(acc[i][0]), float(acc[i][1]), float(acc[i][2])
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if not (math.isfinite(length)) or length < 1e-12:
            out.append((0.0, 0.0, 1.0))
        else:
            out.append((nx / length, ny / length, nz / length))
    return out


def build_personalized_glb(
    fit: FitResult,
    albedo: CompleteUv | RenderedImage,
    atlas_png: ImageBytes | None = None,
    eye_texture: EyeTexture | None = None,
) -> Ok[GnmMesh] | Err[DomainError]:
    """GLB FLAME con 2 primitivas: piel (<3931) y ojo ([3931:5023)).

    PBR real: baseColor blanca + baseColorTexture, sin emisivo. La piel usa
    el albedo (CompleteUv legacy o RenderedImage completion; ambos UV_LEN por
    as_bytes) o `atlas_png` si se da; el ojo usa `eye_texture` real cuando
    se provee (bake_eye_texture desde eye_ball_tex.png) o blanca
    determinista como fallback local. Las UVs van en convencion glTF
    (`v = 1 - v_uv`). Tris a caballo son Err explicito, nunca drop
    silencioso: dropped==len(all)-len(skin)-len(eye) debe ser 0.
    Normales suaves promediadas por vertice en ambas primitivas
    (atributo NORMAL, unitarias y finitas): sin ellas el visor sombrea
    facetado plano. Cada seccion del BIN va padded a 4 antes de offsets
    (pos/uv/normal/skin/eye/pngs).
    """
    try:
        displaced = displaced_positions(fit)
        if isinstance(displaced, Err):
            return displaced
        positions = displaced.value
        if len(positions) != VERT_COUNT:
            return Err(MlFailed(detail=MlDecode(details="glb failed: verts != 5023")))
        loaded = load_flame_template()
        if isinstance(loaded, Err):
            return loaded
        _, template_uvs, template_tris = loaded.value
        split_uvs = [(float(u), 1.0 - float(v)) for u, v in template_uvs]
        skin_tris = [t for t in template_tris if t[0] < EYE_VERT_START and t[1] < EYE_VERT_START and t[2] < EYE_VERT_START]
        eye_tris = [t for t in template_tris if t[0] >= EYE_VERT_START and t[1] >= EYE_VERT_START and t[2] >= EYE_VERT_START]
        # Tris a caballo -> Err explicito, nunca drop silencioso.
        # dropped==len(all)-len(skin)-len(eye) debe ser 0.
        dropped = len(template_tris) - len(skin_tris) - len(eye_tris)
        if dropped != 0:
            return Err(MlFailed(detail=MlDecode(details=f"glb failed: {dropped} tris a caballo piel/ojo")))
        if not skin_tris or not eye_tris:
            return Err(MlFailed(detail=MlDecode(details="glb failed: sin tris piel/ojo")))
        if atlas_png is not None:
            skin_png = atlas_png.as_bytes()
        else:
            img = Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(albedo.as_bytes()))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            skin_png = buf.getvalue()
        if eye_texture is not None:
            eye_img = Image.frombytes("RGB", (UV_WIDTH, UV_HEIGHT), bytes(eye_texture.as_bytes()))
            eye_buf_io = io.BytesIO()
            eye_img.save(eye_buf_io, format="PNG")
            eye_png = eye_buf_io.getvalue()
        else:
            eye_png = _eye_png()
        pos_buf = struct.pack(f"<{VERT_COUNT * 3}f", *[c for p in positions for c in p])
        uv_buf = struct.pack(f"<{VERT_COUNT * 2}f", *[c for t in split_uvs for c in t])
        normals = _smooth_vertex_normals(positions, template_tris)
        nrm_buf = struct.pack(f"<{VERT_COUNT * 3}f", *[c for n in normals for c in n])
        skin_flat = [v for tri in skin_tris for v in tri]
        eye_flat = [v for tri in eye_tris for v in tri]
        if not skin_flat or not eye_flat:
            return Err(MlFailed(detail=MlDecode(details="glb failed: indices vacios")))
        if min(skin_flat) < 0 or max(skin_flat) >= EYE_VERT_START:
            return Err(MlFailed(detail=MlDecode(details="glb failed: piel fuera de rango")))
        if min(eye_flat) < EYE_VERT_START or max(eye_flat) >= EYE_VERT_END:
            return Err(MlFailed(detail=MlDecode(details="glb failed: ojo fuera de rango")))
        use_u32 = VERT_COUNT > 65535
        skin_fmt = f"<{len(skin_flat)}I" if use_u32 else f"<{len(skin_flat)}H"
        eye_fmt = f"<{len(eye_flat)}I" if use_u32 else f"<{len(eye_flat)}H"
        idx_comp = 5125 if use_u32 else 5123
        skin_buf = struct.pack(skin_fmt, *skin_flat)
        eye_buf = struct.pack(eye_fmt, *eye_flat)
        # Pad a 4 tras cada seccion antes de offsets (glTF byteOffset%4==0).
        pos_len, uv_len, nrm_len = len(pos_buf), len(uv_buf), len(nrm_buf)
        skin_len, eye_len = len(skin_buf), len(eye_buf)
        skin_png_len, eye_png_len = len(skin_png), len(eye_png)
        pos_pad = _pad_bytes(pos_buf)
        uv_pad = _pad_bytes(uv_buf)
        nrm_pad = _pad_bytes(nrm_buf)
        skin_pad = _pad_bytes(skin_buf)
        eye_pad = _pad_bytes(eye_buf)
        skin_png_pad = _pad_bytes(skin_png)
        eye_png_pad = _pad_bytes(eye_png)
        uv_off = len(pos_pad)
        nrm_off = uv_off + len(uv_pad)
        skin_off = nrm_off + len(nrm_pad)
        eye_off = skin_off + len(skin_pad)
        skin_png_off = eye_off + len(eye_pad)
        eye_png_off = skin_png_off + len(skin_png_pad)
        bin_buf = pos_pad + uv_pad + nrm_pad + skin_pad + eye_pad + skin_png_pad + eye_png_pad
        while len(bin_buf) % 4 != 0:
            bin_buf += b"\x00"
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        skin_count = len(skin_flat)
        eye_count = len(eye_flat)
        json_str = (
            '{"asset":{"version":"2.0","generator":"vultus-flame-fit"},"scene":0,'
            '"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"VultusFaceFlame"}],'
            '"meshes":[{"name":"FaceFlame","primitives":['
            '{"attributes":{"POSITION":0,"NORMAL":2,"TEXCOORD_0":1},"indices":3,"material":0},'
            '{"attributes":{"POSITION":0,"NORMAL":2,"TEXCOORD_0":1},"indices":4,"material":1}'
            "]}],"
            '"materials":['
            '{"name":"SkinPBR","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":0.7,"baseColorTexture":{"index":0}}},'
            '{"name":"EyePBR","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":0.3,"baseColorTexture":{"index":1}}}'
            "],"
            '"textures":[{"source":0,"sampler":0},{"source":1,"sampler":0}],"samplers":[{"magFilter":9729,"minFilter":9729}],'
            '"images":[{"bufferView":5,"mimeType":"image/png"},{"bufferView":6,"mimeType":"image/png"}],'
            f'"buffers":[{{"byteLength":{len(bin_buf)}}}],'
            '"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%d,"target":34962},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d,"target":34962},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d,"target":34962},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d,"target":34963},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d,"target":34963},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d},'
            '{"buffer":0,"byteOffset":%d,"byteLength":%d}]'
            % (pos_len, uv_off, uv_len, nrm_off, nrm_len, skin_off, skin_len, eye_off, eye_len, skin_png_off, skin_png_len, eye_png_off, eye_png_len)
            + ',"accessors":[{"bufferView":0,"componentType":5126,"count":'
            + f"{VERT_COUNT}"
            + ',"type":"VEC3",'
            + f'"max":[{max(xs)},{max(ys)},{max(zs)}],"min":[{min(xs)},{min(ys)},{min(zs)}]'
            + "},"
            + '{"bufferView":1,"componentType":5126,"count":'
            + f"{VERT_COUNT}"
            + ',"type":"VEC2"},'
            + '{"bufferView":2,"componentType":5126,"count":'
            + f"{VERT_COUNT}"
            + ',"type":"VEC3"},'
            + '{"bufferView":3,"componentType":'
            + f"{idx_comp}"
            + ',"count":'
            + f"{skin_count}"
            + ',"type":"SCALAR"},'
            + '{"bufferView":4,"componentType":'
            + f"{idx_comp}"
            + ',"count":'
            + f"{eye_count}"
            + ',"type":"SCALAR"}],'
            + '"extras":{"personalized":true,"flame":true,"verts":'
            + f"{VERT_COUNT}"
            + ',"eye_start":'
            + f"{EYE_VERT_START}"
            + ',"eye_end":'
            + f"{EYE_VERT_END}"
            + "}}"
        )
        json_bytes = json_str.encode("utf-8")
        while len(json_bytes) % 4 != 0:
            json_bytes += b" "
        total = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
        out = b"glTF" + struct.pack("<I", 2) + struct.pack("<I", total)
        out += struct.pack("<I", len(json_bytes)) + b"JSON" + json_bytes
        out += struct.pack("<I", len(bin_buf)) + b"BIN\x00" + bin_buf
        parsed = parse_gnm_mesh(out)
        if isinstance(parsed, Err):
            return Err(MlFailed(detail=MlDecode(details="glb self-check failed")))
        return parsed
    except Exception as exc:  # noqa: BLE001
        return Err(MlFailed(detail=MlDecode(details=f"glb failed: {exc}")))
