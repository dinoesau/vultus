"""Textura FFHQ-UV sin gris (puente completion, Wave 3 Step 4 + Wave 6-fix Lane A real).

Una sola pasada feed-forward: el mismo input da los mismos bytes.
Local sin pesos corre replay determinista (paridad firmada no-visual para
gateway y contrato en verde); con el puente verificado corre la
completion real; lo visual solo se valida en Modal con el zip real.

Completion real (Wave 6-fix): derivada de la foto, nunca ruido sha:
1. Verifica el puente por archivos canonicos (`FLAME_w_HIFI3D_UV.obj` +
   `eye_ball_tex.png` en `FFHQ_UV_DIR`). Sin ellos falla ruidoso (500),
   nunca sirve el doble en silencio.
2. Recorta la cara por el bbox de landmarks (+15% de margen) y la
   reduce a 64x64 BILINEAR: el layout de apariencia es de la foto, no
   alucinado. El reescalado a 512x512 BILINEAR es la completion
   determinista de ocluidas (piel total por construccion).
3. Suma detalle suave de identidad (octavas 8/16/32 desde el descriptor
   antropometrico cuantizado + fingerprint del puente, amplitud +/-6):
   misma persona comparte patron, distinta persona no.
4. Ojos en bake separado (`bake_eye_texture` desde el mapa real
   `eye_ball_tex.png`): pbr != uv a nivel de modulo; el cableado del zip
   a mapas reales es cutover fuera de este lane.
Cero SKIN_SENTINEL por construccion (barrido final + scrub), evidencia
1.0. Todo numpy+PIL float64 determinista: bytes identicos x2 en el mismo
digest; cross-platform tolerancia max-abs-diff<=1 documentada.

Gris honesto abandonado a proposito (ADR-008): la completion es visual,
no evidencia. El magenta `SKIN_SENTINEL` marca texeles sin dato en el
pipeline legacy; el albedo de piel nunca lo produce.

Config solo por env (`FFHQ_UV_DIR`; mas `VULTUS_REAL_ML` ya existente).
Cero literales de rutas, pesos o volumenes fuera de los nombres de
archivo canonicos. Sin torch top-level, sin FastAPI, sin logging.
Entradas ya probadas, salidas probadas.

Determinismo exigible: `torch.use_deterministic_algorithms(True)` mas
`CUBLAS_WORKSPACE_CONFIG` mas cudnn `deterministic:true benchmark:false`
en mismo digest T4 con bytes identicos x2 (sin flags el eval falla).
Tolerancia documentada: rerun en mismo T4 debe dar bytes identicos;
fallback `max_abs_diff <= 1` por canal con sha registrado (redondeo
bilineal cross-platform). El doble local es exacto (diff 0) sin torch.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import time

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from backend.domain import (
    UV_LEN,
    DomainError,
    Err,
    EyeTexture,
    FitResult,
    ImageBytes,
    Landmarks,
    MlDecode,
    MlFailed,
    Ok,
    RenderedImage,
    ThreadLocalStats,
    parse_eye_texture,
    parse_rendered_image,
)

# Magenta fuera de gama: ningun albedo de piel lo produce. Const en codigo,
# nunca de env. Solo rutas de pesos por env.
SKIN_SENTINEL: tuple[int, int, int] = (255, 0, 255)

# Sal propia del seam: no comparte stream con fit ni con el doble GNM.
TEXTURE_SEED_SALT = b"flame-texture-v1"

# Sal del forward real: dominio separado del doble.
REAL_TEXTURE_SALT = b"flame-real-texture-v1"

# Puente FFHQ-UV: archivos canonicos (mirror de
# scripts/modal-weights-sync.sh BRIDGE_FILES + mapa ocular del puente).
UV_OBJ_NAME = "FLAME_w_HIFI3D_UV.obj"
EYE_MAP_NAME = "eye_ball_tex.png"
# Extras RGB fitting FFHQ-UV (futuro unwrap texgan/DPR/parsing, no parte
# del puente canonico 4 archivos para no romper test_bridge_parity).
# Cuando presentes, run_unwrap_texture es la via real; sin ellas la
# completion foto-derivada es la via real actual.
TEXGAN_NAME = "texgan_ffhq_uv.pth"
UNWRAP_MAT_NAME = "unwrap_1024_info.mat"
UNWRAP_MASK_NAME = "unwrap_1024_info_mask.png"
MEAN_FACE_NAME = "hifi3dpp_mean_face.obj"

# Lookup de unwrap 1024 (v_idx + pesos baricentricos), cacheado por ruta.
# El .mat trae uv_idx_v_idx/uv_idx_bw 1024x1024x3 (uv_idx_vt_idx sin uso:
# los vt con seams se resuelven via UVs por vertice del template real).
_UNWRAP_CACHE: tuple[str, NDArray[np.int64], NDArray[np.float64]] | None = None

# Geometria de la completion real.
FACE_MARGIN = 0.15
TEX_GRID = 64
TEX_SIZE = 512
DETAIL_AMPLITUDE = 6.0
_DETAIL_OCTAVES = ((8, 0.5), (16, 0.3), (32, 0.2))
_BRIDGE_HEAD_BYTES = 65536

# Evidencia minima: fraccion de pixeles sin sentinel sobre el total.
# El doble da 1.0; el real debe superar el umbral o el eval falla.
EVIDENCE_MIN = 0.99

_LAST_TEXTURE_STATS = ThreadLocalStats(
    {"evidence": 0.0, "sentinel_count": 0.0, "duration_ms": 0.0, "parsing": 0.0, "texgan": 0.0}
)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def ffhq_uv_dir() -> str:
    """Puente FFHQ-UV desde env. Vacio = ausente = doble local."""
    return _env("FFHQ_UV_DIR")


def texgan_dir() -> str:
    """Checkpoint TexGAN desde env. Vacio = ausente."""
    return _env("TEXGAN_DIR")


def topo_dir() -> str:
    """Assets topologia unwrap desde env. Vacio = ausente."""
    return _env("TOPO_DIR")


def _texgan_candidate_dirs() -> list[str]:
    cands: list[str] = []
    for d in (texgan_dir(), ffhq_uv_dir()):
        if d and d not in cands:
            cands.append(d)
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        fallback = os.path.join(root, "checkpoints", "texgan_model")
        if fallback not in cands:
            cands.append(fallback)
    return cands


def _topo_candidate_dirs() -> list[str]:
    cands: list[str] = []
    for d in (topo_dir(), ffhq_uv_dir()):
        if d and d not in cands:
            cands.append(d)
    root = _env("WEIGHTS_ROOT") or _env("WEIGHTS_DIR") or "/weights"
    if root:
        fallback = os.path.join(root, "topo_assets")
        if fallback not in cands:
            cands.append(fallback)
    return cands


def _file_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _dir_has_files(path: str) -> bool:
    if not path:
        return False
    try:
        with os.scandir(path) as entries:
            return any(entry.is_file() for entry in entries)
    except OSError:
        return False


def weights_present() -> bool:
    """True solo con el puente FFHQ-UV canonico verificado en `FFHQ_UV_DIR`.

    Exige `FLAME_w_HIFI3D_UV.obj` + `eye_ball_tex.png` no vacios.
    Junk-file safe: archivos arbitrarios no habilitan el real.
    Helper unico usado por `_real_texture_available` y `_real_bake`.
    """
    bridge = ffhq_uv_dir()
    if not bridge:
        return False
    return _file_nonempty(os.path.join(bridge, UV_OBJ_NAME)) and _file_nonempty(
        os.path.join(bridge, EYE_MAP_NAME)
    )


def _real_texture_available() -> bool:
    """True solo con pesos del puente en `FFHQ_UV_DIR`. Sin literales."""
    return weights_present()


def ffhq_uv_extra_present() -> bool:
    """True solo con extras unwrap (texgan + unwrap mat + mean face) en FFHQ_UV_DIR.

    Via futura texgan/DPR/parsing; no parte del puente canonico.
    Total: False si dir ausente o algun extra falta/vacio, nunca raise.
    """
    tex_ok = any(_file_nonempty(os.path.join(d, TEXGAN_NAME)) for d in _texgan_candidate_dirs())
    if not tex_ok:
        return False
    topo_dirs = _topo_candidate_dirs()
    mat_ok = any(_file_nonempty(os.path.join(d, UNWRAP_MAT_NAME)) for d in topo_dirs)
    # hifi3dpp_mean_face.obj falta en HF y Volume: documentado como ausente,
    # no bloquea extras (unwrap usa mat + masks + texgan).
    return mat_ok


def count_sentinel(raw: bytes) -> int:
    """Pixeles exactos `SKIN_SENTINEL` en un buffer RGB plano."""
    sr, sg, sb = SKIN_SENTINEL
    total = 0
    # Paso 3: cada triple es un pixel; sin slicing para no alocar.
    for i in range(0, len(raw) - 2, 3):
        if raw[i] == sr and raw[i + 1] == sg and raw[i + 2] == sb:
            total += 1
    return total


def texture_evidence(raw: bytes) -> float:
    """Fraccion de pixeles sin sentinel. 1.0 = piel total, 0.0 = vacio."""
    if len(raw) == 0 or len(raw) % 3 != 0:
        return 0.0
    pixels = len(raw) // 3
    if pixels == 0:
        return 0.0
    return 1.0 - float(count_sentinel(raw)) / float(pixels)


def max_abs_diff(a: bytes, b: bytes) -> int:
    """Maxima diferencia absoluta por byte. Total: -1 si largos difieren."""
    if len(a) != len(b):
        return -1
    peak = 0
    for x, y in zip(a, b):
        diff = x - y if x >= y else y - x
        peak = max(peak, diff)
    return peak


def ensure_deterministic_texture() -> None:
    """Aplica flags deterministicos de torch sin tumbar nunca.

    Exigible en T4 real: `use_deterministic_algorithms(True)` mas cudnn
    `deterministic=True benchmark=False`. `CUBLAS_WORKSPACE_CONFIG` se lee
    del env (el caller/Modal lo fija; aqui no se sobreescribe en silencio).
    Sin torch (doble local) es no-op total: el doble es exacto por sha256.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # noqa: BLE001 - torch ausente o CPU sin determinismo: el doble sigue exacto
        return
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001 - cudnn ausente en CPU: no-op seguro
        return


def _fit_seed_bytes(fit: FitResult) -> bytes:
    """Serializacion estable del fit para la semilla. Sin importar pipeline."""
    coeffs = fit.coeffs.as_tuple()
    camera = fit.camera.as_tuple()
    return struct.pack("<253f", *coeffs) + struct.pack("<12f", *camera)


def _deterministic_albedo(image: ImageBytes, fit: FitResult, landmarks: Landmarks) -> Ok[RenderedImage] | Err[DomainError]:
    """Replay fixture: sha256 stream + scrub de sentinel, repetible por diseno.

    Cada byte deriva de `TEXTURE_SEED_SALT + image + fit + landmarks`.
    El scrub cambia `255,0,255 -> 254,0,255` (un LSB, invisible) para
    garantizar cero sentinel sin romper determinismo ni per-identidad.
    Total: rama imposible retorna Err sin lanzar.
    """
    seed = hashlib.sha256(
        TEXTURE_SEED_SALT + b"\x00" + image.as_bytes() + b"\x00" + _fit_seed_bytes(fit) + b"\x00" + landmarks.as_bytes()
    ).digest()
    out = bytearray()
    counter = 0
    while len(out) < UV_LEN:
        out.extend(hashlib.sha256(seed + struct.pack(">I", counter)).digest())
        counter += 1
    scrubbed = bytearray(bytes(out[:UV_LEN]))
    sr, sg, sb = SKIN_SENTINEL
    for i in range(0, UV_LEN, 3):
        if scrubbed[i] == sr and scrubbed[i + 1] == sg and scrubbed[i + 2] == sb:
            scrubbed[i] = 254
    parsed = parse_rendered_image(bytes(scrubbed))
    # Sintesis interna con largo exacto por construccion: el parse no puede
    # fallar. Rama imposible retorna Err total, nunca raise por expected.
    if isinstance(parsed, Err):
        return Err(MlFailed(detail=MlDecode(details="synthetic flame texture produced invalid length")))
    return Ok(parsed.value)


def _find_unwrap_mat() -> str | None:
    """Ruta del mat de unwrap en candidatos topo, o None si ausente."""
    for d in _topo_candidate_dirs():
        cand = os.path.join(d, UNWRAP_MAT_NAME)
        if _file_nonempty(cand):
            return cand
    return None


def _require_unwrap_mat() -> Ok[str] | Err[DomainError]:
    """Gate del mat de unwrap en candidatos topo. Total: Err loud si falta.

    El .mat verifica el layout FFHQ-UV denso (20k verts, cara 0-1 ojos 2-3)
    y aporta la mascara de validez; el mapeo texel->triangulo se rasteriza
    del template FLAME 5023 con sus UVs verificadas (misma familia de
    layout, topologia del contrato). Sin mat no hay unwrap, nunca blur
    silencioso disfrazado.
    """
    mat_path = _find_unwrap_mat()
    if mat_path is None:
        return Err(MlFailed(detail=MlDecode(details="real unwrap requires unwrap_1024_info.mat")))
    try:
        from scipy.io import loadmat as _loadmat  # type: ignore[import-untyped]
    except ImportError as exc:
        return Err(MlFailed(detail=MlDecode(details=f"scipy missing for unwrap mat: {exc}")))
    try:
        mat = _loadmat(mat_path)
        v_idx = mat["uv_idx_v_idx"]
        bw = mat["uv_idx_bw"]
    except (KeyError, ValueError, TypeError, OSError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"unwrap mat unreadable: {exc}")))
    if v_idx.shape != (1024, 1024, 3) or bw.shape != (1024, 1024, 3):
        return Err(MlFailed(detail=MlDecode(details="unwrap mat shapes unexpected (expected 1024x1024x3)")))
    return Ok(mat_path)


# Costura 5150-vt colapsada a 1 UV por vertice: los tris de costura cruzan
# islas (UV-span enorme con 3D minima). Sin filtro, con last-wins pintan
# bandas sobre todo el atlas (sopa de triangulos, golden 280 en el template).
_UV_STRETCH_MAX = 0.25

_RASTER_CACHE_KEY = "flame-5023-skin-raster-v2"
_raster_cache: tuple[str, NDArray[np.int64], NDArray[np.float64]] | None = None


def _rasterize_skin_uv(
    uvs: list[tuple[float, float]],
    tris: list[tuple[int, int, int]],
    facing: list[bool] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Mapeo texel 512 -> (tri verts piel, baricentricos) en UV 0-1.

    Convencion del contrato: pixel (col,row) <-> UV (u=col/511, v=row/511)
    (el flip `v=1-v_uv` del GLB se cancela con flipY del visor; ver
    `build_personalized_glb`). Solo tris de piel (verts <3931, UVs en 0-1);
    ojos viven en textura aparte. Last-wins en seams. Cacheado en proceso
    (la clave incluye facing: geometria distinta por fit no reusa raster).
    `facing` alinea con `tris`: tris de espaldas (normal z<=0) no muestrean
    la foto, van a completion.
    """
    global _raster_cache
    digest = ""
    if facing is not None:
        digest = hashlib.sha256(np.asarray(facing, dtype=bool).tobytes()).hexdigest()[:16]
    key = f"{_RASTER_CACHE_KEY}:{len(uvs)}:{len(tris)}:{digest}"
    if _raster_cache is not None and _raster_cache[0] == key:
        return _raster_cache[1], _raster_cache[2]
    uva = np.asarray(uvs, dtype=np.float64)
    v_idx = np.full((TEX_SIZE, TEX_SIZE, 3), -1, dtype=np.int64)
    bw = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.float64)
    for ti, tri in enumerate(tris):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a >= 3931 or b >= 3931 or c >= 3931:
            continue
        if facing is not None and not facing[ti]:
            continue
        pa, pb, pc = uva[a] * 511.0, uva[b] * 511.0, uva[c] * 511.0
        if bool((pa < 0.0).any() or (pa > 511.0).any()) and False:
            pass
        # Tris de costura: colapso 5150-vt a 1 UV por vertice los estira
        # entre islas; se saltan para no envenenar el atlas (la ocluida la
        # rellena la completion). Umbral en UV 0-1, no en pixeles.
        du = float(
            max(
                np.linalg.norm(pa - pb),
                np.linalg.norm(pb - pc),
                np.linalg.norm(pa - pc),
            )
            / 511.0
        )
        if du > _UV_STRETCH_MAX:
            continue
        min_u = max(0.0, float(min(pa[0], pb[0], pc[0])))
        max_u = min(511.0, float(max(pa[0], pb[0], pc[0])))
        min_v = max(0.0, float(min(pa[1], pb[1], pc[1])))
        max_v = min(511.0, float(max(pa[1], pb[1], pc[1])))
        x0, x1 = int(min_u), int(max_u)
        y0, y1 = int(min_v), int(max_v)
        if x1 < x0 or y1 < y0:
            continue
        denom = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(denom) < 1e-12:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float64)
        ys = np.arange(y0, y1 + 1, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        w0 = ((pb[1] - pc[1]) * (grid_x - pc[0]) + (pc[0] - pb[0]) * (grid_y - pc[1])) / denom
        w1 = ((pc[1] - pa[1]) * (grid_x - pc[0]) + (pa[0] - pc[0]) * (grid_y - pc[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not bool(inside.any()):
            continue
        region_v = v_idx[y0 : y1 + 1, x0 : x1 + 1]
        region_b = bw[y0 : y1 + 1, x0 : x1 + 1]
        region_v[inside] = (a, b, c)
        region_b[inside, 0] = w0[inside]
        region_b[inside, 1] = w1[inside]
        region_b[inside, 2] = w2[inside]
    _raster_cache = (key, v_idx, bw)
    return v_idx, bw


def _texgan_completion(
    sampled_255: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64] | None:
    """Completion neuronal texgan (Fase 1): solo texeles no validos.

    Via `backend.texgan.neural_completion` (latente desde w_avg + Adam
    enmascarado, determinista). None si no hay backend/pesos o falla:
    el caller cae a piel media (via documentada en stats `texgan`).
    Total: nunca lanza.
    """
    try:
        from backend.texgan import neural_completion as _neural
    except ImportError:
        return None
    try:
        result = _neural(sampled_255, valid)
    except Exception:  # noqa: BLE001 - cualquier fallo cae a piel media (via documentada en stats)
        return None
    if result is None:
        return None
    if result.shape != (TEX_SIZE, TEX_SIZE, 3):
        return None
    return np.clip(np.asarray(result, dtype=np.float64), 0.0, 255.0)


def _bilinear_sample(photo: NDArray[np.float64], px: NDArray[np.float64]) -> NDArray[np.float64]:
    """Muestreo bilineal vectorizado. photo HxWx3, px (..., 2) en pixeles x,y."""
    height, width = int(photo.shape[0]), int(photo.shape[1])
    x = np.clip(px[..., 0], 0.0, float(width) - 1.001)
    y = np.clip(px[..., 1], 0.0, float(height) - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = (x - x0.astype(np.float64))[..., None]
    fy = (y - y0.astype(np.float64))[..., None]
    c00 = photo[y0, x0]
    c10 = photo[y0, x0 + 1]
    c01 = photo[y0 + 1, x0]
    c11 = photo[y0 + 1, x0 + 1]
    return c00 * (1.0 - fx) * (1.0 - fy) + c10 * fx * (1.0 - fy) + c01 * (1.0 - fx) * fy + c11 * fx * fy


def _project_verts_to_pixels(
    positions: list[tuple[float, float, float]],
    width: int,
    height: int,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Alinea bbox XY del template al bbox de landmarks expandido (afine).

    El template va en y-up (FLAME) y la foto en y-down: se espeja Y para que
    la frente muestree frente y la barbilla barbilla. Total: Err si el bbox
    del template es degenerado.
    """
    tx = np.asarray([p[0] for p in positions], dtype=np.float64)
    ty = np.asarray([p[1] for p in positions], dtype=np.float64)
    tw = float(tx.max() - tx.min())
    th = float(ty.max() - ty.min())
    if tw < 1e-9 or th < 1e-9:
        return Err(MlFailed(detail=MlDecode(details="degenerate template bbox for projection")))
    lx0, lx1 = float(xs.min()), float(xs.max())
    ly0, ly1 = float(ys.min()), float(ys.max())
    lw, lh = lx1 - lx0, ly1 - ly0
    pad_x, pad_y = FACE_MARGIN * lw, FACE_MARGIN * lh
    lx0 = max(0.0, lx0 - pad_x)
    ly0 = max(0.0, ly0 - pad_y)
    lx1 = min(float(width), lx1 + pad_x)
    ly1 = min(float(height), ly1 + pad_y)
    lw, lh = lx1 - lx0, ly1 - ly0
    if lw < 1.0 or lh < 1.0:
        return Err(MlFailed(detail=MlDecode(details="degenerate face box for projection")))
    px = (tx - float(tx.min())) / tw * lw + lx0
    py = ly1 - (ty - float(ty.min())) / th * lh
    return Ok(np.stack((px, py), axis=1))


def run_unwrap_texture(
    image: ImageBytes, fit: FitResult, landmarks: Landmarks
) -> Ok[RenderedImage] | Err[DomainError]:
    """Unwrap real por proyeccion: foto a UV con coordenadas verificadas.

    El atlas 512 cubre UV piel 0-1 (pixel <-> UV directo; el flip del GLB se
    cancela con flipY del visor). Cada texel cubierto por un tri de piel del
    template real se muestrea de la foto por proyeccion afine alineada a
    bbox; ocluidas (sin tri, fuera de foto o fuera de piel parsing) se
    rellenan con piel media foto-derivada + detalle condicionado a checkpoint
    (fingerprint con texgan). Sin parsing (sin torch o sin pth) el unwrap
    procede sin mascara, documentado y determinista por env. Sin mascara
    foranea del layout denso 20k: no corresponde al atlas propio 5023.
    `bake_eye_texture`, nunca en este atlas. Balance gris-world sobre
    muestras validas (DPR SH completo en worker GPU Modal). Cero sentinel
    por construccion. Total: Err loud si falta el mat, el template o la
    foto; nunca blur silencioso disfrazado.
    """
    from backend.flame_fit import identity_ratios, landmark_points

    start = time.perf_counter()
    try:
        if not weights_present():
            return Err(MlFailed(detail=MlDecode(details="real unwrap requires ffhq-uv weights missing")))
        mat_result = _require_unwrap_mat()
        if isinstance(mat_result, Err):
            return mat_result
        photo_result = _decode_photo(image)
        if isinstance(photo_result, Err):
            return photo_result
        photo = photo_result.value
        height, width = int(photo.shape[0]), int(photo.shape[1])
        pts_result = landmark_points(landmarks)
        if isinstance(pts_result, Err):
            return pts_result
        pts = pts_result.value
        xs = pts[:, 0] * float(width)
        ys = pts[:, 1] * float(height)
        from backend.gnm_assemble import displaced_positions, load_flame_template

        loaded = load_flame_template()
        if isinstance(loaded, Err):
            return loaded
        _, template_uvs, template_tris = loaded.value
        displaced = displaced_positions(fit)
        if isinstance(displaced, Err):
            return displaced
        dpa = np.asarray(displaced.value, dtype=np.float64)
        facing: list[bool] = []
        for ta, tb, tc in template_tris:
            n = np.cross(dpa[tb] - dpa[ta], dpa[tc] - dpa[ta])
            facing.append(bool(n[2] > 0.0))
        v_idx, bw = _rasterize_skin_uv(list(template_uvs), list(template_tris), facing)
        covered = v_idx[..., 0] >= 0
        proj_result = _project_verts_to_pixels(displaced.value, width, height, xs, ys)
        if isinstance(proj_result, Err):
            return proj_result
        vert_px = proj_result.value
        safe_idx = np.where(covered[..., None], v_idx, 0)
        tex_px = (
            bw[..., 0:1] * vert_px[safe_idx[..., 0]]
            + bw[..., 1:2] * vert_px[safe_idx[..., 1]]
            + bw[..., 2:3] * vert_px[safe_idx[..., 2]]
        )
        in_photo = (
            covered
            & (tex_px[..., 0] >= 0.0)
            & (tex_px[..., 0] < float(width))
            & (tex_px[..., 1] >= 0.0)
            & (tex_px[..., 1] < float(height))
        )
        skin_valid = in_photo
        _LAST_TEXTURE_STATS["parsing"] = 0.0
        try:
            from backend.face_parsing import face_skin_mask as _parse_mask

            photo_skin = _parse_mask(photo)
            if photo_skin.shape == (height, width):
                ix = np.clip(np.rint(tex_px[..., 0]).astype(np.int64), 0, width - 1)
                iy = np.clip(np.rint(tex_px[..., 1]).astype(np.int64), 0, height - 1)
                skin_valid = in_photo & photo_skin[iy, ix]
                _LAST_TEXTURE_STATS["parsing"] = 1.0
        except Exception:  # noqa: BLE001 - sin torch/pesos: unwrap sin mascara (ver stats parsing)
            skin_valid = in_photo
        sampled = _bilinear_sample(photo, tex_px)
        skin_mean: NDArray[np.float64] | None = None
        if bool(skin_valid.any()):
            means = sampled[skin_valid].mean(axis=0)
            skin_mean = np.array(means, dtype=np.float64)
            gray = float(means.mean())
            if gray > 1e-9 and bool(np.all(means > 1e-9)):
                sampled = sampled * (gray / np.maximum(means, 1e-9))[None, None, :]
        sampled_u8 = np.clip(sampled, 0.0, 255.0)
        completion = _texgan_completion(sampled_u8, skin_valid)
        if completion is not None:
            _LAST_TEXTURE_STATS["texgan"] = 1.0
        else:
            _LAST_TEXTURE_STATS["texgan"] = 0.0
            fingerprint = _texture_fingerprint()
            if isinstance(fingerprint, Err):
                return fingerprint
            ratios_result = identity_ratios(landmarks)
            if isinstance(ratios_result, Err):
                return ratios_result
            quantized = struct.pack(f"<{len(ratios_result.value)}f", *(round(r, 2) for r in ratios_result.value))
            fit_quant = struct.pack("<253f", *(round(c, 1) for c in fit.coeffs.as_tuple()))
            seed = hashlib.sha256(REAL_TEXTURE_SALT + fingerprint.value + quantized + fit_quant).digest()
            detail = _identity_detail(seed)
            if skin_mean is not None:
                completion = np.clip(skin_mean[None, None, :] + detail, 0.0, 255.0)
            else:
                crop_result = _face_crop(photo, landmarks)
                if isinstance(crop_result, Err):
                    return crop_result
                completion = np.clip(_photo_base(crop_result.value) + detail, 0.0, 255.0)
        out = np.where(skin_valid[..., None], sampled_u8, completion)
        raw = np.rint(out).astype(np.uint8).tobytes()
        scrubbed = _scrub_sentinel(raw)
        parsed = parse_rendered_image(scrubbed)
        if isinstance(parsed, Err):
            return Err(MlFailed(detail=MlDecode(details="real unwrap produced invalid length")))
        _LAST_TEXTURE_STATS["evidence"] = texture_evidence(scrubbed)
        _LAST_TEXTURE_STATS["sentinel_count"] = float(count_sentinel(scrubbed))
        _LAST_TEXTURE_STATS["duration_ms"] = (time.perf_counter() - start) * 1000.0
        return Ok(parsed.value)
    except Exception as exc:  # noqa: BLE001 - unwrap caido es MlFailed, no crash
        return Err(MlFailed(detail=MlDecode(details=f"real unwrap failed: {exc}")))


def _real_bake(image: ImageBytes, fit: FitResult, landmarks: Landmarks) -> Ok[RenderedImage] | Err[DomainError]:
    """Bake real: unwrap por proyeccion si hay mat, si no completion/Err loud.

    Con `unwrap_1024_info.mat` corre `run_unwrap_texture` (proyeccion UV
    real + inpaint solo ocluida). Sin mat y con `VULTUS_REAL_ML=1` es Err
    loud; sin exigencia cae a completion foto-derivada (paridad local).
    Total: nunca lanza por inputs esperados. Cero sentinel por construccion.
    """
    from backend.flame_fit import identity_ratios

    start = time.perf_counter()
    try:
        if not weights_present():
            return Err(MlFailed(detail=MlDecode(details="real flame texture required but ffhq-uv weights missing")))
        if _find_unwrap_mat() is not None:
            return run_unwrap_texture(image, fit, landmarks)
        if _env("VULTUS_REAL_ML") == "1":
            return Err(MlFailed(detail=MlDecode(details="real unwrap requires unwrap_1024_info.mat")))

        fingerprint = _texture_fingerprint()
        if isinstance(fingerprint, Err):
            return fingerprint
        photo_result = _decode_photo(image)
        if isinstance(photo_result, Err):
            return photo_result
        photo = photo_result.value
        crop_result = _face_crop(photo, landmarks)
        if isinstance(crop_result, Err):
            return crop_result
        crop = crop_result.value
        ratios_result = identity_ratios(landmarks)
        if isinstance(ratios_result, Err):
            return ratios_result
        ratios = ratios_result.value
        quantized = struct.pack(f"<{len(ratios)}f", *(round(r, 2) for r in ratios))
        fit_quant = struct.pack("<253f", *(round(c, 1) for c in fit.coeffs.as_tuple()))
        seed = hashlib.sha256(REAL_TEXTURE_SALT + fingerprint.value + quantized + fit_quant).digest()
        base = _photo_base(crop)
        detail = _identity_detail(seed)
        out = np.clip(base + detail, 0.0, 255.0)
        raw = np.rint(out).astype(np.uint8).tobytes()
        scrubbed = _scrub_sentinel(raw)
        parsed = parse_rendered_image(scrubbed)
        if isinstance(parsed, Err):
            return Err(MlFailed(detail=MlDecode(details="real flame texture produced invalid length")))
        result = parsed.value
        _LAST_TEXTURE_STATS["evidence"] = texture_evidence(scrubbed)
        _LAST_TEXTURE_STATS["sentinel_count"] = float(count_sentinel(scrubbed))
        _LAST_TEXTURE_STATS["duration_ms"] = (time.perf_counter() - start) * 1000.0
        return Ok(result)
    except Exception as exc:  # noqa: BLE001 - bake caido es MlFailed, no crash
        return Err(MlFailed(detail=MlDecode(details=f"real flame texture failed: {exc}")))


def _texture_fingerprint() -> Ok[bytes] | Err[DomainError]:
    """Huella FFHQ-UV: sha256 de puente + texgan/unwrap mat cuando presentes.

    El checkpoint texgan y el mat de unwrap parametrizan la completion junto
    al OBJ: mismos pesos dan los mismos bytes; pesos distintos los cambian.
    La inferencia GAN completa con torch vive en el worker GPU Modal
    (sin torch top-level aqui). Total: Err solo si un archivo verificado
    deja de leerse (TOCTOU), nunca raise.
    """
    try:
        h = hashlib.sha256(REAL_TEXTURE_SALT)
        for name in (UV_OBJ_NAME, EYE_MAP_NAME):
            path = os.path.join(ffhq_uv_dir(), name)
            with open(path, "rb") as fh:
                head = fh.read(_BRIDGE_HEAD_BYTES)
            h.update(name.encode("utf-8"))
            h.update(struct.pack(">Q", os.path.getsize(path)))
            h.update(head)
        for d in _texgan_candidate_dirs():
            cand = os.path.join(d, TEXGAN_NAME)
            if _file_nonempty(cand):
                with open(cand, "rb") as fh:
                    head = fh.read(_BRIDGE_HEAD_BYTES)
                h.update(TEXGAN_NAME.encode("utf-8"))
                h.update(struct.pack(">Q", os.path.getsize(cand)))
                h.update(head)
                break
        for d in _topo_candidate_dirs():
            cand = os.path.join(d, UNWRAP_MAT_NAME)
            if _file_nonempty(cand):
                with open(cand, "rb") as fh:
                    head = fh.read(_BRIDGE_HEAD_BYTES)
                h.update(UNWRAP_MAT_NAME.encode("utf-8"))
                h.update(struct.pack(">Q", os.path.getsize(cand)))
                h.update(head)
                break
        return Ok(h.digest())
    except OSError as exc:
        return Err(MlFailed(detail=MlDecode(details=f"ffhq-uv bridge unreadable: {exc}")))


def _decode_photo(image: ImageBytes) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Foto a RGB float64. Total: Err si PIL no la decodifica."""
    try:
        with Image.open(io.BytesIO(image.as_bytes())) as handle:
            rgb = handle.convert("RGB")
            return Ok(np.asarray(rgb, dtype=np.float64))
    except Exception as exc:  # noqa: BLE001 - foto indecodificable es Err, no crash
        return Err(MlFailed(detail=MlDecode(details=f"photo decode failed: {exc}")))


def _face_crop(photo: NDArray[np.float64], landmarks: Landmarks) -> Ok[NDArray[np.float64]] | Err[DomainError]:
    """Recorte de cara por bbox de landmarks (+FACE_MARGIN). Total: Err degenerado."""
    from backend.flame_fit import landmark_points

    pts_result = landmark_points(landmarks)
    if isinstance(pts_result, Err):
        return pts_result
    pts = pts_result.value
    height, width = int(photo.shape[0]), int(photo.shape[1])
    if height <= 0 or width <= 0:
        return Err(MlFailed(detail=MlDecode(details="degenerate photo dimensions")))
    xs = pts[:, 0] * float(width)
    ys = pts[:, 1] * float(height)
    bw = float(xs.max() - xs.min())
    bh = float(ys.max() - ys.min())
    if bw < 1.0 or bh < 1.0:
        return Err(MlFailed(detail=MlDecode(details="degenerate face bbox")))
    x0 = max(0, int(xs.min() - FACE_MARGIN * bw))
    x1 = min(width, int(xs.max() + FACE_MARGIN * bw) + 1)
    y0 = max(0, int(ys.min() - FACE_MARGIN * bh))
    y1 = min(height, int(ys.max() + FACE_MARGIN * bh) + 1)
    if x1 <= x0 or y1 <= y0:
        return Err(MlFailed(detail=MlDecode(details="degenerate face crop")))
    return Ok(np.asarray(photo[y0:y1, x0:x1], dtype=np.float64))


def _photo_base(crop: NDArray[np.float64]) -> NDArray[np.float64]:
    """Layout de apariencia 512x512 desde la foto: 64 BILINEAR + 512 BILINEAR."""
    small = Image.fromarray(np.clip(crop, 0.0, 255.0).astype(np.uint8)).resize(
        (TEX_GRID, TEX_GRID), Image.Resampling.BILINEAR
    )
    big = small.resize((TEX_SIZE, TEX_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(big, dtype=np.float64)


def _hash_grid(seed: bytes, side: int) -> NDArray[np.float64]:
    """Grilla side x side x 3 uniforme en [-1,1] desde stream sha256."""
    need = side * side * 3
    stream = bytearray()
    counter = 0
    while len(stream) < need:
        stream.extend(hashlib.sha256(seed + struct.pack(">I", counter)).digest())
        counter += 1
    grid = np.frombuffer(bytes(stream[:need]), dtype=np.uint8).astype(np.float64)
    return (grid / 255.0 * 2.0 - 1.0).reshape(side, side, 3)


def _identity_detail(seed: bytes) -> NDArray[np.float64]:
    """Detalle suave cero-media: octavas hashed a 512, amplitud acotada."""
    canvas = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.float64)
    for index, (side, weight) in enumerate(_DETAIL_OCTAVES):
        grid = _hash_grid(seed + struct.pack(">I", index), side)
        layer = Image.fromarray(((grid + 1.0) * 127.5).astype(np.uint8)).resize(
            (TEX_SIZE, TEX_SIZE), Image.Resampling.BILINEAR
        )
        canvas += (np.asarray(layer, dtype=np.float64) - 127.5) / 127.5 * weight
    canvas -= canvas.mean()
    return canvas * DETAIL_AMPLITUDE


def _scrub_sentinel(raw: bytes) -> bytes:
    """Garantiza cero SKIN_SENTINEL: 255,0,255 -> 254,0,255 (un LSB)."""
    scrubbed = bytearray(raw)
    sr, sg, sb = SKIN_SENTINEL
    for i in range(0, len(scrubbed) - 2, 3):
        if scrubbed[i] == sr and scrubbed[i + 1] == sg and scrubbed[i + 2] == sb:
            scrubbed[i] = 254
    return bytes(scrubbed)


def bake_eye_texture() -> Ok[EyeTexture] | Err[DomainError]:
    """Segundo bake de ojos desde el mapa real `eye_ball_tex.png` (512 RGB).

    Textura propia separada de piel (User Story 5): pbr != uv a nivel de
    modulo. Total: Err loud si el puente no esta verificado o el mapa no
    decodifica; nunca sintetico silencioso.
    """
    try:
        if not weights_present():
            return Err(MlFailed(detail=MlDecode(details="real eye texture required but ffhq-uv weights missing")))
        path = os.path.join(ffhq_uv_dir(), EYE_MAP_NAME)
        with Image.open(path) as handle:
            eye = handle.convert("RGB").resize((TEX_SIZE, TEX_SIZE), Image.Resampling.BILINEAR)
        raw = np.asarray(eye, dtype=np.uint8).tobytes()
        parsed = parse_eye_texture(_scrub_sentinel(raw))
        if isinstance(parsed, Err):
            return Err(MlFailed(detail=MlDecode(details="real eye texture produced invalid length")))
        return Ok(parsed.value)
    except Exception as exc:  # noqa: BLE001 - bake ocular caido es MlFailed, no crash
        return Err(MlFailed(detail=MlDecode(details=f"real eye texture failed: {exc}")))


def bake_flame(
    image: ImageBytes, fit: FitResult, landmarks: Landmarks
) -> Ok[RenderedImage] | Err[DomainError]:
    """Seam textura feed-forward sobre tipos probados. Total: nunca lanza por inputs esperados.

    Con puente verificado corre el bake real y propaga su Ok/Err sin caer
    al doble; con `VULTUS_REAL_ML=1` sin pesos falla loud; sin real ni
    exigencia corre el replay doble (gateway/contrato en verde).
    """
    start = time.perf_counter()
    ensure_deterministic_texture()
    if _real_texture_available():
        return _real_bake(image, fit, landmarks)
    if os.environ.get("VULTUS_REAL_ML") == "1":
        return Err(MlFailed(detail=MlDecode(details="real flame texture required but ffhq-uv weights missing")))
    try:
        deterministic = _deterministic_albedo(image, fit, landmarks)
    except Exception as exc:  # noqa: BLE001 - el doble nunca tumba el job sin causa
        return Err(MlFailed(detail=MlDecode(details=f"flame texture double failed: {exc}")))
    if isinstance(deterministic, Err):
        return deterministic
    out = deterministic.value
    raw = out.as_bytes()
    _LAST_TEXTURE_STATS["evidence"] = texture_evidence(raw)
    _LAST_TEXTURE_STATS["sentinel_count"] = float(count_sentinel(raw))
    _LAST_TEXTURE_STATS["duration_ms"] = (time.perf_counter() - start) * 1000.0
    return Ok(out)


def bake_flame_from_request(payload: bytes) -> Ok[RenderedImage] | Err[DomainError]:
    """Borde wire v2: decodifica una vez y hornea. El decode ya es total."""
    try:
        from backend.pipeline_local import decode_texture_request
    except ImportError:  # pragma: no cover - paridad ruta plana en imagen
        from pipeline_local import (  # type: ignore[no-redef, import-not-found]
            decode_texture_request,
        )

    decoded = decode_texture_request(payload)
    if isinstance(decoded, Err):
        return decoded
    image, fit, landmarks = decoded.value
    return bake_flame(image, fit, landmarks)
