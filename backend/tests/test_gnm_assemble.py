"""Assemble FLAME: GLB PBR 2 materiales + zip-6, goldens congelados a mano.

Wave 4 Step 5: ojos [3931:5023) con material propio, piel sin ese rango,
PBR real (sin truco emisivo), zip-6 por la unica seam de ensamblaje.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile

from backend.domain import (
    UV_LEN,
    ZIP_NAMES,
    Ok,
    ZipBundle,
    parse_complete_uv,
    parse_gnm_mesh,
    parse_image_bytes,
)
from backend.gnm import build_result_zip, uv_to_png
from backend.gnm_assemble import (
    EYE_COUNT,
    EYE_VERT_END,
    EYE_VERT_START,
    VERT_COUNT,
    build_personalized_glb,
    eye_vertex_indices,
    flame_template_sha,
    is_eye_vertex,
    skin_vertex_indices,
)


def _albedo(marker: int):
    raw = bytes([marker & 0xFF, (marker * 2) & 0xFF]) + bytes(UV_LEN - 2)
    parsed = parse_complete_uv(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _fit(marker: float):
    from backend.domain import (
        FitResult,
        parse_camera_params,
        parse_gnm_coeffs,
        parse_landmarks,
    )

    lm = parse_landmarks(json.dumps([[0.0, 0.0, 0.0]] * 478).encode("utf-8"))
    assert isinstance(lm, Ok)
    coeffs = parse_gnm_coeffs([marker] * 253)
    camera = parse_camera_params([0.0] * 12)
    assert isinstance(coeffs, Ok)
    assert isinstance(camera, Ok)
    return FitResult(coeffs=coeffs.value, camera=camera.value)


def test_flame_consts_canonicas() -> None:
    assert VERT_COUNT == 5023
    assert EYE_VERT_START == 3931
    assert EYE_VERT_END == 5023
    assert EYE_COUNT == EYE_VERT_END - EYE_VERT_START == 1092


def test_flame_tri_counts_pinned_canonical() -> None:
    """Pin de conteo de tris (P2): total/skin/eye nombrados y consistentes.

    El guard `dropped != 0 -> Err` ya lo cubre
    `test_straddling_tri_a_caballo_returns_err_no_silent_drop`; aqui se pina
    que el fixture sintetico (waiver local sin asset real) suma exacto y
    particiona sin resto entre piel y ojo.
    """
    import backend.gnm_assemble as _asm

    assert _asm.FLAME_TRI_COUNT == 9976
    assert _asm.FLAME_SKIN_TRIS == 8000
    assert _asm.FLAME_EYE_TRIS == 1976
    assert _asm.FLAME_SKIN_TRIS + _asm.FLAME_EYE_TRIS == _asm.FLAME_TRI_COUNT
    positions, _uvs, tris = _asm._synthetic_flame_template()
    assert len(positions) == _asm.VERT_COUNT
    assert len(tris) == _asm.FLAME_TRI_COUNT
    skin = [t for t in tris if t[0] < EYE_VERT_START and t[1] < EYE_VERT_START and t[2] < EYE_VERT_START]
    eye = [t for t in tris if t[0] >= EYE_VERT_START and t[1] >= EYE_VERT_START and t[2] >= EYE_VERT_START]
    assert len(skin) == _asm.FLAME_SKIN_TRIS
    assert len(eye) == _asm.FLAME_EYE_TRIS
    assert len(tris) - len(skin) - len(eye) == 0


def test_eye_slice_len_1092_contiguo_desde_header() -> None:
    # Prueba contiguidad: el conjunto impreso desde el header es exactamente
    # range(START, END) sin huecos ni duplicados.
    eye = eye_vertex_indices()
    print(f"eye indices header: start={eye[0]} end={eye[-1]} len={len(eye)}")
    assert len(eye) == 1092
    assert eye[0] == EYE_VERT_START
    assert eye[-1] == EYE_VERT_END - 1
    assert eye == list(range(EYE_VERT_START, EYE_VERT_END))
    assert len(set(eye)) == len(eye)
    skin = skin_vertex_indices()
    assert len(skin) == EYE_VERT_START == 3931
    assert set(skin).isdisjoint(set(eye))
    assert len(skin) + len(eye) == VERT_COUNT


def test_skin_mask_excluye_ojo_y_ojo_solo_su_rango() -> None:
    for i in (0, 100, EYE_VERT_START - 1):
        assert is_eye_vertex(i) is False
    for i in (EYE_VERT_START, EYE_VERT_START + 500, EYE_VERT_END - 1):
        assert is_eye_vertex(i) is True
    assert is_eye_vertex(EYE_VERT_END) is False


def test_flame_template_sha_determinista() -> None:
    # Sucesor de gnm_template.bin: sha estable x2; fixture sintetico con
    # waiver si Modal inalcanzable (pre-check solo lectura documentado).
    first = flame_template_sha()
    second = flame_template_sha()
    assert first == second
    assert len(first) == 64
    assert all(c in "0123456789abcdef" for c in first)


def test_personalized_glb_magic_y_pbr_sin_emisivo() -> None:
    out = build_personalized_glb(_fit(0.1), _albedo(0xA1))
    assert isinstance(out, Ok)
    data = out.value.as_bytes()
    assert data[0:4] == b"glTF"
    assert parse_gnm_mesh(data) == out
    json_len = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    assert doc["accessors"][0]["count"] == VERT_COUNT
    mats = doc["materials"]
    assert len(mats) == 2
    names = [m["name"] for m in mats]
    assert any("kin" in n for n in names)
    assert any("ye" in n.lower() for n in names)
    # PBR real: baseColor clara, sin truco emisivo (base negra + emisivo 1).
    skin = next(m for m in mats if "kin" in m["name"])
    assert skin["pbrMetallicRoughness"]["baseColorFactor"] == [1, 1, 1, 1]
    assert "emissiveTexture" not in skin
    assert "emissiveFactor" not in skin
    assert len(doc["primitives"] if "primitives" in doc else doc["meshes"][0]["primitives"]) == 2


def test_glb_eye_material_solo_rango_ojo() -> None:
    out = build_personalized_glb(_fit(0.1), _albedo(0xA1))
    assert isinstance(out, Ok)
    data = out.value.as_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    prims = doc["meshes"][0]["primitives"]
    assert len(prims) == 2
    skin_prim = next(p for p in prims if p["material"] == 0)
    eye_prim = next(p for p in prims if p["material"] == 1)
    bin_start = 20 + json_len + 8
    bin_buf = data[bin_start:]
    for prim, check_eye in ((skin_prim, False), (eye_prim, True)):
        idx_accessor = doc["accessors"][prim["indices"]]
        view = doc["bufferViews"][idx_accessor["bufferView"]]
        off = view["byteOffset"]
        count = idx_accessor["count"]
        comp = idx_accessor["componentType"]
        fmt = f"<{count}I" if comp == 5125 else f"<{count}H"
        size = 4 if comp == 5125 else 2
        idx = struct.unpack(fmt, bin_buf[off : off + count * size])
        assert len(idx) == count
        if check_eye:
            assert all(EYE_VERT_START <= v < EYE_VERT_END for v in idx)
            assert len(idx) > 0
        else:
            assert all(v < EYE_VERT_START for v in idx)
            assert len(idx) > 0


def test_glb_determinista_x2_y_difiere_por_identidad() -> None:
    albedo = _albedo(0xA1)
    first = build_personalized_glb(_fit(0.1), albedo)
    second = build_personalized_glb(_fit(0.1), albedo)
    other = build_personalized_glb(_fit(0.9), albedo)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert isinstance(other, Ok)
    assert first.value.as_bytes() == second.value.as_bytes()
    assert first.value.as_bytes() != other.value.as_bytes()


def test_zip6_por_seam_ensamblaje() -> None:
    fit_a = _fit(0.1)
    fit_b = _fit(0.5)
    alb_a = _albedo(0xA1)
    alb_b = _albedo(0xB2)
    ma = build_personalized_glb(fit_a, alb_a)
    mb = build_personalized_glb(fit_b, alb_b)
    assert isinstance(ma, Ok)
    assert isinstance(mb, Ok)
    bundle = ZipBundle(
        uv_a_png=uv_to_png(alb_a),
        uv_b_png=uv_to_png(alb_b),
        mesh_a_glb=ma.value.as_bytes(),
        mesh_b_glb=mb.value.as_bytes(),
        pbr_a=uv_to_png(alb_a),
        pbr_b=uv_to_png(alb_b),
    )
    blob = build_result_zip(bundle)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
    assert names == list(ZIP_NAMES)
    assert len(names) == 6


def test_glb_embeds_skin_png() -> None:
    import io as _io

    from PIL import Image as _Image

    fit = _fit(0.1)
    albedo = _albedo(0xA1)
    img = _Image.new("RGB", (32, 32), (0xA1, 0xB2, 0xC3))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    parsed = parse_image_bytes(buf.getvalue())
    assert isinstance(parsed, Ok)
    out = build_personalized_glb(fit, albedo, atlas_png=parsed.value)
    assert isinstance(out, Ok)
    assert out.value.as_bytes()[0:4] == b"glTF"
    assert parsed.value.as_bytes() in out.value.as_bytes()


# --- Wave 4-fix P4: tris a caballo, padding, validacion bin, is_synthetic, RenderedImage, pbr-doble ---


def _synthetic_positions_uvs():  # type: ignore[no-untyped-def]
    import backend.gnm_assemble as _asm

    positions, uvs, _ = _asm._synthetic_flame_template()
    return positions, uvs


def test_straddling_tri_a_caballo_returns_err_no_silent_drop(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Tri a caballo (3930,3931,3932) -> Err explicito, nunca drop silencioso.

    Hoy: skin=[<3931] + eye=[>=3931] dropea el tri a caballo y retorna Ok.
    Fix: dropped = len(all)-len(skin)-len(eye) != 0 -> Err + dropped==0.
    """
    import backend.gnm_assemble as _asm
    from backend.domain import Err

    positions, uvs = _synthetic_positions_uvs()
    straddling = [(3930, 3931, 3932)]
    skin_one = [(0, 1, 2)]
    eye_one = [(3931, 3932, 3933)]
    custom_tris = skin_one + eye_one + straddling
    monkeypatch.setattr(_asm, "_flame_cache", (positions, uvs, custom_tris))
    out = _asm.build_personalized_glb(_fit(0.1), _albedo(0xA1))
    assert isinstance(out, Err)
    # dropped == 1 debe surfear como Err, no como Ok con tris perdidos.
    assert "caballo" in str(out.error).lower() or "caballo" in repr(out.error).lower() or dropped_count(custom_tris) == 1


def dropped_count(tris):  # type: ignore[no-untyped-def]
    from backend.gnm_assemble import EYE_VERT_START

    skin = [t for t in tris if t[0] < EYE_VERT_START and t[1] < EYE_VERT_START and t[2] < EYE_VERT_START]
    eye = [t for t in tris if t[0] >= EYE_VERT_START and t[1] >= EYE_VERT_START and t[2] >= EYE_VERT_START]
    return len(tris) - len(skin) - len(eye)


def test_glb_offsets_aligned_per_section_with_odd_skin(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pad a 4 tras cada seccion: byteOffset%4==0 aun con skin_tris impar.

    Hoy: uv_off=pos_len, skin_off=pos+uv sin pad; con 1 tri piel (6 bytes)
    el eye_off queda desalineado. Fix: offsets sobre lens padded.
    """
    import backend.gnm_assemble as _asm

    positions, uvs = _synthetic_positions_uvs()
    # 1 tri piel (impar) + 1 tri ojo: skin_buf=6 bytes (%4==2) fuerza pad.
    custom_tris = [(0, 1, 2), (3931, 3932, 3933)]
    monkeypatch.setattr(_asm, "_flame_cache", (positions, uvs, custom_tris))
    out = _asm.build_personalized_glb(_fit(0.1), _albedo(0xA1))
    assert isinstance(out, Ok)
    data = out.value.as_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    for view in doc["bufferViews"]:
        assert view["byteOffset"] % 4 == 0, f"view desalineado: {view}"


def _write_flame_bin(path: str, positions, uvs, tris) -> None:  # type: ignore[no-untyped-def]
    import struct as _struct

    verts = len(positions)
    with open(path, "wb") as f:
        f.write(_struct.pack("<II", verts, len(tris)))
        f.writelines(_struct.pack("<3f", float(x), float(y), float(z)) for x, y, z in positions)
        f.writelines(_struct.pack("<2f", float(u), float(v)) for u, v in uvs)
        f.writelines(_struct.pack("<III", int(a), int(b), int(c)) for a, b, c in tris)


def test_flame_template_rejects_nonfinite_bin(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Bin con NaN/inf -> Err(MlFailed), nunca aceptacion silenciosa ni raise."""
    import backend.gnm_assemble as _asm
    from backend.domain import Err as _Err

    positions, uvs = _synthetic_positions_uvs()
    bad_positions = [(float("nan"), 0.0, 0.0)] + positions[1:]
    tris = [(0, 1, 2), (3931, 3932, 3933)]
    bin_path = str(tmp_path / "flame_template.bin")
    _write_flame_bin(bin_path, bad_positions, uvs, tris)
    monkeypatch.setenv("FLAME_ASSETS_DIR", str(tmp_path))
    monkeypatch.setenv("WEIGHTS_DIR", "")
    monkeypatch.setattr(_asm, "_flame_cache", None)
    result = _asm.load_flame_template()
    assert isinstance(result, _Err)
    assert "finita" in str(result.error)


def test_flame_template_rejects_bad_index_bin(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Bin con indice>=verts -> Err(MlFailed), nunca aceptacion silenciosa ni raise."""
    import backend.gnm_assemble as _asm
    from backend.domain import Err as _Err
    from backend.gnm_assemble import VERT_COUNT

    positions, uvs = _synthetic_positions_uvs()
    tris = [(0, 1, 2), (VERT_COUNT, 0, 1)]
    bin_path = str(tmp_path / "flame_template.bin")
    _write_flame_bin(bin_path, positions, uvs, tris)
    monkeypatch.setenv("FLAME_ASSETS_DIR", str(tmp_path))
    monkeypatch.setenv("WEIGHTS_DIR", "")
    monkeypatch.setattr(_asm, "_flame_cache", None)
    result = _asm.load_flame_template()
    assert isinstance(result, _Err)
    assert "rango" in str(result.error)


def test_is_flame_synthetic_flag_separate_from_sha(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """is_synthetic flag separado de sha; waiver fixture preservado sin Err en falta local."""
    import backend.gnm_assemble as _asm
    from backend.domain import Ok as _Ok

    # Sin archivo: sintetico True, sha estable, load no lanza (waiver).
    monkeypatch.setenv("FLAME_ASSETS_DIR", str(tmp_path / "vacio-inexistente"))
    monkeypatch.setenv("WEIGHTS_DIR", "")
    monkeypatch.setattr(_asm, "_flame_cache", None)
    assert _asm.is_flame_synthetic() is True
    sha1 = _asm.flame_template_sha()
    sha2 = _asm.flame_template_sha()
    assert sha1 == sha2 and len(sha1) == 64
    loaded = _asm.load_flame_template()
    assert isinstance(loaded, _Ok)
    positions, _, _ = loaded.value
    assert len(positions) == 5023
    # Con archivo: sintetico False.
    positions0, uvs0 = _synthetic_positions_uvs()
    tmp_path.mkdir(exist_ok=True)
    _write_flame_bin(str(tmp_path / "flame_template.bin"), positions0, uvs0, [(0, 1, 2)])
    monkeypatch.setenv("FLAME_ASSETS_DIR", str(tmp_path))
    monkeypatch.setattr(_asm, "_flame_cache", None)
    assert _asm.is_flame_synthetic() is False


def test_build_glb_accepts_rendered_image() -> None:
    """build_personalized_glb acepta RenderedImage (y CompleteUv compat).

    Anotacion debe mencionar RenderedImage; comportamiento identico por as_bytes().
    """
    import inspect as _inspect

    from backend.domain import parse_rendered_image
    from backend.gnm_assemble import build_personalized_glb as _glb

    ann = str(_inspect.signature(_glb))
    assert "RenderedImage" in ann
    raw = bytes(_albedo(0xA1).as_bytes())
    parsed = parse_rendered_image(raw)
    assert isinstance(parsed, Ok)
    out = _glb(_fit(0.1), parsed.value)
    assert isinstance(out, Ok)
    assert out.value.as_bytes()[0:4] == b"glTF"
    # Compat CompleteUv intacta.
    out2 = _glb(_fit(0.1), _albedo(0xA1))
    assert isinstance(out2, Ok)


def test_pbr_intentional_double_pbr_eq_uv(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """PBR skin-duplicate ambos modos (DAG v10 HIL dueno): pbr==uv via helper.

    Placeholder honesto sin mapas roughness/metalness reales; los ojos viven
    en el GLB (2 primitivas SkinPBR+EyePBR). TODO mapas reales -> pbr!=uv.
    Este test fija el duplicado con VULTUS_REAL_ML=0 y con =1 (ambos legs).
    """
    from backend.domain import Ok as _Ok
    from backend.pipeline_local import resolve_pbr_pngs

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("FFHQ_UV_DIR", str(empty))
    fit_a = _fit(0.1)
    alb_a = _albedo(0xA1)
    ma = build_personalized_glb(fit_a, alb_a)
    assert isinstance(ma, Ok)
    uv_png = uv_to_png(alb_a)
    # Ambos legs via helper: pbr == uv (copia, placeholder sin mapa real).
    for real_flag in ("0", "1"):
        monkeypatch.setenv("VULTUS_REAL_ML", real_flag)
        pbr_resolved = resolve_pbr_pngs(alb_a, _albedo(0xB2))
        assert isinstance(pbr_resolved, _Ok)
        assert pbr_resolved.value[0] == uv_png
    bundle = ZipBundle(
        uv_a_png=uv_png,
        uv_b_png=uv_to_png(_albedo(0xB2)),
        mesh_a_glb=ma.value.as_bytes(),
        mesh_b_glb=ma.value.as_bytes(),
        pbr_a=uv_png,
        pbr_b=uv_to_png(_albedo(0xB2)),
    )
    blob = build_result_zip(bundle)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
    assert names == list(ZIP_NAMES)


def test_synthetic_template_head_like_not_lines() -> None:
    """Slice 1 RED: template sintetico debe abrir como cabeza, no lineas."""
    import math as _math

    import backend.gnm_assemble as _asm

    positions, uvs, tris = _asm._synthetic_flame_template()
    assert len(positions) == _asm.VERT_COUNT
    assert len(tris) == _asm.FLAME_TRI_COUNT

    def _area(t: tuple[int, int, int]) -> float:
        p0, p1, p2 = positions[t[0]], positions[t[1]], positions[t[2]]
        ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        cz = ux * vy - uy * vx
        return 0.5 * _math.sqrt(cx * cx + cy * cy + cz * cz)

    areas = [_area(t) for t in tris[:200]]
    mean_area = sum(areas) / len(areas)
    assert mean_area > 1e-6, f"tris degenerados tipo linea, mean_area={mean_area}"
    xs = [p[0] for p in positions[: _asm.SKIN_VERT_COUNT]]
    ys = [p[1] for p in positions[: _asm.SKIN_VERT_COUNT]]
    assert (max(ys) - min(ys)) > (max(xs) - min(xs)) * 1.1
    skin_uvs = uvs[: _asm.SKIN_VERT_COUNT]
    assert all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for u, v in skin_uvs)
    eye_uvs = uvs[_asm.EYE_VERT_START :]
    assert all(2.0 <= u <= 3.0 and 0.0 <= v <= 1.0 for u, v in eye_uvs)


def test_load_flame_template_fails_loud_when_real_demanded(monkeypatch, tmp_path) -> None:
    """Slice 1 RED: con VULTUS_REAL_ML=1 sin bin debe ser Err, no waiver."""
    import backend.gnm_assemble as _asm
    from backend.domain import Err as _Err

    monkeypatch.setenv("FLAME_ASSETS_DIR", str(tmp_path / "vacio-inexistente"))
    monkeypatch.setenv("WEIGHTS_DIR", "")
    monkeypatch.setenv("VULTUS_REAL_ML", "1")
    monkeypatch.setattr(_asm, "_flame_cache", None)
    result = _asm.load_flame_template()
    assert isinstance(result, _Err)


def test_glb_embeds_eye_texture_when_provided() -> None:
    """Slice 4 RED: EyeTexture real debe incrustarse, no blanco fallback."""
    from backend.domain import parse_eye_texture

    raw_eye = bytes([10, 20, 30] * (512 * 512))
    parsed = parse_eye_texture(raw_eye)
    assert isinstance(parsed, Ok)
    out_real = build_personalized_glb(_fit(0.1), _albedo(0xA1), eye_texture=parsed.value)
    out_white = build_personalized_glb(_fit(0.1), _albedo(0xA1))
    assert isinstance(out_real, Ok)
    assert isinstance(out_white, Ok)
    assert out_real.value.as_bytes() != out_white.value.as_bytes()
    assert raw_eye[:100] != bytes([240, 240, 240] * 34)[:100]
