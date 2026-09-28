"""Worker local: pipeline par con sidecar falso, goldens literales a mano.

El pipeline consume el sink estrecho (`report`, `complete`, `fail`);
los tests inyectan el sink en memoria, el runner el sink HTTP.
"""

from __future__ import annotations

import json

from backend.domain import (
    CompareResult,
    CompleteUv,
    DomainError,
    Err,
    FitResult,
    ImageBytes,
    JobId,
    Landmarks,
    Ok,
    Progress,
    Stage,
    parse_image_bytes,
    parse_progress,
)
from backend.pipeline_local import ProgressSink, default_config, job_dir
from backend.pipeline_local import run_pair as run_pair_real

MARKER_A = 0xA1
MARKER_B = 0xB2


def _image(marker: int) -> ImageBytes:
    raw = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes([marker]) * 56
    parsed = parse_image_bytes(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks_bytes() -> bytes:
    pts = [[0.0, 0.0, 0.0]] * 478
    return json.dumps(pts).encode("utf-8")


def _golden_complete(head: bytes) -> bytes:
    from backend.domain import UV_LEN

    return bytes(head) + bytes(UV_LEN - len(head))


class InMemorySink:
    """Sink de tests: registra reportes y guarda el resultado."""

    def __init__(self) -> None:
        self.reports: list[tuple[Progress, Stage]] = []
        self.result: CompareResult | None = None
        self.failed = False

    def report(self, progress: Progress, stage: Stage) -> Ok[None] | Err[DomainError]:
        self.reports.append((progress, stage))
        return Ok(None)

    def complete(self, result: CompareResult) -> Ok[None] | Err[DomainError]:
        self.result = result
        return Ok(None)

    def fail(self) -> Ok[None] | Err[DomainError]:
        self.failed = True
        return Ok(None)


class FakeMlOk:
    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        from backend.domain import parse_landmarks

        return parse_landmarks(_landmarks_bytes())

    def fit(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
        from backend.gnm_fit import fit_gnm

        return fit_gnm(image, landmarks)

    def texture(
        self, job_id: JobId, image: ImageBytes, fit: FitResult, landmarks: Landmarks
    ) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import parse_complete_uv

        head = bytes([10, 200]) if MARKER_A in image.as_bytes() else bytes([4, 210])
        return parse_complete_uv(_golden_complete(head))


class FakeMlFailFit:
    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        from backend.domain import parse_landmarks

        return parse_landmarks(_landmarks_bytes())

    def fit(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
        from backend.domain import FitFailed, MlBadStatus

        _ = (job_id, image, landmarks)
        return Err(FitFailed(detail=MlBadStatus(status=500)))

    def texture(
        self, job_id: JobId, image: ImageBytes, fit: FitResult, landmarks: Landmarks
    ) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import parse_complete_uv

        _ = (job_id, image, fit, landmarks)
        return parse_complete_uv(_golden_complete(bytes([10, 200])))


def test_pair_produces_zip6_sin_heatmap_y_limpia_tmp() -> None:
    import io as _io
    import zipfile as _zipfile

    from backend.domain import UV_LEN, ZIP_NAMES, new_job_id
    from backend.gnm import build_result_zip, uv_to_png

    sink: ProgressSink = InMemorySink()
    image_a = _image(MARKER_A)
    image_b = _image(MARKER_B)
    job_id = new_job_id()
    out = run_pair_real(sink, FakeMlOk(), job_id, image_a, image_b, default_config())  # type: ignore[arg-type]
    assert isinstance(out, Ok)
    result = out.value
    assert len(result.uv_a.as_bytes()) == UV_LEN
    assert len(result.uv_b.as_bytes()) == UV_LEN
    assert result.uv_a.as_bytes()[0:2] == bytes([10, 200])
    assert result.uv_b.as_bytes()[0:2] == bytes([4, 210])
    assert not hasattr(result, "heatmap")
    for mesh in (result.mesh_a, result.mesh_b):
        assert mesh.as_bytes()[0:4] == b"glTF"
    assert result.mesh_a != result.mesh_b
    assert isinstance(sink, InMemorySink)
    assert sink.result == result
    assert not sink.failed
    # Zip-6 por la unica seam: mismos 6 nombres del dominio en el zip real.
    bundle_probe = build_result_zip(
        __import__("backend.domain", fromlist=["ZipBundle"]).ZipBundle(
            uv_a_png=uv_to_png(result.uv_a),
            uv_b_png=uv_to_png(result.uv_b),
            mesh_a_glb=result.mesh_a.as_bytes(),
            mesh_b_glb=result.mesh_b.as_bytes(),
            pbr_a=uv_to_png(result.uv_a),
            pbr_b=uv_to_png(result.uv_b),
        )
    )
    with _zipfile.ZipFile(_io.BytesIO(bundle_probe)) as z:
        assert z.namelist() == list(ZIP_NAMES)
        assert len(z.namelist()) == 6
    done = parse_progress(1.0)
    assert isinstance(done, Ok)
    assert [(p.value(), s) for p, s in sink.reports] == [
        (0.40, Stage.FIT),
        (0.75, Stage.TEXTURE),
        (0.95, Stage.ASSEMBLE),
    ]
    assert not job_dir(job_id).exists()


def test_zip6_sync_python_ts_zip() -> None:
    import io as _io
    import zipfile as _zipfile
    from pathlib import Path as _Path

    from backend.domain import ZIP_NAMES
    from backend.domain import ZipBundle as _Bundle
    from backend.domain import parse_complete_uv as _parse_uv
    from backend.gnm import build_result_zip, uv_to_png

    expected = ["uv_a.png", "uv_b.png", "mesh_a.glb", "mesh_b.glb", "pbr_a.png", "pbr_b.png"]
    assert list(ZIP_NAMES) == expected
    ts = (_Path(__file__).resolve().parents[2] / "edge" / "contract.ts").read_text()
    for name in expected:
        assert name in ts
    assert "heatmap" not in ts.split("ZIP_MANIFEST")[1].split("]")[0]
    uv = _parse_uv(bytes(6) + bytes(786432 - 6))
    assert isinstance(uv, Ok)
    png = uv_to_png(uv.value)
    # Zip real con GLBs minimos de la seam: entradas == ZIP_NAMES == TS.
    import json as _json

    from backend.domain import FitResult as _Fit
    from backend.domain import parse_camera_params as _pcam
    from backend.domain import parse_gnm_coeffs as _pcoeff
    from backend.domain import parse_landmarks as _plm
    from backend.gnm_assemble import build_personalized_glb as _glb

    lm = _plm(_json.dumps([[0.0, 0.0, 0.0]] * 478).encode("utf-8"))
    assert isinstance(lm, Ok)
    c0 = _pcoeff([0.0] * 253)
    cam = _pcam([0.0] * 12)
    assert isinstance(c0, Ok) and isinstance(cam, Ok)
    fit = _Fit(coeffs=c0.value, camera=cam.value)
    ga = _glb(fit, uv.value)
    assert isinstance(ga, Ok)
    blob = build_result_zip(
        _Bundle(
            uv_a_png=png,
            uv_b_png=png,
            mesh_a_glb=ga.value.as_bytes(),
            mesh_b_glb=ga.value.as_bytes(),
            pbr_a=png,
            pbr_b=png,
        )
    )
    with _zipfile.ZipFile(_io.BytesIO(blob)) as z:
        assert z.namelist() == expected


def test_sidecar_error_fails_job_and_cleans_tmp() -> None:
    from backend.domain import new_job_id

    sink: ProgressSink = InMemorySink()
    image_a = _image(MARKER_A)
    image_b = _image(MARKER_B)
    job_id = new_job_id()
    out = run_pair_real(sink, FakeMlFailFit(), job_id, image_a, image_b, default_config())  # type: ignore[arg-type]
    assert isinstance(out, Err)
    assert isinstance(sink, InMemorySink)
    assert sink.failed
    assert sink.result is None
    assert not job_dir(job_id).exists()


def test_pipeline_timeouts_match_decisions() -> None:
    cfg = default_config()
    assert cfg.landmarks_timeout_secs == 5.0
    assert cfg.fit_timeout_secs == 10.0
    assert cfg.texture_timeout_secs == 30.0
    assert cfg.total_timeout_secs == 60.0


# --- Wave 2 Step 3: codec v2 versionado (fit request + texture request) ---
# Seam run_pair+sink: el wire que MlSidecarClient.fit/texture usa.
# Flag-day incompatible por diseno: v1 -> VersionMismatch 400 sin dual-read;
# despliegue con drain de cola TTL60/visibility180. Truncado FitFailed 500
# vs trailing VersionMismatch 400.
# run_pair end-to-end sigue en Wave 4 (heatmap); aqui vectores del codec.


def _v2_proven_parts():  # type: ignore[no-untyped-def]
    import json as _json

    from backend.domain import FitResult as _FitResult
    from backend.domain import Ok as _Ok
    from backend.domain import parse_camera_params as _parse_cam
    from backend.domain import parse_gnm_coeffs as _parse_coeffs
    from backend.domain import parse_landmarks as _parse_lm

    lm = _parse_lm(_json.dumps([[0.0, 1.0, 2.0]] * 478).encode("utf-8"))
    assert isinstance(lm, _Ok)
    img = parse_image_bytes(bytes([0xFF, 0xD8, 0xFF, 0x00]) + bytes(60))
    assert isinstance(img, _Ok)
    coeffs = _parse_coeffs([0.0] * 253)
    camera = _parse_cam([0.0] * 12)
    assert isinstance(coeffs, _Ok)
    assert isinstance(camera, _Ok)
    return img.value, _FitResult(coeffs=coeffs.value, camera=camera.value), lm.value


def test_fit_request_v2_roundtrip_ok() -> None:
    from backend.domain import (
        CODEC_VERSION_LEN,
        VERSION_V2,
        decode_fit_request,
        encode_fit_request,
    )

    image, _, landmarks = _v2_proven_parts()
    blob = encode_fit_request(image, landmarks)
    assert blob[0] == VERSION_V2
    assert len(blob) > CODEC_VERSION_LEN + 4
    back = decode_fit_request(blob)
    assert isinstance(back, Ok)
    back_lm, back_img = back.value
    assert back_lm.as_bytes() == landmarks.as_bytes()
    assert back_img.as_bytes() == image.as_bytes()


def test_fit_request_v1_typical_is_version_mismatch() -> None:
    import json as _json

    from backend.domain import VersionMismatch, decode_fit_request, domain_to_status

    lm_raw = _json.dumps([[0.0, 1.0, 2.0]] * 478).encode("utf-8")
    img_raw = bytes([0xFF, 0xD8, 0xFF, 0x00]) + bytes(60)
    v1 = len(lm_raw).to_bytes(4, "big") + lm_raw + img_raw
    assert v1[0] == 0x00
    result = decode_fit_request(v1)
    assert isinstance(result, Err)
    assert isinstance(result.error, VersionMismatch)
    assert domain_to_status(result.error) == 400


def test_fit_request_v1_collision_is_err() -> None:
    from backend.domain import VERSION_V1, decode_fit_request, encode_fit_request

    image, _, landmarks = _v2_proven_parts()
    valid = encode_fit_request(image, landmarks)
    as_v1 = bytes([VERSION_V1]) + bytes(valid[1:])
    assert isinstance(decode_fit_request(as_v1), Err)
    huge_len_prefix = bytes([0x02, 0xFF, 0xFF, 0xFF, 0xFF]) + b"{}"
    assert isinstance(decode_fit_request(huge_len_prefix), Err)


def test_version_mismatch_is_frozen_value() -> None:
    import pytest as _pytest

    from backend.domain import VersionMismatch, domain_to_message, domain_to_status

    err = VersionMismatch()
    with _pytest.raises(AttributeError):
        err.detail = "x"
    assert domain_to_status(err) == 400
    assert "version" in domain_to_message(err)


def test_texture_request_v2_roundtrip_and_exact_consumption() -> None:
    import json as _json

    from backend.domain import VERSION_V2, VersionMismatch, decode_fit_request
    from backend.domain import domain_to_status as _status
    from backend.pipeline_local import (
        decode_texture_request,
        encode_fit_result,
        encode_texture_request,
    )

    image, fit, landmarks = _v2_proven_parts()
    blob = encode_texture_request(image, fit, landmarks)
    assert blob[0] == VERSION_V2
    back = decode_texture_request(blob)
    assert isinstance(back, Ok)
    back_image, back_fit, back_landmarks = back.value
    assert back_image.as_bytes() == image.as_bytes()
    assert back_landmarks.as_bytes() == landmarks.as_bytes()
    assert back_fit.coeffs.as_tuple() == fit.coeffs.as_tuple()
    assert back_fit.camera.as_tuple() == fit.camera.as_tuple()
    trailing = decode_texture_request(blob + b"\x00")
    assert isinstance(trailing, Err)
    assert isinstance(trailing.error, VersionMismatch)
    lm_raw = _json.dumps([[0.0, 1.0, 2.0]] * 478).encode("utf-8")
    img_raw = bytes([0xFF, 0xD8, 0xFF, 0x00]) + bytes(60)
    v1_inner = len(lm_raw).to_bytes(4, "big") + lm_raw + img_raw
    v1_outer = len(v1_inner).to_bytes(4, "big") + v1_inner + encode_fit_result(fit)
    assert v1_outer[0] == 0x00
    v1_result = decode_texture_request(v1_outer)
    assert isinstance(v1_result, Err)
    assert isinstance(v1_result.error, VersionMismatch)
    assert _status(v1_result.error) == 400
    _ = decode_fit_request


def test_fit_deadline_inside_ttl_and_visibility() -> None:
    from backend import modal_app

    assert modal_app.FIT_TIMEOUT_SECS == 10
    assert modal_app.TOTAL_TIMEOUT_SECS == 60
    assert modal_app.QUEUE_VISIBILITY_TIMEOUT_SECS == 180


# --- Wave 3 Step 4: textura sin gris via seam run_pair+sink (gate obligatorio) ---
# Double/fixture da albedo sin SKIN_SENTINEL, bytes-identicos-x2 bajo config
# deterministica. Preferida sobre unidad nueva por plan (file-conflict: Wave 4
# migra callers de gnm_texture, nunca Wave 3).
# Nota: el run_pair end-to-end sigue en Wave 4 (assemble heatmap legacy);
# aqui la pata texture del mismo seam (el metodo que run_pair invoca).


class FakeMlFlameTexture:
    """Fake de seam: landmarks fijos, fit flame real-double, textura flame."""

    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        from backend.domain import parse_landmarks

        _ = (job_id, image)
        return parse_landmarks(_landmarks_bytes())

    def fit(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
        from backend.flame_fit import fit_flame

        _ = job_id
        return fit_flame(image, landmarks)

    def texture(
        self, job_id: JobId, image: ImageBytes, fit: FitResult, landmarks: Landmarks
    ) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import parse_complete_uv
        from backend.flame_texture import bake_flame

        _ = job_id
        baked = bake_flame(image, fit, landmarks)
        if isinstance(baked, Err):
            return baked
        return parse_complete_uv(baked.value.as_bytes())


def test_texture_v2_via_flame_double_no_sentinel_bytes_x2() -> None:
    from backend.domain import UV_LEN, new_job_id
    from backend.flame_texture import EVIDENCE_MIN, count_sentinel, texture_evidence

    fake = FakeMlFlameTexture()
    image_a = _image(MARKER_A)
    image_b = _image(MARKER_B)
    job_id = new_job_id()
    lm_a = fake.landmarks(job_id, image_a)
    lm_b = fake.landmarks(job_id, image_b)
    assert isinstance(lm_a, Ok)
    assert isinstance(lm_b, Ok)
    fit_a = fake.fit(job_id, image_a, lm_a.value)
    fit_b = fake.fit(job_id, image_b, lm_b.value)
    assert isinstance(fit_a, Ok)
    assert isinstance(fit_b, Ok)
    # Pata texture del seam run_pair (mismo wire v2 que MlSidecarClient.texture).
    first_a = fake.texture(job_id, image_a, fit_a.value, lm_a.value)
    first_b = fake.texture(job_id, image_b, fit_b.value, lm_b.value)
    assert isinstance(first_a, Ok)
    assert isinstance(first_b, Ok)
    for uv in (first_a.value, first_b.value):
        raw = uv.as_bytes()
        assert len(raw) == UV_LEN
        assert count_sentinel(raw) == 0
        assert texture_evidence(raw) >= EVIDENCE_MIN
    # Determinismo exigible: segunda pasada da bytes identicos.
    second_a = fake.texture(job_id, image_a, fit_a.value, lm_a.value)
    second_b = fake.texture(job_id, image_b, fit_b.value, lm_b.value)
    assert isinstance(second_a, Ok)
    assert isinstance(second_b, Ok)
    assert second_a.value.as_bytes() == first_a.value.as_bytes()
    assert second_b.value.as_bytes() == first_b.value.as_bytes()
    # Wire v2: el payload que MlSidecarClient.texture envia decodifica exacto.
    from backend.pipeline_local import decode_texture_request, encode_texture_request

    blob = encode_texture_request(image_a, fit_a.value, lm_a.value)
    back = decode_texture_request(blob)
    assert isinstance(back, Ok)


def test_texture_v2_wire_compat_with_bake_from_request() -> None:
    from backend.domain import new_job_id
    from backend.flame_fit import fit_flame
    from backend.flame_texture import bake_flame_from_request, count_sentinel
    from backend.pipeline_local import encode_texture_request

    image = _image(MARKER_A)
    lm = FakeMlFlameTexture().landmarks(new_job_id(), image)
    assert isinstance(lm, Ok)
    fit = fit_flame(image, lm.value)
    assert isinstance(fit, Ok)
    blob = encode_texture_request(image, fit.value, lm.value)
    out = bake_flame_from_request(blob)
    assert isinstance(out, Ok)
    assert count_sentinel(out.value.as_bytes()) == 0


# --- Wave 4-fix P4: gate sentinel/evidence en seam run_pair + pbr-doble doc ---


class FakeMlSentinel:
    """Fake con sentinel magenta: debe fallar el gate count_sentinel==0."""

    def landmarks(self, job_id: JobId, image: ImageBytes) -> Ok[Landmarks] | Err[DomainError]:
        from backend.domain import parse_landmarks

        _ = (job_id, image)
        return parse_landmarks(_landmarks_bytes())

    def fit(self, job_id: JobId, image: ImageBytes, landmarks: Landmarks) -> Ok[FitResult] | Err[DomainError]:
        from backend.flame_fit import fit_flame

        _ = job_id
        return fit_flame(image, landmarks)

    def texture(
        self, job_id: JobId, image: ImageBytes, fit: FitResult, landmarks: Landmarks
    ) -> Ok[CompleteUv] | Err[DomainError]:
        from backend.domain import UV_LEN, parse_complete_uv

        _ = (job_id, image, fit, landmarks)
        # Un pixel sentinel (255,0,255) al inicio + resto ceros.
        raw = bytes([255, 0, 255]) + bytes(UV_LEN - 3)
        return parse_complete_uv(raw)


def test_run_pair_fake_ml_ok_sentinel_evidence_gate() -> None:
    """Gate count_sentinel==0 + evidence>=EVIDENCE_MIN sobre salida seam run_pair.

    FakeMlOk (dorado sin sentinel) pasa; hoy el gate no existe y el sentinel
    pasaria silencioso. Fix: run_pair falla loud ante sentinel.
    """
    from backend.domain import new_job_id
    from backend.flame_texture import EVIDENCE_MIN, count_sentinel, texture_evidence

    sink_ok: ProgressSink = InMemorySink()
    job_ok = new_job_id()
    out_ok = run_pair_real(sink_ok, FakeMlOk(), job_ok, _image(MARKER_A), _image(MARKER_B), default_config())  # type: ignore[arg-type]
    assert isinstance(out_ok, Ok)
    for uv in (out_ok.value.uv_a, out_ok.value.uv_b):
        raw = uv.as_bytes()
        assert count_sentinel(raw) == 0
        assert texture_evidence(raw) >= EVIDENCE_MIN


def test_run_pair_sentinel_fails_loud() -> None:
    from backend.domain import new_job_id

    sink: ProgressSink = InMemorySink()
    job_id = new_job_id()
    out = run_pair_real(sink, FakeMlSentinel(), job_id, _image(MARKER_A), _image(MARKER_B), default_config())  # type: ignore[arg-type]
    assert isinstance(out, Err)
    assert isinstance(sink, InMemorySink)
    assert sink.failed


def test_pbr_skin_duplicate_both_modes(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """PBR skin-duplicate ambos modos (DAG v10 HIL dueno).

    Dobles y real duplican el albedo como placeholder honesto: sin mapas
    roughness/metalness reales; los ojos viven en el GLB (2 primitivas
    SkinPBR+EyePBR). TODO mapas reales futuros -> pbr!=uv.
    Este test fija pbr==uv con VULTUS_REAL_ML=0 y con =1 (ambos legs).

    El pipeline (CompareResult) no produce PBR; el zip (local_runner/modal)
    resuelve via `resolve_pbr_pngs`.
    """
    from backend.domain import UV_LEN, parse_complete_uv
    from backend.gnm import uv_to_png
    from backend.pipeline_local import resolve_pbr_pngs

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("FFHQ_UV_DIR", str(empty))
    uv_a = parse_complete_uv(bytes([0xA1, 0xB2]) + bytes(UV_LEN - 2))
    uv_b = parse_complete_uv(bytes([0xB2, 0xA1]) + bytes(UV_LEN - 2))
    assert isinstance(uv_a, Ok)
    assert isinstance(uv_b, Ok)
    # Ambos legs: pbr == uv (misma PNG, placeholder sin mapa real).
    for real_flag in ("0", "1"):
        monkeypatch.setenv("VULTUS_REAL_ML", real_flag)
        pbr_resolved = resolve_pbr_pngs(uv_a.value, uv_b.value)
        assert isinstance(pbr_resolved, Ok)
        assert pbr_resolved.value[0] == uv_to_png(uv_a.value)
        assert pbr_resolved.value[1] == uv_to_png(uv_b.value)


# --- Wave 6-fix Lane A: vectores reales separados a nivel de seam ---
# Lo que run_pair consume del sidecar (fit_flame + bake_flame) con
# vectores reales: margen estricto + albedo sin sentinel + evidence +
# SSIM misma>distinta. Los dobles quedan en sus tests propios arriba.
# Gate: VULTUS_REAL_ML=1 + LANDMARKS_REAL=1 + puente + LFW + MediaPipe;
# sin ellos skip con motivo (el gate prod Lane B los exige en verde).


def test_flame_real_pair_margin_and_albedo_at_seam(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import io as _io
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    if _os.environ.get("VULTUS_REAL_ML") != "1" or _os.environ.get("LANDMARKS_REAL") != "1":
        import pytest

        pytest.skip("vectores reales solo con VULTUS_REAL_ML=1 + LANDMARKS_REAL=1")
    home = _Path.home()
    monkeypatch.setenv("DECA_DIR", _os.environ.get("DECA_DIR", str(home / "Code" / "weights" / "deca")))
    monkeypatch.setenv(
        "FLAME_ASSETS_DIR",
        _os.environ.get("FLAME_ASSETS_DIR", str(home / "Code" / "weights" / "flame")),
    )
    monkeypatch.setenv(
        "FFHQ_UV_DIR",
        _os.environ.get("FFHQ_UV_DIR", str(home / "Code" / "weights" / "ffhq-uv")),
    )
    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.flame_fit import fit_flame, flame_distance, weights_present
    from backend.flame_texture import (
        EVIDENCE_MIN,
        bake_flame,
        count_sentinel,
        texture_evidence,
    )

    if not weights_present():
        import pytest

        pytest.skip("sin puente DECA/FLAME no hay forward real local")
    try:
        import mediapipe as _mp
        from mediapipe.tasks.python import BaseOptions as _BaseOptions
        from mediapipe.tasks.python import vision as _vision
    except ImportError:
        import pytest

        pytest.skip("sin mediapipe no hay landmarks reales")
    import numpy as _np
    from PIL import Image as _Image

    dataset = _Path("/Users/esau.martinez/Code/datasets/lfw")
    paths = {
        "A": dataset / "George_W_Bush" / "George_W_Bush_0001.jpg",
        "B": dataset / "George_W_Bush" / "George_W_Bush_0002.jpg",
        "C": dataset / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg",
    }
    for tag, path in paths.items():
        if not path.is_file():
            import pytest

            pytest.skip(f"sin triple LFW no hay vectores reales ({tag})")
    task = home / "Code" / "weights" / "mediapipe" / "face_landmarker.task"
    if not task.is_file():
        import pytest

        pytest.skip("sin face_landmarker.task no hay landmarks reales")

    images = {}
    landmarks = {}
    for tag, path in paths.items():
        raw = path.read_bytes()
        parsed_img = parse_image_bytes(raw)
        assert isinstance(parsed_img, Ok)
        images[tag] = parsed_img.value
        rgb = _np.asarray(_Image.open(_io.BytesIO(raw)).convert("RGB"))
        opts = _vision.FaceLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(task)),
            running_mode=_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        with _vision.FaceLandmarker.create_from_options(opts) as landmarker:
            result = landmarker.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
        assert result.face_landmarks, "MediaPipe sin cara en foto congelada (gate rojo)"
        parsed_lm = parse_landmarks(
            _json.dumps(
                [[float(p.x), float(p.y), float(p.z)] for p in result.face_landmarks[0]]
            ).encode("utf-8")
        )
        assert isinstance(parsed_lm, Ok)
        landmarks[tag] = parsed_lm.value

    fits = {}
    albedos = {}
    for tag in ("A", "B", "C"):
        fit_out = fit_flame(images[tag], landmarks[tag])
        assert isinstance(fit_out, Ok)
        fits[tag] = fit_out.value
        bake_out = bake_flame(images[tag], fits[tag], landmarks[tag])
        assert isinstance(bake_out, Ok)
        raw = bake_out.value.as_bytes()
        assert count_sentinel(raw) == 0
        assert texture_evidence(raw) >= EVIDENCE_MIN
        albedos[tag] = raw

    rerun = fit_flame(images["A"], landmarks["A"])
    assert isinstance(rerun, Ok)
    d_self = flame_distance(fits["A"].coeffs, rerun.value.coeffs)
    d_same = flame_distance(fits["A"].coeffs, fits["B"].coeffs)
    d_diff = flame_distance(fits["A"].coeffs, fits["C"].coeffs)
    assert d_self == 0.0
    assert d_self < d_same < d_diff
