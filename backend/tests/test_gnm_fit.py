"""Fitter: seam fit_gnm con fit real ridge sobre la cabeza GNM."""

from __future__ import annotations

import json
import math
import time

import pytest

from backend.domain import (
    FIT_TIMEOUT_SECS,
    Err,
    ImageBytes,
    Landmarks,
    Ok,
    parse_image_bytes,
    parse_landmarks,
)
from backend.gnm_fit import (
    _LAST_FIT_STATS,
    _real_fit_available,
    coeff_distance,
    fit_gnm,
    fit_gnm_from_request,
    project,
)


def _image(marker: int) -> ImageBytes:
    raw = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes([marker]) * 56
    parsed = parse_image_bytes(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks() -> Landmarks:
    pts = [[0.1, 0.2, 0.3]] * 478
    raw = json.dumps(pts).encode("utf-8")
    parsed = parse_landmarks(raw)
    assert isinstance(parsed, Ok)
    return parsed.value


def _landmarks_variant(warped: bool) -> Landmarks:
    """478 sinteticos sin pesos: grilla con o sin warp no-lineal en x.

    El warp (x^1.5) no lo absorbe la camara de similaridad del fit real,
    asi que ambas variantes dan coefs distintos en modo real; en modo
    doble (CI sin npz) difieren por hash. Sin dependencia del npz.
    """
    pts: list[list[float]] = []
    for i in range(478):
        gx = (i % 32) / 31.0
        gy = ((i // 32) % 15) / 14.0
        if warped:
            gx = gx**1.5
        pts.append([0.2 + 0.6 * gx, 0.2 + 0.6 * gy, 0.0])
    parsed = parse_landmarks(json.dumps(pts).encode("utf-8"))
    assert isinstance(parsed, Ok)
    return parsed.value


def test_fit_deterministic_real() -> None:
    image = _image(0xA1)
    landmarks = _landmarks()
    first = fit_gnm(image, landmarks)
    second = fit_gnm(image, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.coeffs.as_tuple() == second.value.coeffs.as_tuple()
    assert first.value.camera.as_tuple() == second.value.camera.as_tuple()
    assert len(first.value.coeffs.as_tuple()) == 253
    assert len(first.value.camera.as_tuple()) == 12
    assert all(math.isfinite(v) for v in first.value.coeffs.as_tuple())
    assert all(math.isfinite(v) for v in first.value.camera.as_tuple())


def test_fit_differs_per_identity_and_distance_positive() -> None:
    image = _image(0xA1)
    landmarks_a = _landmarks_variant(False)
    landmarks_b = _landmarks_variant(True)
    ra = fit_gnm(image, landmarks_a)
    rb = fit_gnm(image, landmarks_b)
    assert isinstance(ra, Ok)
    assert isinstance(rb, Ok)
    assert ra.value.coeffs.as_tuple() != rb.value.coeffs.as_tuple()
    assert coeff_distance(ra.value.coeffs, rb.value.coeffs) > 0.0
    assert coeff_distance(ra.value.coeffs, ra.value.coeffs) == 0.0


def test_fit_request_bad_payload_fails_loudly() -> None:
    assert isinstance(fit_gnm_from_request(b"\x00\x01"), Err)
    assert isinstance(fit_gnm_from_request(b""), Err)


def test_fit_p95_inside_fit_timeout() -> None:
    image = _image(0xA1)
    landmarks = _landmarks_variant(False)
    durations: list[float] = []
    for _ in range(21):
        start = time.perf_counter()
        out = fit_gnm(image, landmarks)
        durations.append(time.perf_counter() - start)
        assert isinstance(out, Ok)
    durations.sort()
    p95 = durations[int(0.95 * (len(durations) - 1))]
    # Gate: el fit real debe caber en FIT_TIMEOUT_SECS; en local corre en ms.
    assert p95 < FIT_TIMEOUT_SECS


def test_fit_uses_real_head_basis() -> None:
    if not _real_fit_available():
        pytest.skip("sin pesos GNM: no hay base real que verificar")
    from backend.gnm_head import eval_landmarks68, eval_mesh, load_gnm_head

    head = load_gnm_head()
    mesh = eval_mesh(tuple([0.0] * 253))
    template = head.template_positions.tolist()
    assert len(mesh) == len(template)
    for row_m, row_t in zip(mesh, template):
        for vm, vt in zip(row_m, row_t):
            assert abs(float(vm) - float(vt)) < 1e-4
    zeros = eval_landmarks68(tuple([0.0] * 253))
    assert len(zeros) == 68
    landmarks = _landmarks_variant(False)
    out = fit_gnm(_image(0xA1), landmarks)
    assert isinstance(out, Ok)
    coefs = out.value.coeffs.as_tuple()
    assert len(coefs) == 253
    assert all(math.isfinite(v) for v in coefs)
    assert any(abs(v) > 1e-6 for v in coefs)
    grid = fit_gnm(_image(0xA1), _landmarks())
    assert isinstance(grid, Ok)
    assert coeff_distance(out.value.coeffs, grid.value.coeffs) > 0.0
    assert _LAST_FIT_STATS["iterations"] == 3.0
    assert math.isfinite(_LAST_FIT_STATS["loss"]) and _LAST_FIT_STATS["loss"] >= 0.0
    assert _LAST_FIT_STATS["duration_ms"] > 0.0


def test_project_identity_maps_xy_hand_literals() -> None:
    import numpy as np

    from backend.domain import parse_camera_params

    parsed = parse_camera_params([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert isinstance(parsed, Ok)
    pts = np.asarray([[0.2, 0.3, 0.5], [0.7, 0.1, -0.2]], dtype=np.float64)
    out = project(parsed.value, pts)
    assert out.shape == (2, 2)
    assert abs(float(out[0, 0]) - 0.2) < 1e-12
    assert abs(float(out[0, 1]) - 0.3) < 1e-12
    assert abs(float(out[1, 0]) - 0.7) < 1e-12
    assert abs(float(out[1, 1]) - 0.1) < 1e-12


def test_project_scale_translate_hand_literals() -> None:
    import numpy as np

    from backend.domain import parse_camera_params

    parsed = parse_camera_params([2.0, 0.0, 0.0, 0.1, 0.0, 2.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    assert isinstance(parsed, Ok)
    pts = np.asarray([[0.5, 0.25, 1.0]], dtype=np.float64)
    out = project(parsed.value, pts)
    assert abs(float(out[0, 0]) - 1.1) < 1e-12
    assert abs(float(out[0, 1]) - 0.7) < 1e-12


def test_estimate_camera_recovers_opposite_y_sign() -> None:
    import numpy as np

    from backend.gnm_fit import _estimate_camera

    rng = np.random.default_rng(7)
    pred = rng.uniform(-0.07, 0.07, size=(68, 3))
    pred[:, 1] += 0.25
    targets = np.stack([2.0 * pred[:, 0] + 0.1, -2.0 * pred[:, 1] + 0.9], axis=1)
    sx, tx, sy, ty = _estimate_camera(pred, targets)
    assert abs(sx - 2.0) < 1e-9
    assert abs(tx - 0.1) < 1e-9
    assert abs(sy + 2.0) < 1e-9
    assert abs(ty - 0.9) < 1e-9
    assert sx > 0.0
    assert sy < 0.0


# --- Wave 2 Step 3: fit FLAME feed-forward (replay determinista local) ---
# Paridad firmada no-visual: mismo input dos veces da mismos bytes, hash
# difiere por identidad en el doble local (d(A,A)=0 < d(A,B)). El doble local
# no tiene margen de identidad; el margen real DECA es gate prod Wave 6.
# Lo visual solo se valida en Modal.


def test_flame_fit_deterministic_bytes_identical_x2() -> None:
    from backend.flame_fit import fit_flame
    from backend.pipeline_local import encode_fit_result

    image = _image(0xA1)
    landmarks = _landmarks()
    first = fit_flame(image, landmarks)
    second = fit_flame(image, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.coeffs.as_tuple() == second.value.coeffs.as_tuple()
    assert first.value.camera.as_tuple() == second.value.camera.as_tuple()
    assert len(first.value.coeffs.as_tuple()) == 253
    assert len(first.value.camera.as_tuple()) == 12
    assert all(math.isfinite(v) for v in first.value.coeffs.as_tuple())
    assert all(math.isfinite(v) for v in first.value.camera.as_tuple())
    assert encode_fit_result(first.value) == encode_fit_result(second.value)


def test_flame_fit_differs_per_identity_and_hash_differs() -> None:
    from backend.flame_fit import fit_flame, flame_distance

    image = _image(0xA1)
    landmarks_a = _landmarks_variant(False)
    landmarks_b = _landmarks_variant(True)
    ra = fit_flame(image, landmarks_a)
    rb = fit_flame(image, landmarks_b)
    assert isinstance(ra, Ok)
    assert isinstance(rb, Ok)
    assert ra.value.coeffs.as_tuple() != rb.value.coeffs.as_tuple()
    assert flame_distance(ra.value.coeffs, rb.value.coeffs) > 0.0
    assert flame_distance(ra.value.coeffs, ra.value.coeffs) == 0.0


def test_flame_fit_request_bad_payload_fails_loudly() -> None:
    from backend.flame_fit import fit_flame_from_request

    assert isinstance(fit_flame_from_request(b"\x00\x01"), Err)
    assert isinstance(fit_flame_from_request(b""), Err)


def test_flame_fit_request_v2_roundtrip_ok() -> None:
    from backend.domain import encode_fit_request
    from backend.flame_fit import fit_flame_from_request

    blob = encode_fit_request(_image(0xA1), _landmarks())
    assert isinstance(fit_flame_from_request(blob), Ok)


def test_flame_fit_real_required_without_weights_fails_loudly(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from backend.domain import domain_to_status
    from backend.flame_fit import fit_flame

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("VULTUS_REAL_ML", "1")
    monkeypatch.setenv("DECA_DIR", str(empty))
    monkeypatch.setenv("FLAME_ASSETS_DIR", str(empty))
    result = fit_flame(_image(0xA1), _landmarks())
    assert isinstance(result, Err)
    assert domain_to_status(result.error) == 500


def test_flame_fit_p95_inside_fit_timeout() -> None:
    from backend.domain import FIT_TIMEOUT_SECS
    from backend.flame_fit import fit_flame

    image = _image(0xA1)
    landmarks = _landmarks_variant(False)
    durations: list[float] = []
    for _ in range(21):
        start = time.perf_counter()
        out = fit_flame(image, landmarks)
        durations.append(time.perf_counter() - start)
        assert isinstance(out, Ok)
    durations.sort()
    p95 = durations[int(0.95 * (len(durations) - 1))]
    assert p95 < FIT_TIMEOUT_SECS


# --- Wave 6-fix Lane A: vectores reales (forward DECA) vs dobles ---
# Dobles (sin VULTUS_REAL_ML/LANDMARKS_REAL): solo propiedades
# deterministicas (x2 identico, d(A,A)==0<d(A,B) por hash, sin margen de
# identidad). Reales (VULTUS_REAL_ML=1 + LANDMARKS_REAL=1 + puente
# canonico + triple LFW + MediaPipe): margen estricto
# d(A,A)==0 < d(A,Bmisma) < d(A,Cdistinta) con fotos congeladas.
# Sin flags o sin assets los tests reales hacen skip con motivo; el gate
# prod (scripts/e2e-flame-real.py, Lane B) los exige en verde.


def _real_bridge_dirs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os as _os
    from pathlib import Path as _Path

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


def _require_real_fit_vectors(monkeypatch) -> dict[str, bytes]:  # type: ignore[no-untyped-def]
    """Activa el puente y devuelve el triple LFW, o hace skip con motivo."""
    import os as _os
    from pathlib import Path as _Path

    if _os.environ.get("VULTUS_REAL_ML") != "1" or _os.environ.get("LANDMARKS_REAL") != "1":
        pytest.skip("vectores reales solo con VULTUS_REAL_ML=1 + LANDMARKS_REAL=1")
    _real_bridge_dirs(monkeypatch)
    from backend.flame_fit import weights_present

    if not weights_present():
        pytest.skip("sin puente DECA/FLAME canonico no hay forward real local")
    dataset = _Path("/Users/esau.martinez/Code/datasets/lfw")
    paths = {
        "A": dataset / "George_W_Bush" / "George_W_Bush_0001.jpg",
        "B": dataset / "George_W_Bush" / "George_W_Bush_0002.jpg",
        "C": dataset / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg",
    }
    for tag, path in paths.items():
        if not path.is_file():
            pytest.skip(f"sin triple LFW no hay vectores reales ({tag})")
    return {tag: path.read_bytes() for tag, path in paths.items()}


def _real_landmarks_for(jpg: bytes) -> Landmarks:
    """Landmarks MediaPipe reales 478 sobre JPEG, o skip/fallo loud."""
    import io as _io

    try:
        import mediapipe as _mp
        from mediapipe.tasks.python import BaseOptions as _BaseOptions
        from mediapipe.tasks.python import vision as _vision
    except ImportError:
        pytest.skip("sin mediapipe no hay landmarks reales")
    from pathlib import Path as _Path

    import numpy as _np
    from PIL import Image as _Image

    task = _Path.home() / "Code" / "weights" / "mediapipe" / "face_landmarker.task"
    if not task.is_file():
        pytest.skip("sin face_landmarker.task no hay landmarks reales")
    rgb = _np.asarray(_Image.open(_io.BytesIO(jpg)).convert("RGB"))
    opts = _vision.FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(task)),
        running_mode=_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with _vision.FaceLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
    assert result.face_landmarks, "MediaPipe sin cara en foto congelada (gate rojo)"
    face = result.face_landmarks[0]
    assert len(face) == 478, f"se esperaban 478 puntos, llegaron {len(face)}"
    parsed = parse_landmarks(
        json.dumps([[float(p.x), float(p.y), float(p.z)] for p in face]).encode("utf-8")
    )
    assert isinstance(parsed, Ok)
    return parsed.value


def test_flame_weights_require_canonical_bridge_files(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Junk files no habilitan el real: solo deca_model.tar + pkl FLAME."""
    from backend.flame_fit import weights_present

    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "random.txt").write_text("junk")
    (junk / "model.bin").write_bytes(bytes(64))
    monkeypatch.setenv("DECA_DIR", str(junk))
    monkeypatch.setenv("FLAME_ASSETS_DIR", str(junk))
    assert weights_present() is False


def test_flame_real_with_junk_weights_fails_loud(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """REAL=1 con junk (sin canonicos) falla loud 500, nunca doble."""
    from backend.domain import domain_to_status
    from backend.flame_fit import fit_flame

    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "random.txt").write_text("junk")
    monkeypatch.setenv("VULTUS_REAL_ML", "1")
    monkeypatch.setenv("DECA_DIR", str(junk))
    monkeypatch.setenv("FLAME_ASSETS_DIR", str(junk))
    result = fit_flame(_image(0xA1), _landmarks())
    assert isinstance(result, Err)
    assert domain_to_status(result.error) == 500


def test_flame_real_fit_margin_strict_with_real_vectors(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Margen estricto real: d(A,A)==0 < d(A,Bmisma) < d(A,Cdistinta)."""
    import time as _time

    from backend.domain import FIT_TIMEOUT_SECS
    from backend.flame_fit import fit_flame, flame_distance

    raws = _require_real_fit_vectors(monkeypatch)
    images = {}
    for tag, raw in raws.items():
        parsed = parse_image_bytes(raw)
        assert isinstance(parsed, Ok)
        images[tag] = parsed.value
    landmarks = {tag: _real_landmarks_for(raw) for tag, raw in raws.items()}
    start = _time.perf_counter()
    fits = {}
    for tag in ("A", "B", "C"):
        out = fit_flame(images[tag], landmarks[tag])
        assert isinstance(out, Ok)
        assert len(out.value.coeffs.as_tuple()) == 253
        assert len(out.value.camera.as_tuple()) == 12
        assert all(math.isfinite(v) for v in out.value.coeffs.as_tuple())
        fits[tag] = out.value
    assert _time.perf_counter() - start < FIT_TIMEOUT_SECS
    rerun = fit_flame(images["A"], landmarks["A"])
    assert isinstance(rerun, Ok)
    d_self = flame_distance(fits["A"].coeffs, rerun.value.coeffs)
    d_same = flame_distance(fits["A"].coeffs, fits["B"].coeffs)
    d_diff = flame_distance(fits["A"].coeffs, fits["C"].coeffs)
    assert d_self == 0.0
    assert d_self < d_same < d_diff


def test_flame_real_fit_deterministic_x2_with_real_vectors(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Forward real repetible: mismo input dos veces da mismos bytes."""
    from backend.flame_fit import fit_flame
    from backend.pipeline_local import encode_fit_result

    raws = _require_real_fit_vectors(monkeypatch)
    parsed = parse_image_bytes(raws["A"])
    assert isinstance(parsed, Ok)
    landmarks = _real_landmarks_for(raws["A"])
    first = fit_flame(parsed.value, landmarks)
    second = fit_flame(parsed.value, landmarks)
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.coeffs.as_tuple() == second.value.coeffs.as_tuple()
    assert first.value.camera.as_tuple() == second.value.camera.as_tuple()
    assert encode_fit_result(first.value) == encode_fit_result(second.value)


def test_deep3d_fingerprint_empty_without_weights(monkeypatch, tmp_path) -> None:
    """Deep3D fingerprint vacio sin pesos, total sin raise."""
    from backend import flame_fit as _ff
    from backend.domain import Ok as _Ok

    monkeypatch.setenv("DEEP3D_DIR", str(tmp_path / "vacio"))
    monkeypatch.setenv("DECA_DIR", str(tmp_path / "vacio"))
    monkeypatch.setenv("WEIGHTS_ROOT", str(tmp_path / "vacio"))
    monkeypatch.setenv("WEIGHTS_DIR", "")
    res = _ff.deep3d_fingerprint()
    assert isinstance(res, _Ok)
    assert res.value == b""


def test_deep3d_fingerprint_consumes_epoch(monkeypatch, tmp_path) -> None:
    """Con epoch + detector, fingerprint no vacio y estable."""
    from backend import flame_fit as _ff
    from backend.domain import Ok as _Ok

    d = tmp_path / "deep3d"
    d.mkdir()
    (d / "epoch_latest.pth").write_bytes(b"fake-epoch-weights-12345")
    (d / "68lm_detector.pb").write_bytes(b"fake-detector-67890")
    monkeypatch.setenv("DEEP3D_DIR", str(d))
    monkeypatch.setenv("DECA_DIR", "")
    monkeypatch.setenv("WEIGHTS_ROOT", str(tmp_path))
    monkeypatch.setenv("WEIGHTS_DIR", "")
    first = _ff.deep3d_fingerprint()
    second = _ff.deep3d_fingerprint()
    assert isinstance(first, _Ok)
    assert first.value != b""
    assert first.value == second.value


def _require_real_fit_vectors(monkeypatch):  # type: ignore[no-untyped-def]
    """Puente DECA/FLAME + triple LFW + MediaPipe, o skip con motivo."""
    import os as _os
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parent.parent.parent
    monkeypatch.setenv("DECA_DIR", _os.environ.get("DECA_DIR", str(repo / "weights" / "deca")))
    monkeypatch.setenv("FLAME_ASSETS_DIR", _os.environ.get("FLAME_ASSETS_DIR", str(repo / "weights" / "flame")))
    from backend.flame_fit import weights_present as _wp

    if not _wp():
        import pytest

        pytest.skip("sin puente DECA/FLAME no hay fit real local")
    dataset = _Path("/Users/esau.martinez/Code/datasets/lfw")
    tags = {
        "A": dataset / "George_W_Bush" / "George_W_Bush_0001.jpg",
        "B": dataset / "George_W_Bush" / "George_W_Bush_0002.jpg",
        "C": dataset / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg",
    }
    for tag, path in tags.items():
        if not path.is_file():
            import pytest as _pytest

            _pytest.skip(f"sin triple LFW ({tag})")
    return {tag: path.read_bytes() for tag, path in tags.items()}


def test_fit_68_ratios_discriminate_identity_real(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """GWT1 RED: ratios 68 mismos discriminan (misma < distinta) con landmarks reales."""
    import json as _json

    try:
        import mediapipe as _mp
        from mediapipe.tasks import python as _base
        from mediapipe.tasks.python import vision as _vis
    except ImportError:
        import pytest as _pytest

        _pytest.skip("sin mediapipe no hay landmarks reales")

    from backend.domain import Ok as _Ok
    from backend.domain import parse_landmarks as _plm
    from backend.flame_fit import identity_ratios_68

    raws = _require_real_fit_vectors(monkeypatch)
    task = "/Users/esau.martinez/Code/weights/mediapipe/face_landmarker.task"
    opts = _vis.FaceLandmarkerOptions(base_options=_base.BaseOptions(model_asset_path=task), num_faces=1)
    import io as _io

    import numpy as _np
    from PIL import Image as _Image

    lms = {}
    with _vis.FaceLandmarker.create_from_options(opts) as landmarker:
        for tag, raw in raws.items():
            rgb = _np.asarray(_Image.open(_io.BytesIO(raw)).convert("RGB"))
            res = landmarker.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
            pts = [[float(q.x), float(q.y), float(q.z)] for q in res.face_landmarks[0]]
            parsed = _plm(_json.dumps(pts).encode())
            assert isinstance(parsed, _Ok)
            lms[tag] = parsed.value
    import math as _math

    def _d(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return _math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    ra = identity_ratios_68(lms["A"])
    rb = identity_ratios_68(lms["B"])
    rc = identity_ratios_68(lms["C"])
    assert isinstance(ra, _Ok) and isinstance(rb, _Ok) and isinstance(rc, _Ok)
    assert _d(ra.value, rb.value) < _d(ra.value, rc.value)


def test_fit_real_loss_is_computed_symmetry_not_zero(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """GWT1 RED: loss real es residual geometrico finito, no 0 fijo."""
    import json as _json

    try:
        import mediapipe as _mp
        from mediapipe.tasks import python as _base
        from mediapipe.tasks.python import vision as _vis
    except ImportError:
        import pytest as _pytest

        _pytest.skip("sin mediapipe no hay landmarks reales")

    from backend.domain import Ok as _Ok
    from backend.domain import parse_image_bytes as _pimg
    from backend.domain import parse_landmarks as _plm
    from backend.flame_fit import _LAST_FIT_STATS, fit_flame

    raws = _require_real_fit_vectors(monkeypatch)
    task = "/Users/esau.martinez/Code/weights/mediapipe/face_landmarker.task"
    opts = _vis.FaceLandmarkerOptions(base_options=_base.BaseOptions(model_asset_path=task), num_faces=1)
    import io as _io

    import numpy as _np
    from PIL import Image as _Image

    with _vis.FaceLandmarker.create_from_options(opts) as landmarker:
        raw = raws["A"]
        rgb = _np.asarray(_Image.open(_io.BytesIO(raw)).convert("RGB"))
        res = landmarker.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb))
        pts = [[float(q.x), float(q.y), float(q.z)] for q in res.face_landmarks[0]]
        img = _pimg(raw)
        lms = _plm(_json.dumps(pts).encode())
        assert isinstance(img, _Ok) and isinstance(lms, _Ok)
        out = fit_flame(img.value, lms.value)
        assert isinstance(out, _Ok)
    assert _LAST_FIT_STATS["iterations"] == 1.0
    assert _LAST_FIT_STATS["loss"] > 0.0
