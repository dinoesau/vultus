"""Seam HQ-512: bake_hq nunca cae a fallback, falla loud sin pesos.

CI-safe (sin torch, sin pesos): gates obligatorios, totality de la pose
perspectiva, pureza de helpers y paridad del raster forense.
"""

from __future__ import annotations

import io
import json
import pathlib

import pytest

from backend.domain import Ok


class _InMemorySink:
    """Sink minimo: registra etapas y guarda el resultado."""

    def __init__(self) -> None:
        self.reports: list = []
        self.result = None
        self.failed = False

    def report(self, progress, stage):  # type: ignore[no-untyped-def]
        self.reports.append((progress, stage))
        return Ok(None)

    def complete(self, result):  # type: ignore[no-untyped-def]
        self.result = result
        return Ok(None)

    def fail(self):  # type: ignore[no-untyped-def]
        self.failed = True
        return Ok(None)


def _synthetic_landmarks():
    from backend.domain import Err as _Err
    from backend.domain import parse_landmarks

    pts = [
        [0.2 + 0.6 * ((i * 37) % 100) / 100.0, 0.2 + 0.6 * ((i * 53) % 100) / 100.0, 0.0]
        for i in range(478)
    ]
    parsed = parse_landmarks(json.dumps(pts).encode())
    assert not isinstance(parsed, _Err)
    return parsed.value


def _synthetic_image():
    from PIL import Image as _Image

    from backend.domain import Err as _Err
    from backend.domain import parse_image_bytes

    img = _Image.new("RGB", (256, 256), (180, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    parsed = parse_image_bytes(buf.getvalue())
    assert not isinstance(parsed, _Err)
    return parsed.value


def test_bake_hq_fails_loud_without_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    from backend.domain import Err
    from backend.flame_fit import fit_flame
    from backend.flame_texture import _LAST_TEXTURE_STATS, bake_hq

    empty = tmp_path / "empty"
    empty.mkdir()
    for var in ("FFHQ_UV_DIR", "TOPO_DIR", "TEXGAN_DIR", "PARSING_DIR", "DPR_DIR", "WEIGHTS_ROOT", "WEIGHTS_DIR"):
        monkeypatch.setenv(var, str(empty))
    image = _synthetic_image()
    landmarks = _synthetic_landmarks()
    fit = fit_flame(image, landmarks)
    assert not isinstance(fit, Err)
    out = bake_hq(image, fit.value, landmarks)
    assert isinstance(out, Err)
    assert "hq" in str(out.error)
    assert float(_LAST_TEXTURE_STATS.get("parsing", 0.0)) == 0.0


def test_bake_hq_requires_parsing_weights_loud(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    from backend.domain import Err, Ok
    from backend.flame_fit import fit_flame
    from backend.flame_texture import bake_hq

    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "FLAME_w_HIFI3D_UV.obj").write_bytes(b"obj-stub")
    (bridge / "eye_ball_tex.png").write_bytes(b"png-stub")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("FFHQ_UV_DIR", str(bridge))
    for var in ("TOPO_DIR", "TEXGAN_DIR", "PARSING_DIR", "DPR_DIR", "WEIGHTS_ROOT", "WEIGHTS_DIR"):
        monkeypatch.setenv(var, str(empty))
    # El mat de unwrap es colaborador, no el gate bajo test: se asume
    # presente para aislar el gate de parsing (white-box del repo).
    import backend.flame_texture as _ft

    monkeypatch.setattr(_ft, "_require_unwrap_mat", lambda: Ok("stub-mat"))
    image = _synthetic_image()
    landmarks = _synthetic_landmarks()
    fit = fit_flame(image, landmarks)
    assert not isinstance(fit, Err)
    out = bake_hq(image, fit.value, landmarks)
    # Sin parsing es Err loud con la causa: jamas unwrap sin mascara.
    assert isinstance(out, Err)
    assert "parsing" in str(out.error)


def test_dilate_mask_grows_cross() -> None:
    import numpy as np

    from backend.flame_texture import _dilate_mask

    base = np.zeros((5, 5), dtype=bool)
    base[2, 2] = True
    assert _dilate_mask(base, 0).sum() == 1
    grown = _dilate_mask(base, 1)
    assert grown.sum() == 5
    assert grown[2, 2] and grown[1, 2] and grown[3, 2] and grown[2, 1] and grown[2, 3]
    assert not grown[0, 0]
    assert _dilate_mask(base, 2).sum() == 13


def test_perspective_vert_px_degenerate_is_err_not_raise() -> None:
    from backend.domain import Err
    from backend.flame_texture import _perspective_vert_px

    landmarks = _synthetic_landmarks()
    out = _perspective_vert_px([(0.0, 0.0, 0.0)] * 5023, landmarks, 256, 256)
    assert isinstance(out, Err)


def _mirrored_landmarks(width: int, height: int, mirror: bool):  # type: ignore[no-untyped-def]
    """Landmarks 478 sinteticos desde el template con afine exacto (espejo opcional).

    Proyecta los 50 puntos del embed con un afine conocido y los planta en
    los indices MediaPipe de MP_68_MAP: el ajuste afine converge con RMSE ~0
    y el espejo da escala negativa (caso Obama B en prod).
    """
    import numpy as np

    from backend.domain import parse_landmarks
    from backend.gnm_assemble import flame68_positions, load_flame_template
    from backend.gnm_head import MP_68_MAP

    loaded = load_flame_template()
    assert isinstance(loaded, Ok)
    positions = loaded.value[0]
    displaced = [(float(p[0]), float(p[1]), float(p[2])) for p in positions]
    placed = flame68_positions(displaced)
    assert isinstance(placed, Ok)
    dlib_rows, obj3d = placed.value
    obj = np.asarray(obj3d, dtype=np.float64)
    ox = (obj[:, 0] - obj[:, 0].min()) / max(1e-9, obj[:, 0].max() - obj[:, 0].min())
    oy = (obj[:, 1] - obj[:, 1].min()) / max(1e-9, obj[:, 1].max() - obj[:, 1].min())
    fx = (0.75 - 0.5 * ox) if mirror else (0.25 + 0.5 * ox)
    fy = 0.25 + 0.5 * oy
    pts = np.full((478, 3), 0.5, dtype=np.float64)
    for k, j in enumerate([int(i) for i in dlib_rows]):
        pts[int(MP_68_MAP[int(j)]), 0] = float(fx[k])
        pts[int(MP_68_MAP[int(j)]), 1] = float(fy[k])
        pts[int(MP_68_MAP[int(j)]), 2] = 0.0
    parsed = parse_landmarks(json.dumps(pts.tolist()).encode())
    assert isinstance(parsed, Ok)
    return displaced, parsed.value


def test_perspective_mirror_falls_back_to_affine_not_err() -> None:
    """Regression prod (job HQ Obama B): escala negativa no es Err.

    El afine absorbe el flip y converge; el pinhole rotacion-pura no puede.
    HQ usa el afine con persp=0: nunca alinea peor que el forense.
    """
    import numpy as np

    from backend.domain import Ok
    from backend.flame_texture import _LAST_TEXTURE_STATS, _perspective_vert_px

    displaced, landmarks = _mirrored_landmarks(256, 256, mirror=True)
    out = _perspective_vert_px(displaced, landmarks, 256, 256)
    assert isinstance(out, Ok)
    vert_px, used_persp = out.value
    assert used_persp is False
    assert vert_px.shape == (5023, 2)
    assert bool(np.isfinite(vert_px).all())
    assert float(_LAST_TEXTURE_STATS.get("persp", -1.0)) == 0.0


def test_perspective_frontal_stays_ok() -> None:
    """Control positivo: sin espejo el seam sigue total (afine o pinhole)."""
    import numpy as np

    from backend.domain import Ok
    from backend.flame_texture import _perspective_vert_px

    displaced, landmarks = _mirrored_landmarks(256, 256, mirror=False)
    out = _perspective_vert_px(displaced, landmarks, 256, 256)
    assert isinstance(out, Ok)
    vert_px, _ = out.value
    assert vert_px.shape == (5023, 2)
    assert bool(np.isfinite(vert_px).all())


def test_rasterize_size_param_keeps_forensic_identical() -> None:
    import numpy as np

    from backend import flame_texture
    from backend.flame_texture import _rasterize_skin_uv

    flame_texture._raster_cache = None
    uvs = [(0.1 + 0.8 * ((i * 13) % 50) / 50.0, 0.1 + 0.8 * ((i * 29) % 50) / 50.0) for i in range(60)]
    tris = [(i, (i + 1) % 60, (i + 2) % 60) for i in range(0, 60, 3)]
    idx_default, bw_default = _rasterize_skin_uv(uvs, tris, None)
    flame_texture._raster_cache = None
    idx_sized, bw_sized = _rasterize_skin_uv(uvs, tris, None, 512)
    assert idx_default.shape == (512, 512, 3)
    np.testing.assert_array_equal(idx_default, idx_sized)
    np.testing.assert_array_equal(bw_default, bw_sized)
    flame_texture._raster_cache = None
    idx_big, bw_big = _rasterize_skin_uv(uvs, tris, None, 1024)
    assert idx_big.shape == (1024, 1024, 3)
    assert bw_big.shape == (1024, 1024, 3)
    flame_texture._raster_cache = None


class _RecordingMl:
    """Fake que graba que pata de textura usa run_pair."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def landmarks(self, job_id, image):  # type: ignore[no-untyped-def]
        from backend.domain import parse_landmarks

        _ = (job_id, image)
        return parse_landmarks(_synthetic_landmarks().as_bytes())

    def fit(self, job_id, image, landmarks):  # type: ignore[no-untyped-def]
        from backend.gnm_fit import fit_gnm

        _ = (job_id, image, landmarks)
        return fit_gnm(image, landmarks)

    def _golden(self, image):  # type: ignore[no-untyped-def]
        from backend.domain import UV_LEN, parse_complete_uv

        marker = bytes([10, 200]) if image.as_bytes()[8] == 0xA1 else bytes([4, 210])
        return parse_complete_uv(bytes(marker) + bytes(UV_LEN - len(marker)))

    def texture(self, job_id, image, fit, landmarks):  # type: ignore[no-untyped-def]
        self.calls.append("texture")
        _ = (job_id, fit, landmarks)
        return self._golden(image)

    def texture_hq(self, job_id, image, fit, landmarks):  # type: ignore[no-untyped-def]
        self.calls.append("texture_hq")
        _ = (job_id, fit, landmarks)
        return self._golden(image)


def _harness_images():  # type: ignore[no-untyped-def]
    from backend.domain import parse_image_bytes

    def img(marker: int):  # type: ignore[no-untyped-def]
        raw = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes([marker]) * 56
        parsed = parse_image_bytes(raw)
        assert isinstance(parsed, Ok)
        return parsed.value

    return img(0xA1), img(0xB2)


def test_run_pair_hq_config_dispatches_texture_hq() -> None:
    from backend.domain import Ok, new_job_id
    from backend.pipeline_local import hq_config, run_pair

    sink = _InMemorySink()
    fake = _RecordingMl()
    image_a, image_b = _harness_images()
    out = run_pair(sink, fake, new_job_id(), image_a, image_b, hq_config())  # type: ignore[arg-type]
    assert isinstance(out, Ok)
    assert fake.calls == ["texture_hq", "texture_hq"]
    stages = [stage.as_str() for _, stage in sink.reports]
    assert stages == ["fit", "texture", "assemble"]
    assert not sink.failed


def test_run_pair_default_config_dispatches_texture() -> None:
    from backend.domain import Ok, new_job_id
    from backend.pipeline_local import default_config, run_pair

    sink = _InMemorySink()
    fake = _RecordingMl()
    image_a, image_b = _harness_images()
    out = run_pair(sink, fake, new_job_id(), image_a, image_b, default_config())  # type: ignore[arg-type]
    assert isinstance(out, Ok)
    assert fake.calls == ["texture", "texture"]
    assert not sink.failed


def test_hq_config_timeouts() -> None:
    from backend.pipeline_local import default_config, hq_config

    hq = hq_config()
    base = default_config()
    assert hq.texture_timeout_secs == 900.0
    assert hq.total_timeout_secs == 3600.0
    assert hq.landmarks_timeout_secs == base.landmarks_timeout_secs
    assert hq.fit_timeout_secs == base.fit_timeout_secs
