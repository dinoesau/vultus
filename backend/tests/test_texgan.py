"""Fase 1 (issue 94): completion neuronal texgan solo en texeles no validos.

RED: este modulo aun no existe (backend/texgan.py). Los tests CI-safe
(sin torch, sin pesos) cubren consts del decoder, validacion de keys del
checkpoint (puro python sobre listas), blend enmascarado numpy y
determinismo de semillas. El test de carga estricta real solo corre con
torch + pth presentes (local/Modal), CI lo salta.
"""

from __future__ import annotations

import os
import pathlib

import pytest


def test_texgan_decoder_consts() -> None:
    from backend.texgan import (
        TEXGAN_FIT_STEPS,
        TEXGAN_NUM_WS,
        TEXGAN_UV_NATIVE,
        TEXGAN_W_DIM,
    )

    assert TEXGAN_W_DIM == 512
    assert TEXGAN_NUM_WS == 18
    assert TEXGAN_UV_NATIVE == 1024
    assert 1 <= TEXGAN_FIT_STEPS <= 100


def test_texgan_expected_keys_match_real_checkpoint_layout() -> None:
    from backend.texgan import expected_state_keys

    keys = expected_state_keys()
    assert len(keys) == 182
    assert "synthesis.b4.const" in keys
    assert "synthesis.b1024.torgb.bias" in keys
    assert "mapping.fc7.bias" in keys
    assert "mapping.w_avg" in keys


def test_texgan_check_state_keys_pure() -> None:
    from backend.texgan import check_state_keys, expected_state_keys

    full = expected_state_keys()
    missing, unexpected = check_state_keys(full)
    assert missing == frozenset() and unexpected == frozenset()
    dropped = set(full)
    dropped.remove("synthesis.b4.const")
    missing2, _ = check_state_keys(dropped)
    assert missing2 == frozenset({"synthesis.b4.const"})
    _, unexpected2 = check_state_keys(full | {"bogus.key"})
    assert unexpected2 == frozenset({"bogus.key"})


def test_texgan_blend_completion_masked() -> None:
    from backend.texgan import blend_completion

    n = 512 * 512
    sampled = bytes([10, 20, 30]) * n
    synth = bytes([200, 210, 220]) * n
    valid = bytearray(n)
    valid[0] = 1
    valid[n // 2] = 1
    out = blend_completion(sampled, synth, bytes(valid))
    assert len(out) == n * 3
    assert out[0:3] == bytes([10, 20, 30])
    assert out[3:6] == bytes([200, 210, 220])
    mid = (n // 2) * 3
    assert out[mid : mid + 3] == bytes([10, 20, 30])
    assert out == blend_completion(sampled, synth, bytes(valid))


def test_texgan_blend_completion_rejects_bad_lengths() -> None:
    from backend.texgan import blend_completion

    n = 512 * 512
    with pytest.raises(ValueError):
        blend_completion(b"\x00" * 10, bytes([1, 2, 3]) * n, bytes(n))
    with pytest.raises(ValueError):
        blend_completion(bytes([1, 2, 3]) * n, bytes([1, 2, 3]) * n, b"\x01" * 10)


def _real_pth_candidates() -> list[str]:
    cands: list[str] = []
    for env in ("TEXGAN_DIR", "FFHQ_UV_DIR", "WEIGHTS_ROOT", "WEIGHTS_DIR"):
        base = os.environ.get(env, "").strip()
        if base:
            cands.append(base)
    cands.append("/weights")
    cands.append(os.path.expanduser("~/Code/weights"))
    cands.append("/Users/esau.martinez/code/weights")
    return cands


def _find_real_pth() -> str | None:
    for base in _real_pth_candidates():
        for sub in ("checkpoints/texgan_model", "."):
            cand = os.path.join(base, sub, "texgan_ffhq_uv.pth")
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
    return None


def _synthetic_inputs() -> tuple[object, object, object]:
    import json

    from PIL import Image

    from backend.domain import parse_image_bytes, parse_landmarks
    from backend.flame_fit import fit_flame

    img = Image.new("RGB", (256, 256), (180, 150, 130))
    px = img.load()
    assert px is not None
    for y in range(60, 200):
        for x in range(60, 200):
            px[x, y] = (200, 170, 150)
    import io as _io

    buf = _io.BytesIO()
    img.save(buf, format="JPEG")
    parsed_img = parse_image_bytes(buf.getvalue())
    from backend.domain import Err as _Err

    assert not isinstance(parsed_img, _Err)
    pts = [[0.2 + 0.6 * ((i * 37) % 100) / 100.0, 0.2 + 0.6 * ((i * 53) % 100) / 100.0, 0.0] for i in range(478)]
    parsed_lm = parse_landmarks(json.dumps(pts).encode())
    assert not isinstance(parsed_lm, _Err)
    fit = fit_flame(parsed_img.value, parsed_lm.value)
    assert not isinstance(fit, _Err)
    return parsed_img.value, fit.value, parsed_lm.value


def _mirror_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    mirror = "/Users/esau.martinez/code/weights"
    monkeypatch.setenv("FFHQ_UV_DIR", f"{mirror}/ffhq-uv")
    monkeypatch.setenv("TOPO_DIR", f"{mirror}/ffhq-uv-hf/topo_assets")
    monkeypatch.setenv("TEXGAN_DIR", f"{mirror}/ffhq-uv-hf/checkpoints/texgan_model")


def test_texgan_unwrap_falls_back_without_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    import numpy as np

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("FFHQ_UV_DIR", str(empty))
    monkeypatch.setenv("TOPO_DIR", str(empty))
    monkeypatch.setenv("TEXGAN_DIR", str(empty))
    monkeypatch.delenv("WEIGHTS_ROOT", raising=False)
    monkeypatch.delenv("WEIGHTS_DIR", raising=False)
    from backend import flame_texture

    assert flame_texture.weights_present() is False
    out = flame_texture._texgan_completion(np.zeros((512, 512, 3)), np.zeros((512, 512), dtype=bool))
    assert out is None


def test_texgan_unwrap_neural_hole_differs_from_flat_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("scipy.io")
    pth = _find_real_pth()
    if pth is None:
        pytest.skip("sin texgan_ffhq_uv.pth local (CI)")
    _mirror_bridge(monkeypatch)
    monkeypatch.setenv("TEXGAN_FIT_STEPS", "3")
    from backend.flame_texture import (
        _LAST_TEXTURE_STATS,
        count_sentinel,
        run_unwrap_texture,
        texture_evidence,
    )

    image, fit, landmarks = _synthetic_inputs()
    first = run_unwrap_texture(image, fit, landmarks)
    from backend.domain import Err as _Err

    assert not isinstance(first, _Err)
    raw_a = first.value.as_bytes()
    assert count_sentinel(raw_a) == 0
    assert texture_evidence(raw_a) >= 0.99
    assert float(_LAST_TEXTURE_STATS.get("texgan", 0.0)) == 1.0
    second = run_unwrap_texture(image, fit, landmarks)
    assert not isinstance(second, _Err)
    assert second.value.as_bytes() == raw_a


def test_texgan_strict_load_real_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    _ = torch
    pth = _find_real_pth()
    if pth is None:
        pytest.skip("sin texgan_ffhq_uv.pth local (CI)")
    from backend.texgan import load_decoder

    dec = load_decoder(pth)
    assert dec.missing_keys() == [] and dec.unexpected_keys() == []
    uv_a = dec.synth_uv_map(dec.w_avg_batch())
    uv_b = dec.synth_uv_map(dec.w_avg_batch())
    assert uv_a.shape == (1, 3, 1024, 1024)
    assert bool(((uv_a >= 0.0) & (uv_a <= 1.0)).all())
    assert torch.equal(uv_a, uv_b)
