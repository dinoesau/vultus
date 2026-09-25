"""Borde sidecar: DomainError a HTTP una vez en el edge, sin str(e) al cliente.

Negocio a 400, NotFound a 404, infra a 500 generico. R2 404 via
_is_not_found_error sin exponer detalle.
"""

from __future__ import annotations

from backend.domain import (
    EmptyPayload,
    FitFailed,
    InvalidCamera,
    InvalidCoeffs,
    InvalidImage,
    InvalidJobId,
    InvalidProgress,
    InvalidR2Key,
    Invariant,
    MlBadStatus,
    MlDecode,
    MlFailed,
    MlTransport,
    NotFound,
    SizeOutOfRange,
    decode_fit_request,
    domain_to_message,
    domain_to_status,
    encode_fit_request,
    parse_image_bytes,
    parse_landmarks,
)
from backend.modal_app import _is_not_found_error


def test_business_maps_to_400() -> None:
    assert domain_to_status(InvalidImage(detail=SizeOutOfRange())) == 400
    assert domain_to_status(InvalidJobId()) == 400
    assert domain_to_status(InvalidR2Key()) == 400
    assert domain_to_status(InvalidProgress()) == 400
    assert domain_to_status(EmptyPayload()) == 400
    assert domain_to_status(InvalidCoeffs()) == 400
    assert domain_to_status(InvalidCamera()) == 400


def test_notfound_maps_to_404() -> None:
    assert domain_to_status(NotFound(job_id="abc")) == 404
    assert domain_to_message(NotFound(job_id="abc")) == "not found: abc"


def test_infra_maps_to_500_generic() -> None:
    ml = MlFailed(detail=MlTransport(details="boom detail"))
    fit = FitFailed(detail=MlBadStatus(status=500))
    inv = Invariant(detail="bug")
    assert domain_to_status(ml) == 500
    assert domain_to_status(fit) == 500
    assert domain_to_status(inv) == 500
    assert domain_to_message(ml) == "internal error"
    assert domain_to_message(fit) == "internal error"
    assert "boom detail" not in domain_to_message(ml)


def test_invalid_base_url_message_keeps_shape_only() -> None:
    from backend.domain import BadScheme, InvalidBaseUrl

    err = InvalidBaseUrl(detail=BadScheme())
    assert domain_to_status(err) == 400
    assert "http" in domain_to_message(err)


def test_is_not_found_error_narrow() -> None:
    class NoSuchKey(Exception):
        pass

    assert _is_not_found_error(NoSuchKey("x")) is True
    assert _is_not_found_error(ValueError("plain")) is False

    class Fake404(Exception):
        def __init__(self) -> None:
            super().__init__("fetch")
            self.response = {
                "Error": {"Code": "NoSuchKey"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            }

    assert _is_not_found_error(Fake404()) is True


def _valid_fit_payload() -> bytes:
    import json as _json

    from backend.domain import Ok as _Ok

    pts = [[0.0, 1.0, 2.0]] * 478
    lm_raw = _json.dumps(pts).encode("utf-8")
    lm = parse_landmarks(lm_raw)
    assert isinstance(lm, _Ok)
    img_raw = bytes([0xFF, 0xD8, 0xFF, 0x00]) + bytes(60)
    img = parse_image_bytes(img_raw)
    assert isinstance(img, _Ok)
    return encode_fit_request(img.value, lm.value)


def test_fit_truncated_maps_via_domain_table_generic_500() -> None:
    from backend.domain import Err as _Err

    valid = _valid_fit_payload()
    # Truncar dentro de landmarks: len < 4 + size declarado -> Err truncado.
    truncated = valid[:10]
    assert len(truncated) < len(valid)
    result = decode_fit_request(truncated)
    assert isinstance(result, _Err)
    status = domain_to_status(result.error)
    detail = domain_to_message(result.error)
    assert status == 500
    assert detail == "internal error"
    assert detail == domain_to_message(result.error)
    assert "truncated" not in detail


def test_fit_short_and_nonbytes_map_generic_500_no_leak() -> None:
    from backend.domain import Err as _Err

    for raw in (b"\x00\x01", b"", 123):
        result = decode_fit_request(raw)  # type: ignore[arg-type]
        assert isinstance(result, _Err)
        assert domain_to_status(result.error) == 500
        assert domain_to_message(result.error) == "internal error"


def test_texture_bad_landmarks_maps_via_domain_table_no_leak() -> None:
    import json as _json

    from backend.domain import Err as _Err
    from backend.domain import FitResult as _FitResult
    from backend.domain import Ok as _Ok
    from backend.domain import parse_camera_params as _parse_cam
    from backend.domain import parse_gnm_coeffs as _parse_coeffs
    from backend.pipeline_local import (
        FIT_RESULT_LEN,
        decode_texture_request,
        encode_fit_result,
        encode_texture_request,
    )

    coeffs = _parse_coeffs([0.0] * 253)
    camera = _parse_cam([0.0] * 12)
    assert isinstance(coeffs, _Ok)
    assert isinstance(camera, _Ok)
    fit = _FitResult(coeffs=coeffs.value, camera=camera.value)
    img_raw = bytes([0xFF, 0xD8, 0xFF, 0x00]) + bytes(60)
    img = parse_image_bytes(img_raw)
    assert isinstance(img, _Ok)
    bad_lm_raw = b'{"todo":"landmarks"}'
    bad_fit_req = len(bad_lm_raw).to_bytes(4, "big") + bad_lm_raw + img_raw
    fit_bytes = encode_fit_result(fit)
    assert len(fit_bytes) == FIT_RESULT_LEN
    tex_bad = len(bad_fit_req).to_bytes(4, "big") + bad_fit_req + fit_bytes
    result = decode_texture_request(tex_bad)
    assert isinstance(result, _Err)
    assert domain_to_message(result.error) == "internal error"
    assert domain_to_status(result.error) == 500
    assert "landmarks" not in domain_to_message(result.error)

    good_lm = parse_landmarks(_json.dumps([[0.0, 1.0, 2.0]] * 478).encode("utf-8"))
    assert isinstance(good_lm, _Ok)
    valid_tex = encode_texture_request(img.value, fit, good_lm.value)
    truncated_tex = valid_tex[:10]
    assert len(truncated_tex) < len(valid_tex)
    result2 = decode_texture_request(truncated_tex)
    assert isinstance(result2, _Err)
    assert domain_to_status(result2.error) == 500
    assert domain_to_message(result2.error) == "internal error"
    assert domain_to_message(result2.error) == domain_to_message(result2.error)


def test_infra_cause_stays_in_logs_client_sees_generic() -> None:
    ml = MlFailed(detail=MlTransport(details="boom detail gateway down"))
    fit = FitFailed(detail=MlDecode(details="fit payload truncated inner"))
    assert domain_to_status(ml) == 500
    assert domain_to_status(fit) == 500
    assert domain_to_message(ml) == "internal error"
    assert domain_to_message(fit) == "internal error"
    assert "boom" not in domain_to_message(ml)
    assert "truncated" not in domain_to_message(fit)
    assert domain_to_message(ml) == domain_to_message(fit)
