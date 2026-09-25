"""Borde secretos: tokens redactados por tipo, nunca str en logs.

SecretStr y CfToken muestran ********** en str/repr; el valor crudo
solo sale por get_secret_value. caplog prueba ausencia de fuga.
"""

from __future__ import annotations

import logging

import pytest

from backend.shell_secrets import CfToken, SecretStr

SECRET = "super-secret-token-123"


def test_secretstr_redacts_str_and_repr() -> None:
    token = SecretStr(_value=SECRET)
    assert SECRET not in str(token)
    assert SECRET not in repr(token)
    assert str(token) == "**********"
    assert token.get_secret_value() == SECRET


def test_cftoken_redacts_str_and_repr() -> None:
    token = CfToken(_inner=SecretStr(_value=SECRET))
    assert SECRET not in str(token)
    assert SECRET not in repr(token)
    assert token.get_secret_value() == SECRET


def test_secret_absent_from_caplog(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("borde-secrets-probe")
    token = CfToken(_inner=SecretStr(_value=SECRET))
    with caplog.at_level(logging.WARNING):
        logger.warning("using token %s", token)
    assert SECRET not in caplog.text
    assert "**********" in caplog.text
