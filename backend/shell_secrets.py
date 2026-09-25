"""Secretos compartidos stdlib: dueno unico para runner y sidecar.

Sin pydantic ni FastAPI ni logging aqui: el runner es stdlib-only.
`SecretStr` redacta `__repr__` y `__str__`; `CfToken` envuelve el token
Cloudflare y nunca viaja dentro de `QueueJob`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SecretStr:
    """Token generico en memoria. Solo via construccion shell."""

    _value: str

    def get_secret_value(self) -> str:
        return self._value

    def __str__(self) -> str:
        return "**********"

    def __repr__(self) -> str:
        return "SecretStr('**********')"


@dataclass(frozen=True, slots=True)
class CfToken:
    """Token Cloudflare. Solo via construccion shell desde env."""

    _inner: SecretStr

    def get_secret_value(self) -> str:
        return self._inner.get_secret_value()

    def __str__(self) -> str:
        return "**********"

    def __repr__(self) -> str:
        return "CfToken('**********')"
