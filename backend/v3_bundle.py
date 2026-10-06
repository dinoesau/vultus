"""Bundle v3 nuevo (malla densa + albedo 1024 + relights).

Tipo separado del zip-6 forense a nivel de tipos y de nombres: prohibido
mezclar. `V3Bundle` solo via `build_v3_bundle`; `ZipBundle` (dominio)
sigue siendo el unico dueno del zip-6.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.domain import DomainError, Err, MlDecode, MlFailed, Ok
from backend.v3_contract import V3_UV_LEN, V3_ZIP_NAMES


@dataclass(frozen=True, slots=True)
class V3Bundle:
    """Bundle v3 de 6 piezas en orden `V3_ZIP_NAMES`. Solo via constructor."""

    mesh_dense_glb: bytes
    albedo_1024_png: bytes
    relight_neutral: bytes
    relight_key: bytes
    relight_fill: bytes
    relight_rim: bytes

    def names(self) -> tuple[str, ...]:
        """Nombres canonicos v3 en orden. Nunca literales fuera de aqui."""
        return V3_ZIP_NAMES

    def parts(self) -> tuple[bytes, ...]:
        """Piezas en el mismo orden que `names()`."""
        return (
            self.mesh_dense_glb,
            self.albedo_1024_png,
            self.relight_neutral,
            self.relight_key,
            self.relight_fill,
            self.relight_rim,
        )


def build_v3_bundle(
    mesh_glb: bytes, albedo_raw: bytes, relights: list[bytes] | tuple[bytes, ...]
) -> Ok[V3Bundle] | Err[DomainError]:
    """Ensambla el bundle v3. Total: `Err` si piezas invalidas.

    `albedo_raw` es el atlas 1024 RGB plano (`V3_UV_LEN` bytes); se guarda
    tal cual (el PNG lo hace el visor/worker, aqui es bundle de bytes).
    Prohibido mezclar con zip-6: los nombres se verifican disjuntos.
    """
    try:
        rel = list(relights)
    except TypeError as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 bundle relights invalid: {exc}")))
    if not isinstance(mesh_glb, (bytes, bytearray)) or len(mesh_glb) < 12:
        return Err(MlFailed(detail=MlDecode(details="v3 bundle mesh invalid")))
    if not isinstance(albedo_raw, (bytes, bytearray)) or len(albedo_raw) != V3_UV_LEN:
        return Err(
            MlFailed(detail=MlDecode(details=f"v3 bundle albedo len {len(albedo_raw)} != {V3_UV_LEN}"))
        )
    if len(rel) != 4 or not all(isinstance(r, (bytes, bytearray)) and len(r) > 0 for r in rel):
        return Err(MlFailed(detail=MlDecode(details="v3 bundle needs 4 relights")))
    try:
        from backend.domain import ZIP_NAMES as _ZIP6
    except ImportError as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 bundle forensic check failed: {exc}")))
    if set(V3_ZIP_NAMES) & set(_ZIP6):
        return Err(MlFailed(detail=MlDecode(details="v3 bundle overlaps zip-6")))
    return Ok(
        V3Bundle(
            mesh_dense_glb=bytes(mesh_glb),
            albedo_1024_png=bytes(albedo_raw),
            relight_neutral=bytes(rel[0]),
            relight_key=bytes(rel[1]),
            relight_fill=bytes(rel[2]),
            relight_rim=bytes(rel[3]),
        )
    )
