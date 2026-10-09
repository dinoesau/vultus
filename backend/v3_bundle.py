"""Bundle v3 nuevo (malla densa + albedo 1024 + relights).

Tipo separado del zip-8 forense a nivel de tipos y de nombres: prohibido
mezclar. `V3Bundle` solo via `build_v3_bundle`; `ZipBundle` (dominio)
sigue siendo el unico dueno del zip-8.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from backend.domain import DomainError, Err, MlDecode, MlFailed, Ok
from backend.v3_contract import V3_UV_LEN, V3_ZIP_NAMES

# Parche UV reservado a ojos: esquina [0.95,1.0]x[0.0,0.05] por bola
# (cuadricula 26x21 por orden de vertice, determinista). La mascara de
# piel nunca muestrea ahi; el atlas lleva sintesis en el parche.
_EYE_PATCH_U0 = 0.95
_EYE_PATCH_V0 = 0.0
_EYE_PATCH_SU = 0.05
_EYE_PATCH_SV = 0.05
_EYE_COLS = 26
_EYE_ROWS = 21


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
    Prohibido mezclar con zip-8: los nombres se verifican disjuntos.
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
        return Err(MlFailed(detail=MlDecode(details="v3 bundle overlaps zip-8")))
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


def eye_patch_uvs(n_eye_verts: int) -> NDArray[np.float64]:
    """UVs del parche de ojos por orden de vertice (determinista, finito)."""
    out: list[tuple[float, float]] = []
    per_ball = _EYE_COLS * _EYE_ROWS
    for k in range(max(0, int(n_eye_verts))):
        ball = k // per_ball
        local = k % per_ball
        col = local % _EYE_COLS
        row = local // _EYE_COLS
        u = _EYE_PATCH_U0 + (float(col) / float(_EYE_COLS)) * _EYE_PATCH_SU
        v = _EYE_PATCH_V0 + (float(row) / float(_EYE_ROWS)) * _EYE_PATCH_SV + float(ball) * 0.0
        out.append((u, min(1.0, v)))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def dense_vertex_normals(
    verts: NDArray[np.float64], tris: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Normales suaves por vertice (promedio de caras, unitarias).

    Vertices degenerados caen a (0,0,1) determinista. Puro numpy.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(tris, dtype=np.int64)
    acc = np.zeros_like(v)
    try:
        p0 = v[f[:, 0]]
        p1 = v[f[:, 1]]
        p2 = v[f[:, 2]]
    except IndexError:
        return np.tile(np.array([0.0, 0.0, 1.0]), (v.shape[0], 1))
    face = np.cross(p1 - p0, p2 - p0)
    face[~np.isfinite(face).all(axis=1)] = 0.0
    np.add.at(acc, f[:, 0], face)
    np.add.at(acc, f[:, 1], face)
    np.add.at(acc, f[:, 2], face)
    lens = np.linalg.norm(acc, axis=1)
    out = np.zeros_like(acc)
    nz = lens > 1e-12
    out[nz] = acc[nz] / lens[nz][:, None]
    out[~nz] = (0.0, 0.0, 1.0)
    return np.asarray(out, dtype=np.float64)


def _pad4(buf: bytes) -> bytes:
    pad = (-len(buf)) % 4
    return buf + b"\x00" * pad if pad else buf


def build_dense_glb(
    verts: NDArray[np.float64], tris: NDArray[np.int64], uvs_head: NDArray[np.float64]
) -> Ok[bytes] | Err[DomainError]:
    """GLB denso de 1 primitiva (POSITION + NORMAL + TEXCOORD_0 + indices).

    `uvs_head` cubre los primeros 20481 verts (cabeza); los ojos agregados
    van al parche reservado (ver `eye_patch_uvs`). Magia `glTF`, offsets
    multiplo de 4, determinista x2. Total: `Err` ante formas invalidas.
    """
    try:
        v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(tris, dtype=np.int64)
        uh = np.asarray(uvs_head, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        return Err(MlFailed(detail=MlDecode(details=f"v3 dense glb invalid input: {exc}")))
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] <= 20481:
        return Err(MlFailed(detail=MlDecode(details="v3 dense glb verts invalid (cabeza + ojos)")))
    if f.ndim != 2 or f.shape[1] != 3 or int(f.min()) < 0 or int(f.max()) >= int(v.shape[0]):
        return Err(MlFailed(detail=MlDecode(details="v3 dense glb tris invalid")))
    if uh.shape != (20481, 2) or not bool(np.isfinite(uh).all()):
        return Err(MlFailed(detail=MlDecode(details="v3 dense glb head uvs invalid")))
    if not bool(np.isfinite(v).all()):
        return Err(MlFailed(detail=MlDecode(details="v3 dense glb verts non-finite")))
    n = int(v.shape[0])
    eye_uvs = eye_patch_uvs(n - 20481)
    uvs = np.concatenate([uh, eye_uvs], axis=0)
    if uvs.shape != (n, 2):
        return Err(MlFailed(detail=MlDecode(details="v3 dense glb eye uv count mismatch")))
    normals = dense_vertex_normals(v, f)
    pos_buf = _pad4(np.ascontiguousarray(v, dtype=np.float32).tobytes())
    nrm_buf = _pad4(np.ascontiguousarray(normals, dtype=np.float32).tobytes())
    uv_buf = _pad4(np.ascontiguousarray(uvs, dtype=np.float32).tobytes())
    idx_buf = _pad4(np.ascontiguousarray(f, dtype=np.uint32).tobytes())
    uv_off = len(pos_buf)
    nrm_off = uv_off + len(uv_buf)
    idx_off = nrm_off + len(nrm_buf)
    total_bin = len(pos_buf) + len(uv_buf) + len(nrm_buf) + len(idx_buf)
    xs = v[:, 0]
    ys = v[:, 1]
    zs = v[:, 2]
    json_str = (
        '{"asset":{"version":"2.0","generator":"vultus-v3"},"scene":0,'
        '"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],'
        '"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":2,"TEXCOORD_0":1},'
        '"indices":3,"material":0}]}],'
        '"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],'
        '"metallicFactor":0.0,"roughnessFactor":0.8}}],'
        f'"accessors":[{{"bufferView":0,"componentType":5126,"count":{n},"type":"VEC3",'
        f'"max":[{float(xs.max())},{float(ys.max())},{float(zs.max())}],'
        f'"min":[{float(xs.min())},{float(ys.min())},{float(zs.min())}]}},'
        f'{{"bufferView":1,"componentType":5126,"count":{n},"type":"VEC2"}},'
        f'{{"bufferView":2,"componentType":5126,"count":{n},"type":"VEC3"}},'
        f'{{"bufferView":3,"componentType":5125,"count":{int(f.shape[0] * 3)},"type":"SCALAR"}}],'
        f'"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{len(pos_buf)}}},'
        f'{{"buffer":0,"byteOffset":{uv_off},"byteLength":{len(uv_buf)}}},'
        f'{{"buffer":0,"byteOffset":{nrm_off},"byteLength":{len(nrm_buf)}}},'
        f'{{"buffer":0,"byteOffset":{idx_off},"byteLength":{len(idx_buf)}}}]'
        f',"buffers":[{{"byteLength":{total_bin}}}]}}'
    )
    json_bytes = json_str.encode("utf-8")
    json_pad = _pad4(json_bytes)
    bin_buf = pos_buf + uv_buf + nrm_buf + idx_buf
    total = 12 + 8 + len(json_pad) + 8 + len(bin_buf)
    out = (
        b"glTF"
        + struct.pack("<I", 2)
        + struct.pack("<I", total)
        + struct.pack("<I", len(json_pad))
        + b"JSON"
        + json_pad
        + struct.pack("<I", len(bin_buf))
        + b"BIN\x00"
        + bin_buf
    )
    if out[:4] != b"glTF" or len(out) != total:
        return Err(MlFailed(detail=MlDecode(details="v3 dense glb framing invalid")))
    return Ok(bytes(out))
