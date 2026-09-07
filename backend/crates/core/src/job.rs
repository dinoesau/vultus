use std::marker::PhantomData;

use nutype::nutype;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::error::{CoreError, ImageError, Result};

pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
pub const RESULT_TTL_SECONDS: u64 = 60;

/// Branded JobId. Only constructible via `new` / `parse`, so invalid
/// states are unrepresentable past the API edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(Uuid);

impl JobId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn parse(s: &str) -> Result<Self> {
        Uuid::parse_str(s.trim())
            .map(Self)
            .map_err(|_| CoreError::InvalidJobId)
    }

    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Vista prestada zero-cost: sin heap, misma prueba que `ImageBytes`.
/// Parsea prestado en el borde y promueve a owned una sola vez.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageBytesRef<'a>(&'a [u8]);

impl<'a> ImageBytesRef<'a> {
    pub fn parse(raw: &'a [u8]) -> std::result::Result<Self, ImageError> {
        if raw.is_empty() || raw.len() > MAX_IMAGE_BYTES {
            return Err(ImageError::SizeOutOfRange);
        }
        if !is_jpeg(raw) && !is_png(raw) {
            return Err(ImageError::UnsupportedFormat);
        }
        Ok(Self(raw))
    }

    pub fn as_bytes(self) -> &'a [u8] {
        self.0
    }

    pub fn to_owned_image(self) -> ImageBytes {
        ImageBytes(self.0.to_vec())
    }
}

/// Validated image bytes. Parse at the edge, never validate in core.
#[derive(Debug, Clone)]
pub struct ImageBytes(Vec<u8>);

impl ImageBytes {
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        ImageBytesRef::parse(&bytes)
            .map(|r| r.to_owned_image())
            .map_err(CoreError::InvalidImage)
    }

    /// Prestamo sin re-validar: el valor ya fue probado.
    pub fn as_ref_view(&self) -> ImageBytesRef<'_> {
        ImageBytesRef(self.0.as_slice())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

fn is_jpeg(b: &[u8]) -> bool {
    b.len() >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}

fn is_png(b: &[u8]) -> bool {
    b.len() >= 8 && b[0..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Processing,
    Done,
    Failed,
    Expired,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Processing => "processing",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Expired => "expired",
        }
    }
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Progress 0.0..=1.0 emitted per stage. Branded so out-of-range
/// values cannot cross into core.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Progress(f32);

impl Progress {
    pub fn parse(v: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&v) || v.is_nan() {
            return Err(CoreError::InvalidProgress);
        }
        Ok(Self(v))
    }

    pub fn zero() -> Self {
        Self(0.0)
    }

    pub fn value(self) -> f32 {
        self.0
    }
}

/// Stage ordenado del pipeline. Stringly `&str` prohibido en `Queue`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Queued,
    Landmarks,
    Flame,
    Freeuv,
    Bake,
    Done,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Landmarks => "landmarks",
            Self::Flame => "flame",
            Self::Freeuv => "freeuv",
            Self::Bake => "bake",
            Self::Done => "done",
        }
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// TTL en segundos via macro: invariante visible, boilerplate generado.
/// Manual para magic bytes/UUID, macro para rangos simples.
#[nutype(
    validate(greater_or_equal = 1, less_or_equal = 3600),
    derive(Debug, Clone, Copy, PartialEq, Eq, Display, TryFrom, Into)
)]
pub struct TtlSecs(u64);

impl TtlSecs {
    pub fn parse(v: u64) -> Result<Self> {
        Self::try_from(v).map_err(|_| CoreError::Invariant("ttl out of range"))
    }

    pub fn value(self) -> u64 {
        self.into()
    }

    pub fn default_ttl() -> Self {
        Self::try_from(RESULT_TTL_SECONDS).expect("60 esta en 1..=3600")
    }

    /// Intervalo del reaper stateless: TTL/2, minimo 1s.
    /// Un solo lugar para el ciclo de vida; `main` y tests lo reusan.
    pub fn reaper_interval(self) -> std::time::Duration {
        std::time::Duration::from_secs(self.value().div_ceil(2).max(1))
    }

    /// Ventana extra para distinguir `Expired` de `NotFound` antes de purgar.
    /// Espejo de `Store::purge_expired` (2x TTL) y de `ProgressDO` en edge.
    pub fn purge_after(self) -> std::time::Duration {
        std::time::Duration::from_secs(self.value().saturating_mul(2))
    }
}

impl Default for TtlSecs {
    fn default() -> Self {
        Self::default_ttl()
    }
}

/// Type-state del ciclo `Queued -> Processing -> Done|Failed|Expired`.
/// Transiciones ilegales no compilan, handles rancios se destruyen por move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Queued;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Processing;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Done;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Failed;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Expired;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Job<State> {
    job_id: JobId,
    progress: Progress,
    stage: Stage,
    state: PhantomData<State>,
}

impl Job<Queued> {
    pub fn new(job_id: JobId) -> Self {
        Self {
            job_id,
            progress: Progress::zero(),
            stage: Stage::Queued,
            state: PhantomData,
        }
    }

    pub fn start(self) -> Job<Processing> {
        Job {
            job_id: self.job_id,
            progress: self.progress,
            stage: Stage::Landmarks,
            state: PhantomData,
        }
    }

    pub fn job_id(self) -> JobId {
        self.job_id
    }

    pub fn status(self) -> JobStatus {
        JobStatus::Queued
    }
}

impl Job<Processing> {
    pub fn set_progress(self, progress: Progress, stage: Stage) -> Self {
        Self {
            job_id: self.job_id,
            progress,
            stage,
            state: PhantomData,
        }
    }

    pub fn complete(self) -> Job<Done> {
        Job {
            job_id: self.job_id,
            progress: self.progress,
            stage: Stage::Done,
            state: PhantomData,
        }
    }

    pub fn fail(self) -> Job<Failed> {
        Job {
            job_id: self.job_id,
            progress: self.progress,
            stage: self.stage,
            state: PhantomData,
        }
    }

    pub fn expire(self) -> Job<Expired> {
        Job {
            job_id: self.job_id,
            progress: self.progress,
            stage: self.stage,
            state: PhantomData,
        }
    }

    pub fn job_id(self) -> JobId {
        self.job_id
    }

    pub fn status(self) -> JobStatus {
        JobStatus::Processing
    }
}

impl Job<Done> {
    pub fn receipt(self) -> String {
        format!("done {}", self.job_id)
    }

    pub fn status(self) -> JobStatus {
        JobStatus::Done
    }
}

impl Job<Failed> {
    pub fn status(self) -> JobStatus {
        JobStatus::Failed
    }
}

impl Job<Expired> {
    pub fn status(self) -> JobStatus {
        JobStatus::Expired
    }
}

/// R2Key validada: no vacia, sin `..`, max 1024 chars.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct R2Key(String);

impl R2Key {
    pub fn parse(raw: String) -> Result<Self> {
        let t = raw.trim();
        if t.is_empty() || t.len() > 1024 || t.contains("..") {
            return Err(CoreError::InvalidR2Key);
        }
        Ok(Self(t.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Prod pointer: Queues limit 128KB/msg, bytes live in R2.
/// Campos privados: solo construible via `new` con `R2Key` ya probados.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R2Keys {
    image_a: R2Key,
    image_b: R2Key,
}

impl R2Keys {
    pub fn new(image_a: R2Key, image_b: R2Key) -> Self {
        Self { image_a, image_b }
    }

    pub fn image_a(&self) -> &R2Key {
        &self.image_a
    }

    pub fn image_b(&self) -> &R2Key {
        &self.image_b
    }
}

/// Comando de entrada a `Queue`: par de imagenes ya probadas.
/// Evita soltar bytes en el adapter y hace el seam testeable.
#[derive(Debug, Clone)]
pub struct EnqueueCommand {
    image_a: ImageBytes,
    image_b: ImageBytes,
}

impl EnqueueCommand {
    pub fn new(image_a: ImageBytes, image_b: ImageBytes) -> Self {
        Self { image_a, image_b }
    }

    pub fn image_a(&self) -> &ImageBytes {
        &self.image_a
    }

    pub fn image_b(&self) -> &ImageBytes {
        &self.image_b
    }

    pub fn into_pair(self) -> (ImageBytes, ImageBytes) {
        (self.image_a, self.image_b)
    }
}

/// Numero de landmarks exigido por MediaPipe (ver CONTEXT).
pub const LANDMARKS_LEN: usize = 478;

/// Landmarks 478x3 validados como JSON `[[x,y,z], ...]`.
/// Rechaza stubs `{"todo":...}` y bytes aleatorios.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Landmarks(Vec<u8>);

impl Landmarks {
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        if bytes.is_empty() {
            return Err(CoreError::Empty);
        }
        let pts: Vec<[f32; 3]> = serde_json::from_slice(&bytes).map_err(|e| {
            CoreError::Ml(crate::error::MlError::Decode {
                details: e.to_string(),
            })
        })?;
        if pts.len() != LANDMARKS_LEN {
            return Err(CoreError::Ml(crate::error::MlError::Decode {
                details: format!("expected {LANDMARKS_LEN} points, got {}", pts.len()),
            }));
        }
        if pts.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
            return Err(CoreError::Ml(crate::error::MlError::Decode {
                details: "non-finite landmark".to_string(),
            }));
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Dims canonicas de UV raw segun CONTEXT y PIPELINE: 512x512 RGB.
/// `FlawUv`, `CompleteUv` y `Heatmap` prueban esta longitud en `parse`.
/// Fallos van a `Ml::Decode` (500 infra) porque UVs solo vienen del sidecar,
/// nunca del cliente directo.
pub const UV_WIDTH: usize = 512;
pub const UV_HEIGHT: usize = 512;
pub const UV_CHANNELS: usize = 3;
pub const UV_LEN: usize = UV_WIDTH * UV_HEIGHT * UV_CHANNELS;

macro_rules! opaque_uv_bytes {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $name(Vec<u8>);

        impl $name {
            pub fn parse(bytes: Vec<u8>) -> Result<Self> {
                if bytes.len() != UV_LEN {
                    return Err(CoreError::Ml(crate::error::MlError::Decode {
                        details: format!("expected {UV_LEN} uv bytes, got {}", bytes.len()),
                    }));
                }
                Ok(Self(bytes))
            }

            pub fn as_bytes(&self) -> &[u8] {
                &self.0
            }

            pub fn into_bytes(self) -> Vec<u8> {
                self.0
            }

            pub fn len(&self) -> usize {
                self.0.len()
            }

            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
        }
    };
}

opaque_uv_bytes!(FlawUv);
opaque_uv_bytes!(CompleteUv);
opaque_uv_bytes!(Heatmap);

/// Magic GLB (`glTF`) y longitud minima de cabecera (magic 4 + version 4 + len 4).
pub const GLB_MAGIC: [u8; 4] = [0x67, 0x6C, 0x54, 0x46];
pub const GLB_MIN_LEN: usize = 12;
pub const GLB_VERSION: u32 = 2;

/// Nombres exactos del bundle Fase 2 (contrato zip Rust/Python/UI).
pub const ZIP_UV_A: &str = "uv_a.png";
pub const ZIP_UV_B: &str = "uv_b.png";
pub const ZIP_HEATMAP: &str = "heatmap.png";
pub const ZIP_MESH_A: &str = "mesh_a.glb";
pub const ZIP_MESH_B: &str = "mesh_b.glb";

/// Mesh GNM validado: contenedor GLB con magic `glTF` y longitud total coherente.
/// Parse en el borde del bake; el core solo acepta meshes ya probados.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GnmMesh(Vec<u8>);

impl GnmMesh {
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < GLB_MIN_LEN {
            return Err(CoreError::Ml(crate::error::MlError::Decode {
                details: format!("glb truncated: got {} bytes", bytes.len()),
            }));
        }
        if bytes[0..4] != GLB_MAGIC {
            return Err(CoreError::Ml(crate::error::MlError::Decode {
                details: "glb missing glTF magic".to_string(),
            }));
        }
        let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        if total != bytes.len() {
            return Err(CoreError::Ml(crate::error::MlError::Decode {
                details: format!("glb length mismatch: header {total} != {}", bytes.len()),
            }));
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Template facial GNM real (grid UV 64x64 con volumen, expresion neutra fija).
/// Congelado offline por scripts/extract_face_template.py porque el pkl de
/// FLAME no trae UVs destino utilizables (Paso 0 redefinido).
pub const TEMPLATE_GRID: usize = 64;
pub const TEMPLATE_VERTS: usize = 4225;
pub const TEMPLATE_TRIS: usize = 8192;
pub const TEMPLATE_TEXELS: usize = 512 * 512;
pub const LUT_V2_ENTRY_BYTES: usize = 16;

/// LUT v2 baricentrica (generada offline por scripts/compute_bfm_to_gnm.py).
/// Por texel destino: u32 tri + 3x f32 pesos. El runtime solo hace lookup.
static BFM_TO_GNM_LUT_V2: &[u8] = include_bytes!("../../../assets/bfm_to_gnm_v2.bin");

/// Template embebido en el binario (unica fuente Rust; Python lo lee del Volume).
static GNM_TEMPLATE_BYTES: &[u8] = include_bytes!("../../../assets/gnm_template.bin");

/// Plantilla facial probada: posiciones + UVs + indices con invariantes.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceTemplate {
    positions: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<[u32; 3]>,
}

impl FaceTemplate {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(CoreError::Ml(crate::error::MlError::Decode {
                details: "template truncado".to_string(),
            }));
        }
        let verts = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let tris = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        if verts != TEMPLATE_VERTS || tris != TEMPLATE_TRIS {
            return Err(CoreError::Ml(crate::error::MlError::Decode {
                details: format!(
                    "template counts {verts}/{tris} != {TEMPLATE_VERTS}/{TEMPLATE_TRIS}"
                ),
            }));
        }
        let expect = 8 + verts * 12 + verts * 8 + tris * 12;
        if bytes.len() != expect {
            return Err(CoreError::Ml(crate::error::MlError::Decode {
                details: format!("template len {} != {expect}", bytes.len()),
            }));
        }
        let mut off = 8;
        let mut positions = Vec::with_capacity(verts);
        for _ in 0..verts {
            let x =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            let y = f32::from_le_bytes([
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]);
            let z = f32::from_le_bytes([
                bytes[off + 8],
                bytes[off + 9],
                bytes[off + 10],
                bytes[off + 11],
            ]);
            off += 12;
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(CoreError::Ml(crate::error::MlError::Decode {
                    details: "posicion no finita".to_string(),
                }));
            }
            positions.push([x, y, z]);
        }
        let mut uvs = Vec::with_capacity(verts);
        for _ in 0..verts {
            let u =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            let v = f32::from_le_bytes([
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]);
            off += 8;
            if !u.is_finite() || !v.is_finite() {
                return Err(CoreError::Ml(crate::error::MlError::Decode {
                    details: "uv no finita".to_string(),
                }));
            }
            uvs.push([u, v]);
        }
        let mut indices = Vec::with_capacity(tris);
        for _ in 0..tris {
            let a =
                u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            let b = u32::from_le_bytes([
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]);
            let c = u32::from_le_bytes([
                bytes[off + 8],
                bytes[off + 9],
                bytes[off + 10],
                bytes[off + 11],
            ]);
            off += 12;
            if a as usize >= verts || b as usize >= verts || c as usize >= verts {
                return Err(CoreError::Ml(crate::error::MlError::Decode {
                    details: "indice fuera de rango".to_string(),
                }));
            }
            indices.push([a, b, c]);
        }
        Ok(Self {
            positions,
            uvs,
            indices,
        })
    }

    pub fn verts(&self) -> usize {
        self.positions.len()
    }

    pub fn tris(&self) -> usize {
        self.indices.len()
    }

    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    pub fn uvs(&self) -> &[[f32; 2]] {
        &self.uvs
    }

    pub fn indices(&self) -> &[[u32; 3]] {
        &self.indices
    }
}

fn face_template() -> &'static FaceTemplate {
    static ONCE: std::sync::OnceLock<FaceTemplate> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| FaceTemplate::parse(GNM_TEMPLATE_BYTES).expect("template embebido valido"))
}

/// LUT v2 validada una vez: longitud exacta, tri en rango y pesos finitos.
/// Falla ruidoso al primer bake con asset corrupto en vez de panico por
/// indexacion en caliente (job atascado hasta TTL).
fn lut_v2() -> &'static [u8] {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        assert_eq!(
            BFM_TO_GNM_LUT_V2.len(),
            TEMPLATE_TEXELS * LUT_V2_ENTRY_BYTES,
            "lut v2 len"
        );
        for t in 0..TEMPLATE_TEXELS {
            let off = t * LUT_V2_ENTRY_BYTES;
            let b = BFM_TO_GNM_LUT_V2;
            let tri = u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]) as usize;
            assert!(
                tri < TEMPLATE_TRIS,
                "lut v2 tri fuera de rango en texel {t}"
            );
            for k in 0..3 {
                let w = f32::from_le_bytes([
                    b[off + 4 + k * 4],
                    b[off + 5 + k * 4],
                    b[off + 6 + k * 4],
                    b[off + 7 + k * 4],
                ]);
                assert!(w.is_finite(), "lut v2 peso no finito en texel {t}");
            }
        }
    });
    BFM_TO_GNM_LUT_V2
}

fn lut_entry(texel: usize) -> (usize, f32, f32, f32) {
    let off = texel * LUT_V2_ENTRY_BYTES;
    let b = lut_v2();
    let tri = u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]) as usize;
    let w0 = f32::from_le_bytes([b[off + 4], b[off + 5], b[off + 6], b[off + 7]]);
    let w1 = f32::from_le_bytes([b[off + 8], b[off + 9], b[off + 10], b[off + 11]]);
    let w2 = f32::from_le_bytes([b[off + 12], b[off + 13], b[off + 14], b[off + 15]]);
    (tri, w0, w1, w2)
}

fn sample_offset(u: f32, v: f32) -> usize {
    let sx = ((u * 512.0) as usize).min(511);
    let sy = ((v * 512.0) as usize).min(511);
    (sy * 512 + sx) * 3
}

/// Bake baricentrico BFM->GNM v2: por texel destino, lookup (tri + pesos) y
/// sampleo en los 3 vertices del template sobre la imagen BFM.
/// Puro e infallible: `CompleteUv` ya prueba `UV_LEN`, la tabla preserva longitud.
pub fn bake_bfm_to_gnm(uv_bfm: &CompleteUv) -> CompleteUv {
    let src = uv_bfm.as_bytes();
    let tpl = face_template();
    let mut out = vec![0u8; UV_LEN];
    for t in 0..TEMPLATE_TEXELS {
        let (tri, w0, w1, w2) = lut_entry(t);
        let [a, b, c] = tpl.indices[tri];
        let [ua, va] = tpl.uvs[a as usize];
        let [ub, vb] = tpl.uvs[b as usize];
        let [uc, vc] = tpl.uvs[c as usize];
        let oa = sample_offset(ua, va);
        let ob = sample_offset(ub, vb);
        let oc = sample_offset(uc, vc);
        for k in 0..3 {
            let val = w0 * src[oa + k] as f32 + w1 * src[ob + k] as f32 + w2 * src[oc + k] as f32;
            out[t * 3 + k] = (val + 0.5).floor().clamp(0.0, 255.0) as u8;
        }
    }
    CompleteUv::parse(out).expect("bake preserva UV_LEN: entrada ya prueba UV_LEN")
}

fn encode_baked_png(raw: &[u8]) -> Vec<u8> {
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(UV_WIDTH as u32, UV_HEIGHT as u32, raw.to_vec()).expect("uv dims");
    let mut out = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut out);
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .expect("png encode");
    }
    out
}

/// Builder GLB real puro: geometria del template + `TEXCOORD_0` + textura PNG
/// horneada embebida y ligada al material (`baseColorTexture`).
/// Expresion neutra fija. Infallible: el GLB construido siempre trae magic y
/// longitud coherente.
pub fn build_gnm_glb(baked: &CompleteUv) -> GnmMesh {
    let tpl = face_template();
    let png = encode_baked_png(baked.as_bytes());
    let mut pos_bytes: Vec<u8> = Vec::with_capacity(tpl.positions.len() * 12);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in tpl.positions.iter() {
        for k in 0..3 {
            pos_bytes.extend_from_slice(&p[k].to_le_bytes());
            if p[k] < min[k] {
                min[k] = p[k];
            }
            if p[k] > max[k] {
                max[k] = p[k];
            }
        }
    }
    let mut uv_bytes: Vec<u8> = Vec::with_capacity(tpl.uvs.len() * 8);
    for t in tpl.uvs.iter() {
        uv_bytes.extend_from_slice(&t[0].to_le_bytes());
        uv_bytes.extend_from_slice(&t[1].to_le_bytes());
    }
    let mut idx_bytes: Vec<u8> = Vec::with_capacity(tpl.indices.len() * 6);
    for tri in tpl.indices.iter() {
        for v in tri {
            idx_bytes.extend_from_slice(&(*v as u16).to_le_bytes());
        }
    }
    let pos_len = pos_bytes.len();
    let uv_len_b = uv_bytes.len();
    let idx_len = idx_bytes.len();
    let uv_off = pos_len;
    let idx_off = pos_len + uv_len_b;
    let png_off = idx_off + idx_len;
    let mut bin: Vec<u8> = Vec::with_capacity(png_off + png.len() + 3);
    bin.extend_from_slice(&pos_bytes);
    bin.extend_from_slice(&uv_bytes);
    bin.extend_from_slice(&idx_bytes);
    bin.extend_from_slice(&png);
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let idx_count = tpl.indices.len() * 3;
    let json = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"vultus-gnm-real"}},"scene":0,"#,
            r#""scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"name":"VultusFaceNeutral"}}],"#,
            r#""meshes":[{{"name":"FaceNeutral","primitives":[{{"attributes":{{"POSITION":0,"TEXCOORD_0":1}},"indices":2,"material":0}}]}}],"#,
            r#""materials":[{{"name":"BakedSkin","pbrMetallicRoughness":{{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":0.9,"baseColorTexture":{{"index":0}}}}}}],"#,
            r#""textures":[{{"source":0,"sampler":0}}],"samplers":[{{"magFilter":9729,"minFilter":9729}}],"#,
            r#""images":[{{"bufferView":3,"mimeType":"image/png"}}],"#,
            r#""buffers":[{{"byteLength":{bin_len}}}],"#,
            r#""bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{pos_len},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{uv_off},"byteLength":{uvb_len},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{idx_off},"byteLength":{idx_len},"target":34963}},"#,
            r#"{{"buffer":0,"byteOffset":{png_off},"byteLength":{png_len}}}],"#,
            r#""accessors":[{{"bufferView":0,"componentType":5126,"count":{verts},"type":"VEC3","max":[{max0},{max1},{max2}],"min":[{min0},{min1},{min2}]}},"#,
            r#"{{"bufferView":1,"componentType":5126,"count":{verts},"type":"VEC2"}},"#,
            r#"{{"bufferView":2,"componentType":5123,"count":{idx_count},"type":"SCALAR"}}],"#,
            r#""extras":{{"expression":"neutral","bakedLen":{uv_len},"verts":{verts},"tris":{tris}}}}}"#
        ),
        bin_len = bin.len(),
        pos_len = pos_len,
        uv_off = uv_off,
        uvb_len = uv_len_b,
        idx_off = idx_off,
        idx_len = idx_len,
        png_off = png_off,
        png_len = png.len(),
        verts = TEMPLATE_VERTS,
        idx_count = idx_count,
        max0 = max[0],
        max1 = max[1],
        max2 = max[2],
        min0 = min[0],
        min1 = min[1],
        min2 = min[2],
        uv_len = UV_LEN,
        tris = TEMPLATE_TRIS,
    );
    let mut json_bytes = json.into_bytes();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(0x20);
    }
    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb: Vec<u8> = Vec::with_capacity(total_len);
    glb.extend_from_slice(&GLB_MAGIC);
    glb.extend_from_slice(&GLB_VERSION.to_le_bytes());
    glb.extend_from_slice(&(total_len as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\x00");
    glb.extend_from_slice(&bin);
    debug_assert_eq!(glb.len(), total_len);
    GnmMesh::parse(glb).expect("builder produce GLB valido con magic y longitud")
}

/// Paquete resultado efimero del par: dos UV canonicas + heatmap + dos meshes GNM.
/// Vive en `job.rs` (no en `pipeline.rs`) para que `queue` lo almacene
/// sin dependencia circular `queue <-> pipeline`.
/// Campos privados: solo construible via `new` con tipos ya probados.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareResult {
    uv_a: CompleteUv,
    uv_b: CompleteUv,
    heatmap: Heatmap,
    mesh_a: GnmMesh,
    mesh_b: GnmMesh,
}

impl CompareResult {
    pub fn new(
        uv_a: CompleteUv,
        uv_b: CompleteUv,
        heatmap: Heatmap,
        mesh_a: GnmMesh,
        mesh_b: GnmMesh,
    ) -> Self {
        Self {
            uv_a,
            uv_b,
            heatmap,
            mesh_a,
            mesh_b,
        }
    }

    pub fn uv_a(&self) -> &CompleteUv {
        &self.uv_a
    }

    pub fn uv_b(&self) -> &CompleteUv {
        &self.uv_b
    }

    pub fn heatmap(&self) -> &Heatmap {
        &self.heatmap
    }

    pub fn mesh_a(&self) -> &GnmMesh {
        &self.mesh_a
    }

    pub fn mesh_b(&self) -> &GnmMesh {
        &self.mesh_b
    }

    pub fn into_parts(self) -> (CompleteUv, CompleteUv, Heatmap, GnmMesh, GnmMesh) {
        (self.uv_a, self.uv_b, self.heatmap, self.mesh_a, self.mesh_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_jpeg_magic_bytes_accepted() {
        let mut b = vec![0xFF, 0xD8, 0xFF, 0x00];
        b.resize(64, 0);
        assert!(ImageBytes::parse(b).is_ok());
    }

    #[test]
    fn test_random_bytes_rejected() {
        assert!(ImageBytes::parse(vec![1, 2, 3, 4]).is_err());
    }

    #[test]
    fn test_oversize_rejected() {
        let mut b = vec![0xFF, 0xD8, 0xFF];
        b.resize(MAX_IMAGE_BYTES + 1, 0);
        assert!(ImageBytes::parse(b).is_err());
    }

    #[test]
    fn test_ref_promotes_once() {
        let mut b = vec![0xFF, 0xD8, 0xFF, 0x00];
        b.resize(64, 0);
        let view = ImageBytesRef::parse(&b).unwrap();
        let owned = view.to_owned_image();
        assert_eq!(owned.as_bytes(), b.as_slice());
        assert_eq!(owned.as_ref_view().as_bytes(), b.as_slice());
    }

    #[test]
    fn test_job_typestate_orders_transitions() {
        let q = Job::<Queued>::new(JobId::new());
        assert_eq!(q.status(), JobStatus::Queued);
        let p = q.start();
        assert_eq!(p.status(), JobStatus::Processing);
        let p = p.set_progress(Progress::zero(), Stage::Flame);
        let d = p.complete();
        assert_eq!(d.status(), JobStatus::Done);
        assert!(d.receipt().starts_with("done "));
    }

    #[test]
    fn test_ttl_default_is_60() {
        assert_eq!(TtlSecs::default().value(), 60);
        assert!(TtlSecs::parse(0).is_err());
        assert!(TtlSecs::parse(3601).is_err());
    }

    #[test]
    fn test_ttl_reaper_is_half_with_floor() {
        assert_eq!(
            TtlSecs::parse(60).expect("ttl").reaper_interval(),
            std::time::Duration::from_secs(30)
        );
        assert_eq!(
            TtlSecs::parse(1).expect("ttl").reaper_interval(),
            std::time::Duration::from_secs(1)
        );
        assert_eq!(
            TtlSecs::parse(60).expect("ttl").purge_after(),
            std::time::Duration::from_secs(120)
        );
    }

    fn landmarks_fixture() -> Vec<u8> {
        let pts = vec![[0.0f32, 1.0, 2.0]; LANDMARKS_LEN];
        serde_json::to_vec(&pts).expect("fixture")
    }

    #[test]
    fn test_landmarks_accepts_478_and_rejects_stub() {
        assert!(Landmarks::parse(landmarks_fixture()).is_ok());
        assert!(Landmarks::parse(b"{\"todo\":\"landmarks\"}".to_vec()).is_err());
        assert!(Landmarks::parse(vec![1, 2, 3]).is_err());
        assert!(Landmarks::parse(vec![]).is_err());
    }

    #[test]
    fn test_job_status_as_str_covers_all() {
        assert_eq!(JobStatus::Queued.as_str(), "queued");
        assert_eq!(JobStatus::Processing.as_str(), "processing");
        assert_eq!(JobStatus::Done.as_str(), "done");
        assert_eq!(JobStatus::Failed.as_str(), "failed");
        assert_eq!(JobStatus::Expired.as_str(), "expired");
    }

    #[test]
    fn test_r2keys_holds_validated_keys() {
        let a = R2Key::parse("jobs/1/a".to_string()).expect("a");
        let b = R2Key::parse("jobs/1/b".to_string()).expect("b");
        let keys = R2Keys::new(a.clone(), b.clone());
        assert_eq!(keys.image_a(), &a);
        assert_eq!(keys.image_b(), &b);
    }

    #[test]
    fn test_enqueue_command_holds_pair() {
        let mut raw = vec![0xFF, 0xD8, 0xFF, 0x00];
        raw.resize(64, 0);
        let a = ImageBytes::parse(raw.clone()).expect("a");
        let b = ImageBytes::parse(raw).expect("b");
        let cmd = EnqueueCommand::new(a, b);
        assert_eq!(cmd.image_a().as_bytes().len(), 64);
        assert_eq!(cmd.image_b().as_bytes().len(), 64);
    }

    fn golden_uv(head: [u8; 2], fill: u8) -> CompleteUv {
        let mut v = vec![fill; UV_LEN];
        v[..2].copy_from_slice(&head);
        CompleteUv::parse(v).expect("uv dorada")
    }

    #[test]
    fn test_gnm_mesh_accepts_valid_and_rejects_bad() {
        let baked = golden_uv([10, 200], 7);
        let mesh = build_gnm_glb(&baked);
        // PNG embebido comprime los goldens planos: cota geometria+PNG, no raw.
        assert!(mesh.len() > 100_000);
        assert!(mesh.len() < 2_000_000);
        assert_eq!(&mesh.as_bytes()[0..4], &GLB_MAGIC);
        assert!(GnmMesh::parse(mesh.clone().into_bytes()).is_ok());
        assert!(GnmMesh::parse(vec![]).is_err());
        assert!(GnmMesh::parse(vec![1, 2, 3]).is_err());
        let mut no_magic = mesh.clone().into_bytes();
        no_magic[0..4].copy_from_slice(b"BAD!");
        // Reescribe longitud para aislar el fallo a magic.
        let total = (no_magic.len() as u32).to_le_bytes();
        no_magic[8..12].copy_from_slice(&total);
        assert!(GnmMesh::parse(no_magic).is_err());
        let mut truncated = mesh.into_bytes();
        truncated.truncate(20);
        assert!(GnmMesh::parse(truncated).is_err());
    }

    #[test]
    fn test_bake_differs_from_identity_with_golden() {
        // LUT v2 baricentrica: texel (0,0) mezcla vertice (0,0)=(10,200,7)
        // con dos vecinos (7,7,7) via pesos (0.875,0.0625,0.0625):
        // R=0.875*10+0.125*7=9.625->10, G=0.875*200+0.125*7=175.875->176, B=7.
        // Literales a mano, nunca recomputados.
        let input = golden_uv([10, 200], 7);
        let baked = bake_bfm_to_gnm(&input);
        assert_eq!(baked.len(), UV_LEN);
        assert_ne!(baked.as_bytes(), input.as_bytes());
        assert_eq!(&baked.as_bytes()[..3], &[10, 176, 7]);
        // Texel (7,0): peso del vertice singular 0 -> (7,7,7).
        assert_eq!(&baked.as_bytes()[21..24], &[7, 7, 7]);
        // Texel (4,4): triangulo 1 sin vertice singular -> (7,7,7).
        assert_eq!(&baked.as_bytes()[6156..6159], &[7, 7, 7]);
    }

    #[test]
    fn test_build_glb_has_magic_and_bounded_size() {
        let baked = golden_uv([10, 200], 0);
        let mesh = build_gnm_glb(&baked);
        assert_eq!(&mesh.as_bytes()[0..4], &[0x67, 0x6C, 0x54, 0x46]);
        assert!(mesh.len() > 100_000);
        assert!(mesh.len() < 2_000_000);
        let total = u32::from_le_bytes(mesh.as_bytes()[8..12].try_into().expect("header")) as usize;
        assert_eq!(total, mesh.len());
        let json_len =
            u32::from_le_bytes(mesh.as_bytes()[12..16].try_into().expect("json len")) as usize;
        let json = &mesh.as_bytes()[20..20 + json_len];
        let text = String::from_utf8_lossy(json);
        assert!(text.contains("TEXCOORD_0"), "sin TEXCOORD_0");
        assert!(text.contains("baseColorTexture"), "sin textura ligada");
        assert!(text.contains("image/png"), "sin imagen PNG");
        assert!(text.contains("\"count\":4225"), "sin conteo 4225");
        assert!(text.contains("\"count\":24576"), "sin conteo 24576");
        let v: serde_json::Value = serde_json::from_slice(json).expect("json glb valido");
        assert_eq!(
            v["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"],
            1
        );
        let png_magic = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert!(
            mesh.as_bytes().windows(8).any(|w| w == png_magic),
            "sin PNG embebido"
        );
    }

    #[test]
    fn test_face_template_has_golden_counts() {
        let bytes = include_bytes!("../../../assets/gnm_template.bin");
        let tpl = FaceTemplate::parse(bytes).expect("template dorado");
        assert_eq!(tpl.verts(), 4225);
        assert_eq!(tpl.tris(), 8192);
        assert_eq!(tpl.positions().len(), 4225);
        assert_eq!(tpl.uvs().len(), 4225);
        assert_eq!(tpl.indices().len(), 8192);
    }

    #[test]
    fn test_face_template_rejects_oob_index() {
        let mut bytes = include_bytes!("../../../assets/gnm_template.bin").to_vec();
        let last = bytes.len() - 4;
        bytes[last..].copy_from_slice(&99999u32.to_le_bytes());
        assert!(FaceTemplate::parse(&bytes).is_err());
    }

    #[test]
    fn test_face_template_rejects_nonfinite_uv() {
        let mut bytes = include_bytes!("../../../assets/gnm_template.bin").to_vec();
        let uv_start = 8 + 4225 * 12;
        bytes[uv_start..uv_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(FaceTemplate::parse(&bytes).is_err());
    }

    #[test]
    fn test_compare_result_exposes_meshes() {
        let uv_a = golden_uv([10, 200], 0);
        let uv_b = golden_uv([4, 210], 0);
        let heat = Heatmap::parse(golden_uv([6, 10], 0).into_bytes()).expect("heat");
        let mesh_a = build_gnm_glb(&uv_a);
        let mesh_b = build_gnm_glb(&uv_b);
        let res = CompareResult::new(uv_a, uv_b, heat, mesh_a.clone(), mesh_b.clone());
        assert_eq!(res.mesh_a(), &mesh_a);
        assert_eq!(res.mesh_b(), &mesh_b);
        let (_, _, _, ma, mb) = res.into_parts();
        assert_eq!(ma, mesh_a);
        assert_eq!(mb, mesh_b);
    }

    proptest! {
        #[test]
        fn jpeg_with_filler_always_parses(len in 0usize..512) {
            let mut b = vec![0xFF, 0xD8, 0xFF];
            b.resize(3 + len, 0xAB);
            prop_assert!(ImageBytes::parse(b.clone()).is_ok());
            prop_assert!(ImageBytesRef::parse(&b).is_ok());
        }

        #[test]
        fn png_with_filler_always_parses(len in 0usize..512) {
            let mut b = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            b.resize(8 + len, 0xCD);
            prop_assert!(ImageBytes::parse(b.clone()).is_ok());
        }

        #[test]
        fn parse_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..1024)) {
            let _ = ImageBytes::parse(bytes.clone());
            let _ = ImageBytesRef::parse(&bytes);
            let _ = Landmarks::parse(bytes.clone());
            let _ = FlawUv::parse(bytes.clone());
            let _ = CompleteUv::parse(bytes);
        }

        #[test]
        fn valid_progress_always_parses(v in 0.0f32..=1.0) {
            prop_assume!(!v.is_nan());
            prop_assert!(Progress::parse(v).is_ok());
        }

        #[test]
        fn invalid_progress_never_parses(v in any::<f32>()) {
            prop_assume!(v.is_nan() || !(0.0..=1.0).contains(&v));
            prop_assert!(Progress::parse(v).is_err());
        }

        #[test]
        fn valid_ttl_always_parses(v in 1u64..=3600) {
            prop_assert!(TtlSecs::parse(v).is_ok());
        }

        #[test]
        fn r2key_trims_and_roundtrips(s in "[a-z0-9/]{1,32}") {
            let raw = format!("  {s}  ");
            let k = R2Key::parse(raw).unwrap();
            prop_assert_eq!(k.as_str(), s.as_str());
        }

        #[test]
        fn r2key_rejects_dotdot(s in "[a-z]{1,16}") {
            let raw = format!("{s}/../{s}");
            prop_assert!(R2Key::parse(raw).is_err());
        }
    }
}
