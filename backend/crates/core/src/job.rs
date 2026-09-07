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

/// LUT BFM->GNM embebida (generada offline por scripts/compute_bfm_to_gnm.py).
/// 256 B versionada por contenido; el runtime solo hace lookup por texel.
static BFM_TO_GNM_LUT: [u8; 256] = *include_bytes!("../../../assets/bfm_to_gnm.bin");

/// Bake baricentrico BFM->GNM v1: lookup por texel via LUT precomputada.
/// Puro e infallible: `CompleteUv` ya prueba `UV_LEN`, la tabla preserva longitud.
pub fn bake_bfm_to_gnm(uv_bfm: &CompleteUv) -> CompleteUv {
    let bytes: Vec<u8> = uv_bfm
        .as_bytes()
        .iter()
        .map(|b| BFM_TO_GNM_LUT[*b as usize])
        .collect();
    CompleteUv::parse(bytes).expect("bake preserva UV_LEN: entrada ya prueba UV_LEN")
}

/// Builder GLB minimo puro: triangulo neutro + textura horneada en BIN.
/// Expresion neutra fija (Fase 2 sin editor arbitrario). Infallible: el GLB
/// construido siempre trae magic y longitud coherente.
pub fn build_gnm_glb(baked: &CompleteUv) -> GnmMesh {
    let mut bin: Vec<u8> = Vec::with_capacity(44 + UV_LEN);
    for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    for i in [0u16, 1, 2] {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    bin.extend_from_slice(&[0u8; 2]);
    bin.extend_from_slice(baked.as_bytes());
    debug_assert_eq!(bin.len() % 4, 0);
    let json = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"vultus-fase2"}},"scene":0,"#,
            r#""scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"name":"VultusFaceNeutral"}}],"#,
            r#""meshes":[{{"name":"FaceNeutral","primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],"#,
            r#""materials":[{{"name":"BakedSkin","pbrMetallicRoughness":{{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":0.9}}}}],"#,
            r#""buffers":[{{"byteLength":{bin_len}}}],"#,
            r#""bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36,"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":36,"byteLength":6,"target":34963}},"#,
            r#"{{"buffer":0,"byteOffset":44,"byteLength":{uv_len},"name":"BakedUV"}}],"#,
            r#""accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","max":[1,1,0],"min":[0,0,0]}},"#,
            r#"{{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}],"#,
            r#""extras":{{"expression":"neutral","bakedLen":{uv_len}}}}}"#
        ),
        bin_len = bin.len(),
        uv_len = UV_LEN,
    );
    let mut json_bytes = json.into_bytes();
    while json_bytes.len() % 4 != 0 {
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
        assert!(mesh.len() > UV_LEN);
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
        // LUT v1: (i*5+17)%256. Literales a mano, nunca recomputados.
        // LUT[10]=67, LUT[200]=249, LUT[7]=52.
        let input = golden_uv([10, 200], 7);
        let baked = bake_bfm_to_gnm(&input);
        assert_eq!(baked.len(), UV_LEN);
        assert_ne!(baked.as_bytes(), input.as_bytes());
        assert_eq!(&baked.as_bytes()[..2], &[67, 249]);
        assert!(baked.as_bytes()[2..].iter().all(|&b| b == 52));
    }

    #[test]
    fn test_build_glb_has_magic_and_bounded_size() {
        let baked = golden_uv([10, 200], 0);
        let mesh = build_gnm_glb(&baked);
        assert_eq!(&mesh.as_bytes()[0..4], &[0x67, 0x6C, 0x54, 0x46]);
        assert!(mesh.len() > UV_LEN);
        assert!(mesh.len() < 2_000_000);
        let total = u32::from_le_bytes(mesh.as_bytes()[8..12].try_into().expect("header")) as usize;
        assert_eq!(total, mesh.len());
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
