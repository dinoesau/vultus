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

/// Paquete resultado efimero del par: dos UV canonicas + heatmap.
/// Vive en `job.rs` (no en `pipeline.rs`) para que `queue` lo almacene
/// sin dependencia circular `queue <-> pipeline`.
/// Campos privados: solo construible via `new` con tipos ya probados.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareResult {
    uv_a: CompleteUv,
    uv_b: CompleteUv,
    heatmap: Heatmap,
}

impl CompareResult {
    pub fn new(uv_a: CompleteUv, uv_b: CompleteUv, heatmap: Heatmap) -> Self {
        Self {
            uv_a,
            uv_b,
            heatmap,
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

    pub fn into_parts(self) -> (CompleteUv, CompleteUv, Heatmap) {
        (self.uv_a, self.uv_b, self.heatmap)
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
