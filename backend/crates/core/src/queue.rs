use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::error::{CoreError, Result};
use super::job::{EnqueueCommand, JobId, JobStatus, Progress, R2Key, R2Keys, Stage, TtlSecs};
use super::tmp::cleanup_job_dir;

/// Contrato Seam 2. Mismo adapter para local (memoria) y prod (Queues+R2), sin Redis.
/// En local se encolan bytes; en prod solo `{job_id, r2_keys}` (Queues <128KB).
/// `purge_expired` y `ttl` viven en el trait para que el reaper de `main`
/// trabaje sobre `Arc<dyn Queue>` sin ramificar por driver.
#[async_trait]
pub trait Queue: Send + Sync {
    async fn enqueue(&self, cmd: EnqueueCommand) -> Result<EnqueuedJob>;
    async fn status(&self, job_id: &JobId) -> Result<JobStatus>;
    async fn progress(&self, job_id: &JobId) -> Result<(Progress, Stage)>;
    async fn set_progress(&self, job_id: &JobId, progress: Progress, stage: Stage) -> Result<()>;
    async fn stored_lens(&self, job_id: &JobId) -> Result<(usize, usize)>;
    /// Purga expirados tras 2x TTL y limpia `/tmp/{job_id}` best-effort.
    async fn purge_expired(&self) -> usize;
    /// TTL que este adapter usa al encolar. Fuente para el reaper (`TTL/2`).
    fn ttl(&self) -> TtlSecs;
}

/// Reloj inyectable para ciclo de vida stateless.
/// Prod usa `SystemClock`; tests usan `ManualClock` sin `sleep`.
/// Seam interno del module, no parte de la interface `Queue`.
pub trait Clock: Send + Sync + std::fmt::Debug {
    fn now(&self) -> Instant;
}

/// Reloj de produccion: `Instant::now` directo.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Reloj manual para tests: tiempo controlado sin flakiness.
/// Comparte `now` via `Arc` para que queue y test avancen juntos.
#[derive(Debug, Clone)]
pub struct ManualClock {
    now: Arc<std::sync::Mutex<Instant>>,
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl ManualClock {
    pub fn new() -> Self {
        Self {
            now: Arc::new(std::sync::Mutex::new(Instant::now())),
        }
    }

    pub fn with_start(start: Instant) -> Self {
        Self {
            now: Arc::new(std::sync::Mutex::new(start)),
        }
    }

    /// Avanza el reloj. Un solo lugar para simular expiracion.
    pub fn advance(&self, delta: Duration) {
        let mut guard = match self.now.lock() {
            Ok(g) => g,
            Err(poison) => poison.into_inner(),
        };
        *guard += delta;
    }

    pub fn set(&self, at: Instant) {
        let mut guard = match self.now.lock() {
            Ok(g) => g,
            Err(poison) => poison.into_inner(),
        };
        *guard = at;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        match self.now.lock() {
            Ok(g) => *g,
            Err(poison) => *poison.into_inner(),
        }
    }
}

/// Recibo de `enqueue`: `job_id` siempre, `r2_keys` solo en prod.
/// Campos privados: solo construible via `Queue::enqueue` dentro del crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnqueuedJob {
    job_id: JobId,
    /// None en local (bytes directos en queue), Some en prod (R2 pointer).
    r2_keys: Option<R2Keys>,
}

impl EnqueuedJob {
    pub(crate) fn new(job_id: JobId, r2_keys: Option<R2Keys>) -> Self {
        Self { job_id, r2_keys }
    }

    pub fn job_id(&self) -> JobId {
        self.job_id
    }

    pub fn r2_keys(&self) -> Option<&R2Keys> {
        self.r2_keys.as_ref()
    }

    pub fn is_r2_pointer(&self) -> bool {
        self.r2_keys.is_some()
    }
}

#[derive(Debug, Clone)]
struct MemoryEntry {
    status: JobStatus,
    progress: Progress,
    stage: Stage,
    image_a_len: usize,
    image_b_len: usize,
    created_at: Instant,
    ttl: TtlSecs,
}

impl MemoryEntry {
    fn is_expired_at(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.created_at) >= Duration::from_secs(self.ttl.value())
    }

    fn is_purgeable_at(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.created_at)
            >= Duration::from_secs(self.ttl.value().saturating_mul(2))
    }
}

fn not_found(job_id: &JobId) -> CoreError {
    CoreError::not_found(job_id.to_string())
}

/// Estado compartido tras ambos adapters de Seam 2.
/// Un solo lugar para ciclo de vida: inserta, lee estado, avanza progreso.
/// Los adapters solo difieren en `enqueue` (bytes directos vs R2 pointer).
/// El reloj se inyecta para tests deterministas sin `sleep`.
#[derive(Debug, Clone)]
struct Store {
    inner: Arc<RwLock<HashMap<JobId, MemoryEntry>>>,
    clock: Arc<dyn Clock>,
}

impl Store {
    fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            clock,
        }
    }

    async fn insert_queued(
        &self,
        job_id: JobId,
        image_a_len: usize,
        image_b_len: usize,
        ttl: TtlSecs,
    ) {
        self.inner.write().await.insert(
            job_id,
            MemoryEntry {
                status: JobStatus::Queued,
                progress: Progress::zero(),
                stage: Stage::Queued,
                image_a_len,
                image_b_len,
                created_at: self.clock.now(),
                ttl,
            },
        );
    }

    async fn status(&self, job_id: &JobId) -> Result<JobStatus> {
        let now = self.clock.now();
        let inner = self.inner.read().await;
        let e = inner.get(job_id).ok_or_else(|| not_found(job_id))?;
        if e.is_expired_at(now) {
            return Ok(JobStatus::Expired);
        }
        Ok(e.status)
    }

    async fn progress(&self, job_id: &JobId) -> Result<(Progress, Stage)> {
        let now = self.clock.now();
        let inner = self.inner.read().await;
        let e = inner.get(job_id).ok_or_else(|| not_found(job_id))?;
        if e.is_expired_at(now) {
            return Err(not_found(job_id));
        }
        Ok((e.progress, e.stage))
    }

    async fn set_progress(&self, job_id: &JobId, progress: Progress, stage: Stage) -> Result<()> {
        let now = self.clock.now();
        let mut w = self.inner.write().await;
        let e = w.get_mut(job_id).ok_or_else(|| not_found(job_id))?;
        if e.is_expired_at(now) {
            return Err(not_found(job_id));
        }
        e.progress = progress;
        e.stage = stage;
        e.status = JobStatus::Processing;
        Ok(())
    }

    async fn stored_lens(&self, job_id: &JobId) -> Result<(usize, usize)> {
        let now = self.clock.now();
        let inner = self.inner.read().await;
        let e = inner.get(job_id).ok_or_else(|| not_found(job_id))?;
        if e.is_expired_at(now) {
            return Err(not_found(job_id));
        }
        Ok((e.image_a_len, e.image_b_len))
    }

    /// Purga expirados y limpia `/tmp/{job_id}` best-effort.
    /// Retorna cantidad purgada. Nunca falla: stateless no pagina por limpieza.
    async fn purge_expired(&self) -> usize {
        let now = self.clock.now();
        let expired: Vec<JobId> = {
            let inner = self.inner.read().await;
            inner
                .iter()
                .filter(|(_, e)| e.is_purgeable_at(now))
                .map(|(id, _)| *id)
                .collect()
        };
        if expired.is_empty() {
            return 0;
        }
        {
            let mut w = self.inner.write().await;
            for id in &expired {
                w.remove(id);
            }
        }
        for id in &expired {
            cleanup_job_dir(id);
        }
        expired.len()
    }
}

impl Default for Store {
    fn default() -> Self {
        Self::new(Arc::new(SystemClock))
    }
}

/// Nucleo compartido de Seam 2: `Store` + `ttl`.
/// Toda la logica de ciclo de vida vive aqui.
/// Los adapters solo aportan `enqueue` distinto y delegan el resto.
/// Esto concentra complejidad en un module deep tras una interface chica.
#[derive(Debug, Clone, Default)]
struct Shared {
    store: Store,
    ttl: TtlSecs,
}

impl Shared {
    fn with_ttl(ttl: TtlSecs) -> Self {
        Self {
            store: Store::default(),
            ttl,
        }
    }

    fn with_ttl_and_clock(ttl: TtlSecs, clock: Arc<dyn Clock>) -> Self {
        Self {
            store: Store::new(clock),
            ttl,
        }
    }

    async fn status(&self, job_id: &JobId) -> Result<JobStatus> {
        self.store.status(job_id).await
    }

    async fn progress(&self, job_id: &JobId) -> Result<(Progress, Stage)> {
        self.store.progress(job_id).await
    }

    async fn set_progress(&self, job_id: &JobId, progress: Progress, stage: Stage) -> Result<()> {
        self.store.set_progress(job_id, progress, stage).await
    }

    async fn stored_lens(&self, job_id: &JobId) -> Result<(usize, usize)> {
        self.store.stored_lens(job_id).await
    }

    async fn purge_expired(&self) -> usize {
        self.store.purge_expired().await
    }

    fn ttl(&self) -> TtlSecs {
        self.ttl
    }
}

/// Adapter en memoria para tests y tracer bullet Fase 0.
/// Paridad con `fakeredis` del diseno Python original.
/// Guarda longitudes para probar que los bytes fluyen y no se tiran.
#[derive(Debug, Clone, Default)]
pub struct MemoryQueue {
    shared: Shared,
}

impl MemoryQueue {
    pub fn with_ttl(ttl: TtlSecs) -> Self {
        Self {
            shared: Shared::with_ttl(ttl),
        }
    }

    pub fn with_ttl_and_clock(ttl: TtlSecs, clock: Arc<dyn Clock>) -> Self {
        Self {
            shared: Shared::with_ttl_and_clock(ttl, clock),
        }
    }
}

#[async_trait]
impl Queue for MemoryQueue {
    async fn enqueue(&self, cmd: EnqueueCommand) -> Result<EnqueuedJob> {
        let (a, b) = cmd.into_pair();
        let job_id = JobId::new();
        self.shared
            .store
            .insert_queued(
                job_id,
                a.as_bytes().len(),
                b.as_bytes().len(),
                self.shared.ttl,
            )
            .await;
        Ok(EnqueuedJob::new(job_id, None))
    }

    async fn status(&self, job_id: &JobId) -> Result<JobStatus> {
        self.shared.status(job_id).await
    }

    async fn progress(&self, job_id: &JobId) -> Result<(Progress, Stage)> {
        self.shared.progress(job_id).await
    }

    async fn set_progress(&self, job_id: &JobId, progress: Progress, stage: Stage) -> Result<()> {
        self.shared.set_progress(job_id, progress, stage).await
    }

    async fn stored_lens(&self, job_id: &JobId) -> Result<(usize, usize)> {
        self.shared.stored_lens(job_id).await
    }

    async fn purge_expired(&self) -> usize {
        self.shared.purge_expired().await
    }

    fn ttl(&self) -> TtlSecs {
        self.shared.ttl()
    }
}

/// Segundo adapter: simula prod `Queues+R2` con patron R2 pointer.
/// Mismo contrato, distinto transporte: retorna `Some(R2Keys)`.
/// Dos adapters = seam real, no hipotetico.
#[derive(Debug, Clone, Default)]
pub struct R2PointerQueue {
    shared: Shared,
}

impl R2PointerQueue {
    pub fn with_ttl(ttl: TtlSecs) -> Self {
        Self {
            shared: Shared::with_ttl(ttl),
        }
    }

    pub fn with_ttl_and_clock(ttl: TtlSecs, clock: Arc<dyn Clock>) -> Self {
        Self {
            shared: Shared::with_ttl_and_clock(ttl, clock),
        }
    }
}

#[async_trait]
impl Queue for R2PointerQueue {
    async fn enqueue(&self, cmd: EnqueueCommand) -> Result<EnqueuedJob> {
        let (a, b) = cmd.into_pair();
        let job_id = JobId::new();
        let r2_keys = R2Keys::new(
            R2Key::parse(format!("jobs/{job_id}/a"))?,
            R2Key::parse(format!("jobs/{job_id}/b"))?,
        );
        self.shared
            .store
            .insert_queued(
                job_id,
                a.as_bytes().len(),
                b.as_bytes().len(),
                self.shared.ttl,
            )
            .await;
        Ok(EnqueuedJob::new(job_id, Some(r2_keys)))
    }

    async fn status(&self, job_id: &JobId) -> Result<JobStatus> {
        self.shared.status(job_id).await
    }

    async fn progress(&self, job_id: &JobId) -> Result<(Progress, Stage)> {
        self.shared.progress(job_id).await
    }

    async fn set_progress(&self, job_id: &JobId, progress: Progress, stage: Stage) -> Result<()> {
        self.shared.set_progress(job_id, progress, stage).await
    }

    async fn stored_lens(&self, job_id: &JobId) -> Result<(usize, usize)> {
        self.shared.stored_lens(job_id).await
    }

    async fn purge_expired(&self) -> usize {
        self.shared.purge_expired().await
    }

    fn ttl(&self) -> TtlSecs {
        self.shared.ttl()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::ImageBytes;

    fn png_image() -> ImageBytes {
        let mut b = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        b.resize(64, 0);
        ImageBytes::parse(b).expect("png fixture")
    }

    fn cmd() -> EnqueueCommand {
        EnqueueCommand::new(png_image(), png_image())
    }

    fn ttl1() -> TtlSecs {
        TtlSecs::parse(1).expect("ttl 1")
    }

    fn manual_queue() -> (MemoryQueue, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new());
        let clock_obj: Arc<dyn Clock> = clock.clone();
        let q = MemoryQueue::with_ttl_and_clock(ttl1(), clock_obj);
        (q, clock)
    }

    #[tokio::test]
    async fn test_memory_queue_keeps_bytes_and_tracks_progress() {
        let q = MemoryQueue::default();
        let job = q.enqueue(cmd()).await.expect("enqueue");
        assert!(!job.is_r2_pointer());
        assert_eq!(q.stored_lens(&job.job_id()).await.expect("lens"), (64, 64));
        let (p0, s0) = q.progress(&job.job_id()).await.expect("progress");
        assert_eq!(p0, Progress::zero());
        assert_eq!(s0, Stage::Queued);
        let p = Progress::parse(0.4).expect("p");
        q.set_progress(&job.job_id(), p, Stage::Flame)
            .await
            .expect("set");
        assert_eq!(
            q.status(&job.job_id()).await.expect("status"),
            JobStatus::Processing
        );
        let (p1, s1) = q.progress(&job.job_id()).await.expect("progress");
        assert_eq!(p1, p);
        assert_eq!(s1, Stage::Flame);
    }

    #[tokio::test]
    async fn test_r2_pointer_queue_returns_keys() {
        let q = R2PointerQueue::default();
        let job = q.enqueue(cmd()).await.expect("enqueue");
        assert!(job.is_r2_pointer());
        let keys = job.r2_keys().expect("keys");
        assert!(keys.image_a().as_str().contains(&job.job_id().to_string()));
        assert_eq!(q.stored_lens(&job.job_id()).await.expect("lens"), (64, 64));
    }

    #[tokio::test]
    async fn test_unknown_job_is_not_found() {
        let q = MemoryQueue::default();
        let id = JobId::new();
        assert!(q.status(&id).await.is_err());
        assert!(q.progress(&id).await.is_err());
    }

    #[tokio::test]
    async fn test_job_expires_after_ttl_and_lens_gone() {
        let (q, clock) = manual_queue();
        let job = q.enqueue(cmd()).await.expect("enqueue");
        assert_eq!(
            q.status(&job.job_id()).await.expect("status"),
            JobStatus::Queued
        );
        clock.advance(Duration::from_secs(2));
        assert_eq!(
            q.status(&job.job_id()).await.expect("expired status"),
            JobStatus::Expired
        );
        assert!(q.stored_lens(&job.job_id()).await.is_err());
        assert!(q.progress(&job.job_id()).await.is_err());
    }

    #[tokio::test]
    async fn test_expired_job_is_purged_after_double_ttl() {
        let clock = Arc::new(ManualClock::new());
        let clock_obj: Arc<dyn Clock> = clock.clone();
        let q = MemoryQueue::with_ttl_and_clock(ttl1(), clock_obj);
        let job = q.enqueue(cmd()).await.expect("enqueue");
        clock.advance(Duration::from_secs(3));
        let purged = q.purge_expired().await;
        assert_eq!(purged, 1);
        assert!(q.status(&job.job_id()).await.is_err());
    }

    #[tokio::test]
    async fn test_purge_and_ttl_work_via_dyn_queue() {
        // El reaper de `main` solo ve `Arc<dyn Queue>`: ambos adapters
        // deben servir `purge_expired` + `ttl` tras la misma seam.
        let clock: Arc<ManualClock> = Arc::new(ManualClock::new());
        let mk = || -> Arc<dyn Clock> { clock.clone() as Arc<dyn Clock> };
        let ttl = ttl1();
        let queues: Vec<Arc<dyn Queue>> = vec![
            Arc::new(MemoryQueue::with_ttl_and_clock(ttl, mk())),
            Arc::new(R2PointerQueue::with_ttl_and_clock(ttl, mk())),
        ];
        for q in queues {
            assert_eq!(q.ttl(), ttl);
            let job = q.enqueue(cmd()).await.expect("enqueue");
            clock.advance(Duration::from_secs(3));
            assert_eq!(q.purge_expired().await, 1);
            assert!(q.status(&job.job_id()).await.is_err());
        }
    }

    #[tokio::test]
    async fn test_manual_clock_advance_is_deterministic() {
        let clock = ManualClock::new();
        let t0 = Clock::now(&clock);
        clock.advance(Duration::from_secs(5));
        let t1 = Clock::now(&clock);
        assert!(t1.saturating_duration_since(t0) >= Duration::from_secs(5));
    }
}
