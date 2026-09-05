use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::error::{CoreError, Result};
use super::job::{EnqueueCommand, JobId, JobStatus, Progress, R2Key, R2Keys, Stage};

/// Contrato Seam 2. Mismo adapter para local (Redis) y prod (Queues+R2).
/// En local se encolan bytes; en prod solo `{job_id, r2_keys}` (Queues <128KB).
#[async_trait]
pub trait Queue: Send + Sync {
    async fn enqueue(&self, cmd: EnqueueCommand) -> Result<EnqueuedJob>;
    async fn status(&self, job_id: &JobId) -> Result<JobStatus>;
    async fn progress(&self, job_id: &JobId) -> Result<(Progress, Stage)>;
    async fn set_progress(&self, job_id: &JobId, progress: Progress, stage: Stage) -> Result<()>;
    async fn stored_lens(&self, job_id: &JobId) -> Result<(usize, usize)>;
}

/// Recibo de `enqueue`: `job_id` siempre, `r2_keys` solo en prod.
/// Campos privados: solo construible via `Queue::enqueue`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnqueuedJob {
    job_id: JobId,
    /// None en local (bytes directos en queue), Some en prod (R2 pointer).
    r2_keys: Option<R2Keys>,
}

impl EnqueuedJob {
    pub fn new(job_id: JobId, r2_keys: Option<R2Keys>) -> Self {
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
}

fn not_found(job_id: &JobId) -> CoreError {
    CoreError::not_found(job_id.to_string())
}

/// Estado compartido tras ambos adapters de Seam 2.
/// Un solo lugar para ciclo de vida: inserta, lee estado, avanza progreso.
/// Los adapters solo difieren en `enqueue` (bytes directos vs R2 pointer).
#[derive(Debug, Default, Clone)]
struct Store {
    inner: Arc<RwLock<HashMap<JobId, MemoryEntry>>>,
}

impl Store {
    async fn insert_queued(&self, job_id: JobId, image_a_len: usize, image_b_len: usize) {
        self.inner.write().await.insert(
            job_id,
            MemoryEntry {
                status: JobStatus::Queued,
                progress: Progress::zero(),
                stage: Stage::Queued,
                image_a_len,
                image_b_len,
            },
        );
    }

    async fn status(&self, job_id: &JobId) -> Result<JobStatus> {
        self.inner
            .read()
            .await
            .get(job_id)
            .map(|e| e.status)
            .ok_or_else(|| not_found(job_id))
    }

    async fn progress(&self, job_id: &JobId) -> Result<(Progress, Stage)> {
        self.inner
            .read()
            .await
            .get(job_id)
            .map(|e| (e.progress, e.stage))
            .ok_or_else(|| not_found(job_id))
    }

    async fn set_progress(
        &self,
        job_id: &JobId,
        progress: Progress,
        stage: Stage,
    ) -> Result<()> {
        let mut w = self.inner.write().await;
        let e = w.get_mut(job_id).ok_or_else(|| not_found(job_id))?;
        e.progress = progress;
        e.stage = stage;
        e.status = JobStatus::Processing;
        Ok(())
    }

    async fn stored_lens(&self, job_id: &JobId) -> Result<(usize, usize)> {
        self.inner
            .read()
            .await
            .get(job_id)
            .map(|e| (e.image_a_len, e.image_b_len))
            .ok_or_else(|| not_found(job_id))
    }
}

/// Adapter en memoria para tests y tracer bullet Fase 0.
/// Paridad con `fakeredis` del diseno Python original.
/// Guarda longitudes para probar que los bytes fluyen y no se tiran.
#[derive(Debug, Default, Clone)]
pub struct MemoryQueue {
    store: Store,
}

#[async_trait]
impl Queue for MemoryQueue {
    async fn enqueue(&self, cmd: EnqueueCommand) -> Result<EnqueuedJob> {
        let (a, b) = cmd.into_pair();
        let job_id = JobId::new();
        self.store
            .insert_queued(job_id, a.as_bytes().len(), b.as_bytes().len())
            .await;
        Ok(EnqueuedJob::new(job_id, None))
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
}

/// Segundo adapter: simula prod `Queues+R2` con patron R2 pointer.
/// Mismo contrato, distinto transporte: retorna `Some(R2Keys)`.
/// Dos adapters = seam real, no hipotetico.
#[derive(Debug, Default, Clone)]
pub struct R2PointerQueue {
    store: Store,
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
        self.store
            .insert_queued(job_id, a.as_bytes().len(), b.as_bytes().len())
            .await;
        Ok(EnqueuedJob::new(job_id, Some(r2_keys)))
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
}
