use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::error::{CoreError, Result};
use super::job::{JobId, JobStatus, Progress, R2Keys};

/// Contrato Seam 2. Mismo adapter para local (Redis) y prod (Queues+R2).
/// En local se encolan bytes; en prod solo `{job_id, r2_keys}` (Queues <128KB).
#[async_trait]
pub trait Queue: Send + Sync {
    async fn enqueue(&self, image_a: Vec<u8>, image_b: Vec<u8>) -> Result<EnqueuedJob>;
    async fn status(&self, job_id: &JobId) -> Result<JobStatus>;
    async fn set_progress(&self, job_id: &JobId, progress: Progress, stage: &str) -> Result<()>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnqueuedJob {
    pub job_id: JobId,
    /// Some en local (bytes directos), None en prod (usar r2_keys).
    pub r2_keys: Option<R2Keys>,
}

#[derive(Debug, Clone)]
struct MemoryEntry {
    status: JobStatus,
    progress: f32,
    stage: String,
}

/// Adapter en memoria para tests y tracer bullet Fase 0.
/// Paridad con `fakeredis` del diseño Python original.
#[derive(Debug, Default, Clone)]
pub struct MemoryQueue {
    inner: Arc<RwLock<HashMap<String, MemoryEntry>>>,
}

#[async_trait]
impl Queue for MemoryQueue {
    async fn enqueue(&self, _a: Vec<u8>, _b: Vec<u8>) -> Result<EnqueuedJob> {
        let job_id = JobId::new();
        self.inner.write().await.insert(
            job_id.as_str(),
            MemoryEntry {
                status: JobStatus::Queued,
                progress: 0.0,
                stage: "queued".to_string(),
            },
        );
        Ok(EnqueuedJob {
            job_id,
            r2_keys: None,
        })
    }

    async fn status(&self, job_id: &JobId) -> Result<JobStatus> {
        self.inner
            .read()
            .await
            .get(&job_id.as_str())
            .map(|e| e.status)
            .ok_or_else(|| CoreError::NotFound(job_id.as_str()))
    }

    async fn set_progress(&self, job_id: &JobId, progress: Progress, stage: &str) -> Result<()> {
        let mut w = self.inner.write().await;
        let e = w
            .get_mut(&job_id.as_str())
            .ok_or_else(|| CoreError::NotFound(job_id.as_str()))?;
        e.progress = progress.value();
        e.stage = stage.to_string();
        e.status = JobStatus::Processing;
        Ok(())
    }
}
