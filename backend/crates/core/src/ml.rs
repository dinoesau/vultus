use super::error::{CoreError, Result};
use super::job::JobId;

/// Frontera Rust --bytes--> Python ML.
/// Rust nunca importa torch/diffusers/mediapipe.
/// Todo el ML GPU vive tras este cliente HTTP (sidecar Python en Modal).
#[derive(Debug, Clone)]
pub struct MlSidecarClient {
    pub base_url: String,
    inner: reqwest::Client,
}

impl MlSidecarClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            inner: reqwest::Client::new(),
        }
    }

    async fn post_bytes(&self, path: &str, job_id: &JobId, bytes: &[u8]) -> Result<Vec<u8>> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .inner
            .post(&url)
            .header("X-Job-Id", job_id.as_str())
            .header("Content-Type", "application/octet-stream")
            .body(bytes.to_vec())
            .send()
            .await
            .map_err(|e| CoreError::Ml(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(CoreError::Ml(format!("sidecar {}", resp.status())));
        }
        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| CoreError::Ml(e.to_string()))
    }

    /// Worker 1: MediaPipe 478 landmarks (CPU en sidecar Python).
    pub async fn landmarks(&self, job_id: &JobId, image: &[u8]) -> Result<Vec<u8>> {
        self.post_bytes("/ml/landmarks", job_id, image).await
    }

    /// Worker 2: FLAME fitting (GPU en sidecar Python).
    pub async fn flame(&self, job_id: &JobId, image: &[u8], landmarks: &[u8]) -> Result<Vec<u8>> {
        let mut payload = landmarks.to_vec();
        payload.extend_from_slice(image);
        self.post_bytes("/ml/flame", job_id, &payload).await
    }

    /// Worker 3: FreeUV inpainting (GPU en sidecar Python, cuello de botella).
    pub async fn freeuv(&self, job_id: &JobId, flaw_uv: &[u8]) -> Result<Vec<u8>> {
        self.post_bytes("/ml/freeuv", job_id, flaw_uv).await
    }
}
