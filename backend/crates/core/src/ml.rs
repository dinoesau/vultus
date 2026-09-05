use super::error::{BaseUrlError, CoreError, MlError, Result};
use super::job::{CompleteUv, FlawUv, ImageBytes, JobId, Landmarks};

/// URL base del sidecar ML ya probada: `http(s)://host` sin `/` final.
/// Evita `format!("{}{}", base, path)` con doble slash o esquema roto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseUrl(String);

impl BaseUrl {
    pub fn parse(raw: &str) -> std::result::Result<Self, BaseUrlError> {
        let t = raw.trim();
        if t.is_empty() {
            return Err(BaseUrlError::Empty);
        }
        if !(t.starts_with("http://") || t.starts_with("https://")) {
            return Err(BaseUrlError::BadScheme);
        }
        Ok(Self(t.trim_end_matches('/').to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn join(&self, path: &str) -> String {
        format!("{}{}", self.0, path)
    }
}

impl std::fmt::Display for BaseUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Payload `flame`: `u32 BE len + landmarks_json + image_bytes`.
/// Encode/decode en un solo modulo para paridad Rust-Python.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlamePayload;

impl FlamePayload {
    pub fn encode(landmarks: &Landmarks, image: &ImageBytes) -> Vec<u8> {
        let lm = landmarks.as_bytes();
        let img = image.as_bytes();
        let mut out = Vec::with_capacity(4 + lm.len() + img.len());
        out.extend_from_slice(&(lm.len() as u32).to_be_bytes());
        out.extend_from_slice(lm);
        out.extend_from_slice(img);
        out
    }

    pub fn decode(bytes: Vec<u8>) -> Result<(Landmarks, ImageBytes)> {
        if bytes.len() < 4 {
            return Err(CoreError::Ml(MlError::Decode {
                details: "flame payload <4 bytes".to_string(),
            }));
        }
        let n = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if bytes.len() < 4 + n {
            return Err(CoreError::Ml(MlError::Decode {
                details: "flame payload truncated".to_string(),
            }));
        }
        let (lm_raw, img_raw) = (bytes[4..4 + n].to_vec(), bytes[4 + n..].to_vec());
        let lm = Landmarks::parse(lm_raw)?;
        let img = ImageBytes::parse(img_raw).map_err(|e| {
            CoreError::Ml(MlError::Decode {
                details: e.to_string(),
            })
        })?;
        Ok((lm, img))
    }
}

/// Frontera Rust --bytes--> Python ML.
/// Rust nunca importa torch/diffusers/mediapipe.
/// Todo el ML GPU vive tras este cliente HTTP (sidecar Python en Modal).
#[derive(Debug, Clone)]
pub struct MlSidecarClient {
    base_url: BaseUrl,
    inner: reqwest::Client,
}

impl MlSidecarClient {
    pub fn new(base_url: BaseUrl) -> Self {
        Self {
            base_url,
            inner: reqwest::Client::new(),
        }
    }

    pub fn base_url(&self) -> &BaseUrl {
        &self.base_url
    }

    async fn post_bytes(&self, path: &str, job_id: &JobId, bytes: &[u8]) -> Result<Vec<u8>> {
        let url = self.base_url.join(path);
        let resp = self
            .inner
            .post(&url)
            .header("X-Job-Id", job_id.to_string())
            .header("Content-Type", "application/octet-stream")
            .body(bytes.to_vec())
            .send()
            .await
            .map_err(|e| {
                CoreError::Ml(MlError::Transport {
                    details: e.to_string(),
                })
            })?;
        if !resp.status().is_success() {
            return Err(CoreError::Ml(MlError::BadStatus {
                status: resp.status().as_u16(),
            }));
        }
        let body = resp.bytes().await.map_err(|e| {
            CoreError::Ml(MlError::Transport {
                details: e.to_string(),
            })
        })?;
        if body.is_empty() {
            return Err(CoreError::Ml(MlError::Empty));
        }
        Ok(body.to_vec())
    }

    /// Worker 1: MediaPipe 478 landmarks (CPU en sidecar Python).
    pub async fn landmarks(&self, job_id: &JobId, image: &ImageBytes) -> Result<Landmarks> {
        let raw = self
            .post_bytes("/ml/landmarks", job_id, image.as_bytes())
            .await?;
        Landmarks::parse(raw)
    }

    /// Worker 2: FLAME fitting (GPU en sidecar Python).
    pub async fn flame(
        &self,
        job_id: &JobId,
        image: &ImageBytes,
        landmarks: &Landmarks,
    ) -> Result<FlawUv> {
        let payload = FlamePayload::encode(landmarks, image);
        let raw = self.post_bytes("/ml/flame", job_id, &payload).await?;
        FlawUv::parse(raw)
    }

    /// Worker 3: FreeUV inpainting (GPU en sidecar Python, cuello de botella).
    pub async fn freeuv(&self, job_id: &JobId, flaw_uv: &FlawUv) -> Result<CompleteUv> {
        let raw = self
            .post_bytes("/ml/freeuv", job_id, flaw_uv.as_bytes())
            .await?;
        CompleteUv::parse(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_image() -> ImageBytes {
        let mut b = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        b.resize(64, 0);
        ImageBytes::parse(b).expect("png fixture")
    }

    fn landmarks_bytes() -> Vec<u8> {
        let pts = vec![[0.0f32, 0.0, 0.0]; crate::job::LANDMARKS_LEN];
        serde_json::to_vec(&pts).expect("landmarks fixture")
    }

    #[test]
    fn test_base_url_trims_slash_and_rejects_scheme() {
        let u = BaseUrl::parse("https://ml.internal:8081/").expect("valid");
        assert_eq!(u.as_str(), "https://ml.internal:8081");
        assert!(BaseUrl::parse("ml.internal:8081").is_err());
        assert!(BaseUrl::parse("  ").is_err());
    }

    #[test]
    fn test_flame_payload_roundtrips() {
        let lm = Landmarks::parse(landmarks_bytes()).expect("lm");
        let img = png_image();
        let enc = FlamePayload::encode(&lm, &img);
        let (lm2, img2) = FlamePayload::decode(enc).expect("decode");
        assert_eq!(lm, lm2);
        assert_eq!(img.as_bytes(), img2.as_bytes());
    }

    #[test]
    fn test_flame_payload_rejects_truncated() {
        assert!(FlamePayload::decode(vec![0, 0]).is_err());
        assert!(FlamePayload::decode(vec![0, 0, 0, 10, 1, 2]).is_err());
    }
}
