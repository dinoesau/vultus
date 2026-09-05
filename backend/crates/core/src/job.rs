use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::error::{CoreError, Result};

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
        Uuid::parse_str(s)
            .map(Self)
            .map_err(|_| CoreError::InvalidImage("invalid job_id"))
    }

    pub fn as_str(&self) -> String {
        self.0.to_string()
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

/// Validated image bytes. Parse at the edge, never validate in core.
#[derive(Debug, Clone)]
pub struct ImageBytes(Vec<u8>);

impl ImageBytes {
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
            return Err(CoreError::InvalidImage("size out of range"));
        }
        if !is_jpeg(&bytes) && !is_png(&bytes) {
            return Err(CoreError::InvalidImage("not jpeg nor png"));
        }
        Ok(Self(bytes))
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

/// Progress 0.0..=1.0 emitted per stage. Branded so out-of-range
/// values cannot cross into core.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Progress(f32);

impl Progress {
    pub fn parse(v: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&v) || v.is_nan() {
            return Err(CoreError::InvalidImage("progress out of range"));
        }
        Ok(Self(v))
    }

    pub fn value(self) -> f32 {
        self.0
    }
}

/// Prod pointer: Queues limit 128KB/msg, bytes live in R2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R2Keys {
    pub image_a: String,
    pub image_b: String,
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
