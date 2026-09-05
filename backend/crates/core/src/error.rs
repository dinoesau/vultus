use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ImageError {
    #[error("size out of range")]
    SizeOutOfRange,
    #[error("not jpeg nor png")]
    UnsupportedFormat,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BaseUrlError {
    #[error("base_url must start with http:// or https://")]
    BadScheme,
    #[error("base_url is empty")]
    Empty,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MlError {
    #[error("ml transport: {details}")]
    Transport { details: String },
    #[error("ml sidecar status: {status}")]
    BadStatus { status: u16 },
    #[error("ml decode: {details}")]
    Decode { details: String },
    #[error("ml empty payload")]
    Empty,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum QueueError {
    #[error("queue backend: {details}")]
    Backend { details: String },
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CoreError {
    #[error("invalid image: {0}")]
    InvalidImage(#[from] ImageError),
    #[error("invalid job_id")]
    InvalidJobId,
    #[error("invalid progress")]
    InvalidProgress,
    #[error("invalid r2_key")]
    InvalidR2Key,
    #[error("invalid base_url: {0}")]
    InvalidBaseUrl(#[from] BaseUrlError),
    #[error("empty payload")]
    Empty,
    #[error("queue error: {0}")]
    Queue(#[from] QueueError),
    #[error("ml sidecar error: {0}")]
    Ml(#[from] MlError),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invariant violated: {0}")]
    Invariant(&'static str),
}

impl CoreError {
    pub fn not_found(id: impl Into<String>) -> Self {
        Self::NotFound(id.into())
    }

    pub fn queue_backend(details: impl Into<String>) -> Self {
        Self::Queue(QueueError::Backend {
            details: details.into(),
        })
    }

    pub fn ml_transport(details: impl Into<String>) -> Self {
        Self::Ml(MlError::Transport {
            details: details.into(),
        })
    }
}
