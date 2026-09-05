use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid image: {0}")]
    InvalidImage(&'static str),
    #[error("queue error: {0}")]
    Queue(String),
    #[error("ml sidecar error: {0}")]
    Ml(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invariant violated: {0}")]
    Invariant(&'static str),
}
