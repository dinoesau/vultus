use std::sync::Arc;

use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use thiserror::Error;
use vultus_core::{CoreError, EnqueueCommand, ImageBytes, JobId, Queue};

#[derive(Debug, Error)]
pub enum AppError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error(transparent)]
    Domain(#[from] CoreError),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, detail) = match &self {
            Self::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            Self::Domain(
                CoreError::InvalidImage(_)
                | CoreError::InvalidJobId
                | CoreError::InvalidProgress
                | CoreError::InvalidR2Key
                | CoreError::InvalidBaseUrl(_)
                | CoreError::Empty,
            ) => (StatusCode::BAD_REQUEST, self.to_string()),
            Self::Domain(CoreError::NotFound(_)) => (StatusCode::NOT_FOUND, self.to_string()),
            Self::Domain(CoreError::Queue(_) | CoreError::Ml(_) | CoreError::Invariant(_)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal error".to_string(),
            ),
        };
        (status, Json(serde_json::json!({"detail": detail}))).into_response()
    }
}

#[derive(Clone)]
pub struct AppState {
    queue: Arc<dyn Queue>,
}

impl AppState {
    pub fn new(queue: impl Queue + 'static) -> Self {
        Self {
            queue: Arc::new(queue),
        }
    }

    pub fn queue(&self) -> &Arc<dyn Queue> {
        &self.queue
    }
}

#[derive(Serialize)]
pub struct CompareResponse {
    pub job_id: String,
    pub status: &'static str,
}

#[derive(Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub status: &'static str,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/compare", post(compare))
        .route("/v1/jobs/:id", get(job_status))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

/// Seam 1: valida en borde con `ImageBytes::parse`, nunca en core.
/// Retorna 202 + job_id si ambas imágenes son jpeg/png <=8MB.
async fn compare(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, AppError> {
    let mut image_a: Option<Vec<u8>> = None;
    let mut image_b: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("invalid multipart: {e}")))?
    {
        let name = field
            .name()
            .ok_or_else(|| AppError::BadRequest("missing field name".to_string()))?
            .to_string();
        let bytes = field
            .bytes()
            .await
            .map_err(|e| AppError::BadRequest(format!("invalid field bytes: {e}")))?
            .to_vec();
        match name.as_str() {
            "image_a" => image_a = Some(bytes),
            "image_b" => image_b = Some(bytes),
            _ => {}
        }
    }

    let (a_raw, b_raw) = match (image_a, image_b) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return Err(AppError::BadRequest(
                "missing image_a or image_b".to_string(),
            ));
        }
    };

    let a = ImageBytes::parse(a_raw).map_err(AppError::Domain)?;
    let b = ImageBytes::parse(b_raw).map_err(AppError::Domain)?;

    let job = state.queue.enqueue(EnqueueCommand::new(a, b)).await?;
    let body = CompareResponse {
        job_id: job.job_id().to_string(),
        status: "queued",
    };
    Ok((StatusCode::ACCEPTED, Json(body)))
}

async fn job_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let job_id = JobId::parse(&id).map_err(AppError::Domain)?;
    let status = state.queue.status(&job_id).await?;
    let body = JobResponse {
        job_id: job_id.to_string(),
        status: status.as_str(),
    };
    Ok((StatusCode::OK, Json(body)))
}
