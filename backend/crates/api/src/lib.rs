use std::sync::Arc;

use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use vultus_core::{ImageBytes, JobId, MemoryQueue, Queue};

#[derive(Clone)]
pub struct AppState {
    pub queue: Arc<MemoryQueue>,
}

impl AppState {
    pub fn new(queue: MemoryQueue) -> Self {
        Self {
            queue: Arc::new(queue),
        }
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
    pub status: String,
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
async fn compare(State(state): State<AppState>, mut multipart: Multipart) -> impl IntoResponse {
    let mut image_a: Option<Vec<u8>> = None;
    let mut image_b: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        let bytes = field.bytes().await.unwrap_or_default().to_vec();
        match name.as_str() {
            "image_a" => image_a = Some(bytes),
            "image_b" => image_b = Some(bytes),
            _ => {}
        }
    }

    let (a, b) = match (image_a, image_b) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"detail": "missing image_a or image_b"})),
            )
                .into_response();
        }
    };

    let (a, b) = match (ImageBytes::parse(a), ImageBytes::parse(b)) {
        (Ok(a), Ok(b)) => (a.into_bytes(), b.into_bytes()),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"detail": "invalid image"})),
            )
                .into_response();
        }
    };

    match state.queue.enqueue(a, b).await {
        Ok(job) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({"job_id": job.job_id.as_str(), "status": "queued"})),
        )
            .into_response(),
        Err(e) => {
            let msg = format!("{e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"detail": msg})),
            )
                .into_response()
        }
    }
}

async fn job_status(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    let job_id = match JobId::parse(&id) {
        Ok(j) => j,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"detail": "invalid job_id"})),
            )
                .into_response();
        }
    };
    match state.queue.status(&job_id).await {
        Ok(s) => (
            StatusCode::OK,
            Json(serde_json::json!({"job_id": job_id.as_str(), "status": format!("{s:?}").to_lowercase()})),
        )
            .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"detail": "not found"})),
        )
            .into_response(),
    }
}
