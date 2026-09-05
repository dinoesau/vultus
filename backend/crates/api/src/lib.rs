pub mod config;

use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart, Path, State,
    },
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use thiserror::Error;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use vultus_core::{CoreError, EnqueueCommand, ImageBytes, JobId, JobStatus, Queue, TtlSecs};

pub use config::{Config, ConfigError, Port, QueueDriver};

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
    ttl: TtlSecs,
}

impl AppState {
    pub fn new(queue: impl Queue + 'static) -> Self {
        Self {
            queue: Arc::new(queue),
            ttl: TtlSecs::default(),
        }
    }

    pub fn with_ttl(queue: impl Queue + 'static, ttl: TtlSecs) -> Self {
        Self {
            queue: Arc::new(queue),
            ttl,
        }
    }

    /// Construye desde un `Arc<dyn Queue>` ya elegido por driver.
    /// Evita ramificar el reaper por adapter en `main`.
    pub fn from_arc(queue: Arc<dyn Queue>, ttl: TtlSecs) -> Self {
        Self { queue, ttl }
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

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub queue: &'static str,
    pub ttl_secs: u64,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/compare", post(compare))
        .route("/v1/jobs/:id", get(job_status))
        .route("/v1/jobs/:id/events", get(job_events))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    // Ping real al Store: un id aleatorio debe dar NotFound si el Store responde.
    // Cualquier otro error seria 500, pero el trait solo da NotFound aqui.
    let probe = JobId::new();
    let queue_ok = match state.queue.status(&probe).await {
        Err(CoreError::NotFound(_)) => true,
        Ok(_) => true,
        Err(_) => false,
    };
    if queue_ok {
        (
            StatusCode::OK,
            Json(HealthResponse {
                status: "ok",
                queue: "ok",
                ttl_secs: state.ttl.value(),
            }),
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(HealthResponse {
                status: "error",
                queue: "error",
                ttl_secs: state.ttl.value(),
            }),
        )
    }
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
    tracing::info!(job_id = %job.job_id(), r2_pointer = job.is_r2_pointer(), "job enqueued");
    let body = CompareResponse {
        job_id: job.job_id().to_string(),
        status: JobStatus::Queued.as_str(),
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

/// WS de progreso: valida `JobId`, verifica existencia, luego hace upgrade.
/// Emite snapshot inicial + poll cada 500ms. Cierra si el cliente se va.
async fn job_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ws: WebSocketUpgrade,
) -> Result<impl IntoResponse, AppError> {
    let job_id = JobId::parse(&id).map_err(AppError::Domain)?;
    // 404 temprano si el job no existe o ya expiro del todo.
    // `Expired` si se distingue: aun existe como expirado, permitimos WS para verlo.
    state.queue.status(&job_id).await?;
    Ok(ws.on_upgrade(move |socket| handle_socket(socket, state.queue.clone(), job_id)))
}

async fn handle_socket(mut socket: WebSocket, queue: Arc<dyn Queue>, job_id: JobId) {
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));
    // Hasta 60 ticks (~30s) para no dejar conexiones colgadas en Fase 0.
    for _ in 0..60 {
        interval.tick().await;
        let (status, progress, stage) = match queue.status(&job_id).await {
            Err(_) => break,
            Ok(s) => {
                let (p, st) = queue.progress(&job_id).await.unwrap_or_else(|_| {
                    (vultus_core::Progress::zero(), vultus_core::Stage::Queued)
                });
                (s, p, st)
            }
        };
        let payload = serde_json::json!({
            "job_id": job_id.to_string(),
            "status": status.as_str(),
            "progress": progress.value(),
            "stage": stage.as_str(),
        });
        if socket
            .send(Message::Text(payload.to_string()))
            .await
            .is_err()
        {
            break;
        }
        if status == JobStatus::Done || status == JobStatus::Failed || status == JobStatus::Expired
        {
            break;
        }
    }
}
