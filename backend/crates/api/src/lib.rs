pub mod config;

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart, Path, State,
    },
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use thiserror::Error;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use vultus_core::{
    CoreError, EnqueueCommand, ImageBytes, JobId, JobStatus, MlSidecarClient, PipelineConfig,
    Queue, TtlSecs, UV_HEIGHT, UV_WIDTH,
};

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
    sidecar: Option<MlSidecarClient>,
}

impl AppState {
    pub fn new(queue: impl Queue + 'static) -> Self {
        Self {
            queue: Arc::new(queue),
            ttl: TtlSecs::default(),
            sidecar: None,
        }
    }

    pub fn with_ttl(queue: impl Queue + 'static, ttl: TtlSecs) -> Self {
        Self {
            queue: Arc::new(queue),
            ttl,
            sidecar: None,
        }
    }

    /// Construye desde un `Arc<dyn Queue>` ya elegido por driver.
    /// Evita ramificar el reaper por adapter en `main`.
    pub fn from_arc(queue: Arc<dyn Queue>, ttl: TtlSecs) -> Self {
        Self {
            queue,
            ttl,
            sidecar: None,
        }
    }

    pub fn with_ttl_and_sidecar(
        queue: impl Queue + 'static,
        ttl: TtlSecs,
        sidecar: Option<MlSidecarClient>,
    ) -> Self {
        Self {
            queue: Arc::new(queue),
            ttl,
            sidecar,
        }
    }

    pub fn from_arc_with_sidecar(
        queue: Arc<dyn Queue>,
        ttl: TtlSecs,
        sidecar: Option<MlSidecarClient>,
    ) -> Self {
        Self {
            queue,
            ttl,
            sidecar,
        }
    }

    pub fn queue(&self) -> &Arc<dyn Queue> {
        &self.queue
    }

    pub fn sidecar(&self) -> Option<&MlSidecarClient> {
        self.sidecar.as_ref()
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
    pub sidecar: &'static str,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/compare", post(compare))
        .route("/v1/jobs/:id", get(job_status))
        .route("/v1/jobs/:id/result", get(job_result))
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
    let sidecar_state: &'static str = match state.sidecar.as_ref() {
        None => "disabled",
        Some(client) => {
            // Probe rapido 1s a base_url/docs. Cualquier respuesta HTTP
            // (incluso 404) prueba reachability; solo error de transporte es error.
            let base = client.base_url().as_str().to_string();
            let url = format!("{base}/docs");
            let http = reqwest::Client::builder()
                .timeout(Duration::from_secs(1))
                .build();
            match http {
                Err(_) => "error",
                Ok(http) => match http.get(&url).send().await {
                    Ok(_) => "ok",
                    Err(_) => "error",
                },
            }
        }
    };
    let ok = queue_ok && sidecar_state != "error";
    let (status_code, body) = if ok {
        (
            StatusCode::OK,
            HealthResponse {
                status: "ok",
                queue: "ok",
                ttl_secs: state.ttl.value(),
                sidecar: sidecar_state,
            },
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            HealthResponse {
                status: "error",
                queue: if queue_ok { "ok" } else { "error" },
                ttl_secs: state.ttl.value(),
                sidecar: sidecar_state,
            },
        )
    };
    (status_code, Json(body))
}

/// Seam 1: valida en borde con `ImageBytes::parse`, nunca en core.
/// Retorna 202 + job_id si ambas imágenes son jpeg/png <=8MB.
/// Si hay sidecar, spawnea `run_pair` en background sin bloquear el 202.
/// Sin sidecar el job queda queued (tracer Fase0, no flakea tests).
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

    let job = state
        .queue
        .enqueue(EnqueueCommand::new(a.clone(), b.clone()))
        .await?;
    tracing::info!(job_id = %job.job_id(), r2_pointer = job.is_r2_pointer(), "job enqueued");

    if let Some(ml) = state.sidecar.clone() {
        let queue = state.queue.clone();
        let job_id = job.job_id();
        tokio::spawn(async move {
            let start = std::time::Instant::now();
            let cfg = PipelineConfig::default();
            match vultus_core::pipeline::run_pair(&queue, &ml, &job_id, &a, &b, &cfg).await {
                Ok(_) => {
                    tracing::info!(job_id = %job_id, duration_ms = start.elapsed().as_millis(), "pipeline done");
                }
                Err(e) => {
                    tracing::warn!(job_id = %job_id, duration_ms = start.elapsed().as_millis(), error = %e, "pipeline failed");
                }
            }
        });
    }

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

/// GET /v1/jobs/:id/result: zip canonico en memoria con 3 PNG RGB 512x512.
/// `fetch_result` da NotFound si expiro/purgo o aun no hay resultado: 404
/// para no esperar en vano (`/status` sigue mostrando `expired`).
/// Sin mesh.glb ni report.pdf (Fase2/3). Log solo lens + duration, nunca bytes.
async fn job_result(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let start = std::time::Instant::now();
    let job_id = JobId::parse(&id).map_err(AppError::Domain)?;
    let result = state.queue.fetch_result(&job_id).await?;
    let (uv_a, uv_b, heatmap) = result.into_parts();
    let a_png = encode_uv_png(uv_a.as_bytes()).map_err(AppError::Domain)?;
    let b_png = encode_uv_png(uv_b.as_bytes()).map_err(AppError::Domain)?;
    let h_png = encode_uv_png(heatmap.as_bytes()).map_err(AppError::Domain)?;
    let zip_bytes = build_result_zip(&a_png, &b_png, &h_png).map_err(AppError::Domain)?;
    let zip_len = zip_bytes.len();
    tracing::info!(
        job_id = %job_id,
        zip_len = zip_len,
        duration_ms = start.elapsed().as_millis(),
        "result zip served"
    );
    let filename = format!("attachment; filename=\"result-{job_id}.zip\"");
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/zip"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&filename)
            .map_err(|_| AppError::Domain(CoreError::Invariant("filename")))?,
    );
    Ok((StatusCode::OK, headers, zip_bytes))
}

/// Convierte UV raw RGB UV_LEN a PNG 512x512 RGB en memoria, sin disco.
fn encode_uv_png(raw: &[u8]) -> Result<Vec<u8>, CoreError> {
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(UV_WIDTH as u32, UV_HEIGHT as u32, raw.to_vec())
            .ok_or(CoreError::Invariant("uv dims"))?;
    let mut out = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut out);
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|_| CoreError::Invariant("png encode failed"))?;
    }
    Ok(out)
}

/// Empaqueta los 3 PNG con nombres exactos, compresion Stored (determinista).
fn build_result_zip(a_png: &[u8], b_png: &[u8], h_png: &[u8]) -> Result<Vec<u8>, CoreError> {
    use std::io::Write;
    use zip::{write::SimpleFileOptions, CompressionMethod};
    let cursor = std::io::Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(cursor);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("uv_a.png", options)
        .map_err(|_| CoreError::Invariant("zip encode failed"))?;
    zip.write_all(a_png)
        .map_err(|_| CoreError::Invariant("zip encode failed"))?;
    zip.start_file("uv_b.png", options)
        .map_err(|_| CoreError::Invariant("zip encode failed"))?;
    zip.write_all(b_png)
        .map_err(|_| CoreError::Invariant("zip encode failed"))?;
    zip.start_file("heatmap.png", options)
        .map_err(|_| CoreError::Invariant("zip encode failed"))?;
    zip.write_all(h_png)
        .map_err(|_| CoreError::Invariant("zip encode failed"))?;
    let cursor = zip
        .finish()
        .map_err(|_| CoreError::Invariant("zip encode failed"))?;
    Ok(cursor.into_inner())
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
    // Hasta 120 ticks (~60s) para cubrir total_timeout 60s sin colgar.
    // Cierra en terminal Done/Failed/Expired.
    for _ in 0..120 {
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
