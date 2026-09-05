use vultus_api::{router, AppState};
use vultus_core::MemoryQueue;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let state = AppState::new(MemoryQueue::default());
    let app = router(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    tracing::info!("vultus-api on :8000");
    axum::serve(listener, app).await.unwrap();
}
