use anyhow::Context;
use vultus_api::{router, AppState};
use vultus_core::MemoryQueue;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let state = AppState::new(MemoryQueue::default());
    let app = router(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000")
        .await
        .context("failed to bind :8000")?;
    tracing::info!("vultus-api on :8000");
    axum::serve(listener, app).await.context("server crashed")?;
    Ok(())
}
