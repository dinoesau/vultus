use std::sync::Arc;

use anyhow::Context;
use vultus_api::{router, AppState, Config, QueueDriver};
use vultus_core::{MemoryQueue, Queue, R2PointerQueue, TtlSecs};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let config = Config::parse_env().context("invalid config from env")?;
    tracing::info!(
        port = %config.port,
        driver = ?config.driver,
        ttl_secs = config.ttl.value(),
        "vultus-api starting"
    );

    // Reaper stateless: una sola tarea sobre `Arc<dyn Queue>`.
    // El intervalo vive en `TtlSecs::reaper_interval` (TTL/2) y la purga
    // en `Queue::purge_expired` (2x TTL). Sin ramas por driver.
    let queue: Arc<dyn Queue> = match config.driver {
        QueueDriver::Memory => Arc::new(MemoryQueue::with_ttl(config.ttl)),
        QueueDriver::R2Pointer => Arc::new(R2PointerQueue::with_ttl(config.ttl)),
    };
    spawn_reaper(queue.clone(), config.ttl);
    let state = AppState::from_arc(queue, config.ttl);

    let app = router(state);
    let listener = tokio::net::TcpListener::bind(config.bind_addr())
        .await
        .context("failed to bind")?;
    tracing::info!("vultus-api on {}", config.bind_addr());
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server crashed")?;
    Ok(())
}

fn spawn_reaper(queue: Arc<dyn Queue>, ttl: TtlSecs) {
    let interval = ttl.reaper_interval();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            let n = queue.purge_expired().await;
            if n > 0 {
                tracing::info!(purged = n, "ttl reaper purged jobs");
            }
        }
    });
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
