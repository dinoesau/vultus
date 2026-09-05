use futures_util::StreamExt;
use std::net::SocketAddr;
use vultus_api::{router, AppState};
use vultus_core::{JobId, MemoryQueue, Progress, Queue, Stage};

fn png_bytes() -> Vec<u8> {
    let mut b = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    b.resize(64, 0);
    b
}

async fn spawn_server() -> (SocketAddr, MemoryQueue) {
    let queue = MemoryQueue::default();
    let app = router(AppState::new(queue.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    (addr, queue)
}

async fn create_job(addr: SocketAddr) -> String {
    let client = reqwest::Client::new();
    let form = reqwest::multipart::Form::new()
        .part(
            "image_a",
            reqwest::multipart::Part::bytes(png_bytes())
                .file_name("a.png")
                .mime_str("image/png")
                .expect("mime"),
        )
        .part(
            "image_b",
            reqwest::multipart::Part::bytes(png_bytes())
                .file_name("b.png")
                .mime_str("image/png")
                .expect("mime"),
        );
    let resp = client
        .post(format!("http://{addr}/v1/compare"))
        .multipart(form)
        .send()
        .await
        .expect("post compare");
    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);
    let body: serde_json::Value = resp.json().await.expect("json");
    body.get("job_id")
        .and_then(|v| v.as_str())
        .expect("job_id")
        .to_string()
}

async fn read_first_ws_event(addr: SocketAddr, job_id: &str) -> serde_json::Value {
    let url = format!("ws://{addr}/v1/jobs/{job_id}/events");
    let (mut ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws handshake");
    let msg = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
        .await
        .expect("ws timeout")
        .expect("stream ended")
        .expect("ws msg");
    let text = msg.into_text().expect("text");
    serde_json::from_str(&text).expect("json event")
}

#[tokio::test]
async fn test_ws_emits_queued_snapshot_for_new_job() {
    let (addr, _queue) = spawn_server().await;
    let job_id = create_job(addr).await;
    let event = read_first_ws_event(addr, &job_id).await;
    assert_eq!(
        event.get("job_id").and_then(|v| v.as_str()),
        Some(job_id.as_str())
    );
    assert_eq!(event.get("status").and_then(|v| v.as_str()), Some("queued"));
    assert_eq!(event.get("stage").and_then(|v| v.as_str()), Some("queued"));
    assert_eq!(event.get("progress").and_then(|v| v.as_f64()), Some(0.0));
}

#[tokio::test]
async fn test_ws_reflects_progress_after_set_progress() {
    let (addr, queue) = spawn_server().await;
    let job_id = create_job(addr).await;
    let id = JobId::parse(&job_id).expect("job id");
    queue
        .set_progress(&id, Progress::parse(0.4).expect("p"), Stage::Flame)
        .await
        .expect("set_progress");
    let event = read_first_ws_event(addr, &job_id).await;
    assert_eq!(
        event.get("status").and_then(|v| v.as_str()),
        Some("processing")
    );
    assert_eq!(event.get("stage").and_then(|v| v.as_str()), Some("flame"));
    let progress = event
        .get("progress")
        .and_then(|v| v.as_f64())
        .expect("progress");
    assert!((progress - 0.4).abs() < 0.001, "progress {progress}");
}

#[tokio::test]
async fn test_ws_unknown_job_handshake_fails() {
    let (addr, _queue) = spawn_server().await;
    let unknown = "11111111-1111-4111-8111-111111111111";
    let url = format!("ws://{addr}/v1/jobs/{unknown}/events");
    let result = tokio_tungstenite::connect_async(&url).await;
    assert!(result.is_err(), "unknown job must not upgrade to WS");
}
