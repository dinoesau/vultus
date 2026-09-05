use axum::http::StatusCode;
use axum_test::TestServer;
use vultus_api::{router, AppState};
use vultus_core::{MemoryQueue, R2PointerQueue};

fn png_bytes() -> Vec<u8> {
    // PNG magic + filler mínimo válido para `ImageBytes::parse`.
    let mut b = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    b.resize(64, 0);
    b
}

fn compare_parts(png: Vec<u8>) -> axum_test::multipart::MultipartForm {
    axum_test::multipart::MultipartForm::new()
        .add_part(
            "image_a",
            axum_test::multipart::Part::bytes(png.clone())
                .mime_type("image/png")
                .file_name("a.png"),
        )
        .add_part(
            "image_b",
            axum_test::multipart::Part::bytes(png)
                .mime_type("image/png")
                .file_name("b.png"),
        )
}

#[tokio::test]
async fn test_create_job_returns_202() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    let resp = server
        .post("/v1/compare")
        .multipart(compare_parts(png_bytes()))
        .await;
    resp.assert_status(StatusCode::ACCEPTED);
    let body: serde_json::Value = resp.json();
    assert!(body.get("job_id").is_some());
    assert_eq!(body.get("status").and_then(|v| v.as_str()), Some("queued"));
}

#[tokio::test]
async fn test_create_then_status_is_queued() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    let resp = server
        .post("/v1/compare")
        .multipart(compare_parts(png_bytes()))
        .await;
    let body: serde_json::Value = resp.json();
    let job_id = body.get("job_id").and_then(|v| v.as_str()).expect("job_id");
    let status = server.get(&format!("/v1/jobs/{job_id}")).await;
    status.assert_status(StatusCode::OK);
    let sbody: serde_json::Value = status.json();
    assert_eq!(sbody.get("status").and_then(|v| v.as_str()), Some("queued"));
}

#[tokio::test]
async fn test_r2_pointer_queue_serves_same_seam() {
    let server = TestServer::new(router(AppState::new(R2PointerQueue::default()))).unwrap();
    let resp = server
        .post("/v1/compare")
        .multipart(compare_parts(png_bytes()))
        .await;
    resp.assert_status(StatusCode::ACCEPTED);
}

#[tokio::test]
async fn test_invalid_image_returns_400_without_enqueue() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    let resp = server
        .post("/v1/compare")
        .multipart(
            axum_test::multipart::MultipartForm::new()
                .add_part(
                    "image_a",
                    axum_test::multipart::Part::bytes(vec![1, 2, 3]).file_name("a.bin"),
                )
                .add_part(
                    "image_b",
                    axum_test::multipart::Part::bytes(vec![4, 5, 6]).file_name("b.bin"),
                ),
        )
        .await;
    resp.assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_missing_image_returns_400() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    let resp = server
        .post("/v1/compare")
        .multipart(axum_test::multipart::MultipartForm::new().add_part(
            "image_a",
            axum_test::multipart::Part::bytes(png_bytes()).file_name("a.png"),
        ))
        .await;
    resp.assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_bad_uuid_returns_400() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    server
        .get("/v1/jobs/not-a-uuid")
        .await
        .assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_unknown_job_returns_404() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    server
        .get("/v1/jobs/11111111-1111-4111-8111-111111111111")
        .await
        .assert_status(StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_health_ok() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    server.get("/health").await.assert_status(StatusCode::OK);
}
