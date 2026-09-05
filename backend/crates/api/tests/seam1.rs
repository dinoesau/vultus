use axum::http::StatusCode;
use axum_test::TestServer;
use vultus_api::{router, AppState};
use vultus_core::MemoryQueue;

fn png_bytes() -> Vec<u8> {
    // PNG magic + filler mínimo válido para `ImageBytes::parse`.
    let mut b = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    b.resize(64, 0);
    b
}

#[tokio::test]
async fn test_create_job_returns_202() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    let png = png_bytes();
    let resp = server
        .post("/v1/compare")
        .multipart(
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
                ),
        )
        .await;
    resp.assert_status(StatusCode::ACCEPTED);
    let body: serde_json::Value = resp.json();
    assert!(body.get("job_id").is_some());
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
async fn test_health_ok() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    server.get("/health").await.assert_status(StatusCode::OK);
}
