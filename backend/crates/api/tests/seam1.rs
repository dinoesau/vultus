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
    let resp = server.get("/health").await;
    resp.assert_status(StatusCode::OK);
    let body: serde_json::Value = resp.json();
    assert_eq!(body.get("status").and_then(|v| v.as_str()), Some("ok"));
    assert_eq!(body.get("queue").and_then(|v| v.as_str()), Some("ok"));
    assert_eq!(body.get("ttl_secs").and_then(|v| v.as_u64()), Some(60));
}

#[tokio::test]
async fn test_events_bad_uuid_returns_400() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    server
        .get("/v1/jobs/not-a-uuid/events")
        .await
        .assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_events_without_ws_upgrade_is_rejected() {
    // Sin headers WS, `WebSocketUpgrade` rechaza con 400 antes del handler.
    // El 404 real con handshake WS se prueba en `tests/ws_events.rs::test_ws_unknown_job_handshake_fails`.
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    server
        .get("/v1/jobs/11111111-1111-4111-8111-111111111111/events")
        .await
        .assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_job_expires_to_expired_status() {
    use std::sync::Arc;
    use vultus_core::{ManualClock, TtlSecs};
    let ttl = TtlSecs::parse(1).expect("ttl");
    let clock = Arc::new(ManualClock::new());
    let queue = MemoryQueue::with_ttl_and_clock(ttl, clock.clone());
    let server = TestServer::new(router(AppState::with_ttl(queue, ttl))).unwrap();
    let resp = server
        .post("/v1/compare")
        .multipart(compare_parts(png_bytes()))
        .await;
    resp.assert_status(StatusCode::ACCEPTED);
    let body: serde_json::Value = resp.json();
    let job_id = body.get("job_id").and_then(|v| v.as_str()).expect("job_id");
    clock.advance(std::time::Duration::from_secs(2));
    let status = server.get(&format!("/v1/jobs/{job_id}")).await;
    status.assert_status(StatusCode::OK);
    let sbody: serde_json::Value = status.json();
    assert_eq!(
        sbody.get("status").and_then(|v| v.as_str()),
        Some("expired")
    );
}

// Fase1-UV-canonico (aditivo): zip canonico + 404s, sin tocar los 11 anteriores.
#[tokio::test]
async fn test_result_serves_canonical_zip_with_golden_pngs() {
    use std::io::Read;
    use std::sync::Arc;
    use vultus_core::{
        CompareResult, CompleteUv, EnqueueCommand, Heatmap, ImageBytes, Queue, TtlSecs, UV_LEN,
    };
    fn golden(head: [u8; 2]) -> Vec<u8> {
        let mut v = vec![0u8; UV_LEN];
        v[..2].copy_from_slice(&head);
        v
    }
    let queue = MemoryQueue::default();
    let state = AppState::from_arc(
        Arc::new(queue.clone()) as Arc<dyn Queue>,
        TtlSecs::default(),
    );
    let server = TestServer::new(router(state)).unwrap();
    let a = ImageBytes::parse(png_bytes()).expect("a");
    let b = ImageBytes::parse(png_bytes()).expect("b");
    let job_id = queue
        .enqueue(EnqueueCommand::new(a, b))
        .await
        .expect("enqueue")
        .job_id();
    let uv_a = CompleteUv::parse(golden([10, 200])).expect("uv_a");
    let uv_b = CompleteUv::parse(golden([4, 210])).expect("uv_b");
    let heat = Heatmap::parse(golden([6, 10])).expect("heat");
    queue
        .complete_with_result(&job_id, CompareResult::new(uv_a, uv_b, heat))
        .await
        .expect("complete");

    let resp = server.get(&format!("/v1/jobs/{job_id}/result")).await;
    resp.assert_status(StatusCode::OK);
    let ctype = resp
        .header("content-type")
        .to_str()
        .expect("ctype")
        .to_string();
    assert_eq!(ctype, "application/zip");
    let cdisp = resp
        .header("content-disposition")
        .to_str()
        .expect("cdisp")
        .to_string();
    assert!(cdisp.contains("attachment"), "cdisp {cdisp}");
    assert!(
        cdisp.contains(&format!("result-{job_id}.zip")),
        "cdisp {cdisp}"
    );
    let bytes = resp.as_bytes().to_vec();
    assert!(!bytes.is_empty(), "zip no vacio");

    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).expect("unzip");
    assert_eq!(archive.len(), 3);
    let mut names = Vec::new();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).expect("entry");
        names.push(file.name().to_string());
        let mut data = Vec::new();
        file.read_to_end(&mut data).expect("read png");
        // Magic PNG literal, nunca recomputado.
        assert!(
            data.len() > 8 && data[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            "magic png en {}",
            file.name()
        );
        let img = image::load_from_memory(&data).expect("decode png");
        assert_eq!(img.width(), 512, "ancho {}", file.name());
        assert_eq!(img.height(), 512, "alto {}", file.name());
        let raw = img.to_rgb8().into_raw();
        assert_eq!(raw.len(), UV_LEN, "len {}", file.name());
        let expected_head: [u8; 2] = match file.name() {
            "uv_a.png" => [10, 200],
            "uv_b.png" => [4, 210],
            "heatmap.png" => [6, 10],
            other => panic!("nombre inesperado {other}"),
        };
        assert_eq!(&raw[..2], &expected_head, "head {}", file.name());
        assert!(
            raw[2..].iter().all(|&x| x == 0),
            "resto cero en {}",
            file.name()
        );
    }
    names.sort();
    assert_eq!(names, vec!["heatmap.png", "uv_a.png", "uv_b.png"]);
}

#[tokio::test]
async fn test_result_unknown_job_returns_404() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    server
        .get("/v1/jobs/11111111-1111-4111-8111-111111111111/result")
        .await
        .assert_status(StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_result_bad_uuid_returns_400() {
    let server = TestServer::new(router(AppState::new(MemoryQueue::default()))).unwrap();
    server
        .get("/v1/jobs/not-a-uuid/result")
        .await
        .assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_result_expired_returns_404() {
    use std::sync::Arc;
    use vultus_core::{
        CompareResult, CompleteUv, EnqueueCommand, Heatmap, ImageBytes, ManualClock, Queue,
        TtlSecs, UV_LEN,
    };
    fn golden(head: [u8; 2]) -> Vec<u8> {
        let mut v = vec![0u8; UV_LEN];
        v[..2].copy_from_slice(&head);
        v
    }
    let ttl = TtlSecs::parse(1).expect("ttl");
    let clock = Arc::new(ManualClock::new());
    let queue = MemoryQueue::with_ttl_and_clock(ttl, clock.clone());
    let server = TestServer::new(router(AppState::from_arc(
        Arc::new(queue.clone()) as Arc<dyn Queue>,
        ttl,
    )))
    .unwrap();
    let a = ImageBytes::parse(png_bytes()).expect("a");
    let b = ImageBytes::parse(png_bytes()).expect("b");
    let job_id = queue
        .enqueue(EnqueueCommand::new(a, b))
        .await
        .expect("enqueue")
        .job_id();
    queue
        .complete_with_result(
            &job_id,
            CompareResult::new(
                CompleteUv::parse(golden([10, 200])).expect("uv_a"),
                CompleteUv::parse(golden([4, 210])).expect("uv_b"),
                Heatmap::parse(golden([6, 10])).expect("heat"),
            ),
        )
        .await
        .expect("complete");
    clock.advance(std::time::Duration::from_secs(2));
    server
        .get(&format!("/v1/jobs/{job_id}/result"))
        .await
        .assert_status(StatusCode::NOT_FOUND);
}
