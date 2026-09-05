use std::sync::Arc;
use std::time::Duration;

use super::error::{CoreError, Result};
use super::job::{CompareResult, CompleteUv, FlawUv, Heatmap, ImageBytes, JobId, Landmarks};
use super::ml::MlSidecarClient;
use super::queue::Queue;
use super::tmp::{cleanup_job_dir, job_dir};

/// Salida del par: alias de `CompareResult` (vive en `job.rs` para que
/// `queue` lo almacene sin dependencia circular `queue <-> pipeline`).
pub type PipelineOutput = CompareResult;

/// Configuracion de timeouts del orquestador (ver decisiones memorizadas).
/// Por llamada: landmarks 5s, flame 10s, freeuv 30s. Deadline total 60s = TTL.
/// El SLO <20s p95 warm es eval; el timeout es red, no SLO.
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    landmarks_timeout: Duration,
    flame_timeout: Duration,
    freeuv_timeout: Duration,
    total_timeout: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            landmarks_timeout: Duration::from_secs(5),
            flame_timeout: Duration::from_secs(10),
            freeuv_timeout: Duration::from_secs(30),
            total_timeout: Duration::from_secs(60),
        }
    }
}

impl PipelineConfig {
    pub fn landmarks_timeout(self) -> Duration {
        self.landmarks_timeout
    }

    pub fn flame_timeout(self) -> Duration {
        self.flame_timeout
    }

    pub fn freeuv_timeout(self) -> Duration {
        self.freeuv_timeout
    }

    pub fn total_timeout(self) -> Duration {
        self.total_timeout
    }
}

/// Orquestador profundo Fase 1 (Seam 3): paralelismo + timeouts + progreso +
/// limpieza tras una interfaz estrecha (`PipelineConfig`, `run_pair`).
/// Concurrencia con `join!` A/B en Rust, sin semaforo (erroneo con replicas);
/// la serializacion vive solo en el sidecar. Bake Fase 1 es identidad:
/// FreeUV ya entrega `CompleteUv` canonica, sin GNM real.
pub async fn run_pair(
    queue: &Arc<dyn Queue>,
    ml: &MlSidecarClient,
    job_id: &JobId,
    image_a: &ImageBytes,
    image_b: &ImageBytes,
    cfg: &PipelineConfig,
) -> Result<PipelineOutput> {
    let outcome = tokio::time::timeout(
        cfg.total_timeout,
        run_pair_inner(queue, ml, job_id, image_a, image_b, cfg),
    )
    .await;
    match outcome {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(err)) => {
            let _ = queue.fail_job(job_id).await;
            cleanup_job_dir(job_id);
            Err(err)
        }
        Err(_) => {
            let _ = queue.fail_job(job_id).await;
            cleanup_job_dir(job_id);
            Err(CoreError::ml_transport("pipeline total timeout"))
        }
    }
}

async fn run_pair_inner(
    queue: &Arc<dyn Queue>,
    ml: &MlSidecarClient,
    job_id: &JobId,
    image_a: &ImageBytes,
    image_b: &ImageBytes,
    cfg: &PipelineConfig,
) -> Result<PipelineOutput> {
    let _guard = TmpGuard::create(job_id);
    queue
        .set_progress(job_id, progress(0.15), super::job::Stage::Landmarks)
        .await?;
    let (landmarks_a, landmarks_b) = tokio::join!(
        call_landmarks(ml, job_id, image_a, cfg),
        call_landmarks(ml, job_id, image_b, cfg),
    );
    let (landmarks_a, landmarks_b) = (landmarks_a?, landmarks_b?);
    queue
        .set_progress(job_id, progress(0.40), super::job::Stage::Flame)
        .await?;
    let (flaw_a, flaw_b) = tokio::join!(
        call_flame(ml, job_id, image_a, &landmarks_a, cfg),
        call_flame(ml, job_id, image_b, &landmarks_b, cfg),
    );
    let (flaw_a, flaw_b) = (flaw_a?, flaw_b?);
    queue
        .set_progress(job_id, progress(0.75), super::job::Stage::Freeuv)
        .await?;
    let (uv_a, uv_b) = tokio::join!(
        call_freeuv(ml, job_id, &flaw_a, cfg),
        call_freeuv(ml, job_id, &flaw_b, cfg),
    );
    let (uv_a, uv_b) = (uv_a?, uv_b?);
    queue
        .set_progress(job_id, progress(0.95), super::job::Stage::Bake)
        .await?;
    let heatmap = heatmap_abs_diff(&uv_a, &uv_b);
    let output = PipelineOutput::new(uv_a, uv_b, heatmap);
    queue.complete_with_result(job_id, output.clone()).await?;
    Ok(output)
}

/// Cadena secuencial de una cara: landmarks -> flame -> freeuv, cada etapa
/// con su timeout. Sin progreso ni limpieza: `run_pair` las maneja a nivel
/// job porque el progreso global no puede intercalarse por cara con `join!`.
pub async fn process_one_face(
    ml: &MlSidecarClient,
    job_id: &JobId,
    image: &ImageBytes,
    cfg: &PipelineConfig,
) -> Result<CompleteUv> {
    let landmarks = call_landmarks(ml, job_id, image, cfg).await?;
    let flaw = call_flame(ml, job_id, image, &landmarks, cfg).await?;
    call_freeuv(ml, job_id, &flaw, cfg).await
}

async fn call_landmarks(
    ml: &MlSidecarClient,
    job_id: &JobId,
    image: &ImageBytes,
    cfg: &PipelineConfig,
) -> Result<Landmarks> {
    tokio::time::timeout(cfg.landmarks_timeout, ml.landmarks(job_id, image))
        .await
        .map_err(|_| CoreError::ml_transport("landmarks timeout"))?
}

async fn call_flame(
    ml: &MlSidecarClient,
    job_id: &JobId,
    image: &ImageBytes,
    landmarks: &Landmarks,
    cfg: &PipelineConfig,
) -> Result<FlawUv> {
    tokio::time::timeout(cfg.flame_timeout, ml.flame(job_id, image, landmarks))
        .await
        .map_err(|_| CoreError::ml_transport("flame timeout"))?
}

async fn call_freeuv(
    ml: &MlSidecarClient,
    job_id: &JobId,
    flaw_uv: &FlawUv,
    cfg: &PipelineConfig,
) -> Result<CompleteUv> {
    tokio::time::timeout(cfg.freeuv_timeout, ml.freeuv(job_id, flaw_uv))
        .await
        .map_err(|_| CoreError::ml_transport("freeuv timeout"))?
}

/// Heatmap Fase 1: `|a-b|` por byte tal cual, sin normalizacion ni colormap.
/// Misma regla que `compute_heatmap` en `workers_cpu` (core no puede
/// depender de ese crate: el depende de core). Infallible: ambas UV prueban
/// `UV_LEN`, el heatmap hereda la longitud.
fn heatmap_abs_diff(uv_a: &CompleteUv, uv_b: &CompleteUv) -> Heatmap {
    let bytes: Vec<u8> = uv_a
        .as_bytes()
        .iter()
        .zip(uv_b.as_bytes().iter())
        .map(|(x, y)| x.abs_diff(*y))
        .collect();
    Heatmap::parse(bytes).expect("heatmap preserva UV_LEN: entradas ya prueban UV_LEN")
}

fn progress(value: f32) -> super::job::Progress {
    super::job::Progress::parse(value).expect("progreso literal dentro de 0.0..=1.0")
}

/// Guardia RAII: `job_dir` se crea al iniciar y se borra al salir por
/// cualquier camino (exito, error, timeout total, panic por `Drop`).
/// Best-effort: `cleanup_job_dir` nunca pagina.
struct TmpGuard<'a> {
    job_id: &'a JobId,
}

impl<'a> TmpGuard<'a> {
    fn create(job_id: &'a JobId) -> Self {
        let _ = std::fs::create_dir_all(job_dir(job_id));
        Self { job_id }
    }
}

impl Drop for TmpGuard<'_> {
    fn drop(&mut self) {
        cleanup_job_dir(self.job_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{EnqueueCommand, JobStatus, Progress, Stage, UV_LEN};
    use crate::ml::BaseUrl;
    use crate::queue::MemoryQueue;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    const MARKER_A: u8 = 0xA1;
    const MARKER_B: u8 = 0xB2;
    const FLAW_A: u8 = 0x11;
    const FLAW_B: u8 = 0x22;

    fn test_image(marker: u8) -> ImageBytes {
        let mut b = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        b.resize(64, marker);
        ImageBytes::parse(b).expect("imagen de test valida")
    }

    fn landmarks_body() -> Vec<u8> {
        let pts = vec![[0.0f32, 0.0, 0.0]; crate::job::LANDMARKS_LEN];
        serde_json::to_vec(&pts).expect("fixture landmarks")
    }

    fn golden_complete(head: [u8; 2]) -> Vec<u8> {
        let mut v = vec![0u8; UV_LEN];
        v[..2].copy_from_slice(&head);
        v
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FakeMode {
        Ok,
        FailLandmarks,
    }

    fn find_headers_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
    }

    async fn read_request(stream: &mut TcpStream) -> Option<(String, Vec<u8>)> {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = stream.read(&mut tmp).await.ok()?;
            if n == 0 {
                return None;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(start) = find_headers_end(&buf) {
                let head = String::from_utf8_lossy(&buf[..start]).to_string();
                let mut lines = head.lines();
                let request_line = lines.next().unwrap_or("");
                let path = request_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .to_string();
                let mut len = 0usize;
                for line in lines {
                    let lower = line.to_ascii_lowercase();
                    if let Some(v) = lower.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
                while buf.len() < start + len {
                    let n = stream.read(&mut tmp).await.ok()?;
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                let end = (start + len).min(buf.len());
                return Some((path, buf[start..end].to_vec()));
            }
            if buf.len() > 10 * 1024 * 1024 {
                return None;
            }
        }
    }

    fn route(path: &str, body: &[u8], mode: &FakeMode) -> (&'static str, Vec<u8>) {
        match (path, mode) {
            ("/ml/landmarks", FakeMode::FailLandmarks) => {
                ("500 Internal Server Error", b"boom".to_vec())
            }
            ("/ml/landmarks", _) => ("200 OK", landmarks_body()),
            ("/ml/flame", _) => {
                let (_, img) =
                    crate::ml::FlamePayload::decode(body.to_vec()).expect("payload flame valido");
                let flaw = if img.as_bytes().contains(&MARKER_A) {
                    FLAW_A
                } else {
                    FLAW_B
                };
                ("200 OK", vec![flaw; UV_LEN])
            }
            ("/ml/freeuv", _) => {
                let head = if body.first() == Some(&FLAW_A) {
                    [10u8, 200]
                } else {
                    [4u8, 210]
                };
                ("200 OK", golden_complete(head))
            }
            _ => ("404 Not Found", b"nope".to_vec()),
        }
    }

    async fn serve(listener: TcpListener, mode: FakeMode) {
        loop {
            let (mut stream, _) = listener.accept().await.expect("accept del fake sidecar");
            tokio::spawn(async move {
                let Some((path, body)) = read_request(&mut stream).await else {
                    return;
                };
                let (status, resp_body) = route(&path, &body, &mode);
                let head = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    resp_body.len()
                );
                let _ = stream.write_all(head.as_bytes()).await;
                let _ = stream.write_all(&resp_body).await;
            });
        }
    }

    async fn fake_client(mode: FakeMode) -> (MlSidecarClient, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind del fake sidecar");
        let addr = listener.local_addr().expect("addr del fake sidecar");
        let handle = tokio::spawn(serve(listener, mode));
        let base = BaseUrl::parse(&format!("http://{addr}")).expect("base url del fake");
        (MlSidecarClient::new(base), handle)
    }

    #[tokio::test]
    async fn test_pair_produces_canonical_uvs_and_golden_heatmap() {
        let (ml, server) = fake_client(FakeMode::Ok).await;
        let queue: Arc<dyn Queue> = Arc::new(MemoryQueue::default());
        let image_a = test_image(MARKER_A);
        let image_b = test_image(MARKER_B);
        let cmd = EnqueueCommand::new(image_a.clone(), image_b.clone());
        let job_id = queue.enqueue(cmd).await.expect("enqueue").job_id();

        let out = run_pair(
            &queue,
            &ml,
            &job_id,
            &image_a,
            &image_b,
            &PipelineConfig::default(),
        )
        .await
        .expect("run_pair");

        // Literales dorados a mano, nunca recomputados con la funcion bajo test.
        assert_eq!(out.uv_a().len(), UV_LEN);
        assert_eq!(out.uv_b().len(), UV_LEN);
        assert_eq!(&out.uv_a().as_bytes()[..2], &[10, 200]);
        assert_eq!(&out.uv_b().as_bytes()[..2], &[4, 210]);
        assert!(out.uv_a().as_bytes()[2..].iter().all(|&b| b == 0));
        assert!(out.uv_b().as_bytes()[2..].iter().all(|&b| b == 0));
        assert_eq!(out.heatmap().len(), UV_LEN);
        assert_eq!(&out.heatmap().as_bytes()[..2], &[6, 10]);
        assert!(out.heatmap().as_bytes()[2..].iter().all(|&b| b == 0));

        assert_eq!(
            queue.status(&job_id).await.expect("status"),
            JobStatus::Done
        );
        let (progress, stage) = queue.progress(&job_id).await.expect("progress");
        assert_eq!(progress, Progress::parse(1.0).expect("progreso 1.0"));
        assert_eq!(stage, Stage::Done);
        let stored = queue
            .fetch_result(&job_id)
            .await
            .expect("resultado guardado");
        assert_eq!(stored, out);
        assert!(!job_dir(&job_id).exists());
        server.abort();
    }

    #[tokio::test]
    async fn test_sidecar_error_fails_job_and_cleans_tmp() {
        let (ml, server) = fake_client(FakeMode::FailLandmarks).await;
        let queue: Arc<dyn Queue> = Arc::new(MemoryQueue::default());
        let image_a = test_image(MARKER_A);
        let image_b = test_image(MARKER_B);
        let cmd = EnqueueCommand::new(image_a.clone(), image_b.clone());
        let job_id = queue.enqueue(cmd).await.expect("enqueue").job_id();

        assert!(run_pair(
            &queue,
            &ml,
            &job_id,
            &image_a,
            &image_b,
            &PipelineConfig::default()
        )
        .await
        .is_err());
        assert_eq!(
            queue.status(&job_id).await.expect("status"),
            JobStatus::Failed
        );
        assert!(!job_dir(&job_id).exists());
        server.abort();
    }

    #[test]
    fn test_pipeline_timeouts_match_decisions() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.landmarks_timeout(), Duration::from_secs(5));
        assert_eq!(cfg.flame_timeout(), Duration::from_secs(10));
        assert_eq!(cfg.freeuv_timeout(), Duration::from_secs(30));
        assert_eq!(cfg.total_timeout(), Duration::from_secs(60));
    }
}
