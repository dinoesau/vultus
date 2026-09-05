//! Paridad edge-core (C): Rust es fuente de verdad, `edge/contract.ts` es espejo.
//! Si este test falla, actualiza `edge/contract.ts` o el Rust correspondiente.
//! Fuente Rust: `vultus_core::{MAX_IMAGE_BYTES, RESULT_TTL_SECONDS, Stage, Progress, TtlSecs}`.
//! Espejo TS: `edge/contract.ts`, usado por `edge/worker.ts` y `edge/progress-do.ts`.

use std::path::PathBuf;
use vultus_core::{Progress, Stage, TtlSecs, MAX_IMAGE_BYTES, RESULT_TTL_SECONDS};

fn repo_file(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../")
        .join(rel)
}

fn read_repo(rel: &str) -> String {
    let path = repo_file(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("no se pudo leer {} ({path:?}): {e}", rel))
}

#[test]
fn test_edge_contract_matches_rust_constants() {
    let ts = read_repo("edge/contract.ts");

    // Tamano maximo: Rust bytes exactos, TS misma expresion.
    assert_eq!(MAX_IMAGE_BYTES, 8 * 1024 * 1024);
    assert!(
        ts.contains("8 * 1024 * 1024"),
        "contract.ts debe definir MAX_IMAGE_BYTES como 8 * 1024 * 1024"
    );

    // TTL: default 60, rango 1..=3600 en ambos lados.
    assert_eq!(RESULT_TTL_SECONDS, 60);
    assert_eq!(TtlSecs::default().value(), 60);
    assert!(ts.contains("RESULT_TTL_SECONDS = 60"), "TTL default 60");
    assert!(ts.contains("TTL_MIN_SECS = 1"), "TTL min 1");
    assert!(ts.contains("TTL_MAX_SECS = 3600"), "TTL max 3600");
    assert!(TtlSecs::parse(0).is_err());
    assert!(TtlSecs::parse(3601).is_err());

    // Ventana purge 2x TTL: Rust `purge_after` = 120s, edge `progress-do.ts` re-arma alarma.
    assert_eq!(
        TtlSecs::default().purge_after(),
        std::time::Duration::from_secs(120)
    );
}

#[test]
fn test_edge_stages_match_rust_order() {
    let ts = read_repo("edge/contract.ts");
    let rust_order = [
        Stage::Queued.as_str(),
        Stage::Landmarks.as_str(),
        Stage::Flame.as_str(),
        Stage::Freeuv.as_str(),
        Stage::Bake.as_str(),
        Stage::Done.as_str(),
    ];
    assert_eq!(
        rust_order,
        ["queued", "landmarks", "flame", "freeuv", "bake", "done"]
    );
    // Orden en TS debe ser el mismo; buscamos el bloque STAGES en orden.
    let mut cursor = 0;
    for stage in rust_order {
        let needle = format!("\"{stage}\"");
        let found = ts[cursor..]
            .find(needle.as_str())
            .unwrap_or_else(|| panic!("STAGES debe contener {needle} en orden"));
        cursor += found + needle.len();
    }
}

#[test]
fn test_edge_magic_matches_rust_parsers() {
    let ts = read_repo("edge/contract.ts").to_lowercase();
    // JPEG FF D8 FF y PNG 89 50 4E 47 0D 0A 1A 0A en ambos lados.
    for needle in [
        "0xff", "0xd8", "0x89", "0x50", "0x4e", "0x47", "0x0d", "0x0a", "0x1a",
    ] {
        assert!(
            ts.contains(needle),
            "contract.ts debe contener magic {needle} como Rust is_jpeg/is_png"
        );
    }
    // Progress 0..=1 en ambos lados.
    assert!(
        read_repo("edge/contract.ts").contains("n >= 0 && n <= 1"),
        "isValidProgress debe espejar Progress::parse 0.0..=1.0"
    );
    assert!(Progress::parse(0.4).is_ok());
    assert!(Progress::parse(-0.1).is_err());
    assert!(Progress::parse(f32::NAN).is_err());
}

#[test]
fn test_edge_worker_uses_r2_pointer_pattern() {
    let worker = read_repo("edge/worker.ts");
    // Paridad con `R2PointerQueue::enqueue`: `jobs/{job_id}/a|b`.
    assert!(
        worker.contains("jobs/${job_id}/a"),
        "worker.ts debe usar jobs/${{job_id}}/a como R2PointerQueue"
    );
    assert!(
        worker.contains("jobs/${job_id}/b"),
        "worker.ts debe usar jobs/${{job_id}}/b como R2PointerQueue"
    );
}

#[test]
fn test_edge_progress_do_matches_store_lifecycle() {
    let do_ts = read_repo("edge/progress-do.ts");
    // Primera alarma -> expired visible, segunda -> purga. Espejo de Store 2x TTL.
    assert!(
        do_ts.contains("\"expired\"") || do_ts.contains("'expired'") || do_ts.contains("expired"),
        "progress-do.ts debe marcar expired como Rust JobStatus::Expired"
    );
    assert!(
        do_ts.contains("deleteAll"),
        "progress-do.ts debe purgar como Store::purge_expired"
    );
    assert!(
        do_ts.contains("setAlarm"),
        "progress-do.ts debe re-armar alarma para ventana 2x TTL"
    );
}
