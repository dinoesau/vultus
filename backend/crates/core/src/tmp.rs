use super::job::JobId;

/// Directorio efimero por job: `{tmp}/vultus-{job_id}`.
/// Nunca persistente. Best-effort: ignora `NotFound`, loguea resto en caller.
pub fn job_dir(job_id: &JobId) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("vultus-{job_id}"))
}

/// Borra `/tmp/vultus-{job_id}` si existe. Nunca pagina: stateless limpia en silencio.
pub fn cleanup_job_dir(job_id: &JobId) {
    let dir = job_dir(job_id);
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            tracing::warn!(job_id = %job_id, path = %dir.display(), error = %e, "tmp cleanup failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_dir_is_under_tmp_and_missing_cleanup_is_noop() {
        let id = JobId::new();
        let dir = job_dir(&id);
        assert!(dir.to_string_lossy().contains(&id.to_string()));
        // No debe panicar aunque no exista.
        cleanup_job_dir(&id);
    }

    #[test]
    fn test_cleanup_removes_dir() {
        let id = JobId::new();
        let dir = job_dir(&id);
        std::fs::create_dir_all(&dir).expect("setup tmp");
        std::fs::write(dir.join("x.bin"), b"hi").expect("setup file");
        cleanup_job_dir(&id);
        assert!(!dir.exists());
    }
}
