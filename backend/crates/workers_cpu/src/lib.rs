use thiserror::Error;
use vultus_core::assert::assert_ok;

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("size mismatch: {0}")]
    SizeMismatch(&'static str),
    #[error("empty input")]
    Empty,
}

/// Worker 4 CPU (Rust): `heatmap = |UV_A - UV_B|` por byte.
/// Puro, sin I/O. Testeable con literales golden.
pub fn compute_heatmap(uv_a: &[u8], uv_b: &[u8]) -> Result<Vec<u8>, WorkerError> {
    if uv_a.is_empty() || uv_b.is_empty() {
        return Err(WorkerError::Empty);
    }
    if uv_a.len() != uv_b.len() {
        return Err(WorkerError::SizeMismatch("uv_a.len != uv_b.len"));
    }
    Ok(uv_a
        .iter()
        .zip(uv_b.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .collect())
}

/// Bake baricéntrico BFM->GNM (Fase 0: identidad verificada por shape).
/// La matriz real precomputada llega en Fase 2; aquí solo el contrato.
pub fn bake_bfm_to_gnm(uv_bfm: &[u8]) -> Result<Vec<u8>, WorkerError> {
    assert_ok(!uv_bfm.is_empty(), "bake input vacío").map_err(|_| WorkerError::Empty)?;
    Ok(uv_bfm.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_uv_produces_black_heatmap() {
        let uv = vec![10u8, 20, 30, 40];
        let heat = compute_heatmap(&uv, &uv).unwrap();
        assert_eq!(heat, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_known_diff_produces_known_heatmap() {
        // Literal golden verificado a mano, no recomputado.
        let a = vec![10u8, 200];
        let b = vec![4u8, 210];
        assert_eq!(compute_heatmap(&a, &b).unwrap(), vec![6, 10]);
    }

    #[test]
    fn test_mismatched_sizes_fail() {
        assert!(compute_heatmap(&[1, 2], &[1]).is_err());
    }
}
