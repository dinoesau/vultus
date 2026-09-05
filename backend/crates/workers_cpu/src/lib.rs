use vultus_core::{CompleteUv, FlawUv, Heatmap};

/// Worker 4 CPU (Rust): `heatmap = |UV_A - UV_B|` por byte.
/// Puro, sin I/O. Infallible: `CompleteUv` ya prueba `UV_LEN`, dos UV
/// cualesquiera tienen igual longitud y el heatmap hereda `UV_LEN`.
pub fn compute_heatmap(uv_a: &CompleteUv, uv_b: &CompleteUv) -> Heatmap {
    let bytes: Vec<u8> = uv_a
        .as_bytes()
        .iter()
        .zip(uv_b.as_bytes().iter())
        .map(|(x, y)| x.abs_diff(*y))
        .collect();
    Heatmap::parse(bytes).expect("heatmap preserva UV_LEN: entradas ya prueban UV_LEN")
}

/// Bake baricentrico BFM->GNM (Fase 0: identidad verificada por shape).
/// La matriz real precomputada llega en Fase 2; aqui solo el contrato.
/// Infallible: `FlawUv` ya prueba `UV_LEN`, la copia la preserva.
pub fn bake_bfm_to_gnm(uv_bfm: &FlawUv) -> CompleteUv {
    CompleteUv::parse(uv_bfm.as_bytes().to_vec())
        .expect("bake preserva UV_LEN: entrada ya prueba UV_LEN")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vultus_core::UV_LEN;

    fn uv_full(fill: u8) -> CompleteUv {
        CompleteUv::parse(vec![fill; UV_LEN]).unwrap()
    }

    fn uv_with_head(head: &[u8], fill: u8) -> CompleteUv {
        let mut v = vec![fill; UV_LEN];
        v[..head.len()].copy_from_slice(head);
        CompleteUv::parse(v).unwrap()
    }

    #[test]
    fn test_identical_uv_produces_black_heatmap() {
        let uv = uv_full(10);
        let heat = compute_heatmap(&uv, &uv);
        assert_eq!(heat.len(), UV_LEN);
        assert!(heat.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_known_diff_produces_known_heatmap() {
        // Literal golden verificado a mano, no recomputado.
        let a = uv_with_head(&[10u8, 200], 7);
        let b = uv_with_head(&[4u8, 210], 7);
        let heat = compute_heatmap(&a, &b);
        assert_eq!(&heat.as_bytes()[..2], &[6, 10]);
        assert!(heat.as_bytes()[2..].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_wrong_uv_length_rejected_at_parse() {
        assert!(CompleteUv::parse(vec![1, 2]).is_err());
        assert!(FlawUv::parse(vec![]).is_err());
        assert!(Heatmap::parse(vec![0u8; UV_LEN - 1]).is_err());
        assert!(CompleteUv::parse(b"{\"todo\":\"complete-uv\"}".to_vec()).is_err());
    }
}
