use vultus_core::{CompleteUv, GnmMesh, Heatmap};

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

/// Bake baricentrico BFM->GNM Fase 2: lookup por texel via LUT precomputada.
/// Delega al nucleo puro en `vultus-core::job` (unica fuente, sin duplicar);
/// este crate es la seam del runtime CPU tras la misma frontera tipada.
pub fn bake_bfm_to_gnm(uv_bfm: &CompleteUv) -> CompleteUv {
    vultus_core::bake_bfm_to_gnm(uv_bfm)
}

/// Builder GLB puro Fase 2: mesh neutro + textura horneada.
/// Delega al nucleo; el template fijo vive en `vultus-core::job`.
pub fn build_gnm_glb(baked: &CompleteUv) -> GnmMesh {
    vultus_core::build_gnm_glb(baked)
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
    fn test_bake_is_not_identity_with_golden() {
        // LUT v2 baricentrica: mismos literales a mano que el nucleo.
        let input = uv_with_head(&[10u8, 200], 7);
        let baked = bake_bfm_to_gnm(&input);
        assert_eq!(baked.len(), UV_LEN);
        assert_ne!(baked.as_bytes(), input.as_bytes());
        assert_eq!(&baked.as_bytes()[..3], &[10, 176, 7]);
        assert_eq!(&baked.as_bytes()[21..24], &[7, 7, 7]);
        assert_eq!(&baked.as_bytes()[6156..6159], &[7, 7, 7]);
    }

    #[test]
    fn test_build_glb_has_gltf_magic_and_bounded_size() {
        let baked = uv_with_head(&[10u8, 200], 0);
        let mesh = build_gnm_glb(&baked);
        assert_eq!(&mesh.as_bytes()[0..4], &[0x67, 0x6C, 0x54, 0x46]);
        assert!(mesh.len() > 100_000);
        assert!(mesh.len() < 2_000_000);
        let json_len =
            u32::from_le_bytes(mesh.as_bytes()[12..16].try_into().expect("json")) as usize;
        let text = String::from_utf8_lossy(&mesh.as_bytes()[20..20 + json_len]);
        assert!(text.contains("TEXCOORD_0"));
        assert!(text.contains("baseColorTexture"));
    }

    #[test]
    fn test_wrong_uv_length_rejected_at_parse() {
        assert!(CompleteUv::parse(vec![1, 2]).is_err());
        assert!(vultus_core::FlawUv::parse(vec![]).is_err());
        assert!(vultus_core::Heatmap::parse(vec![0u8; UV_LEN - 1]).is_err());
        assert!(CompleteUv::parse(b"{\"todo\":\"complete-uv\"}".to_vec()).is_err());
    }
}
