# Fixtures E2E (par dorado sintético)

`a.png` y `b.png`: PNG 1x1 válidos (magic `89 50 4E 47 0D 0A 1A 0A`, colores
distintos rojo/azul) que pasan `ImageBytes::parse` (solo magic) y los dobles
deterministas del sidecar (`_is_png`), fluyendo por el pipeline hasta `done`.

Alcance: el E2E automático con este par afirma pipeline técnico `done`
(3 paneles, slider, zip descargable), no verdad forense.

Deuda checkpoint 2: las 10 parejas pose 0-30 con caras reales se congelan aquí
solo tras revisión humana a ojo. No inventar caras ni declarar equivalencia
forense sin esa revisión.

## Par dorado real (deploy-real-models, congelado 2026-09-05)

LFW es solo investigación: ningún JPEG entra a git. Manifiesto con ruta + hash:

- A: `George_W_Bush/George_W_Bush_0001.jpg` (11984 bytes, JPEG 250x250)
  sha256 `b559818d8704954f81e2df57e9fb5dc0962dd8811cc4ff27cbbd2afc7c12a576`
- B: `George_W_Bush/George_W_Bush_0002.jpg` (12457 bytes, JPEG 250x250)
  sha256 `f04d53698da366ca8562b2d24ad9ed058116621b8fd0d51fcb46a6e5e470e0f3`
- Dataset: `/Users/esau.martinez/Code/datasets/lfw` (5749 personas, 13233 JPEG).
- Sidecar: `https://dinoesau--vultus-workers-ml-sidecar-app.modal.run` (warm).
- Revisión a ojo: PASS. `uv_a` con ojos, cejas y boca reales; heatmap con
  diferencias estructuradas, no negro. Ojos algo fantasmales a 12 pasos SD,
  aceptado para Fase 1.
- Stats esperadas: landmarks 478 finitos; flaw-uv 786432 bytes std ~84;
  complete-uv 786432 bytes std ~50; zip `{uv_a,uv_b,heatmap}.png` 512x512;
  pipeline warm ~16.5s, compare a done ~18s (SLO p95 menor a 20s, solo warm).
- Cableado Rust: sin cambios de código, solo `ML_SIDECAR_URL` al endpoint Modal.
