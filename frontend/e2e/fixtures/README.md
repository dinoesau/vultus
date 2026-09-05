# Fixtures E2E (par dorado sintético)

`a.png` y `b.png`: PNG 1x1 válidos (magic `89 50 4E 47 0D 0A 1A 0A`, colores
distintos rojo/azul) que pasan `ImageBytes::parse` (solo magic) y los dobles
deterministas del sidecar (`_is_png`), fluyendo por el pipeline hasta `done`.

Alcance: el E2E automático con este par afirma pipeline técnico `done`
(3 paneles, slider, zip descargable), no verdad forense.

Deuda checkpoint 2: las 10 parejas pose 0-30 con caras reales se congelan aquí
solo tras revisión humana a ojo. No inventar caras ni declarar equivalencia
forense sin esa revisión.
