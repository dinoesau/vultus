# Coverage - Registro por fase

Extiende este registro en cada fase via Self-update antes del run.

- **Fase 0**: E2E-1..E2E-5 (ver `acceptance.md`). Deuda conocida: sin consumidor real `Queued->Done`, `set_progress` solo simulado en tests.
- **Fase 1**: añadir `POST /ml/*` reales, `GET /v1/jobs/{id}/result` zip, visor UV mas heatmap.
- **Fase 1 (E2E-6, cubierto)**: par dorado sintético `frontend/e2e/fixtures/a.png|b.png` llega a `done`, slider responde, zip descarga (`frontend/e2e/compare.spec.ts`, `acceptDownloads: true` en `playwright.config.ts`). Deuda: las 10 parejas pose 0-30 se congelan tras revisión humana checkpoint 2; el E2E automático afirma pipeline técnico, no verdad forense.
- **Fase 2+**: añadir visor 3D, metricas PDF, rate-limit, Turnstile/WAF.

Regla: superficie nueva sin escenario es bloqueador del veredicto, no nota al pie.
