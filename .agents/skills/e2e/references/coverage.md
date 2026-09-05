# Coverage - Registro por fase

Extiende este registro en cada fase via Self-update antes del run.

- **Fase 0**: E2E-1..E2E-5 (ver `acceptance.md`). Deuda conocida: sin consumidor real `Queued->Done`, `set_progress` solo simulado en tests.
- **Fase 1**: añadir `POST /ml/*` reales, `GET /v1/jobs/{id}/result` zip, visor UV mas heatmap.
- **Fase 2+**: añadir visor 3D, metricas PDF, rate-limit, Turnstile/WAF.

Regla: superficie nueva sin escenario es bloqueador del veredicto, no nota al pie.
