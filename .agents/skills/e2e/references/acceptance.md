# Acceptance - Given-When-Then

Escribe los escenarios aplicables en el chat antes de correr.
Formato estricto: Given (estado inicial) - When (accion) - Then (resultado observable con valores literales).

## Escenarios Fase 0

**E2E-1 Stack levanta**

- Given `docker compose up -d --build`
- When `curl :8000/health`, `curl :4321`, `curl :8081/docs`
- Then `api {"status":"ok","queue":"ok","ttl_secs":60}`, frontend contiene `Vultus`, sidecar expone `/ml/landmarks|flame|freeuv`

**E2E-2 Job dummy**

- Given stack sano
- When `POST /v1/compare` con 2 PNG minimos + `GET /v1/jobs/{id}` + `WS /v1/jobs/{id}/events`
- Then `202 {job_id, status:queued}`, `200 {status:queued|processing}`, WS primer evento `{status:queued, stage:queued, progress:0.0}`

**E2E-3 Contratos de error**

- Given stack sano
- When imagen invalida, multipart incompleto, uuid roto, job desconocido
- Then `400` imagen/faltante/uuid, `404` desconocido, sin encolar

**E2E-4 Stateless TTL**

- Given job creado con `TtlSecs=60`
- When pasa `TTL` sin purga y luego `2xTTL` con `purge_expired`
- Then primero `status:expired` mas `stored_lens/progress -> NotFound`, luego `status -> NotFound`, `job_dir` inexistente, sin redis

**E2E-5 Paridad edge**

- Given `wrangler dev`
- When `POST /v1/compare` + `GET /v1/jobs/{id}` + uuid roto + desconocido
- Then `202 queued`, `200 queued` desde el DO, `400` uuid, `404` desconocido (nunca `queued` fantasma en prod)

## Plantilla para escenarios nuevos

```text
**E2E-N Nombre**

- Given <estado inicial verificable>
- When <accion exacta con comando>
- Then <resultado observable con literales, nunca recomputado>
```

Reglas: un escenario por comportamiento, literales golden en el Then, sin probar internos.
Si un Then falla, la fase no se declara completa.
