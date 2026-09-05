---
name: e2e
description: 'Runs the full end-to-end verification for the Vultus stack: Docker health, smoke script, Playwright, and edge wrangler dev with real curl. Writes Given-When-Then criteria plus a 12-factor audit before running. Self-updates when new features need testing. Use when the user says E2E, end-to-end, smoke, verifica de punta a punta, or before declaring a phase complete.'
license: MIT
allowed-tools: Bash
---

# E2E

Verificacion real de punta a punta, nunca `config` ni `dry-run`.
Orden fijo: stack -> smoke -> Playwright -> edge.

## Workflow

1. Actualiza la skill si hay superficie nueva (ver Self-update). Nunca saltes lo no cubierto.
2. Lee `references/acceptance.md` y escribe los escenarios aplicables en el chat antes de correr.
3. Lee `references/twelve-factor.md` y escribe la auditoria en el chat antes de correr.
4. Corre `scripts/e2e-stack.sh` desde la raiz del repo. Si `playwright: command not found`, corre `npm install` y `npx playwright install chromium` en `frontend/` y repite `npm run test:e2e`.
5. Corre `scripts/e2e-edge.sh` desde la raiz del repo.
6. Emite el veredicto por escenario (verde/rojo con evidencia) y limpia (ver Cleanup).

## Self-update protocol

Ejecuta este paso siempre antes del run:

1. Lista superficies nuevas sin escenario en `references/coverage.md`: endpoints, WS, visores, reportes, rate-limit, auth, colas reales.
2. Si falta alguna, edita primero los ficheros de esta skill: añade el escenario GWT en `references/acceptance.md`, el comando exacto en `scripts/` y la entrada en `references/coverage.md`.
3. Corre el E2E ya con la skill actualizada. La actualizacion va en el mismo commit o en uno previo con `test(e2e): extiende cobertura a X`.

## Cleanup

- Mata el `wrangler dev` de fondo (lo hace `scripts/e2e-edge.sh`).
- Borra `frontend/test-results/`.
- Si `npm install` genero `frontend/package-lock.json` nuevo, commitealo con `chore(frontend): fija lockfile tras E2E verde`.
- Nunca commitees `.env`, `*.pem`, `*.key` ni `test-results/`.

## References

- `references/acceptance.md`: metodologia Given-When-Then y escenarios E2E-1..E2E-5 mas plantilla para nuevos.
- `references/twelve-factor.md`: checklist de auditoria 12-Factor.
- `references/coverage.md`: registro de cobertura por fase y deuda conocida.
- `scripts/e2e-stack.sh`: ejecuta `compose ps`, healths y `smoke-fase0.sh`.
- `scripts/e2e-edge.sh`: ejecuta `wrangler dev` efimero mas matriz curl.
