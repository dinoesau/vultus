#!/usr/bin/env bash
set -euo pipefail
# E2E stack: compose ps + healths + smoke. Corre desde la raiz del repo.
# Uso: .agents/skills/e2e/scripts/e2e-stack.sh
docker compose ps
echo "---API HEALTH---"
curl -s http://localhost:8000/health; echo
echo "---FRONT---"
curl -s http://localhost:4321 | head -c 300; echo
echo "---SIDECAR---"
curl -s http://localhost:8081/docs | head -c 200; echo
bash scripts/smoke-fase0.sh
