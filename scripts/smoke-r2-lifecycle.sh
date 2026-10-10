#!/usr/bin/env bash
# smoke-r2-lifecycle.sh - verifica que la regla lifecycle red 1 dia esta
# aplicada en R2 (prod + preview). No verifica comentarios en toml: lista la
# regla viva via wrangler y exige id + prefijo + dias canonicos.
#
# Uso:
#   R2_BUCKETS="vultus-jobs vultus-jobs-preview" bash scripts/smoke-r2-lifecycle.sh
#   R2_BUCKET=vultus-jobs bash scripts/smoke-r2-lifecycle.sh
#
# Requiere credenciales Cloudflare en el entorno (CLOUDFLARE_API_TOKEN +
# CLOUDFLARE_ACCOUNT_ID). Solo lectura: delega en r2-lifecycle-apply.sh --check.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
APPLY="$ROOT/scripts/r2-lifecycle-apply.sh"

if [ -n "${R2_BUCKETS:-}" ]; then
  TARGETS="$R2_BUCKETS"
elif [ -n "${R2_BUCKET:-}" ]; then
  TARGETS="$R2_BUCKET"
else
  echo "SMOKE-LIFECYCLE: FAIL env R2_BUCKET/R2_BUCKETS vacio" >&2
  exit 1
fi

echo "smoke lifecycle contra buckets: $TARGETS"
R2_BUCKETS="$TARGETS" bash "$APPLY" --check
echo "smoke lifecycle ok (regla expire-jobs jobs/ 1 dia viva, no solo comentario)"
