#!/usr/bin/env bash
# r2-lifecycle-apply.sh - regla lifecycle R2 versionada (red 1 dia sobre jobs/).
#
# Borrado logico 60s vive en ProgressDO/Python (DO TTL 60 + DEL tras entrega).
# R2 no soporta expiracion < 1 dia: esta regla es solo red de seguridad para
# que R2 se limpie aunque falle el borrado logico.
#
# Regla canonica: id `expire-jobs`, prefijo `jobs/`, `--expire-days 1`
# (minimo R2). Se aplica en prod (`vultus-jobs`) y preview
# (`vultus-jobs-preview`). Dev emulado no tiene lifecycle.
#
# Config solo por env (cero literales de bucket en codigo):
#   R2_BUCKET      (o R2_BUCKETS plural separado por espacios para prod+preview)
#   LIFECYCLE_RULE (default expire-jobs)
#   LIFECYCLE_PREFIX (default jobs/)
#   EXPIRE_DAYS    (default 1, minimo R2; distinto de 1 se rechaza)
#   WRANGLER_CMD   (default "npx --yes wrangler@4", overridable en tests)
#
# Uso:
#   R2_BUCKET=vultus-jobs bash scripts/r2-lifecycle-apply.sh --check
#   R2_BUCKET=vultus-jobs bash scripts/r2-lifecycle-apply.sh --apply
#   R2_BUCKETS="vultus-jobs vultus-jobs-preview" bash scripts/r2-lifecycle-apply.sh --apply
#
# --check es solo lectura (exit 0 solo si la regla existe con prefijo+dias
# correctos). --apply es idempotente (existe -> skip, no existe -> add).
# Nunca borra nada.
set -euo pipefail

LIFECYCLE_VERSION=1
DEFAULT_RULE="expire-jobs"
DEFAULT_PREFIX="jobs/"
DEFAULT_DAYS="1"

usage() {
  cat <<EOF
r2-lifecycle-apply.sh v${LIFECYCLE_VERSION} - red R2 1 dia sobre jobs/
Env: R2_BUCKET (o R2_BUCKETS="prod preview"), LIFECYCLE_RULE (default expire-jobs),
     LIFECYCLE_PREFIX (default jobs/), EXPIRE_DAYS (default 1), WRANGLER_CMD override tests
  --check    solo lectura: exige regla + prefijo + dias correctos (exit 0/1)
  --apply    idempotente: existe -> skip, falta -> lifecycle add (exit 0/1)
  --version  imprime version
  --help     esta ayuda
Ejemplo: R2_BUCKETS="vultus-jobs vultus-jobs-preview" bash scripts/r2-lifecycle-apply.sh --apply
EOF
}

fail() {
  echo "LIFECYCLE: FAIL $1" >&2
  exit 1
}

resolve_buckets() {
  if [ -n "${R2_BUCKETS:-}" ]; then
    printf '%s' "$R2_BUCKETS"
  elif [ -n "${R2_BUCKET:-}" ]; then
    printf '%s' "$R2_BUCKET"
  else
    fail "env R2_BUCKET/R2_BUCKETS vacio (config solo por env)"
  fi
}

wrangler() {
  if [ -n "${WRANGLER_CMD:-}" ]; then
    # shellcheck disable=SC2086
    $WRANGLER_CMD "$@"
  else
    npx --yes wrangler@4 "$@"
  fi
}

lifecycle_output() {
  local bucket="$1"
  wrangler r2 bucket lifecycle list "$bucket" 2>/dev/null || fail "wrangler lifecycle list inalcanzable (bucket=$bucket)"
}

# check_bucket <bucket> <rule> <prefix> <days>: exit 0 si la regla existe con
# prefijo+dias correctos, 1 si no. Imprime PASS/FAIL por bucket.
check_bucket() {
  local bucket="$1" rule="$2" prefix="$3" days="$4"
  local out
  out=$(lifecycle_output "$bucket")
  if ! printf '%s' "$out" | grep -q "$rule"; then
    echo "LIFECYCLE: [CHECK] FAIL bucket=$bucket sin regla $rule"
    return 1
  fi
  if ! printf '%s' "$out" | grep -q "$prefix"; then
    echo "LIFECYCLE: [CHECK] FAIL bucket=$bucket regla $rule sin prefijo $prefix"
    return 1
  fi
  if ! printf '%s' "$out" | grep -q "$days"; then
    echo "LIFECYCLE: [CHECK] FAIL bucket=$bucket regla $rule sin expire-days $days"
    return 1
  fi
  echo "LIFECYCLE: [CHECK] PASS bucket=$bucket regla $rule prefijo $prefix expire $days"
  return 0
}

apply_bucket() {
  local bucket="$1" rule="$2" prefix="$3" days="$4"
  if check_bucket "$bucket" "$rule" "$prefix" "$days" 2>/dev/null; then
    echo "LIFECYCLE: [APPLY] SKIP bucket=$bucket regla $rule ya existe, se omite"
    return 0
  fi
  wrangler r2 bucket lifecycle add "$bucket" "$rule" "$prefix" --expire-days "$days" -y \
    || fail "wrangler lifecycle add fallo (bucket=$bucket rule=$rule)"
  echo "LIFECYCLE: [APPLY] PASS bucket=$bucket regla $rule prefijo $prefix expire $days aplicada"
}

cmd="${1:---help}"
RULE="${LIFECYCLE_RULE:-$DEFAULT_RULE}"
PREFIX="${LIFECYCLE_PREFIX:-$DEFAULT_PREFIX}"
DAYS="${EXPIRE_DAYS:-$DEFAULT_DAYS}"

case "$cmd" in
  --version)
    echo "r2-lifecycle-apply.sh v$LIFECYCLE_VERSION"
    ;;
  --help|-h)
    usage
    ;;
  --check|--apply)
    if [ "$DAYS" != "1" ]; then
      fail "EXPIRE_DAYS=$DAYS invalido (minimo R2 = 1; logico 60s vive en DO/Python, no aqui)"
    fi
    if [ "$PREFIX" != "jobs/" ]; then
      fail "LIFECYCLE_PREFIX=$PREFIX invalido (canonico jobs/; v3 usa excepcion aparte)"
    fi
    rc=0
    buckets_output="$(resolve_buckets)" || exit 1
    for b in $buckets_output; do
      if [ "$cmd" = "--check" ]; then
        check_bucket "$b" "$RULE" "$PREFIX" "$DAYS" || rc=1
      else
        apply_bucket "$b" "$RULE" "$PREFIX" "$DAYS" || rc=1
      fi
    done
    if [ "$rc" -eq 0 ]; then
      echo "LIFECYCLE: PASS $cmd buckets: $buckets_output"
    else
      echo "LIFECYCLE: FAIL $cmd (ver lineas FAIL)" >&2
    fi
    exit "$rc"
    ;;
  *)
    echo "LIFECYCLE: FAIL subcomando desconocido: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
