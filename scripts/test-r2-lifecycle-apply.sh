#!/usr/bin/env bash
# test-r2-lifecycle-apply.sh - suite del script admin con `wrangler` mockeado.
# Cero red, cero bucket real: WRANGLER_CMD apunta a un falso en PATH que
# responde `lifecycle list` desde fixtures y registra `lifecycle add`.
# Cubre: check PASS con regla correcta, check FAIL sin regla / prefijo
# distinto / dias distintos, apply SKIP idempotente, apply ADD cuando falta,
# rechazo de EXPIRE_DAYS != 1 y PREFIX != jobs/, subcomando malo (exit 2),
# env faltante (FAIL), prod+preview (R2_BUCKETS plural).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
APPLY="$ROOT/scripts/r2-lifecycle-apply.sh"

pass=0
fail_n=0

ok() {
  pass=$((pass + 1))
  echo "TEST: PASS $1"
}

bad() {
  fail_n=$((fail_n + 1))
  echo "TEST: FAIL $1" >&2
}

# assert_run <nombre> <exit-esperado> <grep-esperado> -- <comando...>
assert_run() {
  local name="$1" want_exit="$2" want_grep="$3"
  shift 3
  if [ "$1" = "--" ]; then shift; fi
  local out rc
  out=$("$@" 2>&1) && rc=0 || rc=$?
  if [ "$rc" -ne "$want_exit" ]; then
    bad "$name (exit=$rc esperado=$want_exit; out: $out)"
    return 0
  fi
  if ! printf '%s' "$out" | grep -q "$want_grep"; then
    bad "$name (sin '$want_grep'; out: $out)"
    return 0
  fi
  ok "$name"
}

work=$(mktemp -d "${TMPDIR:-/tmp}/vultus-lifecycle-test.XXXXXX")
mkdir -p "$work/bin"
: >"$work/calls.log"
export MOCK_CALLS="$work/calls.log"
cat >"$work/bin/wrangler" <<'MOCK'
#!/usr/bin/env bash
# Mock: MOCK_SCENARIO rige list; add registra y responde ok.
set -euo pipefail
if [ "${1:-}" = "r2" ] && [ "${2:-}" = "bucket" ] && [ "${3:-}" = "lifecycle" ] && [ "${4:-}" = "list" ]; then
  case "${MOCK_SCENARIO:-good}" in
    empty) printf '[]' ;;
    wrong-prefix) printf '[{"id": "expire-jobs", "prefix": "other/", "expire_days": 1}]' ;;
    wrong-days) printf '[{"id": "expire-jobs", "prefix": "jobs/", "expire_days": 7}]' ;;
    *) printf '[{"id": "expire-jobs", "prefix": "jobs/", "expire_days": 1}]' ;;
  esac
  exit 0
fi
if [ "${1:-}" = "r2" ] && [ "${2:-}" = "bucket" ] && [ "${3:-}" = "lifecycle" ] && [ "${4:-}" = "add" ]; then
  echo "ADD: $*" >>"${MOCK_CALLS:-/dev/null}"
  echo "added $5 $6 $7"
  exit 0
fi
echo "mock: no soportado: $*" >&2
exit 3
MOCK
chmod +x "$work/bin/wrangler"
export WRANGLER_CMD="$work/bin/wrangler"

export MOCK_SCENARIO=good
assert_run "check PASS regla correcta" 0 "CHECK.*PASS.*expire-jobs" -- \
  env R2_BUCKET=vultus-jobs bash "$APPLY" --check

assert_run "apply SKIP idempotente sin add" 0 "SKIP.*ya existe" -- \
  env R2_BUCKET=vultus-jobs bash "$APPLY" --apply
if [ -s "$work/calls.log" ]; then
  bad "apply SKIP invoco add (debía omitir): $(cat "$work/calls.log")"
else
  ok "apply SKIP: cero add invocados"
fi

export MOCK_SCENARIO=empty
assert_run "check FAIL sin regla" 1 "sin regla" -- \
  env R2_BUCKET=vultus-jobs bash "$APPLY" --check

: >"$work/calls.log"
assert_run "apply ADD cuando falta" 0 "aplicada" -- \
  env R2_BUCKET=vultus-jobs bash "$APPLY" --apply
if grep -q "ADD:" "$work/calls.log"; then
  ok "apply ADD registro llamada add"
else
  bad "apply ADD no registro llamada add"
fi
if grep -q "jobs/" "$work/calls.log" && grep -q "expire-days 1" "$work/calls.log"; then
  ok "apply ADD con prefijo jobs/ y expire-days 1"
else
  bad "apply ADD sin prefijo/dias canonicos: $(cat "$work/calls.log")"
fi

export MOCK_SCENARIO=wrong-prefix
assert_run "check FAIL prefijo distinto" 1 "sin prefijo" -- \
  env R2_BUCKET=vultus-jobs bash "$APPLY" --check

export MOCK_SCENARIO=wrong-days
assert_run "check FAIL dias distintos" 1 "sin expire-days" -- \
  env R2_BUCKET=vultus-jobs bash "$APPLY" --check

export MOCK_SCENARIO=good
assert_run "prod+preview plural PASS" 0 "LIFECYCLE: PASS" -- \
  env R2_BUCKETS="vultus-jobs vultus-jobs-preview" bash "$APPLY" --check

assert_run "EXPIRE_DAYS distinto de 1 se rechaza" 1 "minimo R2" -- \
  env R2_BUCKET=vultus-jobs EXPIRE_DAYS=7 bash "$APPLY" --check

assert_run "PREFIX distinto de jobs/ se rechaza" 1 "canonico jobs" -- \
  env R2_BUCKET=vultus-jobs LIFECYCLE_PREFIX="v3/" bash "$APPLY" --check

assert_run "subcomando malo exit 2" 2 "subcomando desconocido" -- \
  bash "$APPLY" --no-existe

assert_run "sin env FAIL" 1 "R2_BUCKET/R2_BUCKETS vacio" -- \
  env -u R2_BUCKET -u R2_BUCKETS bash "$APPLY" --check

rm -rf "$work"
echo "TEST: $pass passed, $fail_n failed"
[ "$fail_n" -eq 0 ]
