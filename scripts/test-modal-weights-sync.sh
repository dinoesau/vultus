#!/usr/bin/env bash
# test-modal-weights-sync.sh - suite del script admin con `modal` mockeado (Wave 6-fix Lane B).
#
# Cero red, cero Volume real: un `modal` falso en PATH responde ls (generado
# desde un arbol fixture en disco) y get por archivo a stdout, y registra
# cp/rm (que nunca deben correrse aqui). Cubre: bueno (backup por contenido
# PASS), mismo-tamano-distinto-contenido FAIL (prueba hash por contenido, no
# solo tamano), archivo recursivo faltante FAIL, corrupto (JSON invalido FAIL),
# schema (schema inesperado FAIL), refusos de --rm-gnm (sin cutover / sin
# doble confirmacion / confirmacion erronea), subcomando malo (exit 2) y env
# faltante (FAIL).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SYNC="$ROOT/scripts/modal-weights-sync.sh"

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

setup_mock() {
  # $1 = dir de trabajo; crea bin/modal mock + arbol fixture MOCK_VOL_ROOT.
  local work="$1"
  local vol="$work/vol"
  mkdir -p "$work/bin" "$vol/mediapipe" "$vol/flame" "$vol/deca" "$vol/ffhq-uv"
  mkdir -p "$vol/gnm/sub" "$vol/gnm.bak-t1/sub" "$vol/gnm.bak-evil/sub" "$vol/gnm.bak-short/sub"
  printf 'task' >"$vol/mediapipe/face_landmarker.task"
  printf 'pkl' >"$vol/flame/flame2023_Open.pkl"
  printf 'tar' >"$vol/deca/deca_model.tar"
  printf 'obj' >"$vol/ffhq-uv/FLAME_w_HIFI3D_UV.obj"
  printf 'eye' >"$vol/ffhq-uv/eye_ball_tex.png"
  mkdir -p "$vol/checkpoints/texgan_model" "$vol/checkpoints/deep3d_model" "$vol/topo_assets"
  printf 'texgan' >"$vol/checkpoints/texgan_model/texgan_ffhq_uv.pth"
  printf 'deep3d' >"$vol/checkpoints/deep3d_model/epoch_latest.pth"
  printf 'unwrap' >"$vol/topo_assets/unwrap_1024_info.mat"
  printf 'meanface' >"$vol/topo_assets/hifi3dpp_mean_face.obj"
  printf '0123456789ABCDEF' >"$vol/gnm/a.bin"
  printf '0123456789' >"$vol/gnm/same.bin"
  printf 'recursivo' >"$vol/gnm/sub/c.bin"
  cp "$vol/gnm/a.bin" "$vol/gnm.bak-t1/a.bin"
  cp "$vol/gnm/same.bin" "$vol/gnm.bak-t1/same.bin"
  cp "$vol/gnm/sub/c.bin" "$vol/gnm.bak-t1/sub/c.bin"
  cp "$vol/gnm/a.bin" "$vol/gnm.bak-evil/a.bin"
  printf 'XXXXXXXXXX' >"$vol/gnm.bak-evil/same.bin"
  cp "$vol/gnm/sub/c.bin" "$vol/gnm.bak-evil/sub/c.bin"
  cp "$vol/gnm/a.bin" "$vol/gnm.bak-short/a.bin"
  cp "$vol/gnm/same.bin" "$vol/gnm.bak-short/same.bin"
  cat >"$work/bin/modal" <<'MOCK'
#!/usr/bin/env bash
# modal mockeado: MOCK_SCENARIO=good|corrupt|schema; MOCK_VOL_ROOT con el arbol;
# MOCK_LOG registra cp/rm por archivo. ls se genera desde disco; get solo por
# archivo a stdout. V1: cualquier `-r` falla como el Volume real
# (`recursive is not supported for V1 volumes`); cp copia archivo a archivo
# (crea padres); rm solo archivos (dirs sin -r fallan como en V1 real).
set -euo pipefail
sub1=${1:-}; sub2=${2:-}
if [ "$sub1" = "volume" ] && [ "$sub2" = "ls" ]; then
  case "${MOCK_SCENARIO:-good}" in
    corrupt) printf 'esto no es json'; exit 0 ;;
    schema) printf '[{"filename": 42, "type": "file"}]'; exit 0 ;;
  esac
  MOCK_LS_PATH=${4:-/} python3 -c "
import os
root = os.environ['MOCK_VOL_ROOT']
rel = os.environ['MOCK_LS_PATH'].lstrip('/') or '.'
full = os.path.join(root, rel)
if not os.path.isdir(full):
    print('[]')
    raise SystemExit(0)
import json
out = []
for child in sorted(os.listdir(full)):
    cf = os.path.join(full, child)
    name = child if rel == '.' else rel + '/' + child
    out.append({'filename': name, 'type': 'dir' if os.path.isdir(cf) else 'file',
                'size': str(os.path.getsize(cf)) + ' B'})
print(json.dumps(out))
"
  exit 0
fi
if [ "$sub1" = "volume" ] && [ "$sub2" = "get" ]; then
  remote=${4:-}; dest=${5:-}
  full="${MOCK_VOL_ROOT:-}$remote"
  if [ "$dest" = "-" ] && [ -f "$full" ]; then cat "$full"; exit 0; fi
  exit 1
fi
if [ "$sub1" = "volume" ] && [ "$sub2" = "cp" ]; then
  for a in "$@"; do
    if [ "$a" = "-r" ]; then echo "recursive is not supported for V1 volumes" >&2; exit 1; fi
  done
  echo "CP: $*" >>"${MOCK_LOG:-/dev/null}"
  src="${MOCK_VOL_ROOT:-}$4"
  dst="${MOCK_VOL_ROOT:-}$5"
  if [ -f "$src" ]; then mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; exit 0; fi
  echo "mock cp: origen no es archivo: $4" >&2
  exit 1
fi
if [ "$sub1" = "volume" ] && [ "$sub2" = "rm" ]; then
  for a in "$@"; do
    if [ "$a" = "-r" ]; then echo "recursive is not supported for V1 volumes" >&2; exit 1; fi
  done
  echo "RM: $*" >>"${MOCK_LOG:-/dev/null}"
  target="${MOCK_VOL_ROOT:-}$4"
  if [ -f "$target" ]; then rm "$target"; exit 0; fi
  echo "Cannot remove directory without enabling recursive removal." >&2
  exit 1
fi
echo "mock: subcomando no soportado: $*" >&2
exit 3
MOCK
  chmod +x "$work/bin/modal"
}

work=$(mktemp -d "${TMPDIR:-/tmp}/vultus-sync-test.XXXXXX")
export MOCK_LOG="$work/calls.log"
: >"$MOCK_LOG"
setup_mock "$work"
export PATH="$work/bin:$PATH"
export MOCK_VOL_ROOT="$work/vol"
export MODAL_VOLUME=mockvol R2_BUCKET=mockbucket

export MOCK_SCENARIO=good
assert_run "backup bueno por contenido" 0 "BACKUP.*PASS.*por contenido" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t1 bash "$SYNC" --backup

assert_run "mismo tamano distinto contenido FAIL" 1 "difiere" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=evil bash "$SYNC" --backup

assert_run "archivo recursivo faltante FAIL" 1 "difiere" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=short bash "$SYNC" --backup

# Espejo de flame_texture.py (ambos eye files): sin eye_ball_tex.png el puente falla.
mv "$work/vol/ffhq-uv/eye_ball_tex.png" "$work/vol/ffhq-uv/eye_ball_tex.png.hide"
assert_run "ojo ausente FAIL puente" 1 "eye_ball_tex" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t1 bash "$SYNC" --backup
mv "$work/vol/ffhq-uv/eye_ball_tex.png.hide" "$work/vol/ffhq-uv/eye_ball_tex.png"

# Espejo de flame_fit.py (OR de pkl): solo el alias generic_model.pkl tambien pasa.
mv "$work/vol/flame/flame2023_Open.pkl" "$work/vol/flame/flame2023_Open.pkl.hide"
printf 'pkl-alias' >"$work/vol/flame/generic_model.pkl"
assert_run "pkl alias OR PASS" 0 "BACKUP.*PASS.*por contenido" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t1 bash "$SYNC" --backup
rm "$work/vol/flame/generic_model.pkl"
mv "$work/vol/flame/flame2023_Open.pkl.hide" "$work/vol/flame/flame2023_Open.pkl"

export MOCK_SCENARIO=corrupt
assert_run "JSON corrupto FAIL ruidoso" 1 "JSON invalido" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t1 bash "$SYNC" --backup

export MOCK_SCENARIO=schema
assert_run "schema inesperado FAIL ruidoso" 1 "schema inesperado" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t1 bash "$SYNC" --backup

export MOCK_SCENARIO=good
assert_run "rm sin cutover rehusa" 1 "rehuso volume rm" -- \
  env BACKUP_TAG=t1 CONFIRM_RM_GNM=borrar-gnm-t1 bash "$SYNC" --rm-gnm

assert_run "rm sin doble confirmacion rehusa" 1 "doble confirmacion" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t1 bash "$SYNC" --rm-gnm

assert_run "rm con confirmacion erronea rehusa" 1 "doble confirmacion" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t1 CONFIRM_RM_GNM=si-borralo bash "$SYNC" --rm-gnm

assert_run "subcomando malo exit 2" 2 "subcomando desconocido" -- \
  bash "$SYNC" --no-existe

assert_run "sin env FAIL" 1 "env MODAL_VOLUME vacio" -- \
  env -u MODAL_VOLUME bash "$SYNC" --check

if [ -s "$MOCK_LOG" ]; then
  bad "solo-lectura y refusos invocaron cp/rm (log debe estar vacio): $(cat "$MOCK_LOG")"
else
  ok "solo-lectura y refusos: cero cp/rm invocados"
fi

# Fase 2: operaciones reales contra el mock (copia/restaura/borra de verdad).
# --backup con tag fresco t2: copia por archivo y verifica por contenido.
assert_run "backup fresco copia y verifica" 0 "backup ok" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t2 bash "$SYNC" --backup

# Sin -r en ningun lado (regresion V1: recursive is not supported).
if grep -q " -r " "$MOCK_LOG"; then
  bad "el script uso -r (prohibido en V1): $(grep " -r " "$MOCK_LOG")"
else
  ok "cero flags -r en cp/rm (V1-safe)"
fi

# --restore: rompe un archivo de gnm y restaura desde el backup t2.
printf 'ROTO' >"$work/vol/gnm/a.bin"
assert_run "restore repara y verifica" 0 "restore ok" -- \
  env BACKUP_TAG=t2 CONFIRM_RESTORE=restaurar-gnm-t2 bash "$SYNC" --restore
[ "$(cat "$work/vol/gnm/a.bin")" = "0123456789ABCDEF" ] && ok "restore dejo bytes identicos" || bad "restore dejo bytes distintos"

# --restore sin confirmacion rehusa.
assert_run "restore sin confirmacion rehusa" 1 "doble confirmacion" -- \
  env BACKUP_TAG=t2 bash "$SYNC" --restore

# --rm-gnm real al final (borra los archivos del mock gnm, backup intacto).
assert_run "rm borra archivos y verifica vacio" 0 "cero archivos" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t2 CONFIRM_RM_GNM=borrar-gnm-t2 bash "$SYNC" --rm-gnm
[ -z "$(find "$work/vol/gnm" -type f)" ] && ok "mock gnm sin archivos tras rm" || bad "mock gnm aun tiene archivos"
[ -f "$work/vol/gnm.bak-t2/a.bin" ] && ok "backup intacto tras rm" || bad "backup danado por rm"

# --check post-rm: origen vacio + backup standalone + volumen vacio = PASS.
assert_run "check post-rm PASS" 0 "PASS post-cutover" -- \
  env FLAME_CUTOVER=1 BACKUP_TAG=t2 bash "$SYNC" --check

rm -rf "$work"
echo "TEST: $pass passed, $fail_n failed"
[ "$fail_n" -eq 0 ]
