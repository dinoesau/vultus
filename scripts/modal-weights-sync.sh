#!/usr/bin/env bash
# modal-weights-sync.sh - proceso admin versionado para la limpieza del Volume (Wave 6 Step 7).
#
# Orden fijo (ADR-008 + plan-flame-deca-render Step 7):
#   1. puente-up:   assets DECA/FLAME/FFHQ-UV presentes en el Volume.
#   2. cutover:     flag FLAME_CUTOVER=1 (codigo FLAME desplegado y sirviendo el zip real).
#   3. backup:      copia por ARCHIVO de /gnm a /gnm.bak-<tag> + manifest
#                  sha256 por CONTENIDO verificado (nombre + tamano + sha256
#                  por archivo via `modal volume get` a tmp; nunca solo tamano).
#   4. rm:          solo entonces borrado por ARCHIVO de /gnm (doble confirmacion).
#
# Nota V1: `modal volume cp/rm -r` NO existe en Volumes V1
# (`recursive is not supported for V1 volumes`); este script jamas usa `-r`:
# copia/borra archivo por archivo (los padres se autocrean al copiar).
# Los directorios vacios son inborrables en V1: la verificacion post-rm es
# "cero archivos bajo /gnm", no "directorio ausente".
#
# Config solo por env (cero literales de volumen/bucket en codigo):
#   MODAL_VOLUME   (requerido)  nombre del Volume, ej. vultus-weights.
#   R2_BUCKET      (requerido)  bucket de jobs, solo contexto operativo del gate.
#   FLAME_CUTOVER  (requerido "1" para backup/rm)  flag de cutover de codigo.
#   BACKUP_TAG     (requerido para backup/rm/--check post-cutover)  ej. 2026-09-28.
#   CONFIRM_RM_GNM (requerido "borrar-gnm-<tag>" solo para --rm-gnm) doble confirmacion.
#
# Layout interno del Volume (dato observado via `modal volume ls`, espejo del
# env de modal_app.py: /weights/{flame,deca,ffhq-uv,gnm,mediapipe}):
#   mediapipe/face_landmarker.task, flame/flame2023_Open.pkl o
#   flame/generic_model.pkl (alias OR, espejo de backend/flame_fit.py
#   FLAME_PKL_NAMES), deca/deca_model.tar,
#   ffhq-uv/FLAME_w_HIFI3D_UV.obj + ffhq-uv/eye_ball_tex.png (ambos, espejo
#   de backend/flame_texture.py UV_OBJ_NAME + EYE_MAP_NAME), gnm/ (legacy).
#
# Restore documentado (rollback si el cutover falla tras el rm), V1-safe por archivo:
#   MODAL_VOLUME=... BACKUP_TAG=... CONFIRM_RESTORE=restaurar-gnm-<tag> bash scripts/modal-weights-sync.sh --restore
#   modal volume ls "$MODAL_VOLUME" /gnm --json   # verifica vuelta de archivos bajo gnm/
#
# Uso:
#   MODAL_VOLUME=... R2_BUCKET=... bash scripts/modal-weights-sync.sh --check
#   MODAL_VOLUME=... R2_BUCKET=... FLAME_CUTOVER=1 BACKUP_TAG=... bash scripts/modal-weights-sync.sh --backup
#   MODAL_VOLUME=... R2_BUCKET=... FLAME_CUTOVER=1 BACKUP_TAG=... CONFIRM_RM_GNM=borrar-gnm-<tag> bash scripts/modal-weights-sync.sh --rm-gnm
#   MODAL_VOLUME=... BACKUP_TAG=... CONFIRM_RESTORE=restaurar-gnm-<tag> bash scripts/modal-weights-sync.sh --restore
#
# --check es solo lectura (ls + get a tmp para hashes) y falla si falta el puente,
# el flag de cutover, el backup verificado por contenido, o si /gnm sigue presente.
# Nunca borra nada.
set -euo pipefail

SYNC_VERSION=4
# Puente canonico (espejo exacto de weights_present):
# - backend/flame_fit.py: deca_model.tar en DECA_DIR + (flame2023_Open.pkl OR
#   generic_model.pkl) en FLAME_ASSETS_DIR.
# - backend/flame_texture.py: FLAME_w_HIFI3D_UV.obj + eye_ball_tex.png en FFHQ_UV_DIR.
# BRIDGE_FILES son los 4 exactos obligatorios; el pkl FLAME va por OR en
# BRIDGE_FLAME_PKL_ALTS (uno basta, ambos valen).
BRIDGE_FILES="mediapipe/face_landmarker.task deca/deca_model.tar ffhq-uv/FLAME_w_HIFI3D_UV.obj ffhq-uv/eye_ball_tex.png checkpoints/texgan_model/texgan_ffhq_uv.pth checkpoints/deep3d_model/epoch_latest.pth topo_assets/unwrap_1024_info.mat topo_assets/hifi3dpp_mean_face.obj flame/flame_template.bin flame/flame_hifi_transfer.npz flame/flame68_embed.npz"
BRIDGE_FLAME_PKL_ALTS="flame/flame2023_Open.pkl flame/generic_model.pkl"
LEGACY_PREFIX="gnm"

usage() {
  cat <<EOF
modal-weights-sync.sh v${SYNC_VERSION} - limpieza segura del Volume (puente-up -> cutover -> backup -> rm gnm)
Env: MODAL_VOLUME (req), R2_BUCKET (req), FLAME_CUTOVER=1 (req backup/rm), BACKUP_TAG (req backup/rm), CONFIRM_RM_GNM=borrar-gnm-<tag> (req rm)
  --check    solo lectura: puente + cutover + backup verificado por contenido + gnm sin archivos (exit 0 solo post-cutover)
  --backup   copia /gnm a /gnm.bak-<tag> por archivo (V1-safe, sin -r) y verifica manifest por contenido (requiere puente + cutover)
  --rm-gnm   borra archivos de /gnm por archivo (V1-safe, sin -r; dirs vacios quedan) (REHUSA sin backup verificado por contenido del tag dado + doble confirmacion)
  --restore  restaura /gnm.bak-<tag> a /gnm por archivo (requiere backup verificado + doble confirmacion CONFIRM_RESTORE=restaurar-gnm-<tag>)
  --version  imprime version
  --help     esta ayuda
Restore: modal volume cp -r "\$MODAL_VOLUME" /gnm.bak-<tag> /gnm
EOF
}

fail() {
  echo "SYNC: FAIL $1" >&2
  exit 1
}

need_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    fail "env $name vacio (config solo por env)"
  fi
}

# Lista --json de un prefijo; imprime el json o falla.
vol_ls_json() {
  local prefix="$1"
  modal volume ls "$MODAL_VOLUME" "/$prefix" --json 2>/dev/null || fail "modal volume ls inalcanzable ($MODAL_VOLUME/$prefix)"
}

# Parse estricto del --json de `modal volume ls` (nunca grep sobre JSON):
# valida schema (lista de {filename: str, type: str}) y ejecuta la consulta.
# Modos: has <want> (exit 0 si existe exacto, 1 si no), list (filenames
# ordenados), files (lineas "type<TAB>filename<TAB>size" solo con size str).
# JSON invalido o schema inesperado => FAIL ruidoso (exit 1), nunca match parcial.
ls_query() {
  local mode="$1"
  local want="${2:-}"
  LS_WANT="$want" python3 -c "
import json, os, sys
want = os.environ.get('LS_WANT', '')
mode = sys.argv[1]
try:
    items = json.load(sys.stdin)
except ValueError:
    print('SYNC: FAIL volume ls devolvio JSON invalido', file=sys.stderr)
    sys.exit(1)
if not isinstance(items, list):
    print('SYNC: FAIL volume ls schema inesperado (no es lista)', file=sys.stderr)
    sys.exit(1)
names = []
rows = []
for entry in items:
    if not isinstance(entry, dict):
        print('SYNC: FAIL volume ls schema inesperado (item no es objeto)', file=sys.stderr)
        sys.exit(1)
    filename = entry.get('filename')
    etype = entry.get('type')
    if not isinstance(filename, str) or not isinstance(etype, str):
        print('SYNC: FAIL volume ls schema inesperado (filename/type no son str)', file=sys.stderr)
        sys.exit(1)
    names.append(filename)
    size = entry.get('size', '')
    if not isinstance(size, str):
        print('SYNC: FAIL volume ls schema inesperado (size no es str)', file=sys.stderr)
        sys.exit(1)
    rows.append((etype, filename, size))
if mode == 'has':
    sys.exit(0 if want in names else 1)
if mode == 'list':
    print('\n'.join(sorted(names)))
elif mode == 'files':
    for etype, filename, size in sorted(rows):
        print(f'{etype}\t{filename}\t{size}')
else:
    print('SYNC: FAIL modo interno desconocido', file=sys.stderr)
    sys.exit(1)
" "$mode"
}

# True (0) si el path exacto existe en el Volume (match exacto, sin basename).
vol_has() {
  local path="$1"
  vol_ls_json "$(dirname "$path")" | ls_query has "$path"
}

# Descarga un archivo del Volume a stdout y devuelve su sha256 (3 intentos).
# `modal volume get` de directorios es fragil; por archivo a stdout es fiable.
get_file_sha() {
  local remote_path="$1"
  local attempt out
  for attempt in 1 2 3; do
    if out=$(modal volume get "$MODAL_VOLUME" "/$remote_path" - 2>/dev/null | sha256sum | cut -d' ' -f1); then
      if [ -n "$out" ]; then
        printf '%s' "$out"
        return 0
      fi
    fi
    echo "SYNC: reintentando get de /$remote_path (intento $attempt/3)" >&2
    sleep 2
  done
  return 1
}

# Manifest por CONTENIDO de un prefijo: filas "rel size sha256" ordenadas.
# Enumera recursivamente con `ls` estricto y hashea cada archivo via `get -`
# (solo lectura). Nunca compara solo por tamano: mismo tamano con distinto
# contenido difiere. Devuelve 1 (sin `fail` directo: el llamador decide el
# mensaje; `exit` dentro de $(...) solo mataria al subshell).
content_manifest_of() {
  local prefix="$1"
  local queue head json entries entry etype fname fsize rel sha rows
  queue=("$prefix")
  head=0
  rows=""
  while [ "$head" -lt "${#queue[@]}" ]; do
    json=$(vol_ls_json "${queue[$head]}") || return 1
    head=$((head + 1))
    entries=$(printf '%s' "$json" | ls_query files) || return 1
    while IFS= read -r entry; do
      etype=${entry%%$'\t'*}
      fname=${entry#*$'\t'}
      fsize=${fname##*$'\t'}
      fname=${fname%$'\t'*}
      if [ "$etype" = "dir" ]; then
        queue+=("$fname")
      elif [ "$etype" = "file" ]; then
        rel=${fname#"$prefix/"}
        sha=$(get_file_sha "$fname") || return 1
        rows="${rows}${rel} ${fsize} ${sha}"$'\n'
      fi
    done <<<"$entries"
  done
  printf '%s' "$rows" | sort
}

# Nombres de ARCHIVO bajo un prefijo, uno por linea, ordenados (sin dirs).
# Base de copy_tree/rm_tree/files_empty. V1-safe: solo `ls`, nunca `-r`.
list_files() {
  local prefix="$1"
  local queue head json entries entry etype fname
  queue=("$prefix")
  head=0
  while [ "$head" -lt "${#queue[@]}" ]; do
    json=$(vol_ls_json "${queue[$head]}") || return 1
    head=$((head + 1))
    entries=$(printf '%s' "$json" | ls_query files) || return 1
    while IFS= read -r entry; do
      [ -n "$entry" ] || continue
      etype=${entry%%$'\t'*}
      fname=${entry#*$'\t'}
      fname=${fname%$'\t'*}
      if [ "$etype" = "dir" ]; then
        queue+=("$fname")
      elif [ "$etype" = "file" ]; then
        printf '%s\n' "$fname"
      fi
    done <<<"$entries"
  done | sort -u
}

# True (0) si no hay archivos bajo el prefijo (dirs vacios no cuentan: V1).
files_empty() {
  local prefix="$1"
  local files
  files=$(list_files "$prefix") || return 1
  [ -z "$files" ]
}

# Copia un arbol archivo por archivo (V1-safe, jamas `-r`; padres se autocrean).
copy_tree() {
  local src="$1" dst="$2"
  local f rel
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    rel=${f#"$src/"}
    [ "$rel" != "$f" ] || fail "ruta fuera del prefijo: $f (src $src)"
    modal volume cp "$MODAL_VOLUME" "/$f" "/$dst/$rel" || return 1
  done < <(list_files "$src")
}

# Borra los archivos bajo un prefijo uno por uno (V1-safe, jamas `-r`).
# Imprime el conteo borrado a stdout (ultima linea).
rm_tree() {
  local prefix="$1"
  local f n
  n=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    modal volume rm "$MODAL_VOLUME" "/$f" || return 1
    n=$((n + 1))
  done < <(list_files "$prefix")
  printf '%d' "$n"
}

check_bridge() {
  local missing=0 f alt pkl_ok
  for f in $BRIDGE_FILES; do
    if vol_has "$f"; then
      echo "SYNC: [BRIDGE] PASS $f"
    else
      echo "SYNC: [BRIDGE] FAIL falta $f"
      missing=1
    fi
  done
  # OR de pkl FLAME (espejo de FLAME_PKL_NAMES): basta uno de los alias.
  pkl_ok=1
  for alt in $BRIDGE_FLAME_PKL_ALTS; do
    if vol_has "$alt"; then
      echo "SYNC: [BRIDGE] PASS $alt (alias OR)"
      pkl_ok=0
      break
    fi
  done
  if [ "$pkl_ok" -ne 0 ]; then
    echo "SYNC: [BRIDGE] FAIL falta flame pkl (flame2023_Open.pkl o generic_model.pkl)"
    missing=1
  fi
  return $missing
}

check_backup() {
  local tag="$1"
  local src dst
  src=$(content_manifest_of "$LEGACY_PREFIX") || fail "no se pudo listar contenido de /$LEGACY_PREFIX"
  [ -n "$src" ] || fail "origen /$LEGACY_PREFIX vacio (nada que respaldar)"
  dst=$(content_manifest_of "$LEGACY_PREFIX.bak-$tag") || fail "backup /$LEGACY_PREFIX.bak-$tag ausente o ilegible"
  [ -n "$dst" ] || fail "backup /$LEGACY_PREFIX.bak-$tag vacio"
  if [ "$src" = "$dst" ]; then
    local sha
    sha=$(printf '%s' "$dst" | sha256sum | cut -d' ' -f1)
    echo "SYNC: [BACKUP] PASS /$LEGACY_PREFIX.bak-$tag verificado por contenido (manifest sha256=$sha)"
    return 0
  fi
  echo "SYNC: [BACKUP] FAIL manifest de contenido difiere entre /$LEGACY_PREFIX y /$LEGACY_PREFIX.bak-$tag"
  echo "--- origen ---"; printf '%s\n' "$src"
  echo "--- backup ---"; printf '%s\n' "$dst"
  return 1
}

# Backup standalone para estado post-rm: el origen ya esta vacio por diseno,
# asi que se verifica que el backup exista, no este vacio, e imprime su sha
# (la igualdad origen==backup se probo en --backup/--rm-gnm antes del rm).
check_backup_standalone() {
  local tag="$1"
  local dst sha
  dst=$(content_manifest_of "$LEGACY_PREFIX.bak-$tag") || fail "backup /$LEGACY_PREFIX.bak-$tag ausente o ilegible"
  [ -n "$dst" ] || fail "backup /$LEGACY_PREFIX.bak-$tag vacio"
  sha=$(printf '%s' "$dst" | sha256sum | cut -d' ' -f1)
  echo "SYNC: [BACKUP] PASS /$LEGACY_PREFIX.bak-$tag post-rm standalone (manifest sha256=$sha, $(( $(printf '%s\n' "$dst" | wc -l) )) archivos)"
}

cmd="${1:---help}"
case "$cmd" in
  --version)
    echo "modal-weights-sync.sh v$SYNC_VERSION"
    ;;
  --help|-h)
    usage
    ;;
  --check)
    need_env MODAL_VOLUME
    need_env R2_BUCKET
    echo "SYNC: v$SYNC_VERSION volume=$MODAL_VOLUME bucket=$R2_BUCKET (solo lectura)"
    rc=0
    check_bridge || rc=1
    if [ "${FLAME_CUTOVER:-0}" != "1" ]; then
      echo "SYNC: [CUTOVER] FAIL flag FLAME_CUTOVER!=1 (codigo FLAME aun no cutover)"
      rc=1
    else
      echo "SYNC: [CUTOVER] PASS FLAME_CUTOVER=1"
    fi
    if [ -n "${BACKUP_TAG:-}" ]; then
      if files_empty "$LEGACY_PREFIX" 2>/dev/null; then
        check_backup_standalone "$BACKUP_TAG" || rc=1
      else
        check_backup "$BACKUP_TAG" || rc=1
      fi
    else
      echo "SYNC: [BACKUP] FAIL BACKUP_TAG vacio (sin tag no hay backup verificable)"
      rc=1
    fi
    if files_empty "$LEGACY_PREFIX" 2>/dev/null; then
      echo "SYNC: [VOLUME] PASS cero archivos bajo /$LEGACY_PREFIX (solo pesos vivos + dirs vacios V1)"
    else
      echo "SYNC: [VOLUME] FAIL /$LEGACY_PREFIX aun tiene archivos (pre-rm; post-cutover debe estar vacio)"
      rc=1
    fi
    if [ "$rc" -eq 0 ]; then
      echo "SYNC: PASS post-cutover (puente + cutover + backup + solo vivos)"
    else
      echo "SYNC: FAIL (ver lineas FAIL; skip nunca es PASS para el gate prod)"
    fi
    exit "$rc"
    ;;
  --backup)
    need_env MODAL_VOLUME
    need_env R2_BUCKET
    [ "${FLAME_CUTOVER:-0}" = "1" ] || fail "rehuso --backup sin FLAME_CUTOVER=1 (orden: puente-up -> cutover -> backup)"
    need_env BACKUP_TAG
    check_bridge || fail "puente incompleto, backup abortado"
    if [ -n "$(list_files "$LEGACY_PREFIX.bak-$BACKUP_TAG" 2>/dev/null)" ]; then
      echo "SYNC: backup ya existe, re-verificando manifest por contenido"
    else
      echo "SYNC: copiando /$LEGACY_PREFIX a /$LEGACY_PREFIX.bak-$BACKUP_TAG (por archivo, V1-safe)"
      copy_tree "$LEGACY_PREFIX" "$LEGACY_PREFIX.bak-$BACKUP_TAG" \
        || fail "copia por archivo fallo"
    fi
    check_backup "$BACKUP_TAG" || fail "backup no verificable tras cp"
    echo "SYNC: backup ok. Restore si el cutover falla: MODAL_VOLUME=\"$MODAL_VOLUME\" BACKUP_TAG=\"$BACKUP_TAG\" CONFIRM_RESTORE=\"restaurar-gnm-$BACKUP_TAG\" bash scripts/modal-weights-sync.sh --restore"
    ;;
  --rm-gnm)
    need_env MODAL_VOLUME
    need_env R2_BUCKET
    [ "${FLAME_CUTOVER:-0}" = "1" ] || fail "rehuso volume rm $LEGACY_PREFIX sin FLAME_CUTOVER=1"
    need_env BACKUP_TAG
    [ "${CONFIRM_RM_GNM:-}" = "borrar-gnm-$BACKUP_TAG" ] || fail "rehuso volume rm $LEGACY_PREFIX sin doble confirmacion (CONFIRM_RM_GNM=borrar-gnm-$BACKUP_TAG)"
    check_backup "$BACKUP_TAG" || fail "rehuso volume rm $LEGACY_PREFIX sin backup verificado por contenido (BACKUP_TAG=$BACKUP_TAG)"
    echo "SYNC: backup verificado por contenido + doble confirmacion, borrando archivos de /$LEGACY_PREFIX (dirs vacios quedan en V1)"
    rm_tree "$LEGACY_PREFIX" || fail "borrado por archivo fallo"
    if files_empty "$LEGACY_PREFIX"; then
      echo "SYNC: rm ok, cero archivos bajo /$LEGACY_PREFIX (solo pesos vivos + dirs vacios V1)"
    else
      fail "/$LEGACY_PREFIX aun tiene archivos tras rm"
    fi
    echo "SYNC: rm ok. Restore: MODAL_VOLUME=\"$MODAL_VOLUME\" BACKUP_TAG=\"$BACKUP_TAG\" CONFIRM_RESTORE=\"restaurar-gnm-$BACKUP_TAG\" bash scripts/modal-weights-sync.sh --restore"
    ;;
  --restore)
    need_env MODAL_VOLUME
    need_env BACKUP_TAG
    [ "${CONFIRM_RESTORE:-}" = "restaurar-gnm-$BACKUP_TAG" ] || fail "rehuso restore sin doble confirmacion (CONFIRM_RESTORE=restaurar-gnm-$BACKUP_TAG)"
    check_bridge || fail "puente incompleto, restore abortado (no se toca /$LEGACY_PREFIX)"
    echo "SYNC: restaurando /$LEGACY_PREFIX.bak-$BACKUP_TAG a /$LEGACY_PREFIX (por archivo, V1-safe)"
    copy_tree "$LEGACY_PREFIX.bak-$BACKUP_TAG" "$LEGACY_PREFIX" \
      || fail "restauracion por archivo fallo"
    check_backup "$BACKUP_TAG" || fail "restore no verificable (manifest difiere tras copia)"
    echo "SYNC: restore ok, /$LEGACY_PREFIX verificado por contenido contra el backup"
    ;;
  *)
    echo "SYNC: FAIL subcomando desconocido: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
