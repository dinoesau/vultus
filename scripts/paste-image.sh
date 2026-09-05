#!/usr/bin/env bash
set -euo pipefail
# Vuelca la imagen del portapapeles a PNG para adjuntar en OpenCode v2.
# Uso: scripts/paste-image.sh [destino.png]

DEST="${1:-/tmp/opencode-paste-$(date +%Y%m%d-%H%M%S).png}"
mkdir -p "$(dirname "$DEST")"

if ! pngpaste "$DEST" 2>/dev/null; then
  echo "error: no hay imagen PNG en el portapapeles. Copiala con Cmd+C o Cmd+Shift+Ctrl+4." >&2
  exit 1
fi

SIZE=$(wc -c < "$DEST" | tr -d ' ')
if [ "$SIZE" -eq 0 ] || [ "$SIZE" -gt 20971520 ]; then
  echo "error: imagen fuera de limite (size=${SIZE} bytes, max 20MiB)." >&2
  exit 1
fi

case "$DEST" in
  /*) ABS="$DEST" ;;
  *) ABS="$PWD/$DEST" ;;
esac
echo "file://$ABS"
echo "Pega esa linea en el prompt de OpenCode (requiere modelo multimodal)." >&2
