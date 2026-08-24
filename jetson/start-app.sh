#!/usr/bin/env bash
# Gradio uygulamasini Jetson host venv'inde baslatir.
# Venv yoksa olusturur (ilk calistirmada pip kurulumu ~2-5 dk).
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv"

# .deps-ok sentineli: pip yarida kesilirse bir sonraki calistirmada kurulum tekrarlanir
if [ ! -f "$VENV/.deps-ok" ]; then
  echo "venv kuruluyor: $VENV"
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r "$REPO/requirements.txt"
  touch "$VENV/.deps-ok"
fi

# Yuklenen videolar tmpfs'e degil diske yazilsin
export GRADIO_TEMP_DIR="$HOME/.cache/gradio-vqa"
mkdir -p "$GRADIO_TEMP_DIR"

cd "$REPO"
exec "$VENV/bin/python" -m app.main --host 0.0.0.0 --port 7860 "$@"
