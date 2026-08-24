#!/usr/bin/env bash
# Tek bir modeli ayaga kaldirir: ayni motorun diger konteynerlerini durdurur
# (port paylasimi), hedefi baslatir, /v1/models 200 olana kadar bekler.
#
# Kullanim: bash jetson/start-model.sh <model-anahtari>
#   Anahtarlar config.yaml'daki models: altindaki adlardir. Listelemek icin argumansiz cagir.
set -u; . "$(dirname "$0")/lib.sh"

KEY="${1:-}"
MODELS="$(models_tsv)" || exit 1
if [ -z "$KEY" ]; then
  echo "kullanim: $0 <model-anahtari>"; echo "mevcut modeller:"
  printf '%s\n' "$MODELS" | awk -F'\t' '{printf "  %-20s (%s, %s)\n", $1, $2, $3}'
  exit 2
fi

ROW=$(printf '%s\n' "$MODELS" | awk -F'\t' -v k="$KEY" '$1==k')
[ -n "$ROW" ] || { bad "model bulunamadi: $KEY (argumansiz cagirip listeye bak)"; exit 2; }
ENGINE=$(printf '%s' "$ROW" | cut -f2)
HF_ID=$(printf '%s' "$ROW" | cut -f3)
CONTAINER=$(printf '%s' "$ROW" | cut -f4)
PORT=$(engine_port "$ENGINE") || { bad "bilinmeyen engine: $ENGINE"; exit 1; }

AV=$(avail_mb)
[ "$AV" -lt 3000 ] && warn "MemAvailable ${AV}MB (<3000) - baslatma riskli, once free-memory.sh dusun"

# ayni motorun diger konteynerleri portu birakmali
while IFS=$'\t' read -r k e h c x xd; do
  if [ "$e" = "$ENGINE" ] && [ "$c" != "$CONTAINER" ]; then
    if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = "true" ]; then
      echo "ayni motorda calisan $c durduruluyor..."
      docker stop -t 20 "$c" >/dev/null
    fi
  fi
done <<< "$MODELS"

docker start "$CONTAINER" >/dev/null
wait_healthy "$CONTAINER" "http://127.0.0.1:${PORT}/v1/models" 300 || exit 1
echo "$KEY ($HF_ID) hazir: http://$(hostname -I | awk '{print $1}'):${PORT}/v1"
free -h | sed -n '2p'
