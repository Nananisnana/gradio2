#!/usr/bin/env bash
# config.yaml'daki TUM model konteynerlerini durdurur (baska hicbir konteynere dokunmaz).
set -u; . "$(dirname "$0")/lib.sh"
CONTAINERS=$(models_tsv | cut -f4 | tr '\n' ' ')
# shellcheck disable=SC2086
docker stop -t 20 $CONTAINERS 2>/dev/null
echo "Durduruldu:"
for c in $CONTAINERS; do
  docker ps -a --format '{{.Names}}: {{.Status}}' | grep "^$c:" || true
done
free -h | sed -n '2p'
