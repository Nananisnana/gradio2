#!/usr/bin/env bash
set -u; . "$(dirname "$0")/lib.sh"
echo "== model konteynerleri (config.yaml) =="
while IFS=$'\t' read -r key engine hf_id container extra extra_dl; do
  st=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || echo "YOK")
  printf '  %-20s %-8s %-24s %s\n' "$key" "$engine" "$container" "$st"
done <<< "$(models_tsv)"
echo "== motor sagligi / servis edilen model =="
for pair in "vllm:${VLLM_PORT}" "sglang:${SGL_PORT}"; do
  n=${pair%%:*}; p=${pair##*:}
  code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:$p/v1/models" 2>/dev/null || echo 000)
  m=$(curl -s -m 3 "http://127.0.0.1:$p/v1/models" 2>/dev/null | tr -d '\n' | cut -c1-140)
  echo "$n :$p  status=$code  ${m:-"-"}"
done
echo "== bellek =="
free -h
echo "(GPU/guc icin ayri terminalde: tegrastats)"
