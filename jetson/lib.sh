#!/usr/bin/env bash
# Ortak yardimcilar. Kullanim: . "$(dirname "$0")/lib.sh"
#
# Port tanimlari TEK yerde (motor basina tek port; ayni motorun modelleri portu
# paylasir, ayni anda tek model calisir). Alternatif porta gecis (README #9)
# ayni env degiskenleriyle TUM scriptlerde gecerli olur.
VLLM_PORT="${VLLM_PORT:-8000}"
SGL_PORT="${SGL_PORT:-30000}"

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_YAML="${CONFIG_YAML:-$_LIB_DIR/../config.yaml}"

C_G='\033[32m'; C_R='\033[31m'; C_Y='\033[33m'; C_0='\033[0m'
ok()   { printf " ${C_G}PASS${C_0}  %s\n" "$1"; }
bad()  { printf " ${C_R}FAIL${C_0}  %s\n" "$1"; }
warn() { printf " ${C_Y}WARN${C_0}  %s\n" "$1"; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

# Model kayitlarini config.yaml'dan okur (tek dogruluk kaynagi).
# Cikti satirlari:
#   key<TAB>engine<TAB>hf_id<TAB>container<TAB>serve_extra<TAB>extra_downloads(virgullu)
models_tsv() {
  python3 - "$CONFIG_YAML" <<'PY'
import sys
try:
    import yaml
except ImportError:
    sys.exit("HATA: python3 yaml modulu yok -> sudo apt install -y python3-yaml")
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for k, m in cfg["models"].items():
    extra_dl = ",".join(m.get("extra_downloads") or [])
    print("\t".join([k, m["engine"], m["hf_id"], m["container"],
                     m.get("serve_extra", ""), extra_dl]))
PY
}

config_default_model() {
  python3 - "$CONFIG_YAML" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg.get("default_model") or next(iter(cfg["models"])))
PY
}

engine_port() { # $1=engine -> port
  case "$1" in
    vllm) echo "$VLLM_PORT" ;;
    sglang) echo "$SGL_PORT" ;;
    *) echo ""; return 1 ;;
  esac
}

# wait_healthy <container> <health_url> <timeout_s>
# Hazir olma tanimi (app ile ortak sozlesme): GET {base_url}/v1/models -> 200
wait_healthy() {
  local name=$1 url=$2 timeout=$3 t=0
  echo "Bekleniyor: $name -> $url (azami ${timeout}sn; ilk yuklemede 2-4 dk normal)"
  while [ "$t" -lt "$timeout" ]; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != "true" ]; then
      bad "$name konteyneri durdu (crash). Son 80 satir log:"
      docker logs --tail 80 "$name" 2>&1 | tail -n 80
      return 1
    fi
    if curl -sf -m 2 "$url" >/dev/null 2>&1; then
      ok "$name saglikli (${t}sn)"
      return 0
    fi
    sleep 3; t=$((t+3))
  done
  bad "$name ${timeout}sn icinde saglikli olmadi. Son 80 satir log:"
  docker logs --tail 80 "$name" 2>&1 | tail -n 80
  return 1
}
