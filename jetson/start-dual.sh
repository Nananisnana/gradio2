#!/usr/bin/env bash
# DUAL MOD DENEYI (varsayilan DEGIL - guvenli varsayilan switch't ir):
# iki motorun birer modelini SIRAYLA baslatir - asla ayni anda initialize etme:
# yukleme aninin tepe bellek kullanimi cakisir ve OOM uretir.
#   vLLM slotu  -> config.yaml'daki default_model (vllm motorunda olmali)
#   SGLang slotu-> config.yaml'daki ilk sglang modeli
# ONKOSUL: container'lar dusuk fraksiyonla olusturulmus olmali:
#   VLLM_FRAC=0.3 SGL_FRAC=0.3 bash jetson/setup.sh --recreate
set -u; . "$(dirname "$0")/lib.sh"
AV=$(avail_mb)
[ "$AV" -lt 3500 ] && warn "MemAvailable ${AV}MB (<3500) - dual mod muhtemelen sigmaz; yine de deneniyor"

# fraksiyon kontrolu: container'lar switch varsayilaniyla (0.5) olusturulduysa
# dual'da toplam butce kesin asilir - once dusuk frac ile recreate iste
frac_of() { docker inspect -f '{{join .Config.Cmd " "}}' "$1" 2>/dev/null | grep -oE '(gpu-memory-utilization|mem-fraction-static) [0-9.]+' | awk '{print $2}'; }
HIGH=0
while IFS=$'\t' read -r k e h c x xd; do
  f=$(frac_of "$c"); [ -n "$f" ] || continue
  awk -v v="$f" 'BEGIN{exit !(v>0.35)}' && { warn "$c fraksiyonu $f (>0.35) - dual icin yuksek"; HIGH=1; }
done <<< "$(models_tsv)"
if [ "$HIGH" -eq 1 ]; then
  bad "Once dusuk fraksiyonla yeniden olustur: VLLM_FRAC=0.3 SGL_FRAC=0.3 bash jetson/setup.sh --recreate"
  exit 1
fi

MODELS="$(models_tsv)" || exit 1
DEFAULT=$(config_default_model)
DEFAULT_ENGINE=$(printf '%s\n' "$MODELS" | awk -F'\t' -v k="$DEFAULT" '$1==k{print $2}')
if [ "$DEFAULT_ENGINE" != "vllm" ]; then
  warn "default_model ($DEFAULT) vllm motorunda degil; vllm slotu icin ilk vllm modeli kullanilacak"
  DEFAULT=$(printf '%s\n' "$MODELS" | awk -F'\t' '$2=="vllm"{print $1; exit}')
fi
SGL_MODEL_KEY=$(printf '%s\n' "$MODELS" | awk -F'\t' '$2=="sglang"{print $1; exit}')
[ -n "$DEFAULT" ] && [ -n "$SGL_MODEL_KEY" ] || { bad "config.yaml'da her iki motor icin de model olmali"; exit 1; }

bash "$(dirname "$0")/start-model.sh" "$DEFAULT" || exit 1
echo "--- vLLM slotu tamam, SGLang slotu baslatiliyor ---"
if ! bash "$(dirname "$0")/start-model.sh" "$SGL_MODEL_KEY"; then
  bad "SGLang dual modda kalkamadi. Secenekler:"
  echo "  (a) fraction dusur:  SGL_FRAC=0.28 bash jetson/setup.sh --recreate-model $SGL_MODEL_KEY"
  echo "  (b) switch moda gec: bash jetson/stop-all.sh; config.yaml -> mode: switch;"
  echo "      sonra frac 0.5 ile recreate (README bolum 5)"
  exit 1
fi
ok "DUAL MOD AKTIF ($DEFAULT + $SGL_MODEL_KEY)"
free -h
echo "Simdi 2. terminalde izle: tegrastats  (RAM 7300/7620'ye yaklasirsa tehlike)"
echo "Iptal kriteri: MemAvailable <500MB veya dmesg'de oom-kill -> README bolum 5 KARAR NOKTASI"
