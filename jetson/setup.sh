#!/usr/bin/env bash
# Tek seferlik kurulum (idempotent, tekrar calistirmasi guvenli):
#   1) jetson-containers klonla + install.sh
#   2) imajlari cek (ilk cekim 30-60 dk - bu beklerken 2. terminalde venv kur)
#   3) config.yaml'daki TUM modelleri on-indir (kaldigi yerden devam eder)
#   4) HER MODEL icin bir konteyner OLUSTUR (docker create - calistirmaz)
#
# Model listesi config.yaml'dan okunur (tek dogruluk kaynagi) - models_tsv, lib.sh.
# Ayni motorun modelleri ayni portu paylasir; ayni anda yalniz biri calistirilir.
#
# Bayraklar:
#   --recreate               tum konteynerleri sil + yeniden olustur
#   --recreate-model <key>   yalniz o modelin konteyneri (ornek: internvl25-1b)
#   --vllm-fallback          vllm motorunun konteynerlerini ghcr (NVIDIA) imajiyla
#                            olustur (SADECE r36.4.x; --recreate ile birlikte kullan)
#
# Ornekler:
#   bash jetson/setup.sh                                        # ilk kurulum (switch: frac 0.5)
#   VLLM_FRAC=0.3 SGL_FRAC=0.3 bash jetson/setup.sh --recreate  # dual deneyi oncesi
set -euo pipefail; . "$(dirname "$0")/lib.sh"

VLLM_IMG="${VLLM_IMG:-dustynv/vllm:0.9.2-r36.4-cu128-24.04}"
SGL_IMG="${SGL_IMG:-dustynv/sglang:0.4.7-r36.4-cu128-24.04}"
VLLM_FB_IMG="${VLLM_FB_IMG:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}"
# Varsayilan fraksiyonlar SWITCH moduna gore (tek sunucu ayakta).
# ONEMLI (sahada dogrulandi): Jetson'da bellek BIRLESIK - vLLM'in fraksiyonu
# sistemin o an kullandigi bellegi de sayar. 0.5 tavani (3.8GB), sistem ~3GB
# kullaniyorken KV cache'e yer birakmaz ("No available memory for the cache
# blocks"). 0.7 tavani (5.3GB) kucuk modellere ~2GB pay birakir ve makinenin
# geri kalanina 2.3GB dokunulmamis alan kalir.
# DUAL deneyi icin dusuk fraksiyonla yeniden olustur:
#   VLLM_FRAC=0.45 SGL_FRAC=0.3 bash jetson/setup.sh --recreate
# 0.85 (sahada kalibre): CUDA'nin "bos" gordugu deger free sutunudur (available
# DEGIL - buff/cache CUDA'ya gorunmez). GUI+tarayici acikken cuda-kullanimi ~5.3G
# olabilir; 0.7 tavani (5.2G) bunun altinda kalip mesru sekilde "No available
# memory" uretir. 0.85 tavani (6.3G) + --swap-space 0 ile kucuk modele bol pay kalir.
VLLM_FRAC="${VLLM_FRAC:-0.9}"
SGL_FRAC="${SGL_FRAC:-0.5}"
# KV cache ihtiyacini kisan ikinci dugme (dar bellekte 1024 deneyin):
VLLM_MAX_LEN="${VLLM_MAX_LEN:-2048}"
DATA="$HOME/jetson-containers/data"

RE_ALL=0; RE_MODEL=""; FB=0
while [ "$#" -gt 0 ]; do case "$1" in
  --recreate) RE_ALL=1 ;;
  --recreate-model) shift; RE_MODEL="${1:-}"; [ -n "$RE_MODEL" ] || { echo "--recreate-model bir model anahtari ister"; exit 2; } ;;
  --vllm-fallback) FB=1 ;;
  *) echo "bilinmeyen arg: $1"; exit 2 ;;
esac; shift; done

echo "== 1/4 jetson-containers =="
if [ ! -d "$HOME/jetson-containers" ]; then
  git clone --depth 1 https://github.com/dusty-nv/jetson-containers "$HOME/jetson-containers"
fi
if ! command -v jetson-containers >/dev/null 2>&1; then
  # NOT [EKIBE BILGI]: install.sh sudo ile python bagimliliklari + sistem paketleri kurar.
  # GERI ALMA: repo klasorunu silmek install.sh'in kurdugu sistem paketlerini GERI ALMAZ;
  # tam liste icin install.sh icine bak (apt ile tek tek kaldirilabilir).
  bash "$HOME/jetson-containers/install.sh"
fi
mkdir -p "$DATA/models/huggingface"

MODELS="$(models_tsv)"   # key engine hf_id container serve_extra extra_downloads (python3-yaml ister)

echo "== 2/4 imajlar =="
docker image inspect "$VLLM_IMG" >/dev/null 2>&1 || docker pull "$VLLM_IMG"
docker image inspect "$SGL_IMG"  >/dev/null 2>&1 || docker pull "$SGL_IMG"
if [ "$FB" -eq 1 ]; then
  docker image inspect "$VLLM_FB_IMG" >/dev/null 2>&1 || docker pull "$VLLM_FB_IMG"
fi

# dustynv 0.9.2 imajinda bazi saf-Python paketleri eksik (sahada dogrulandi:
# pandas -> vllm CLI, num2words -> SmolVLM islemcisi; timm/einops -> InternVL
# remote-code icin onleyici). Imajin ustune INCE BIR YEREL KATMAN insa edilir -
# bir kez, ~1-2 dk, sonrasi offline. Yeni bir eksik cikarsa listeye ekle ve
# 'docker rmi vllm-fixed:local' deyip setup.sh'i tekrar calistir.
VLLM_PIP_EXTRAS="${VLLM_PIP_EXTRAS:-pandas num2words timm einops}"
# v3 notlari (hepsi sahada dogrulandi):
# - pip, calisma anindaki yorumlayiciya sabitli (/opt/venv) - build'de ENTRYPOINT
#   calismadigi icin ciplak 'python3 -m pip' sistem Python'una kurabiliyordu.
# - imajin pip'i Jetson'a ozel bir depoya (pypi.jetson-ai-lab.dev) ayarli ve o
#   alan adi olu -> --index-url ile standart PyPI'a yonlendirilir (bu paketlerin
#   hepsi PyPI'da ARM uyumlu mevcuttur).
# - --network=host: build, makinenin agini/DNS'ini dogrudan kullansin.
VLLM_RUN_IMG="vllm-fixed:local-v3"
if ! docker image inspect "$VLLM_RUN_IMG" >/dev/null 2>&1; then
  echo "== 2b: eksik python paketleri icin ince katman ($VLLM_PIP_EXTRAS) =="
  docker build --network=host -t "$VLLM_RUN_IMG" - <<EOF
FROM $VLLM_IMG
RUN if [ -x /opt/venv/bin/python3 ]; then PY=/opt/venv/bin/python3; else PY=python3; fi \\
    && \$PY -m pip install --no-cache-dir --index-url https://pypi.org/simple $VLLM_PIP_EXTRAS \\
    && \$PY -c "import num2words, pandas; print('ince katman dogrulandi')"
EOF
fi

# dustynv/sglang 0.4.7 imajinda sgl_kernel (SGLang'in derlenmis CUDA cekirdegi)
# EKSIK (sahada dogrulandi: acilista ModuleNotFoundError). Jetson icin derlenmis
# wheel, jetson-ai-lab'in YASAYAN indeksinde (.io; imajin isaret ettigi .dev olu)
# mevcut -> vllm'deki gibi ince katmanla kurulur ve import build icinde dogrulanir.
SGL_RUN_IMG="sglang-fixed:local-v1"
if ! docker image inspect "$SGL_RUN_IMG" >/dev/null 2>&1; then
  echo "== 2c: sglang ince katmani (sgl-kernel, jetson-ai-lab.io/jp6/cu128) =="
  docker build --network=host -t "$SGL_RUN_IMG" - <<EOF
FROM $SGL_IMG
RUN if [ -x /opt/venv/bin/python3 ]; then PY=/opt/venv/bin/python3; else PY=python3; fi \\
    && \$PY -m pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 sgl-kernel \\
    && \$PY -c "import sgl_kernel; print('sgl_kernel dogrulandi')"
EOF
fi

echo "== 3/4 model on-indirme (config.yaml'daki tum modeller + bagimliliklari) =="
# extra_downloads: modelin isaret ettigi harici repolar da indirilir
# (orn. LLaVA'nin SigLIP tower'i - indirilmezse ilk acilis container icinde
# indirmeye kalkar ve wait_healthy zaman asimina takilabilir)
HF_IDS=$({ printf '%s\n' "$MODELS" | cut -f3; printf '%s\n' "$MODELS" | cut -f6 | tr ',' '\n'; } \
  | grep -v '^$' | sort -u | paste -sd, -)
docker run --rm --network host \
  -v "$DATA:/data" -e HF_HOME=/data/models/huggingface \
  --entrypoint python3 "$VLLM_IMG" -c "
from huggingface_hub import snapshot_download
for repo in '$HF_IDS'.split(','):
    print('indiriliyor:', repo)
    snapshot_download(repo)
print('modeller hazir')
"

echo "== 4/4 konteynerler (model basina bir adet; create - calistirmaz) =="
recreate_guard() { # $1=isim $2=recreate bayragi ; 0 donerse yeniden olusturulacak
  if docker container inspect "$1" >/dev/null 2>&1; then
    if [ "$2" -eq 1 ]; then
      echo "  $1 siliniyor (recreate)"; docker rm -f "$1" >/dev/null
    else
      echo "  $1 zaten var (atlandi; yeni flag icin --recreate / --recreate-model)"; return 1
    fi
  fi
  return 0
}

while IFS=$'\t' read -r key engine hf_id container serve_extra extra_dl; do
  [ -n "$key" ] || continue
  RE=0
  [ "$RE_ALL" -eq 1 ] && RE=1
  [ "$RE_MODEL" = "$key" ] && RE=1
  recreate_guard "$container" "$RE" || continue

  case "$engine" in
    vllm)
      PORT="$VLLM_PORT"
      if [ "$FB" -eq 0 ]; then
        # NOT: 'vllm serve' CLI'si KULLANILMIYOR - dustynv 0.9.2 imajinda CLI,
        # benchmark modulleri uzerinden pandas ister ve pandas imajda yok ->
        # acilista ModuleNotFoundError ile coker (sahada dogrulandi).
        # api_server modulu ayni OpenAI sunucusunu pandas'siz acar.
        # serve_extra kelime bolunmesi kasitli (ornek: --trust-remote-code)
        # shellcheck disable=SC2086
        docker create --name "$container" \
          --runtime nvidia --network host --ipc host --restart no --oom-score-adj 500 \
          -v "$DATA:/data" -e HF_HOME=/data/models/huggingface \
          "$VLLM_RUN_IMG" \
          python3 -m vllm.entrypoints.openai.api_server --model "$hf_id" \
            --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --enforce-eager \
            --gpu-memory-utilization "$VLLM_FRAC" --max-num-seqs 1 --swap-space 0 \
            --max-model-len "$VLLM_MAX_LEN" --max-num-batched-tokens 1024 \
            --limit-mm-per-prompt '{"image":1}' \
            $serve_extra
      else
        # ghcr fallback: HF cache yolu farkli. Imaj ENTRYPOINT'i zaten 'vllm serve'
        # iceriyorsa bastaki 'vllm serve' kaldirilmali - once kontrol et:
        #   docker inspect --format '{{.Config.Entrypoint}}' "$VLLM_FB_IMG"
        # shellcheck disable=SC2086
        docker create --name "$container" \
          --runtime nvidia --network host --ipc host --restart no --oom-score-adj 500 \
          -v "$DATA/models/huggingface:/root/.cache/huggingface" \
          "$VLLM_FB_IMG" \
          vllm serve "$hf_id" \
            --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --enforce-eager \
            --gpu-memory-utilization "$VLLM_FRAC" --max-num-seqs 1 --swap-space 0 \
            --max-model-len "$VLLM_MAX_LEN" --max-num-batched-tokens 1024 \
            --limit-mm-per-prompt '{"image":1}' \
            $serve_extra
      fi
      ok "$container olusturuldu ($hf_id, port=$PORT, frac=$VLLM_FRAC, fallback=$FB)"
      ;;
    sglang)
      PORT="$SGL_PORT"
      # --disable-cuda-graph: 0.4.x Triton PTX crash onlemi (issue #939)
      # chat-template config.yaml'daki serve_extra'dan gelir (LLaVA icin ZORUNLU)
      # shellcheck disable=SC2086
      docker create --name "$container" \
        --runtime nvidia --network host --ipc host --restart no --oom-score-adj 500 \
        -v "$DATA:/data" -e HF_HOME=/data/models/huggingface \
        "$SGL_RUN_IMG" \
        python3 -m sglang.launch_server \
          --model-path "$hf_id" \
          --host 0.0.0.0 --port "$PORT" --dtype half \
          --mem-fraction-static "$SGL_FRAC" --context-length 2048 \
          --disable-cuda-graph \
          $serve_extra
      ok "$container olusturuldu ($hf_id, port=$PORT, frac=$SGL_FRAC)"
      ;;
    *)
      bad "bilinmeyen engine: $engine (model: $key)"; exit 1 ;;
  esac
done <<< "$MODELS"

echo
echo "Kurulum tamam. Siradaki adim: bash jetson/start-model.sh $(config_default_model)"
