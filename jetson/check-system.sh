#!/usr/bin/env bash
# Read-only on kontrol: HICBIR degisiklik yapmaz, sudo gerektirmez.
set -u; . "$(dirname "$0")/lib.sh"
FAILS=0; WARNS=0
f(){ bad "$1"; FAILS=$((FAILS+1)); }
w(){ warn "$1"; WARNS=$((WARNS+1)); }

echo "=== 1) L4T / JetPack surumu ==="
L4T=$(head -n1 /etc/nv_tegra_release 2>/dev/null || echo "OKUNAMADI")
echo "    $L4T"
REL=$(echo "$L4T" | sed -n 's/^# R\([0-9]\+\).*/\1/p')
REV=$(echo "$L4T" | sed -n 's/.*REVISION: \([0-9.]\+\).*/\1/p')
PATHSEL="?"
if [ "$REL" = "36" ]; then
  case "$REV" in
    4.*) ok "L4T r36.$REV -> YOL A (tum imajlar uyumlu, ghcr fallback ACIK)"; PATHSEL=A ;;
    2.*|3.*) w "L4T r36.$REV -> YOL B (r36.4 imajlari sorun cikarabilir, karar matrisi asagida)"; PATHSEL=B ;;
    *) w "r36 ama revizyon belirsiz ('$REV') -> 'apt show nvidia-l4t-core' ile teyit et"; PATHSEL=B ;;
  esac
else
  f "JetPack 6 degil (R${REL:-?}). DUR - ekiple gorusulmeden ilerleme."; PATHSEL=STOP
fi

echo "=== 2) RAM (diger yazilimlar calisirken) ==="
AV=$(avail_mb); echo "    MemAvailable: ${AV} MB"
free -h | sed -n '1,3p' | sed 's/^/    /'
if   [ "$AV" -ge 4000 ]; then ok "RAM: dual mod denenebilir"
elif [ "$AV" -ge 3000 ]; then w  "RAM 3-4GB: dual zor, switch mod muhtemel"
else f "RAM <3GB: yalnizca switch mod + dusuk fraction"; fi

echo "=== 3) Disk (docker data-root) ==="
DROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
df -h "$DROOT" 2>/dev/null | sed 's/^/    /'
FREE=$(df -BG --output=avail "$DROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if   [ "${FREE:-0}" -ge 40 ]; then ok "Disk ${FREE}GB bos (>=40GB)"
elif [ "${FREE:-0}" -ge 25 ]; then w "Disk ${FREE}GB: yeterli ama kaynak-derleme eskalasyonuna dar"
else f "Disk <25GB: imajlar sigmayabilir"; fi

echo "=== 4) Docker + nvidia runtime + grup + python3-yaml ==="
if python3 -c "import yaml" 2>/dev/null; then ok "python3 yaml modulu var (scriptler config.yaml'i okur)"
else f "python3 yaml modulu YOK -> sudo apt install -y python3-yaml"; fi
if docker version --format 'docker {{.Server.Version}}' 2>/dev/null; then ok "docker calisiyor"
else f "docker yok veya erisim yok"; fi
if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then ok "nvidia runtime kayitli"
else f "nvidia runtime YOK (README sorun giderme #7)"; fi
if id -nG | tr ' ' '\n' | grep -qx docker; then ok "kullanici docker grubunda"
else f "docker grubu eksik (README sorun giderme #8)"; fi

echo "=== 5) Portlar ($VLLM_PORT/$SGL_PORT/7860) ==="
for p in "$VLLM_PORT" "$SGL_PORT" 7860; do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${p}$"; then
    f "port $p DOLU (baska bir servis mi? 'sudo ss -ltnp' ile bak; alternatif port plani README'de)"
  else ok "port $p bos"; fi
done

echo "=== 6) Bilgi (karar icin, PASS/FAIL degil) ==="
echo "    Swap/zram:"; swapon --show 2>/dev/null | sed 's/^/      /' || true
echo "    GUI hedefi: $(systemctl get-default 2>/dev/null || echo bilinmiyor)"
echo "    En cok RAM kullanan surecler (cihazin taban cizgisini not et):"
ps -eo pid,comm,rss --sort=-rss | head -6 | awk 'NR>1{printf "      %s %s %.0fMB\n",$1,$2,$3/1024}'
[ -d "$HOME/jetson-containers" ] && echo "    jetson-containers: VAR" || echo "    jetson-containers: YOK (setup.sh kuracak)"
docker image ls --format '    imaj mevcut: {{.Repository}}:{{.Tag}}' 2>/dev/null | grep -E 'vllm|sglang' || echo "    LLM imaji henuz yok (setup.sh cekecek)"

echo
echo "=========== KARAR MATRISI ==========="
echo " YOL A (r36.4.x): dustynv vllm 0.9.2 + dustynv sglang 0.4.7; vLLM cokerse ghcr fallback ACIK."
echo " YOL B (r36.2/3): dustynv imajlarini yine de dene; 'device kernel image is invalid' gorursen:"
echo "   ghcr fallback KAPALI. Secenekler:"
echo "   (1) JetPack OTA yukseltme r36.4.x [EKIBE SOR - cihazdaki diger yazilimlar icin regresyon riski, ~1 saat + reboot]"
echo "   (2) yukseltmeyi ayri ziyarete planla; bugun yalnizca calisan bileseni kur."
echo " STOP: JetPack 6 degilse hicbir imaj uyumlu degil."
echo "====================================="
echo "SONUC: $FAILS FAIL / $WARNS WARN  (YOL: $PATHSEL)"
[ "$FAILS" -eq 0 ] || exit 1
