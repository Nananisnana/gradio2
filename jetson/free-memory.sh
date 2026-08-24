#!/usr/bin/env bash
# OPSIYONEL bellek kazanma merdiveni - PAYLASIMLI CIHAZ KURALLARI:
#   * Hicbir sey systemctl disable / fstab ile KALICILASTIRILMAZ.
#   * Her adim oturum kapsamlidir: reboot hepsini devre disi birakir
#     (yalniz swapfile DOSYASI diskte kalir - restore siler).
#   * Her adim tek tek onay ister; kazanc + geri alma komutu gosterilir.
#   * Uygulanan adimlar durum dosyasina yazilir; restore YALNIZ uygulanani geri alir.
#
# Kullanim: bash jetson/free-memory.sh status|apply|restore
set -u; . "$(dirname "$0")/lib.sh"

SWAPFILE="/swapfile-vqa"
STATE="$HOME/.free-memory-vqa.state"

mark()    { grep -qx "$1" "$STATE" 2>/dev/null || echo "$1" >> "$STATE"; }
unmark()  { [ -f "$STATE" ] && sed -i "/^$1$/d" "$STATE"; }
marked()  { grep -qx "$1" "$STATE" 2>/dev/null; }

swap_active() { swapon --show 2>/dev/null | grep -qF "$SWAPFILE"; }

show_status() {
  echo "== swap ==";        swapon --show 2>/dev/null || echo "  (swap yok)"
  echo "== GUI hedefi ==";  systemctl get-default 2>/dev/null
  echo "== bellek ==";      free -h
  echo "== swapfile ==";    ls -la "$SWAPFILE" 2>/dev/null || echo "  ($SWAPFILE yok)"
  echo "== uygulanmis adimlar =="; cat "$STATE" 2>/dev/null || echo "  (yok)"
}

confirm() { # $1=aciklama
  printf '\n%s\n' "$1"
  read -r -p "Uygula? [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ]
}

apply() {
  echo "### Adim 1/4: sayfa onbellegini bosalt"
  if confirm "Kazanc: cache bosalir, NvMap bitisik bellek tahsislerine yardim eder. GERI ALMA: gerekmez."; then
    sync && sudo sysctl vm.drop_caches=3
  fi

  echo "### Adim 2/4: NVMe swapfile 8G (oturum kapsamli, fstab'a YAZILMAZ)"
  if confirm "Kazanc: RAM kazandirmaz ama OOM emniyet agi olur (cihazdaki diger yazilimlari korur).
GERI ALMA: bash jetson/free-memory.sh restore  (reboot'ta swap devre disi kalir, dosya diskte durur)"; then
    if ! swap_active; then
      # dosya var ama swap imzasi yoksa (yarim kalmis onceki calisma) mkswap tekrar kosulur
      if [ ! -f "$SWAPFILE" ] || [ "$(sudo blkid -o value -s TYPE "$SWAPFILE" 2>/dev/null)" != "swap" ]; then
        sudo fallocate -l 8G "$SWAPFILE" && sudo chmod 600 "$SWAPFILE" && sudo mkswap "$SWAPFILE"
      fi
      sudo swapon "$SWAPFILE"
    fi
    if swap_active; then
      ok "swapfile aktif: $SWAPFILE"; mark swapfile
    else
      bad "swapfile AKTIF DEGIL (yukaridaki hataya bak) - Adim 3 zram'i KAPATMAYACAK"
    fi
  fi

  echo "### Adim 3/4: zram kapat [EKIBE BILGI VER]"
  if ! swap_active; then
    warn "NVMe swap aktif degil -> zram kapatma adimi guvenli degil, ATLANDI"
  elif confirm "Kazanc: ~200-400MB + swap sirasinda CPU. GERI ALMA: sudo systemctl restart nvzramconfig  (veya reboot)"; then
    for z in /dev/zram*; do [ -e "$z" ] && sudo swapoff "$z" 2>/dev/null; done
    echo "  zram kapatildi (varsa)"; mark zram
  fi

  echo "### Adim 4/4: GUI kapat (oturum kapsamli) [EKIBE SOR!]"
  echo "!!! Cihazda ekrana bagimli baska yazilim varsa BU ADIMI YAPMA !!!"
  if confirm "Kazanc: ~600-1000MB. GERI ALMA: sudo systemctl isolate graphical.target  (veya reboot)"; then
    systemctl get-default > "$HOME/.free-memory-vqa.target" 2>/dev/null || true
    sudo systemctl isolate multi-user.target
    mark gui
  fi

  echo; show_status
}

restore() {
  echo "### Geri alma (yalniz uygulanan adimlar, 4 -> 3 -> 2 sirasiyla)"
  if marked gui; then
    echo "- GUI geri getiriliyor..."
    sudo systemctl isolate graphical.target && unmark gui
  fi
  if marked zram; then
    echo "- zram geri getiriliyor..."
    if sudo systemctl restart nvzramconfig 2>/dev/null; then
      unmark zram
    else
      warn "nvzramconfig yeniden baslatilamadi - reboot da geri getirir"
    fi
  fi
  if marked swapfile; then
    echo "- swapfile kapatiliyor..."
    if swap_active; then
      if sudo swapoff "$SWAPFILE"; then
        sudo rm -f "$SWAPFILE"; unmark swapfile
      else
        bad "swapoff basarisiz (bellek dolu olabilir - once: bash jetson/stop-all.sh). Dosya SILINMEDI."
      fi
    else
      sudo rm -f "$SWAPFILE"; unmark swapfile
    fi
  fi
  echo; show_status
}

case "${1:-}" in
  status)  show_status ;;
  apply)   apply ;;
  restore) restore ;;
  *) echo "kullanim: $0 status|apply|restore"; exit 2 ;;
esac
