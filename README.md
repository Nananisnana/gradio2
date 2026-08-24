# Jetson Orin Nano — Video Soru-Cevap Uygulaması (vLLM + SGLang, OpenAI formatı)


Gradio uygulaması: video yükle → model seç → soru yaz ("meydanda insan var mı") →
istenen saniyeden kare alınır → seçili backend'e OpenAI Chat Completions formatında
gönderilir → kare + cevap birlikte gösterilir.

## 0. Mimari ve sözleşme

**Model Registry / Engine Registry ayrımı:** Kullanıcı UI'da MODEL seçer (motor değil).
Her model bir motora bağlıdır; motor başına aynı anda TEK model servis edilebilir
(vLLM/SGLang süreç başına tek model yükler) → her modelin kendi önceden-oluşturulmuş
container'ı vardır, aynı motorun modelleri motorun tek portunu paylaşır. Model seçilince
ModelManager o motorun eski container'ını durdurup yenisini başlatır.

```
Gradio app (host venv, 0.0.0.0:7860) — dropdown: 3 MODEL
   │
   ├── vLLM motoru   :8000 ── vllm-smolvlm2-256m   (SmolVLM2-256M, varsayılan)
   │    (tek slot)       └── vllm-internvl25-1b    (InternVL2.5-1B, --trust-remote-code)
   └── SGLang motoru :30000 ─ sglang-llava-05b     (LLaVA-OneVision-0.5B, chatml-llava)
```

| Konu | Değer |
|---|---|
| Model/container listesi | `config.yaml` → `models:` (TEK doğruluk kaynağı; script'ler `python3-yaml` ile okur) |
| Motor portları | `jetson/lib.sh` (`VLLM_PORT`/`SGL_PORT` env ile ezilebilir — tüm script'ler aynı değeri görür) |
| Hazır olma tanımı | `GET /v1/models` → HTTP 200; **aktiflik** tanımı: cevaptaki served-id, modelin `hf_id`'siyle eşleşir (yanlış modele soru gitmez) |
| Çalışma modu | `config.yaml` → `mode:` — **`switch` (GÜVENLİ VARSAYILAN: toplamda tek container)** / `dual` (DENEYSEL: motor başına bir slot — Bölüm 5) / `mock` (test) |
| Uygulamanın docker yetkisi | yalnız `start` / `stop` / `inspect` — sudo'suz (kullanıcı docker grubunda) |
| Model cache | `~/jetson-containers/data/models/huggingface` (tüm container'lara ortak mount) |

## 1. Ön kontrol (5 dk)

Önce repo Jetson'a klonlanır (private repo — `gh auth login` veya bir GitHub token gerekir):

```bash
git clone https://github.com/Nananisnana/gradio2 ~/gradio2
cd ~/gradio2
sed -i 's/\r$//' jetson/*.sh          # Windows CRLF emniyeti
bash jetson/check-system.sh
```

Çıktının sonundaki **KARAR MATRİSİ**ne bak:
- **YOL A** (L4T r36.4.x): her şey uyumlu, devam.
- **YOL B** (r36.2/3): dustynv image'ları yine de dene; "device kernel image is invalid"
  görürsen ghcr fallback KULLANILAMAZ → JetPack yükseltme **[EKİBE SOR]** ya da o gün
  yalnızca çalışan bileşeni kur.
- **STOP**: JetPack 6 değil → ekiple görüşmeden ilerleme.

FAIL satırlarını Bölüm 6'daki tabloyla çöz. Bilgi bloğundaki "en çok RAM kullanan
süreçler" listesini not et (cihazın taban çizgisi — sorun anında kıyas için).

## 2. Kurulum (60–90 dk, çoğu bekleme)

```bash
bash jetson/setup.sh
```

| Adım | Süre |
|---|---|
| jetson-containers clone + install | ~5 dk |
| Image pull (2 image, ~8-12GB/adet) | 30–60 dk |
| Model ön-indirme (3 model + bağımlılıkları, ~6GB) | 10–25 dk |
| Container create (model başına 1, 3 adet) | ~1 dk |

Ön-indirme, modellerin işaret ettiği **harici bağımlılık repolarını da** çeker
(`config.yaml` → `extra_downloads`; örn. LLaVA'nın SigLIP vision tower'ı ~1.5GB —
indirilmezse ilk açılış container içinde indirmeye kalkar ve `wait_healthy` zaman
aşımına takılabilir; offline'da `LocalEntryNotFoundError` üretirdi). Yeni bir model
eklerken config'i harici tower/tokenizer isteyip istemediğine bakarak doldurun.

**PARALEL İPUCU:** image'lar inerken 2. terminalde uygulamanın venv'ini kur:
`bash jetson/start-app.sh` ilk çalıştırmada venv'i kendisi kurar; sadece pip kısmı
bitince Ctrl+C ile çıkabilirsin (sunucular hazır olmadan app'i açık tutmana gerek yok).

`setup.sh` idempotent — yarıda kesilirse tekrar çalıştır, kaldığı yerden devam eder.

## 3. Duman testi — modelleri TEK TEK doğrula

`start-model.sh` anahtarları `config.yaml`'daki model adlarıdır (argümansız çağırınca listeler):

```bash
bash jetson/start-model.sh smolvlm2-256m   && bash jetson/smoke-test.sh vllm
bash jetson/start-model.sh internvl25-1b   && bash jetson/smoke-test.sh vllm
bash jetson/stop-all.sh

bash jetson/start-model.sh llava-05b       && bash jetson/smoke-test.sh sglang
bash jetson/stop-all.sh
```

`start-model.sh` aynı motorun çalışan diğer container'ını otomatik durdurur
(port paylaşımı); `smoke-test.sh` o an aktif modeli `/v1/models`'ten keşfeder.

- İlk başlatmada model yükleme 2–4 dk sürebilir; `wait_healthy` bekler.
- Varsayılan test görseli `tests/assets/smoke-scene.png`: 256×256 sentetik sahne —
  beyaz zemin, **bilinen konumda** kırmızı kare + çeldirici mavi daire. Konum bilindiği
  için grounding aşaması dönen kutunun **doğru bölgeyle örtüşmesini** de doğrular
  (gerçek bir fotoğrafta bu denetlenemezdi — ground-truth yok).
- **Gerçek görüntüyle test (önerilir, sahada):** kendi videonuzdan bir kare verin —
  proje davranışı gerçek veriyle sınanır:
  ```bash
  SMOKE_IMAGE=/yol/meydan-karesi.jpg \
  SMOKE_PROMPT="Are there any people in this image?" \
  SMOKE_EXPECT="person,insan,people,yes,evet" \
  bash jetson/smoke-test.sh vllm
  ```
  (özel görselde bölge doğrulaması atlanır; kutu sayısı raporlanır)
- Smoke-test **aşamalı bir PASS/FAIL raporu** basar:
  ```
  Server ready     PASS   (model-id)
  OpenAI request   PASS   (HTTP 200)
  Response shape   PASS
  Text response    PASS   (cevabın ilk 60 karakteri)
  Vision input     PASS/WARN/FAIL
  Grounding        PASS/WARN/FAIL
  ```
- **Vision input:** cevap görsel içeriğini (beklenen anahtar kelimeleri) anmalı.
  SGLang'de anmıyorsa **FAIL + exit 1** (= eksik chat-template, Bölüm 6 satır 4);
  vLLM'de yalnız WARN (küçük model kalitesi olabilir, exit kodunu etkilemez).
- **Grounding:** uygulamanın GERÇEK sözleşmesi ve parser'ıyla (`app/grounding.py`)
  kutu istenir; varsayılan sahnede kutunun kırmızı karenin gerçek bölgesiyle
  örtüşmesi de denetlenir (örtüşmezse WARN). Rapor bilgilendiricidir, exit kodunu
  etkilemez — küçük modellerin grounding'de FAIL vermesi beklenen sonuçtur
  (Bölüm 4'teki model tablosu). Sıkılaştırmak için:
  `GROUNDING_REQUIRED=1 bash jetson/smoke-test.sh vllm`.
- Exit kodu: zorunlu aşamalardan biri FAIL ise 1 — script CI/otomasyonda da kullanılabilir.

## 4. Uygulama (5 dk)

```bash
bash jetson/start-app.sh
```

- Jetson monitöründen: `http://localhost:7860`
- Aynı ağdaki başka bir PC'den: `http://<jetson-ip>:7860` (IP'yi `hostname -I` gösterir)

### Grounding / kutulama katmanı

Kutulama kararı prompt + model cevabına bırakılır: VLM'e sistem prompt'uyla bir JSON
sözleşmesi verilir (`{"answer": "...", "boxes": [{"label": "car", "bbox": [x1,y1,x2,y2]}]}`,
koordinatlar 0-1000 normalize uzayda). Model kutu döndürürse frame üzerine jenerik
çizici ile işlenir (person/car/dog/phone... herhangi bir etiket); "hava nasıl?" gibi
lokalizasyon gerektirmeyen sorularda `boxes: []` gelir ve düz frame gösterilir.
Parser savunmalıdır: bozuk/eksik JSON'da tüm metin cevap sayılır, kutu çizilmez —
uygulama asla kırılmaz.

**Dürüst beklenti (model yetenekleri):**

| Model | Grounding beklentisi |
|---|---|
| InternVL2.5-1B | En iyi mevcut aday — JSON sözleşmesine uyduğu sahada doğrulandı; kutu üretimi değişken |
| SmolVLM2-256M | Muhtemelen `boxes: []` veya güvenilmez koordinat |
| LLaVA-OneVision-0.5B | Zayıf — çoğunlukla düz cevap beklenir |

Mock modda `göster / var mı / nerede / kutula` içeren sorular sahte 2 kutu üretir
(kutulama yolunu backend'siz test etmek için).

Kullanım: video yükle → oynat → listeden MODEL seç → soruyu yazıp **Enter/Sor** —
**soru gönderildiği anda oynatıcının o anki saniyesi otomatik yakalanır** ve o kare
modele gider (cevabın başındaki `[X.X. sn]` hangi karenin kullanıldığını gösterir).
Belirli bir saniyeyi sormak için videoyu o ana getirip (oynatarak veya durdurarak)
Enter'a basman yeterli. Rozetler 5 sn'de bir
yenilenir. Model seçimi o modeli otomatik başlatır: switch modda (varsayılan) diğer her
şey durdurulur, dual deneyinde yalnız aynı motorun slotu değişir (diğer motor ayakta
kalır); geçiş 1-2 dk sürer ve ilerleme arayüzde canlı gösterilir. Soru yalnız rozeti
AKTİF olan modele gider.

## 5. Dual mod — OPSİYONEL DENEY (varsayılan DEĞİL)

**Güvenli varsayılan `mode: switch`tir:** tek aktif model, container'lar frac 0.5,
OOM riski en düşük — paylaşımlı cihaz için doğru duruş. Normal kullanımda bu bölüme
hiç girmeden Bölüm 4 ile çalışmaya devam edilir; 3 modelin tamamı switch modda
serbestçe gezilir. Dual (iki motor aynı anda) yalnızca bilinçli bir deneydir.

**Denemek istersen — önkoşul: düşük fraksiyonla recreate + config değişikliği:**
```bash
bash jetson/stop-all.sh
VLLM_FRAC=0.3 SGL_FRAC=0.3 bash jetson/setup.sh --recreate
# config.yaml içinde:  mode: switch  ->  mode: dual
bash jetson/start-dual.sh   # sırayla: önce vLLM slotu (default_model), sonra SGLang slotu
# 2. terminalde: tegrastats
```
(`start-dual.sh` fraksiyonları container'lardan okur ve >0.35 görürse başlatmayı reddeder.)

Her iki motora uygulamadan (veya smoke-test ile) 2-3'er istek at; `free -h` ve
swap büyümesini not et. Dual moddayken vLLM slotunda model değiştirmek (örn. SmolVLM →
InternVL) SGLang slotuna dokunmaz ama 1-2 dk'lık container değişimi tetikler. 2B'lik

**OOM belirtileri:** container exit 137 · `sudo dmesg | tail` içinde oom-killer ·
tegrastats RAM 7300+/7620 · masaüstü donması · **cihazdaki diğer yazılımların ölmesi/yeniden başlaması (ACİL: `bash jetson/stop-all.sh` + ekibe haber)**

**İptal kriteri — biri bile gerçekleşirse deneyi bitir, switch'e dön (kalıcı):**
- İki sunucu sağlıklı + birer istek sonrası `MemAvailable < 500 MB`, VEYA
- tek istekte swap > 1.5 GB büyüdü, VEYA
- `sudo dmesg | grep -i oom` herhangi bir kill gösteriyor.

**Switch'e dönüş (deney bitince — başarılı olsa da olmasa da önerilen son durum):**
```bash
bash jetson/stop-all.sh
# config.yaml içinde:  mode: dual  ->  mode: switch
VLLM_FRAC=0.5 SGL_FRAC=0.5 bash jetson/setup.sh --recreate
bash jetson/start-model.sh smolvlm2-256m
bash jetson/start-app.sh
```

Deneyde dual'da kalmak istiyorsan ama dar geliyorsa ara adımlar: `VLLM_FRAC=0.25`
veya `0.20` ile vLLM container'larını recreate et (`--recreate-model smolvlm2-256m`
vb.; SGLang ≥0.28 kalmalı — 1.8GB ağırlık statik havuza sığmak zorunda); yetmezse Bölüm 7.

## 6. Sorun giderme

| # | Belirti | Neden | Çözüm |
|---|---|---|---|
| 1a | vLLM açılışta `No available memory for the cache blocks` (logda `Possibly too large swap space` + "Model loading took ~4 GiB") | **Asıl neden (sahada doğrulandı):** vLLM'in varsayılan `--swap-space 4`'ü, birleşik bellekte havuzun 4 GiB'ını peşinen rezerve eder; ayrıca fraksiyon tavanı sistemin kullandığını da sayar | Bizim create'te `--swap-space 0` var — bu hatayı görüyorsan eski container: `git pull` + `--recreate-model <key>`. Hâlâ darsa `VLLM_FRAC` yükselt (0.7→0.85) ve `free -h` ile taban kullanımına bak |
| 1b | vLLM KV-cache init sırasında sebepsiz çöküyor/donuyor (bellek mesajı YOK) | Nano'da bilinen init sorunu (0.8.6 #1568; 0.9.2 doğrulanmadı) | reboot + drop_caches dene; sürerse `bash jetson/setup.sh --recreate --vllm-fallback` (SADECE r36.4.x) |
| 2 | SGLang açılışta PTX hatası (CUDA graph capture) | Triton PTX sorunu (0.4.4 #939) | `docker inspect sglang-llava-05b` ile `--disable-cuda-graph` var mı bak (bizim create'te var) |
| 3 | Açılışta hemen "device kernel image is invalid" | Host r36.2/3 ↔ r36.4 image uyumsuzluğu | YOL B durumu: ghcr fallback yasak; JetPack yükseltme [EKİBE SOR] |
| 4 | SGLang akıcı cevap veriyor ama **görseli yok sayıyor** | `--chat-template chatml-llava` eksik (config.yaml `serve_extra`) | Container'ı yeniden oluştur (`--recreate-model llava-05b`); smoke-test ile doğrula |
| 5 | `NvMapMemAllocInternalTagged` / contiguous alloc hatası | Bellek parçalanması | `sync && sudo sysctl vm.drop_caches=3`; sürerse reboot [EKİBE SOR — cihazdaki diğer yazılımlar kesintiye uğrar] |
| 6 | `unknown runtime: nvidia` | nvidia-container-toolkit docker'a bağlı değil | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
| 7 | `permission denied ... docker.sock` | Kullanıcı docker grubunda değil | `sudo usermod -aG docker $USER` + oturumu kapat/aç [GERİ ALMA: `sudo gpasswd -d $USER docker`] |
| 8 | Port dolu (8000/30000/7860) | Cihazdaki başka bir servis veya artık süreç | `sudo ss -ltnp` ile bak; başka bir servise aitse alternatif portlara geç: `export VLLM_PORT=8010 SGL_PORT=30010` deyip `bash jetson/setup.sh --recreate` — tüm script'ler (start/smoke/status/check) portları `lib.sh` üzerinden aynı env'den okur, o yüzden **export'u her yeni terminalde tekrarla** (kalıcısı: `~/.bashrc`'ye ekle). `config.yaml`'daki engine base_url'lerini de aynı portlarla değiştir; 7860 doluysa `bash jetson/start-app.sh --port 7861` |
| 9 | vLLM arg hatası: `--limit-mm-per-prompt` | Sürümler arası söz dizimi farkı | JSON yerine `image=1` biçimini dene (`setup.sh` içinde değiştir + `--recreate`) |
| 10 | vLLM açılışta `ModuleNotFoundError` / `ImportError: Package X is required` (görülenler: `pandas`, `num2words`) | dustynv imajında bazı saf-Python paketleri eksik (sahada doğrulandı) | `setup.sh` bunları içeren yerel bir `vllm-fixed:local` katmanı inşa eder ve container'lar onu kullanır. Bu hatayı görüyorsan: `git pull` + `bash jetson/setup.sh --recreate`. YENİ bir eksik paket çıkarsa: `setup.sh`'taki `VLLM_PIP_EXTRAS` listesine ekle → `docker rmi vllm-fixed:local` → `bash jetson/setup.sh --recreate` |
| 11 | Health 200 olmuyor, loglar model indiriyor | HF cache mount yanlış / ön-indirme atlanmış | `ls ~/jetson-containers/data/models/huggingface/hub`; `setup.sh` 3. adımı tekrar çalıştır |
| 12 | `bash: $'\r': command not found` | Windows CRLF satır sonları | `sed -i 's/\r$//' jetson/*.sh` |
| 13 | İlk cevap çok yavaş (30–90 sn) | `--enforce-eager` + ilk ısınma | Normal; app zaman aşımı 120 sn; sonraki istekler hızlanır |


## 7. Bellek kazanma — OPSİYONEL

```bash
bash jetson/free-memory.sh status    # mevcut durum
bash jetson/free-memory.sh apply     # her adım tek tek onay ister
bash jetson/free-memory.sh restore   # hepsini geri alır
```

Merdiven (güvenliden riskliye): 1) drop_caches (zararsız) → 2) NVMe swapfile 8G
(oturum-kapsamlı, fstab'a yazılmaz — OOM emniyet ağı; zram adımı swap doğrulanmadan
kilitlidir) → 3) zram kapatma → 4) GUI kapatma ~600-1000MB
Uygulanan adımlar `~/.free-memory-vqa.state` dosyasına yazılır; `restore` yalnız
uygulananları geri alır. Reboot tüm adımları devre dışı bırakır ama `/swapfile-vqa`
DOSYASI diskte kalır — silmek için `restore` (veya `sudo rm /swapfile-vqa`).

## 8. Eskalasyon merdivenleri

**vLLM:** frac 0.3→0.25→0.2 (`--recreate-model <key>`) → (yalnız YOL A)
`setup.sh --recreate --vllm-fallback` (ghcr NVIDIA image'ı; adlar/portlar aynı →
uygulama etkilenmez; ilk kullanımda entrypoint kontrolü:
`docker inspect --format '{{.Config.Entrypoint}}' ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` —
entrypoint zaten `vllm serve` içeriyorsa `setup.sh`'taki baştaki `vllm serve` kaldırılır) →
model yedeği: sorunlu modeli `config.yaml`'dan sil veya yerine yenisini ekle (örn.
`hf_id: HuggingFaceTB/SmolVLM2-500M-Video-Instruct`), `setup.sh` çalıştır (yeni container'ı
oluşturur ve modeli indirir) — dropdown ve rozetler config'ten otomatik güncellenir.

**SGLang (mutlaka çalışacak):** frac 0.3→0.28 + `--context-length 1024` +
disable-cuda-graph teyidi → taze reboot [EKİBE SOR] + drop_caches, boot'tan hemen sonra
dene (bitişik bellek en geniş) → solo/switch mod (frac 0.5) → **gece kaynak derleme:**
```bash
cd ~/jetson-containers && nohup ./build.sh sglang > ~/sglang-build.log 2>&1 &
```
(saatler sürer; ~30-40GB disk; cihaz yükü düşükken [EKİBE SOR]; ertesi gün
`docker images | grep sglang` ile yeni tag'i bul, `SGL_IMG=<yeni-tag> bash
jetson/setup.sh --recreate-model llava-05b`; chat-template ihtiyacını smoke-test'in
"görseli betimleme" kriteriyle yeniden doğrula) → son çare: pypi.jetson-ai-lab.io
wheel'leri (30 dk ile sınırla, erişilebilirliği istikrarsız).

## 9. Ayrılmadan önce

```bash
bash jetson/status.sh                 # son durumu not et
bash jetson/free-memory.sh restore    # (uyguladıysan) sistem değişikliklerini geri al
ls -la /swapfile-vqa 2>/dev/null      # dosya hâlâ varsa: restore tekrar (veya sudo rm)
```
- İstenen son durumu bırak: genelde `stop-all.sh` (sunucular kapalı, cihaz rahat).
---

## Geliştirme (Windows, Jetson'suz)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Backend'siz UI testi (sahte cevaplar):
$env:VQA_MODE="mock"; .venv\Scripts\python -m app.main
# Gerçek mod davranışı (switch varsayılanı; sunucular kapalı -> rozetler KAPALI, Türkçe hata):
# DİKKAT: VQA_MODE oturumda kalır — önce temizle:
Remove-Item Env:VQA_MODE; .venv\Scripts\python -m app.main
```

Switch orkestrasyonunu Docker Desktop ile sahte container'lar üzerinden test edebilirsin:
```powershell
docker create --name vllm-smolvlm2-256m alpine sleep 1d
docker create --name vllm-internvl25-1b alpine sleep 1d
docker create --name sglang-llava-05b alpine sleep 1d
# config.yaml -> mode: switch ; dropdown'da model değişiminde stop/start sırası ve
# zaman aşımı yolu gözlemlenir (served-id hiç eşleşmez -> timeout mesajı).
```

Not: `pip` sürüm çözümlemesi Jetson'da bir pin'e takılırsa aynı minor hat içinde kal ve
değişikliği bu dosyaya not düş. cv2 wheel'i Jetson'da sorun çıkarırsa venv'i
`python3 -m venv --system-site-packages .venv` ile yeniden kur ve opencv+numpy
pin'lerini kaldır.
