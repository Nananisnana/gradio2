# Teknik Rehber — Kod ve Teknolojilere Hakimiyet

> Bu belge, projedeki her teknolojiyi ve her kod dosyasını en basit düzeyden başlayarak
> anlatır. Sırayla okunmak üzere yazıldı: önce kavramlar, sonra teknolojiler, sonra
> bizim kodun satır satır mantığı, en sonda "neden böyle yaptık" kararları.

---

## 1. Büyük Resim: Enter'dan Cevaba Yolculuk

Kullanıcı videoyu yükler, sorusunu yazar, Enter'a basar. O andan itibaren:

```
Tarayıcı (JavaScript)          → oynatıcının o anki saniyesini yakalar
        ↓
Gradio (Python web arayüzü)    → video + saniye + soru + model seçimini toplar
        ↓
frames.py (OpenCV)             → videodan o saniyenin karesini çıkarır, küçültür
        ↓
vqa_client.py (openai kütüp.)  → kareyi base64 yapar, OpenAI formatında istek kurar
        ↓
HTTP isteği                    → localhost:8000 (vLLM) veya :30000 (SGLang)
        ↓
Çıkarım motoru (Docker içinde) → modeli GPU'da çalıştırır, cevabı üretir
        ↓
grounding.py                   → cevabın JSON'unu ayrıştırır (answer + boxes)
        ↓
annotate.py (OpenCV)           → kutu varsa karenin üstüne çizer
        ↓
Gradio                         → işaretli kare + cevap ekranda
```

Bu belgedeki her bölüm, bu zincirin bir halkasını derinlemesine anlatır.

---

## 2. Temel Kavramlar Sözlüğü

Bunları bilirsen gerisi kolay akar:

- **HTTP isteği:** Bir bilgisayarın diğerine "şu adresten şunu istiyorum" demesi.
  Tarayıcının site açması da, bizim modele soru göndermemiz de aynı mekanizma.
  `GET` = "bana ver" (okuma), `POST` = "sana gönderiyorum, işle" (bizim sorular).
- **Port:** Aynı makinedeki farklı servislerin kapı numaraları. `localhost:8000`
  = "bu makinede, 8000 numaralı kapıdaki servis". vLLM 8000'de, SGLang 30000'de,
  Gradio 7860'ta oturur.
- **localhost / 127.0.0.1:** "Bu makinenin kendisi." Uygulama ile motorlar aynı
  Jetson'da olduğu için birbirlerine bu adresle ulaşır.
- **JSON:** Programların ortak veri dili. `{"ad": "değer"}` şeklinde iç içe
  sözlükler ve listeler. Hem isteklerimiz hem cevaplar JSON.
- **base64:** Ham baytları (örn. bir JPEG dosyasını) sadece harf-rakamdan oluşan
  güvenli bir metne çevirme yöntemi. Resmi JSON'un içine "metin gibi" gömmemizi
  sağlar. ~%33 şişirir ama her sistemden sorunsuz geçer.
- **Ortam değişkeni (env var):** Programlara dışarıdan verilen ayar.
  `VQA_MODE=mock python -m app.main` → program, `VQA_MODE` değişkenini okuyup
  ona göre davranır. Terminal oturumunda kalıcıdır (unut: `unset VQA_MODE`).
- **venv (sanal ortam):** Projeye özel, izole Python paket klasörü. Sistemin
  Python'unu kirletmeden `pip install` yapmayı sağlar. Her işletim sistemi /
  makine kendi venv'ini ister (Windows'unki WSL'de çalışmaz).
- **ARM vs x86:** İşlemci mimarileri — farklı "makine dilleri". Jetson ARM,
  masaüstü PC'ler x86 kullanır. Bu yüzden Jetson için derlenmiş Docker
  image'ları PC'de ÇALIŞMAZ (Windows'ta modellerin hep KAPALI olmasının nedeni).
- **GPU / CUDA:** Grafik işlemci ve NVIDIA'nın onu programlama sistemi. Model
  hesapları binlerce küçük çarpma-toplamadır; GPU bunları paralel yapar. Jetson'da
  özel bir durum var: **CPU ve GPU aynı 8GB belleği paylaşır** (unified memory) —
  masaüstünde GPU'nun kendi ayrı belleği olur.

---

## 3. OpenAI Formatı: Her Şeyin Ortak Dili

### 3.1 Neden bir "format"?

Modele soru sormak = bir sunucuya JSON paketi POST'lamak. OpenAI bu paketin şeklini
ChatGPT için tanımladı; o kadar yaygınlaştı ki fiili standart oldu. "OpenAI-uyumlu
sunucu" şu demek: *aynı adresleri açarım, aynı paket şeklini kabul ederim.* Kazanç:
istemci kodun karşıda kimin olduğunu bilmez — sadece adres değişir. Bizim tek
`vqa_client.py` dosyasının 2 motor × 4 modelle konuşabilmesinin sırrı bu.

### 3.2 İki kapı

- `POST /v1/chat/completions` → soru sor, cevap al (ana kapı)
- `GET /v1/models` → "şu an hangi model yüklüsün?" (bizim sağlık kontrolü + rozetler)

### 3.3 İsteğin anatomisi

```json
{
  "model": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
  "messages": [
    {"role": "system", "content": "Sen bir görsel soru-cevap asistanısın... (grounding sözleşmesi)"},
    {"role": "user", "content": [
        {"type": "text", "text": "meydanda insan var mı"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
    ]}
  ],
  "max_tokens": 256,
  "temperature": 0.2
}
```

- **`messages`** = konuşma kaydı. `role` üç değer alır:
  - `system`: modele sahne talimatı (kullanıcı görmez). Bizim JSON-cevap-ver
    sözleşmemiz burada yaşıyor.
  - `user`: kullanıcının mesajı.
  - `assistant`: modelin önceki cevapları (sohbet geçmişi için; biz her soruyu
    bağımsız sorduğumuzdan kullanmıyoruz).
- **`content` liste olabilir** → çok-modluluğun sırrı. Metin parçası ve görüntü
  parçası yan yana; model ikisini birlikte görür.
- **`data:image/jpeg;base64,...`** = "data URI": görüntünün kendisi, adres yerine
  paketin içinde taşınır.
- **`temperature`**: 0'a yakın = tutarlı/tekrarlanabilir cevap; yüksek = çeşitlilik.
  Tespit işinde hayal gücü istemeyiz → 0.2.
- **`max_tokens`**: cevabın uzunluk tavanı.

### 3.4 Cevabın anatomisi

```json
{"choices": [{"message": {"role": "assistant", "content": "...model ne dediyse..."}}]}
```

Okuduğumuz tek yer: `choices[0].message.content`. Bizim sistem prompt'umuz sayesinde
bu içerik ayrıca `{"answer": ..., "boxes": [...]}` biçiminde JSON olur (Bölüm 7).

### 3.5 Kodda karşılığı

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(model=..., messages=[...])
```

`api_key="EMPTY"`: resmî kütüphane anahtar ister; yerel sunucularımız kontrol
etmez, boş değer koyup geçeriz.

---

## 4. Model Nedir? Ağırlıklar, Parametreler, Kuantizasyon

### 4.1 Ağırlık = öğrenilmiş sayılar

Bir yapay zekâ modeli, özünde **milyarlarca ondalıklı sayıdan** oluşan dev bir
hesap makinesidir. Bu sayılara "ağırlık (weight)" denir; eğitim sürecinde
öğrenilmişlerdir. "SmolVLM2-**256M**" = 256 milyon ağırlık; "Qwen2-VL-**2B**" =
2 milyar. Sayı büyüdükçe model genelde akıllanır ama bellek ve hız bedeli öder.

Kabaca bellek hesabı: her ağırlık 16-bit (2 bayt) tutulursa **2B model ≈ 4GB**
sadece ağırlık için. 8GB paylaşımlı Jetson'da neden minicik modeller seçtiğimiz
bu matematikten görülüyor.

### 4.2 Hugging Face ve safetensors

**Hugging Face**, modellerin GitHub'ı: ağırlık dosyaları (`.safetensors` — güvenli,
hızlı yüklenen bir format), yapılandırma (`config.json`) ve tokenizer dosyaları
orada barınır. `snapshot_download("Qwen/Qwen2-VL-2B-Instruct-AWQ")` = o deponun
tüm dosyalarını indirip önbelleğe (bizde `~/jetson-containers/data/models/huggingface`)
koymak. Önemli tuzak: bazı modellerin `config.json`'ı **başka depoları işaret eder**
— LLaVA'nın görüntü işleyicisi (SigLIP) ayrı bir depodur ve ayrıca indirilmelidir.
`config.yaml`'daki `extra_downloads` alanı bu yüzden var.

### 4.3 VLM nasıl "görür"?

Bir görsel-dil modeli (VLM) iki parçanın evliliğidir:

1. **Vision encoder** (örn. SigLIP): resmi alır, küçük parçalara (patch) böler,
   her parçayı bir sayı vektörüne çevirir — resmi "kelimeleştirir".
2. **Dil modeli** (örn. Qwen2-0.5B): bu görsel "kelimeleri" ve senin metnini
   birlikte okuyup cevabı kelime kelime üretir.

"LLaVA-OneVision-**qwen2-0.5b**" adındaki qwen2, içindeki dil modelinin kimliğidir.

### 4.4 Kuantizasyon (AWQ)

**Kuantizasyon** = ağırlıkları daha az bitle saklamak. AWQ, 16-bit sayıları ~4-bit'e
indirir → 2B model ~4.4GB yerine **~1.5GB**'a düşer, hafif kalite kaybıyla.
"Qwen2-VL-2B-**AWQ**" seçmemizin nedeni bu: 2B zekâsını küçük bellek ayak iziyle
almak. Riski: kuantize hesap özel GPU çekirdekleri ister; bunların Jetson'da
çalışması henüz doğrulanmadı (çalışmazsa modeli listeden sileriz, sistem etkilenmez).

---

## 5. Çıkarım Motorları: vLLM ve SGLang

### 5.1 Motor neden gerekli?

Ağırlıkları diskten okuyup soru cevaplatmayı basit bir Python script'i de yapabilir —
ama saniyeler değil dakikalar sürer ve belleği çarçur eder. **Çıkarım motoru
(inference engine)**, bu işi üretim kalitesinde yapan sunucudur:

- Ağırlıkları GPU'ya bir kez yükler, istekler arasında yüklü tutar ("sıcak" model).
- **KV cache** yönetir: model cevap üretirken her adımda o ana kadarki her şeyi
  yeniden hesaplamamak için ara sonuçları saklar. Bu önbellek bellek yer — motorlar
  bunu akıllıca yönetir (vLLM'in ünü buradan gelir: "PagedAttention" tekniği KV
  cache'i işletim sistemlerinin RAM sayfalaması gibi parça parça yönetir, israfı bitirir).
- Aynı anda çok isteği gruplar (batching) — bizde gereksiz, `--max-num-seqs 1` dedik.
- Dışarıya OpenAI-uyumlu HTTP kapısı açar.

### 5.2 vLLM ve SGLang farkı

İkisi de aynı ligde yarışan motorlar. vLLM daha yaygın/olgun; SGLang daha yeni,
bazı iş yüklerinde daha hızlı. Projede ikisinin de olması senin şartındı — mimarinin
motordan bağımsızlığını kanıtlıyor: **aynı istemci kodu ikisiyle de konuşuyor.**

### 5.3 Kritik gerçek: süreç başına TEK model

Bir vLLM/SGLang süreci başlarken `vllm serve <model>` argümanıyla TEK model yükler
ve ölene kadar onu servis eder. "Çalışırken model değiştir" diye bir komut yoktur.
Bizim **model başına bir container** tasarımımızın kök nedeni bu: model değiştirmek
= eski sunucuyu durdurup yenisini başlatmak (1-2 dk, çünkü ağırlıklar GPU'ya
yeniden yüklenir).

### 5.4 Bizim kullandığımız bayraklar (setup.sh'taki serve komutları)

- `--gpu-memory-utilization 0.5` (vLLM) / `--mem-fraction-static 0.5` (SGLang):
  "Toplam belleğin şu kadarını al." Jetson'da bellek paylaşımlı olduğu için bu
  fren hayati — motor sınırsız bırakılsa cihazdaki diğer yazılımların belleğini yer.
- `--max-model-len 2048`: modelin görebileceği azami bağlam (soru+resim+cevap
  toplam "token" bütçesi). Kısa tuttuk → KV cache küçük kalır.
- `--enforce-eager` (vLLM) / `--disable-cuda-graph` (SGLang): "CUDA graph" denen
  bir hızlandırmayı kapatır. Bu hızlandırma başlangıçta ekstra bellek ister ve
  Jetson'da bilinen çökmelere yol açıyor — güvenlik için kapalı.
- `--chat-template chatml-llava` (yalnız SGLang+LLaVA): **en sinsi bayrak.**
  Chat template, "messages listesini modelin beklediği ham metne çevirme kalıbı"dır.
  Yanlış/eksik kalıpla görüntü, modele hiç ulaşmaz ama model yine de akıcı (uydurma)
  cevap verir — sessiz bozulma. Smoke test'in "cevap görseli anmalı" kriteri tam
  bu hatayı yakalamak için var.
- `--trust-remote-code` (yalnız InternVL): bazı modeller depolarında kendi Python
  kodunu taşır; bu bayrak "o koda güven, çalıştır" der.

---

## 6. Docker ve jetson-containers

### 6.1 Image vs container

- **Image** = dondurulmuş kalıp: işletim sistemi parçaları + CUDA + Python +
  vLLM, hepsi bir pakette. Salt-okunur.
- **Container** = o kalıptan açılmış çalışan (veya durdurulmuş) kopya.

Analoji: image = kurulum DVD'si, container = kurulu ve açılabilen bilgisayar.

### 6.2 Neden Docker?

vLLM'i Jetson'a "çıplak" kurmak, CUDA sürümü/derleme cehennemidir. Docker,
tüm bağımlılıkları paketin içinde getirir: `docker pull` + `docker start` = çalışır.
**jetson-containers** (Dusty Franklin'in projesi) bu image'ların Jetson'ın ARM +
Tegra-CUDA'sı için **önceden derlenmiş** halleridir — bizim `dustynv/vllm` ve
`dustynv/sglang` çekmemizin nedeni. (Aynı image'lar PC'de açılmaz: farklı işlemci dili.)

### 6.3 Bizim docker sözleşmemiz (setup.sh'ın yaptığı)

Her model için **bir kez** `docker create` yapılır — container, tüm serve komutu
gömülü, ama ÇALIŞTIRILMADAN hazırlanır. Sonrasında uygulama sadece:

```
docker start <ad>      # başlat (komut zaten gömülü)
docker stop -t 20 <ad> # 20 sn nazik süre tanıyıp durdur
docker inspect <ad>    # durumunu sor
```

Neden bu tasarım? **Güvenlik ve basitlik:** uygulama asla `docker run` ile
serbest argüman kuramaz (yanlışlıkla ya da hata sonucu tehlikeli bir şey
çalıştıramaz); tüm ayarlar tek dosyada (setup.sh) yaşar.

`docker create` bayraklarımızın anlamı:

- `--network host`: container kendi ağ kutusunda değil, doğrudan Jetson'ın
  ağında — 8000 portu doğrudan makinenin 8000'i.
- `--runtime nvidia`: container'a GPU erişimi ver.
- `-v ~/jetson-containers/data:/data`: Jetson'daki klasörü container'ın içine
  aynala (volume mount). Model önbelleği DIŞARIDA durur → container silinse de
  modeller yeniden inmez, tüm container'lar aynı önbelleği paylaşır.
- `--restart no`: çökerse kendiliğinden yeniden başlamasın — paylaşımlı cihazda
  kontrolsüz yeniden başlama = kontrolsüz bellek tüketimi.
- `--oom-score-adj 500`: Linux'a "bellek biterse İLK ÖNCE beni öldür" demek.
  cihazdaki diğer süreçleri koruyan sigorta.
- `--ipc host`: paylaşımlı bellek segmentleri için (PyTorch'un ihtiyacı).

---

## 7. Registry Mimarisi: config.yaml

**Tek doğruluk kaynağı ilkesi:** adlar, portlar, modeller YALNIZ `config.yaml`'da
yazılıdır. Python uygulaması onu `yaml` kütüphanesiyle, kabuk script'leri
`models_tsv` yardımcısıyla (lib.sh içinde küçük bir python) okur. Bir bilgi iki
yerde yazılsaydı, birini güncelleyip diğerini unutmak kaçınılmaz olurdu.

İki ayrı kayıt defteri:

```yaml
engines:                 # MOTORLAR: kapı numaraları
  vllm:   { base_url: "http://localhost:8000/v1" }
  sglang: { base_url: "http://localhost:30000/v1" }

models:                  # MODELLER: kim, hangi motorda, hangi container'da
  qwen2-vl-2b-awq:
    display_name: "Qwen2-VL-2B AWQ (vLLM)"
    engine: vllm                         # hangi motora bağlı
    hf_id: "Qwen/Qwen2-VL-2B-Instruct-AWQ"  # Hugging Face kimliği
    container: "vllm-qwen2-vl-2b-awq"    # onun docker kutusu
    serve_extra: "..."                   # motora eklenecek özel bayraklar
    extra_downloads: [...]               # işaret ettiği harici depolar
```

Kullanıcı MODEL seçer; model motoru belirler; aynı motorun modelleri aynı portu
paylaşır (aynı anda yalnız biri çalışır). Yeni model eklemek = bu dosyaya bir blok
eklemek; dropdown, rozetler, kurulum, script'ler kendiliğinden uyum sağlar.

Üç çalışma modu (`mode:`):

- **switch (varsayılan):** toplamda tek container ayakta. En güvenli.
- **dual (deney):** motor başına bir slot (vLLM'de 1 + SGLang'de 1) aynı anda.
  Bellek sınırda; bilinçli deney prosedürü ister.
- **mock:** hiçbir sunucu yok; sahte cevaplar (arayüz testi için).
  `VQA_MODE` ortam değişkeni bu satırı geçici ezebilir.

---

## 8. Uygulama Dosyaları — Satır Düzeyinde Gezinti

### 8.1 `app/config.py` — yapılandırmayı okuyan ve DENETLEYEN kapı

- Her kavram bir `dataclass` (alanları sabit, tip belirli küçük veri sınıfı):
  `EngineConfig`, `ModelConfig`, `FrameConfig`, `RequestConfig`, `AppConfig`.
  `frozen=True` = oluşturulduktan sonra değiştirilemez (yanlışlıkla bozulmaz).
- `load_config()` **fail-fast** ilkesiyle çalışır: dosya eksikse, mode geçersizse,
  model bilinmeyen motora işaret ediyorsa, container adları çakışıyorsa —
  uygulama AÇILMADAN, anlaşılır Türkçe hatayla durur. Bozuk ayarla yarım çalışmak,
  hiç çalışmamaktan tehlikelidir.
- Öncelik zinciri: `--config` argümanı > `VQA_CONFIG` env > `./config.yaml`;
  `VQA_MODE` env yalnız `mode`'u ezer.

### 8.2 `app/frames.py` — videodan kare çıkarma

OpenCV'nin `VideoCapture`'ı video dosyasını açar. Önemli kavramlar:

- **fps** (frame per second): saniyedeki kare sayısı. `saniye × fps = kare numarası`.
- **Seek:** `cap.set(CAP_PROP_POS_FRAMES, n)` = "n'inci kareye git". Video
  sıkıştırması yüzünden bu bedava değil: çoğu kare, öncekilere göre "fark" olarak
  saklanır; tam kareler (keyframe) araya serpilidir. OpenCV, hedefin öncesindeki
  keyframe'e gidip oraya kadar çözerek İSTENEN kareyi verir.
- **Ölçülen saniye:** istenen saniyeyi değil, GERÇEKTEN çözülen karenin zamanını
  raporlarız (`_measure_position`, seek'ten sonra / read'den ÖNCE — çünkü `read()`
  pozisyonu okunan karenin ötesine iter). Küçük ama dürüst fark: 24fps videoda
  5.30 istenirse gerçek kare 5.2917'dir; arayüz gerçeği söyler.
- **Geriye yürüyen kurtarma:** istenen kare bozuksa (kesik dosya kuyruğu vb.)
  0'a atlamak yerine 1 kare / 0.1 / 0.25 / ... / 30 sn geriye adım adım denenir;
  0. saniye SON çaredir. Kullanıcının sorduğu ana en yakın okunabilir kare bulunur.
- Kare bulununca: uzun kenar 512 piksele küçültülür (`INTER_AREA` — küçültme için
  doğru interpolasyon), JPEG'e sıkıştırılır (kalite 85), base64 data-URI'ye çevrilir.
- Hatalar `FrameExtractionError(user_message)` olarak fırlar — kullanıcıya hep
  Türkçe, anlaşılır mesaj gider.

### 8.3 `app/grounding.py` — sözleşme ve savunmalı ayrıştırıcı

- `SYSTEM_PROMPT`: modele verilen talimat — "soruyu kullanıcının dilinde cevapla;
  cevap nesne konumlamayı gerektiriyorsa 0-1000 normalize koordinatlarla
  `{"answer", "boxes"}` JSON'u döndür; gerekmiyorsa boxes boş."
- **Neden 0-1000 normalize?** Model, resmi kendi iç boyutunda görür; piksel
  koordinatı isteseydik hangi çözünürlüğe göre olduğu belirsiz kalırdı.
  0-1000 = "resmin yüzde binde kaçı" — çözünürlükten bağımsız. Çizerken kendi
  karemizin genişlik/yüksekliğiyle çarparız.
- `parse_grounding_result()` **savunmalıdır** çünkü küçük modeller talimata
  uymayabilir. Sırasıyla dener: markdown çitlerini (` ```json `) temizle → tümünü
  JSON olarak dene → olmadıysa metnin içindeki ilk `{` ... son `}` bloğunu dene →
  yine olmadıysa TÜM metin cevaptır, kutu yok. Kutu listesinde de tek tek eleme:
  bbox 4 sayı değilse, ters çevrilmişse (x2≤x1) atla. 0-1 float da 0-1000 int de
  kabul edilir. **Bu fonksiyon asla exception fırlatmaz** — modelin en saçma
  çıktısı bile uygulamayı düşüremez.

### 8.4 `app/vqa_client.py` — modele soran katman

- `OpenAIVqaClient.ask(prompt, data_uri)`: system prompt + kullanıcı mesajı
  (metin + görüntü) ile isteği kurar, cevabı `parse_grounding_result`'tan geçirip
  `VqaResult` döndürür.
- **Hata eşleme tablosu:** her teknik hata, kullanıcıya anlamlı Türkçe mesaja
  çevrilir — bağlantı yok ("backend kapalı olabilir"), zaman aşımı ("ilk istek
  yavaş olabilir"), 404 ("sunucu bu model adını tanımıyor"), 400 ("SGLang
  chat-template eksik olabilir"), 5xx ("docker logs <container>" yönlendirmesi).
- `max_retries=0`: istek başarısızsa OTOMATİK tekrar YOK. Nedeni Jetson'a özgü:
  8GB paylaşımlı cihazda kontrolsüz tekrar = OOM riski; kullanıcı isterse elle tekrarlar.
- Zaman aşımı ayrımı: bağlanma 5 sn (kapalı sunucuyu hızlı fark et), cevap
  bekleme 120 sn (ilk istek, ısınma yüzünden yavaştır).
- `MockVqaClient`: modelsiz test için sahte cevap üretir; soruda "var mı / göster /
  nerede / kutula" geçerse SABİT iki sahte kutu döndürür (çizim yolunu test etmek
  için — konumları anlamsızdır, etiketleri "mock-nesne"dir).

### 8.5 `app/annotate.py` — kutu çizici

- Her etikete **deterministik renk**: etiket adının CRC32 özeti → renk tekerinde
  bir ton. Aynı etiket her zaman aynı renk; farklı etiketler farklı. (Python'un
  `hash()`'i kullanılamazdı — her çalıştırmada değişir.)
- Çizgi kalınlığı ve yazı boyutu kare boyutuyla ölçeklenir; etiket yazısı dolgulu
  şeritte, kutunun üstüne sığmazsa altına alınır.
- Girdi kare DEĞİŞTİRİLMEZ (kopya üzerinde çizilir) — aynı kare başka amaçla
  tekrar kullanılabilir kalır.

### 8.6 `app/backends.py` — ModelManager: sistemin beyni

İki sınıf:

**`DockerController`** — docker komutlarının tek geçidi. `subprocess` ile
`docker start/stop/inspect/version` çalıştırır. İncelikleri:
- `available()`: docker var mı? Olumlu cevap kalıcı önbelleklenir; olumsuz cevap
  30 sn sonra yeniden denenir (docker daemon'ı sonradan açılabilir).
- Test edilebilirlik: gerçek subprocess yerine sahte bir "runner" enjekte
  edilebilir — testlerimiz docker'sız çalışır.

**`ModelManager`** — model yaşam döngüsü:
- `served_model(engine)`: motorun `/v1/models` kapısına sorar → şu an gerçekten
  hangi model yüklü? **Aktiflik tanımımız:** dönen kimlik == modelin `hf_id`'si.
  Bu sayede yanlış modele soru gitmesi İMKÂNSIZ (soru sorulmadan önce de kontrol edilir).
- `statuses()`: her model için rozet durumu — AKTİF (motor onu servis ediyor) /
  BAŞLATILIYOR (container ayakta ama henüz hazır değil) / KAPALI.
- `activate(model)`: model değiştirme akışı. Bir **generator**'dır (`yield` ile
  adım adım ilerleme mesajı üretir → arayüzde canlı akar):
  1. Zaten aktifse → bitti.
  2. Durdurulacakları belirle: switch modda DİĞER HER ŞEY; dual modda yalnız
     AYNI MOTORUN diğer container'ları (öteki motorun slotu dokunulmaz).
  3. Durdur → hedefi başlat → `served_model` doğru kimliği dönene dek yokla
     (3 sn arayla, 300 sn tavan). Container bu sırada ölürse "crash" raporla.
  4. Her sonuç bir `SwitchEvent` mesajıdır; fonksiyon asla exception fırlatmaz.
- `threading.Lock`: aynı anda iki geçiş denenirse ikincisi kibarca reddedilir.
  (İki geçişin aynı anda container durdurup başlatması kaosa yol açardı.)

### 8.7 `app/main.py` — Gradio arayüzü ve olay örgüsü

- `gr.Blocks` içinde bileşenler tanımlanır: `gr.Video`, `gr.Dropdown` (modeller),
  `gr.HTML` (rozetler), `gr.Textbox` (soru), `gr.Button`, `gr.Image` + `gr.Textbox`
  (çıktılar), `gr.Timer` (5 sn'de bir rozet tazeleme).
- **Olay bağlama** deseni: `bileşen.olay(fonksiyon, girdiler, çıktılar)` —
  "bu butona tıklanınca şu Python fonksiyonunu şu bileşenlerin değerleriyle çağır,
  dönüşünü şu bileşenlere yaz."
- **Enter-anı yakalama:** `ask_btn.click(on_ask, ..., js=JS_FREEZE_TIME)`.
  `js=` parametresi, Python fonksiyonundan ÖNCE tarayıcıda çalışan bir JavaScript
  ön-işleyicidir; girdi dizisini alır, videonun `currentTime`'ını okuyup saniye
  yuvasına yazar, değiştirilmiş diziyi Python'a iletir. Saniyeyi taşıyan `gr.Number`
  bileşeni **görünmezdir** (`visible=False`) — Gradio'da JS→Python aktarımı için
  bir bileşen yuvası şarttır, ama kullanıcının görmesi gerekmez.
  (Deneyle öğrendiğimiz ders: js-only bir olaya `.then()` zinciri Gradio 5.31'de
  TETİKLENMEZ; js'i asıl fonksiyonla aynı olaya koymak gereken desendir.)
- **Canlı ilerleme:** `on_model_change` bir generator — `activate`'ten gelen her
  mesajı `yield` eder; Gradio her yield'i anında tarayıcıya basar. 1-2 dk'lık
  model geçişi bu sayede "donmuş" görünmez. Geçiş boyunca Sor butonu ve dropdown
  kilitlenir, bitince açılır.
- `concurrency_limit=1`: aynı olaydan aynı anda ikinci bir çalıştırma sıraya girer
  (paylaşımlı cihazda paralel istek istemiyoruz).
- `on_ask` akışı: doğrulamalar (video var mı, soru boş mu, geçiş sürüyor mu,
  model AKTİF mi) → kare çıkar → sor → kutu varsa çiz → `[X.X. sn] cevap (N nesne
  kutulandı)` döndür. Model hatasında kare YİNE gösterilir (yalnız cevap yerinde
  hata mesajı olur).

### 8.8 `app/ui_text.py`

Kullanıcıya görünen TÜM Türkçe metinler tek sözlükte. Neden? Metin değişikliği
tek dosyada yapılır; kod içinde dağınık string aranmaz; ileride ikinci dil
eklemek kolaylaşır.

---

## 9. jetson/ Script'leri — İşletme Katmanı

### 9.1 `lib.sh` — ortak alet çantası

Port tanımları (`VLLM_PORT=8000`, `SGL_PORT=30000` — env ile ezilebilir, TÜM
script'ler buradan okur), renkli PASS/FAIL yazıcılar, `avail_mb` (boş RAM),
`models_tsv` (config.yaml'ı tablo satırlarına çevirir), `wait_healthy`
(container ayakta VE `/v1/models` 200 dönene kadar bekler; container ölürse
son 80 satır logu basar).

### 9.2 `check-system.sh` — salt-okunur ön kontrol

Hiçbir şey değiştirmez. Bakar: L4T/JetPack sürümü (→ YOL A/B/STOP karar matrisi),
boş RAM eşikleri, disk, docker + nvidia runtime + kullanıcı grubu, python3-yaml,
portların boşluğu, en çok RAM yiyen süreçler (cihaz taban çizgisi).

### 9.3 `setup.sh` — tek seferlik, idempotent kurulum

**İdempotent** = tekrar tekrar çalıştırmak güvenli; yapılmışı atlar. Dört adım:
jetson-containers klonu → image pull (yoksa) → TÜM modeller + `extra_downloads`
bağımlılıkları indirme (kaldığı yerden devam eder) → model başına `docker create`.
Bayraklar: `--recreate`, `--recreate-model <ad>`, `--vllm-fallback` (dustynv vLLM
Nano'da çökerse NVIDIA image'ına geçiş). Fraksiyon değişikliği = recreate
(argümanlar tek yerde kalsın diye bilinçli tasarım).

### 9.4 `start-model.sh <ad>` — tek giriş noktası

Aynı motorun çalışan başka container'ı varsa durdurur (port tek), hedefi başlatır,
sağlığı bekler. Argümansız çağrılırsa modelleri listeler.

### 9.5 `start-dual.sh` — deney kapısı

İki motoru SIRAYLA başlatır (aynı anda yükleme = tepe bellek çakışması). Önce
container'lara gömülü fraksiyonları `docker inspect`'le okur; 0.35'ten büyükse
BAŞLATMAYI REDDEDER ("önce 0.3 ile recreate et" der) — switch fraksiyonlarıyla
dual denemek kesin OOM olurdu, kapı bunu fiziken engeller.

### 9.6 `smoke-test.sh vllm|sglang` — aşamalı sağlama

Altı aşama, her biri PASS/WARN/FAIL: Server ready → OpenAI request → Response
shape → Text response → Vision input (test sahnesi: bilinen konumda kırmızı kare +
mavi daire; cevap rengi anmalı — SGLang'de anmazsa chat-template alarmı) →
Grounding (uygulamanın GERÇEK sözleşmesi ve parser'ı import edilir; dönen kutunun
kırmızı karenin gerçek bölgesiyle örtüşmesi denetlenir). Exit kodu otomasyona
uygundur. `SMOKE_IMAGE=...` ile kendi görüntünle test edebilirsin.

### 9.7 `free-memory.sh` — onaylı bellek merdiveni

Her adım tek tek onay ister, hiçbir şey kalıcılaştırılmaz, uygulananlar durum
dosyasına yazılır ve `restore` YALNIZ uygulananı geri alır. Sıra: önbellek
boşaltma → NVMe takas dosyası (OOM emniyet ağı; doğrulanmadan zram adımı kilitli) →
zram kapatma → GUI kapatma (ekrana bağımlı başka yazılım varsa YASAK).

### 9.8 Bellek bütçesi gerçeği

Jetson'da ~7.6GB kullanılabilir; sistem+GUI+diğer yazılımlar ~3-3.5GB alır. Kalan ~4-5GB'de:
switch modda tek motor (frac 0.5 ≈ 3.8GB tavan) rahat; dual modda iki motor ancak
0.3/0.3 fraksiyonlarla sınırda sığar. cihazdaki diğer yazılımların OOM'la ölmesi kabul edilemez
olduğundan tüm varsayılanlar temkinli tarafta.

---

## 10. Test Stratejisi — Neyi Nasıl Doğruladık

Katman katman:

1. **Birim testleri (43 adet, `tests/`):** her modül yalıtımda. Püf noktaları:
   docker'sız test için sahte DockerController; video akışını sahtelemek için
   sahte VideoCapture (bozuk-kuyruk senaryoları böyle test edildi); sentetik
   video (her karenin rengine kare numarası kodlanır → hangi karenin geldiği
   piksellerden okunur).
2. **Mock modu:** arayüzün tam akışı, modelsiz.
3. **Sahte OpenAI sunucusu:** smoke-test mantığı, kontrollü senaryolarla (sağlıklı /
   görseli yok sayan / 500 dönen / yanlış bölgeye kutu koyan) doğrulandı.
4. **Gerçek tarayıcı (Playwright + headless Chromium):** Enter-anı yakalama gibi
   tarayıcı-JS davranışları API üzerinden test EDİLEMEZ (API tarayıcıyı atlar);
   bu yüzden gerçek tarayıcı sürüldü. Gradio'nun js/then davranışı da böyle
   deneyle keşfedildi.
5. **Jetson'da kalanlar:** ARM image'larının fiilen açılması, gerçek bellek
   davranışı, SGLang'in bu cihaz sınıfındaki ilk çalıştırması, AWQ çekirdekleri.

---

## 11. Tasarım Kararları ve Nedenleri (Özet Tablo)

| Karar | Neden |
|---|---|
| Model başına ayrı container | Motorlar süreç başına tek model yükler; app'in `docker run` kurması güvenlik gereği yasak → önceden hazırlanmış kutular arasında start/stop |
| Aktiflik = served-id eşleşmesi | "Port açık" yetmez; YANLIŞ modele soru gitmesini fiziken engeller |
| switch varsayılan, dual deney | Paylaşımlı 8GB'de cihazdaki diğer yazılımların güvenliği her şeyden önce |
| `max_retries=0` | Otomatik tekrar, dar bellekte OOM riskini katlar |
| Ölçülen saniye raporlama | İstenen değil gerçekleşen doğrudur; kuantizasyon/VFR farkları gizlenmez |
| Bozuk karede geriye yürüyüş | 0. kareye atlamak kullanıcının sorusuyla ilgisiz cevap üretir |
| Savunmalı grounding parser | Küçük modeller sözleşmeye uymayabilir; uygulama asla kırılmamalı |
| 0-1000 normalize koordinat | Çözünürlükten bağımsız; model ve çizici farklı boyutlarda çalışabilir |
| Tek doğruluk kaynağı (config.yaml) | Çift kayıt = kaçınılmaz tutarsızlık (inceleme bunu bir kez yakaladı: port kopyaları) |
| Türkçe metinler tek dosyada | Değişiklik tek yerde; koddan bağımsız |
| Fail-fast config doğrulama | Bozuk ayarla yarım çalışmak, hiç çalışmamaktan tehlikeli |
| Aşamalı smoke test + gerçek exit kodu | "curl çıktısına göz at" insan hatasına açık; PASS/FAIL makinece de okunur |
| `--oom-score-adj 500` + `--restart no` | Bellek biterse önce biz ölürüz; ölen kendiliğinden hortlamaz |

---

## 12. Sık Karışan Kavramlar — Hızlı Ayrımlar

- **Model ≠ Motor:** model = öğrenilmiş sayılar (Hugging Face'ten); motor =
  onları çalıştıran sunucu yazılımı (vLLM/SGLang).
- **OpenAI formatı ≠ OpenAI modeli:** format bir konuşma protokolü; modellerimiz
  OpenAI'nin değil ama aynı dili konuşuyorlar.
- **Image ≠ Container:** kalıp ≠ çalışan kopya.
- **KAPALI rozeti ≠ hata:** "o modelin sunucusu şu an ayakta değil" demek;
  Windows'ta hep KAPALI olması normaldir (ARM image'ları PC'de açılamaz).
- **Mock kutuları ≠ gerçek tespit:** mock görüntüye hiç bakmaz; kutuları sabittir,
  etiketi "mock-nesne"dir.
- **`VQA_MODE` ≠ Windows'a özel:** her yerde çalışan geçici mod ezme değişkeni;
  Jetson'da KULLANMA (gerçek modeller dururken sahte cevap alırsın).

---

*Bu belge repodadır (`docs/teknik-rehber.md`); kodla birlikte sürümlenir.
Bir bölüm eksik ya da anlaşılmaz gelirse söyle — derinleştirip güncelleyelim.*
