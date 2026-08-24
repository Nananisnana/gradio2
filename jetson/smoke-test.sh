#!/usr/bin/env bash
# Asamali duman testi - OpenAI format paritesini VE gercek davranisi dogrular:
#   Server ready    GET /v1/models 200 + model id
#   OpenAI request  POST /v1/chat/completions HTTP 200
#   Response shape  choices[0].message.content mevcut
#   Text response   content bos degil
#   Vision input    cevap gorsel icerigini anmali (varsayilan sahnede kirmizi kare +
#                   mavi daire); sglang'de anmazsa FAIL (chat-template eksik, README #4),
#                   vllm'de WARN (kucuk model kalitesi olabilir)
#   Grounding       uygulamanin GERCEK sozlesmesi (app/grounding.py) ile kutu istenir.
#                   Varsayilan sahnede kirmizi karenin KONUMU BILINIR -> donen kutunun
#                   dogru bolgeyle ortusmesi de dogrulanir. Rapor bilgilendiricidir,
#                   cikis kodunu etkilemez (kucuk modellerde FAIL beklenen sonuc);
#                   GROUNDING_REQUIRED=1 ile cikis koduna dahil edilir.
#
# Varsayilan gorsel: tests/assets/smoke-scene.png (256x256 sentetik sahne,
# ground-truth'lu). GERCEK goruntuyle test icin (orn. kendi videonuzdan bir kare):
#   SMOKE_IMAGE=/yol/foto.jpg SMOKE_PROMPT="Are there any people in this image?" \
#   SMOKE_EXPECT="person,insan,people" bash jetson/smoke-test.sh vllm
# (ozel gorselde bolge dogrulamasi atlanir - ground-truth bilinmez, kutu sayisi raporlanir)
#
# Kullanim: bash jetson/smoke-test.sh vllm|sglang
# Cikis kodu: 0 = tum zorunlu asamalar PASS; 1 = en az bir FAIL
set -u; . "$(dirname "$0")/lib.sh"
case "${1:-}" in
  vllm) PORT=$VLLM_PORT ;;
  sglang) PORT=$SGL_PORT ;;
  *) echo "kullanim: $0 vllm|sglang"; exit 2 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"

ENGINE="$1" PORT="$PORT" REPO="$REPO" \
SMOKE_IMAGE="${SMOKE_IMAGE:-}" SMOKE_PROMPT="${SMOKE_PROMPT:-}" SMOKE_EXPECT="${SMOKE_EXPECT:-}" \
GROUNDING_REQUIRED="${GROUNDING_REQUIRED:-0}" python3 - <<'PY'
import base64, json, mimetypes, os, sys, urllib.request, urllib.error

sys.path.insert(0, os.environ["REPO"])
from app.grounding import SYSTEM_PROMPT, parse_grounding_result  # uretimle AYNI sozlesme

PORT = os.environ["PORT"]
ENGINE = os.environ["ENGINE"]
BASE = f"http://127.0.0.1:{PORT}/v1"

DEFAULT_IMAGE = os.path.join(os.environ["REPO"], "tests", "assets", "smoke-scene.png")
CUSTOM = bool(os.environ.get("SMOKE_IMAGE"))
IMAGE_PATH = os.environ.get("SMOKE_IMAGE") or DEFAULT_IMAGE
PROMPT = os.environ.get("SMOKE_PROMPT") or "Describe this image in one short sentence."
# varsayilan sahne: beyaz zemin + kirmizi kare + mavi daire
EXPECT = [w.strip().lower() for w in (os.environ.get("SMOKE_EXPECT") or
          "red,kırmızı,kirmizi,square,kare,blue,mavi,circle,daire,shape").split(",") if w.strip()]
# varsayilan sahnedeki kirmizi karenin ground-truth bolgesi (normalize)
TRUE_BOX = (0.125, 0.375, 0.5, 0.75)
if CUSTOM:
    base_q = os.environ.get("SMOKE_PROMPT") or "What objects are in this image?"
    GROUNDING_PROMPT = base_q + " Return bounding boxes for the relevant objects."
else:
    GROUNDING_PROMPT = "Where is the red square? Return its bounding box."

try:
    raw_img = open(IMAGE_PATH, "rb").read()
except OSError as e:
    print(f"Gorsel okunamadi: {IMAGE_PATH} ({e})"); sys.exit(1)
mime = mimetypes.guess_type(IMAGE_PATH)[0] or "image/png"
DATA_URI = f"data:{mime};base64," + base64.b64encode(raw_img).decode()

results = []  # (asama, durum, not)

def finish():
    print()
    hard_fail = False
    for name, status, note in results:
        if status == "FAIL" and (name != "Grounding" or os.environ.get("GROUNDING_REQUIRED") == "1"):
            hard_fail = True
        line = f"{name:<16} {status}"
        if note:
            line += f"   ({note})"
        print(line)
    sys.exit(1 if hard_fail else 0)

def http_json(url, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())

# 1) Server ready
try:
    _, models = http_json(BASE + "/models", timeout=5)
    model_id = models["data"][0]["id"]
    results.append(("Server ready", "PASS", model_id))
except Exception as e:
    results.append(("Server ready", "FAIL", str(e)[:90]))
    finish()

def chat(messages):
    return http_json(BASE + "/chat/completions", {
        "model": model_id, "messages": messages, "max_tokens": 128, "temperature": 0,
    })

user_msg = {"role": "user", "content": [
    {"type": "text", "text": PROMPT},
    {"type": "image_url", "image_url": {"url": DATA_URI}},
]}

# 2) OpenAI request
try:
    st, resp = chat([user_msg])
    results.append(("OpenAI request", "PASS", f"HTTP {st}"))
except urllib.error.HTTPError as e:
    body = e.read()[:120].decode(errors="replace")
    results.append(("OpenAI request", "FAIL", f"HTTP {e.code}: {body}"))
    finish()
except Exception as e:
    results.append(("OpenAI request", "FAIL", str(e)[:90]))
    finish()

# 3) Response shape + 4) Text response
try:
    content = resp["choices"][0]["message"]["content"]
    results.append(("Response shape", "PASS", ""))
except (KeyError, IndexError, TypeError):
    results.append(("Response shape", "FAIL", json.dumps(resp)[:120]))
    finish()
if isinstance(content, str) and content.strip():
    results.append(("Text response", "PASS", content.strip()[:60]))
else:
    results.append(("Text response", "FAIL", "bos content"))
    finish()

# 5) Vision input: cevap gorsel icerigini (beklenen anahtar kelimeler) anmali
if any(w in content.lower() for w in EXPECT):
    results.append(("Vision input", "PASS", ""))
elif ENGINE == "sglang":
    results.append(("Vision input", "FAIL", "cevap gorseli anmiyor -> chat-template eksik olabilir (README #4)"))
else:
    results.append(("Vision input", "WARN", "cevap beklenen icerigi anmadi (kucuk model kalitesi olabilir)"))

# 6) Grounding: uygulamanin gercek sozlesmesi + parser'i ile
def overlaps_truth(b):
    tx1, ty1, tx2, ty2 = TRUE_BOX
    ix = max(0.0, min(b.x2, tx2) - max(b.x1, tx1))
    iy = max(0.0, min(b.y2, ty2) - max(b.y1, ty1))
    inter = ix * iy
    true_area = (tx2 - tx1) * (ty2 - ty1)
    cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    return inter >= 0.25 * true_area or (tx1 <= cx <= tx2 and ty1 <= cy <= ty2)

def g_messages(merge_system):
    text = (SYSTEM_PROMPT + "\n\n" + GROUNDING_PROMPT) if merge_system else GROUNDING_PROMPT
    user = {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": DATA_URI}},
    ]}
    return [user] if merge_system else [{"role": "system", "content": SYSTEM_PROMPT}, user]

g_note = ""
try:
    try:
        _, gresp = chat(g_messages(merge_system=False))
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        # bazi chat sablonlari system rolunu reddeder (InternVL'de sahada
        # goruldu) -> sozlesmeyi kullanici mesajina birlestirip tek deneme
        # (uygulamadaki istemci de ayni geri dususu yapar)
        _, gresp = chat(g_messages(merge_system=True))
        g_note = "; system rolu reddedildi -> birlestirildi"
    graw = gresp["choices"][0]["message"]["content"] or ""
    parsed = parse_grounding_result(graw)
    if parsed.boxes and CUSTOM:
        results.append(("Grounding", "PASS", f"{len(parsed.boxes)} kutu (ozel gorsel: bolge dogrulanamaz){g_note}"))
    elif parsed.boxes and any(overlaps_truth(b) for b in parsed.boxes):
        results.append(("Grounding", "PASS", f"{len(parsed.boxes)} kutu, dogru bolgede{g_note}"))
    elif parsed.boxes:
        results.append(("Grounding", "WARN", f"{len(parsed.boxes)} kutu ama kirmizi kare bolgesiyle ortusmuyor{g_note}"))
    elif parsed.answer != parsed.raw_response:
        results.append(("Grounding", "WARN", f"gecerli JSON, kutu yok{g_note}"))
    else:
        results.append(("Grounding", "FAIL", f"JSON sozlesmesine uymadi (kucuk modellerde beklenir){g_note}"))
except urllib.error.HTTPError as e:
    body = ""
    try:
        body = e.read()[:100].decode(errors="replace")
    except Exception:
        pass
    results.append(("Grounding", "FAIL", f"HTTP {e.code}: {body}{g_note}"))
except Exception as e:
    results.append(("Grounding", "FAIL", f"{str(e)[:90]}{g_note}"))

finish()
PY
