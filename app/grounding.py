"""Prompt'a bağlı genel grounding katmanı.

Sözleşme: VLM'e sistem prompt'uyla JSON {"answer": ..., "boxes": [...]} istenir.
Kutulama kararı modele bırakılır: lokalizasyon gerektirmeyen soruda boxes boş
kalır. Parser HERHANGI bir nesne etiketi için çalışır (persona özel değildir)
ve küçük modellerin bozuk çıktısına karşı savunmalıdır: JSON bulunamaz veya
geçersizse tüm metin cevap sayılır, boxes boş döner (uygulama asla kırılmaz).

Koordinat uzayı: bbox [x1, y1, x2, y2], 0-1000 normalize (sol-üst köşe 0,0).
İçeride 0-1 float'a çevrilir; çizici frame boyutuyla ölçekler.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    label: str
    x1: float  # 0-1 normalize
    y1: float
    x2: float
    y2: float


@dataclass
class VqaResult:
    answer: str
    boxes: list[Box] = field(default_factory=list)
    raw_response: str = ""


SYSTEM_PROMPT = """You are a visual question answering assistant. Answer the user's question about the given image briefly, in the same language as the question.

If and only if the question benefits from localizing specific objects or regions in the image (e.g. "is there a person?", "show the red car", "where is the phone?"), also return bounding boxes for the relevant objects. If localization is not needed (e.g. "how does the weather look?") or nothing relevant is visible, return an empty boxes list.

Respond with ONLY a JSON object, no markdown fences, no extra text, exactly in this format:
{"answer": "<answer in the user's language>", "boxes": [{"label": "<object name in English>", "bbox": [x1, y1, x2, y2]}]}

bbox coordinates are integers in a 0-1000 normalized space: (0,0) is the top-left and (1000,1000) is the bottom-right of the image."""

_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")


def _normalize_coord(v: float, max_seen: float) -> float:
    # sözleşme 0-1000; bazı modeller 0-1 float döndürür — ikisini de kabul et
    if max_seen <= 1.0:
        return min(max(v, 0.0), 1.0)
    return min(max(v, 0.0), 1000.0) / 1000.0


def _parse_boxes(raw_boxes) -> list[Box]:
    boxes: list[Box] = []
    if not isinstance(raw_boxes, list):
        return boxes
    for b in raw_boxes:
        if not isinstance(b, dict):
            continue
        bbox = b.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            coords = [float(c) for c in bbox]
        except (TypeError, ValueError):
            continue
        max_seen = max(coords) if coords else 0.0
        x1, y1, x2, y2 = (_normalize_coord(c, max_seen) for c in coords)
        if x2 <= x1 or y2 <= y1:
            continue
        label = str(b.get("label") or "nesne")
        boxes.append(Box(label=label, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def parse_grounding_result(raw: str) -> VqaResult:
    """Model çıktısını VqaResult'a çevirir; asla exception fırlatmaz."""
    raw = (raw or "").strip()
    text = _FENCE_RE.sub("", raw).strip()

    candidate = None
    try:
        candidate = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # metnin içine gömülü ilk { ... son } bloğunu dene
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                candidate = json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                candidate = None

    if not isinstance(candidate, dict):
        return VqaResult(answer=raw, boxes=[], raw_response=raw)

    answer = candidate.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = raw
    return VqaResult(
        answer=answer.strip(),
        boxes=_parse_boxes(candidate.get("boxes")),
        raw_response=raw,
    )
