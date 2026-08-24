import numpy as np
import pytest

from app.annotate import annotate_frame
from app.grounding import Box, parse_grounding_result


def test_clean_json_with_boxes():
    raw = '{"answer": "Evet, kırmızı bir araba görünüyor.", "boxes": [{"label": "car", "bbox": [120, 200, 420, 500]}]}'
    res = parse_grounding_result(raw)
    assert res.answer == "Evet, kırmızı bir araba görünüyor."
    assert len(res.boxes) == 1
    b = res.boxes[0]
    assert b.label == "car"
    assert b.x1 == pytest.approx(0.12) and b.y2 == pytest.approx(0.5)
    assert res.raw_response == raw


def test_json_without_boxes():
    res = parse_grounding_result('{"answer": "Hava açık görünüyor.", "boxes": []}')
    assert res.answer == "Hava açık görünüyor."
    assert res.boxes == []


def test_json_in_markdown_fences():
    raw = 'Tabii:\n```json\n{"answer": "Bir kişi var.", "boxes": [{"label": "person", "bbox": [0, 0, 500, 1000]}]}\n```'
    res = parse_grounding_result(raw)
    assert res.answer == "Bir kişi var."
    assert len(res.boxes) == 1


def test_json_embedded_in_prose():
    raw = 'Sonuç şu: {"answer": "İki köpek.", "boxes": [{"label": "dog", "bbox": [10, 10, 100, 100]}, {"label": "dog", "bbox": [500, 500, 900, 900]}]} umarım yardımcı olur'
    res = parse_grounding_result(raw)
    assert res.answer == "İki köpek."
    assert [b.label for b in res.boxes] == ["dog", "dog"]


def test_invalid_json_falls_back_to_plain_text():
    raw = "Görüntüde üç insan görüyorum, meydanda yürüyorlar."
    res = parse_grounding_result(raw)
    assert res.answer == raw
    assert res.boxes == []


def test_broken_box_entries_are_skipped():
    raw = '''{"answer": "Karışık.", "boxes": [
        {"label": "ok", "bbox": [100, 100, 200, 200]},
        {"label": "eksik-bbox"},
        {"label": "kisa", "bbox": [1, 2, 3]},
        {"label": "ters", "bbox": [300, 300, 100, 100]},
        {"label": "sayi-degil", "bbox": ["a", "b", "c", "d"]},
        "dict-degil"
    ]}'''
    res = parse_grounding_result(raw)
    assert [b.label for b in res.boxes] == ["ok"]


def test_zero_one_float_coords_accepted():
    raw = '{"answer": "x", "boxes": [{"label": "cat", "bbox": [0.1, 0.2, 0.8, 0.9]}]}'
    res = parse_grounding_result(raw)
    assert res.boxes[0].x1 == pytest.approx(0.1)
    assert res.boxes[0].y2 == pytest.approx(0.9)


def test_out_of_range_coords_clamped():
    raw = '{"answer": "x", "boxes": [{"label": "b", "bbox": [-50, 0, 1500, 800]}]}'
    res = parse_grounding_result(raw)
    b = res.boxes[0]
    assert b.x1 == 0.0 and b.x2 == 1.0 and b.y2 == pytest.approx(0.8)


def test_missing_answer_falls_back_to_raw():
    raw = '{"boxes": [{"label": "p", "bbox": [0, 0, 10, 10]}]}'
    res = parse_grounding_result(raw)
    assert res.answer == raw  # answer alanı yoksa ham metin cevaptır


def test_empty_input():
    res = parse_grounding_result("")
    assert res.answer == "" and res.boxes == []


def test_truncated_json_salvages_answer():
    # max_tokens'a carpan dongulu cikti (sahada goruldu): JSON yarim, answer kurtarilmali
    raw = '{"answer": "Yes, there is a bridge.", "boxes": [{"label": "The bridge", "bbox": [0.0, 0.0, 1.0, 0.9]}, {"label": "The bridge", "bbox": [0.0, 0.0'
    res = parse_grounding_result(raw)
    assert res.answer == "Yes, there is a bridge."   # ham JSON dokuntusu DEGIL
    assert res.boxes == []


def test_duplicate_boxes_deduped_and_capped():
    dup = ', '.join('{"label": "b", "bbox": [0, 0, 1000, 900]}' for _ in range(30))
    uniq = ', '.join(f'{{"label": "u{i}", "bbox": [{i*10}, 0, {i*10+50}, 100]}}' for i in range(30))
    res1 = parse_grounding_result(f'{{"answer": "x", "boxes": [{dup}]}}')
    assert len(res1.boxes) == 1                      # birebir tekrarlar elendi
    res2 = parse_grounding_result(f'{{"answer": "x", "boxes": [{uniq}]}}')
    assert len(res2.boxes) == 10                     # ust sinir


# --- annotate_frame ---

def _frame():
    return np.full((200, 300, 3), 128, dtype=np.uint8)


def test_annotate_draws_and_preserves_input():
    frame = _frame()
    orig = frame.copy()
    boxes = [Box("person", 0.1, 0.1, 0.5, 0.9), Box("car", 0.6, 0.2, 0.95, 0.7)]
    out = annotate_frame(frame, boxes)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)          # çizim yapıldı
    assert np.array_equal(frame, orig)             # girdi DEĞİŞMEDİ


def test_annotate_empty_boxes_returns_frame_unchanged():
    frame = _frame()
    out = annotate_frame(frame, [])
    assert out is frame


def test_annotate_same_label_same_color_different_labels_differ():
    from app.annotate import _label_color
    assert _label_color("person") == _label_color("person")
    assert _label_color("person") != _label_color("car")
