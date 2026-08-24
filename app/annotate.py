"""Jenerik kutu çizici: VqaResult.boxes'taki HERHANGI bir etiketi frame'e işler.

Etiket başına deterministik renk (crc32 tabanlı — süreçler arası kararlı),
çizgi kalınlığı ve yazı boyutu frame boyutuyla ölçeklenir. Girdi RGB numpy
dizisidir ve DEĞİŞTİRİLMEZ; kopya döndürülür.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np

from .grounding import Box


def _label_color(label: str) -> tuple[int, int, int]:
    hue = zlib.crc32(label.encode("utf-8")) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    r, g, b = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return int(r), int(g), int(b)


def annotate_frame(frame_rgb: np.ndarray, boxes: list[Box]) -> np.ndarray:
    if not boxes:
        return frame_rgb
    out = frame_rgb.copy()
    h, w = out.shape[:2]
    thickness = max(2, min(h, w) // 200)
    font_scale = max(0.4, min(h, w) / 800.0)

    for box in boxes:
        x1, y1 = int(box.x1 * w), int(box.y1 * h)
        x2, y2 = int(box.x2 * w), int(box.y2 * h)
        color = _label_color(box.label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        (tw, th), baseline = cv2.getTextSize(
            box.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        ty = y1 - 4 if y1 - th - baseline - 4 >= 0 else y2 + th + baseline + 4
        cv2.rectangle(
            out,
            (x1, ty - th - baseline),
            (min(x1 + tw + 4, w - 1), ty + baseline),
            color,
            -1,
        )
        cv2.putText(
            out, box.label, (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out
