"""Videodan tek kare çıkarma: saniye -> RGB kare + base64 JPEG data-URI."""
from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from .ui_text import TR


class FrameExtractionError(Exception):
    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


@dataclass
class FrameResult:
    frame_rgb: np.ndarray      # RGB, gr.Image için
    data_uri: str              # "data:image/jpeg;base64,...."
    actual_second: float       # kıskaçlama sonrası gerçek saniye
    duration: float | None     # metadata yoksa None


def _measure_position(cap: cv2.VideoCapture, fps: float, fallback: float) -> float:
    """Decode edilecek karenin GERÇEK zamanını ölçer (seek sonrası, read ÖNCESİ
    çağrılmalı — read() pozisyonu okunan karenin ötesine ilerlettiği için sonrası
    bir kare kaydırır). Öncelik POS_FRAMES/fps'tedir: FFMPEG backend'i seek
    sonrası read öncesi POS_MSEC'te güvenilmez değerler döndürebiliyor (Windows'ta
    ölçüldü: ~1-2ms saçma değerler). POS_MSEC yalnız fps metadata'sı olmayan
    yolda kullanılır; o da geçersizse istenen saniyeye düşülür."""
    if fps and fps > 0:
        idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if idx is not None and idx >= 0:
            return idx / fps
    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if ms and ms > 0:
        return ms / 1000.0
    return fallback


# istenen kare decode edilemezse geriye doğru denenecek ofsetler (sn).
# Her deneme keyframe'den ileri-decode gerektirdiği için pahalıdır; sabit
# küçük adımlar yerine büyüyen adımlar kullanılır. 0. saniye EN SON çaredir.
_FALLBACK_OFFSETS_S = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


def _try_read_at(cap: cv2.VideoCapture, fps: float, target: float):
    """target saniyesine seek edip bir kare okumayı dener.
    Döndürür: (kare | None, ölçülen_saniye)."""
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(target * fps))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
    measured = _measure_position(cap, fps, fallback=target)
    ok, bgr = cap.read()
    return (bgr if ok and bgr is not None else None), measured


def extract_frame(
    video_path: str,
    second: float,
    max_dim: int = 512,
    jpeg_quality: int = 85,
) -> FrameResult:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise FrameExtractionError(TR["err_video_open"])

        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        second = max(0.0, float(second or 0.0))

        if fps and fps > 0 and n_frames and n_frames > 0:
            duration = n_frames / fps
            # son karenin ötesine seek etmemek için 1 kare payı bırak
            second = min(second, max(0.0, duration - 1.0 / fps))
        else:
            duration = None

        # istenen değil, gerçekten decode edilen karenin zamanı raporlanır
        # (frame-grid kuantizasyonu ve VFR videolarda fark eder)
        bgr, measured = _try_read_at(cap, fps, second)

        if bgr is None:
            # istenen kare bozuk/okunamaz: 0'a atlamak yerine EN YAKIN
            # okunabilir kareye geriye doğru yürü (özensiz encoder'lar,
            # kesik dosya kuyrukları); 0. saniye son çare
            one_frame = (1.0 / fps) if fps and fps > 0 else 0.1

            def _seek_key(t: float):
                # dedup GERÇEK seek biriminde: fps biliniyorsa kare indeksi
                # (saniye-yuvarlaması çok yüksek fps'te denemeleri kaçırabilir)
                return int(round(t * fps)) if fps and fps > 0 else round(t, 3)

            zero_key = _seek_key(0.0)
            tried = {_seek_key(second)}
            for offset in (one_frame, *_FALLBACK_OFFSETS_S):
                target = max(0.0, second - offset)
                key = _seek_key(target)
                if key in tried:
                    continue
                tried.add(key)
                bgr, measured = _try_read_at(cap, fps, target)
                if bgr is not None or key == zero_key:
                    break
            if bgr is None and zero_key not in tried:
                bgr, measured = _try_read_at(cap, fps, 0.0)
            if bgr is None:
                raise FrameExtractionError(TR["err_frame_read"])
        second = measured

        h, w = bgr.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            bgr = cv2.resize(
                bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise FrameExtractionError(TR["err_frame_encode"])
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

        return FrameResult(
            frame_rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            data_uri=data_uri,
            actual_second=second,
            duration=duration,
        )
    finally:
        cap.release()
