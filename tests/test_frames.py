import cv2
import numpy as np
import pytest

from app.frames import FrameExtractionError, extract_frame

FPS = 10
SECONDS = 3  # 30 kare; her karenin mavi kanalı = kare indeksi (hangi kare geldi anlaşılır)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    assert writer.isOpened()
    for i in range(FPS * SECONDS):
        frame = np.full((48, 64, 3), (i, 128, 200), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_extract_at_zero(clip):
    fr = extract_frame(clip, 0)
    assert fr.actual_second == 0.0
    assert fr.duration == pytest.approx(SECONDS, abs=0.2)
    assert fr.data_uri.startswith("data:image/jpeg;base64,")
    assert fr.frame_rgb.shape == (48, 64, 3)


def test_extract_mid(clip):
    fr = extract_frame(clip, 1.5)
    # mavi kanal ~ kare indeksi (JPEG kaybı için tolerans)
    blue = int(fr.frame_rgb[0, 0, 2])
    assert abs(blue - 15) <= 3


def test_clamp_beyond_duration(clip):
    fr = extract_frame(clip, 999)
    assert fr.actual_second <= SECONDS
    assert fr.frame_rgb is not None


def test_actual_second_is_measured_not_requested(clip):
    # 10 fps'te 1.234 sn istegi frame-grid'e oturur: frame 12 -> 1.2 sn.
    # Raporlanan deger istek (1.234) DEGIL, olculen decode zamani olmali.
    fr = extract_frame(clip, 1.234)
    assert fr.actual_second == pytest.approx(1.2, abs=0.06)
    assert fr.actual_second != pytest.approx(1.234, abs=1e-6)


def test_measured_time_matches_decoded_frame(clip):
    # olculen zaman ile decode edilen karenin icerigi (mavi kanal = frame no)
    # tutarli olmali: actual_second * fps ~= kare indeksi
    fr = extract_frame(clip, 2.0)
    blue = int(fr.frame_rgb[0, 0, 2])
    # tolerans 5: mp4v kodek mavi kanali birkac birim kaydirabiliyor
    assert abs(blue - round(fr.actual_second * FPS)) <= 5


def test_downscale(clip):
    fr = extract_frame(clip, 0, max_dim=32)
    assert max(fr.frame_rgb.shape[:2]) == 32


def test_no_upscale(clip):
    fr = extract_frame(clip, 0, max_dim=4096)
    assert fr.frame_rgb.shape == (48, 64, 3)


def test_corrupt_file(tmp_path):
    bad = tmp_path / "bozuk.mp4"
    bad.write_bytes(b"bu bir video degil")
    with pytest.raises(FrameExtractionError):
        extract_frame(str(bad), 0)


# --- geriye yürüyen fallback (sahte VideoCapture ile deterministik) ---

class FakeCap:
    """60 sn'lik (600 kare, 10 fps) sahte video: yalnız decodable_below
    altındaki kare indeksleri okunabilir (kesik dosya kuyrugu senaryosu)."""

    def __init__(self, decodable_below):
        self.decodable_below = decodable_below
        self.pos_frames = 0.0
        self.read_attempts = []

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 10.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 600.0
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.pos_frames
        if prop == cv2.CAP_PROP_POS_MSEC:
            return 0.0
        return 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos_frames = float(value)
        return True

    def read(self):
        idx = int(self.pos_frames)
        self.read_attempts.append(idx)
        if idx < self.decodable_below:
            return True, np.full((48, 64, 3), 100, dtype=np.uint8)
        return False, None

    def release(self):
        pass


def _with_fake_cap(monkeypatch, fake):
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: fake)


def test_broken_tail_falls_back_to_nearest_readable(monkeypatch):
    # 58.5 sn'den sonrasi bozuk; 58.7 istenince 0'a ATLAMAMALI,
    # en yakin okunabilir kareye geriye yurumeli
    fake = FakeCap(decodable_below=585)
    _with_fake_cap(monkeypatch, fake)
    fr = extract_frame("sahte.mp4", 58.7)
    assert 57.0 <= fr.actual_second < 58.7      # yakin bir kare bulundu
    assert fr.actual_second != 0.0              # 0'a dusulmedi
    assert len(fake.read_attempts) <= 12        # deneme sayisi sinirli


def test_zero_is_last_resort(monkeypatch):
    # yalniz 0. kare okunabilir: tum geri adimlar denenip 0'a dusulmeli
    fake = FakeCap(decodable_below=1)
    _with_fake_cap(monkeypatch, fake)
    fr = extract_frame("sahte.mp4", 58.7)
    assert fr.actual_second == 0.0
    assert len(fake.read_attempts) <= 12


def test_nothing_readable_raises(monkeypatch):
    fake = FakeCap(decodable_below=0)
    _with_fake_cap(monkeypatch, fake)
    with pytest.raises(FrameExtractionError):
        extract_frame("sahte.mp4", 58.7)
