"""Tüm kullanıcıya görünen Türkçe metinler tek yerde."""

TR = {
    # Genel UI
    "app_title": "Video Soru-Cevap",
    "video_label": "Video yükle",
    "backend_label": "Model",
    "prompt_label": "Soru",
    "ask_btn": "Sor",
    "frame_label": "Çıkarılan kare",
    "answer_label": "Cevap",

    # Durum rozetleri
    "status_active": "AKTİF",
    "status_starting": "BAŞLATILIYOR",
    "status_stopped": "KAPALI",
    "status_unknown": "BİLİNMİYOR",

    # Doğrulama / akış mesajları
    "err_no_video": "Önce bir video yükleyin.",
    "err_no_prompt": "Bir soru yazın.",
    "err_switch_in_progress": "Model geçişi sürüyor, lütfen bekleyin.",
    "err_model_not_active": "Seçili model aktif değil (rozet: {status}). Model, listeden seçilince otomatik başlatılır — rozet AKTİF olana kadar bekleyin.",
    "answer_prefix": "[{second:.1f}. sn] {answer}",
    "boxes_note": " ({n} nesne kutulandı)",

    # Frame çıkarma hataları
    "err_video_open": "Video açılamadı. Dosya bozuk veya desteklenmeyen formatta olabilir.",
    "err_frame_read": "Videodan kare okunamadı. Farklı bir saniye deneyin.",
    "err_frame_encode": "Kare JPEG olarak kodlanamadı.",

    # Backend / istemci hataları
    "err_conn": "Sunucuya bağlanılamadı. Seçili backend kapalı olabilir; durum rozetlerini kontrol edin.",
    "err_timeout": "İstek zaman aşımına uğradı. Modelin ilk isteği yavaş olabilir; tekrar deneyin.",
    "err_model_404": "Model bulunamadı: sunucu '{model}' adını tanımıyor. config.yaml'daki model adını kontrol edin.",
    "err_bad_request": "Sunucu isteği reddetti (400). SGLang için --chat-template ayarı eksik olabilir. Detay: {detail}",
    "err_server": "Sunucu hatası ({code}). Log için: docker logs {container}",
    "err_unexpected": "Beklenmeyen hata: {detail}",
    "err_empty_answer": "Sunucu boş cevap döndürdü.",

    # Geçiş (switch) mesajları
    "switch_already": "Zaten bir model geçişi sürüyor.",
    "switch_no_docker": "Docker bulunamadı (geliştirme ortamında geçiş yapılamaz).",
    "switch_stopping": "{name} durduruluyor...",
    "switch_starting": "{name} başlatılıyor...",
    "switch_loading": "Model yükleniyor... ({elapsed}s/{timeout}s)",
    "switch_ready": "{name} hazır.",
    "switch_crashed": "{name} beklenmedik şekilde durdu (crash). Log için: docker logs {container}",
    "switch_timeout": "Zaman aşımı: {name} {timeout}s içinde hazır olmadı. Log için: docker logs {container}",
    "switch_docker_error": "Docker komutu başarısız: {detail}",

    # Mock
    "mock_answer": "(MOCK) Soru alındı: '{prompt}'. Görüntü {kb} KB. Gerçek model bağlı değil.",
}
