"""config.yaml yükleme ve fail-fast doğrulama.

Model Registry ve Engine Registry ayrıdır: kullanıcı UI'da MODEL seçer; her model
bir motora bağlıdır ve kendi container'ına sahiptir (motor başına aynı anda tek
model servis edilebilir, aynı motorun modelleri portu paylaşır).

Öncelik sırası: --config argümanı > $VQA_CONFIG > ./config.yaml
$VQA_MODE ortam değişkeni mode alanını ezer (Windows'ta mock testi için).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_MODES = ("dual", "switch", "mock")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class EngineConfig:
    key: str
    base_url: str


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    engine: str
    hf_id: str
    container: str
    serve_extra: str = ""
    # modelin isaret ettigi harici HF repolari (orn. LLaVA'nin SigLIP tower'i);
    # setup.sh on-indirme adiminda bunlari da ceker (dependency registry)
    extra_downloads: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameConfig:
    max_dim: int = 512
    jpeg_quality: int = 85


@dataclass(frozen=True)
class RequestConfig:
    read_timeout_s: float = 120.0
    max_tokens: int = 256
    temperature: float = 0.2


@dataclass(frozen=True)
class AppConfig:
    mode: str
    default_model: str
    switch_timeout_s: int = 300
    health_poll_interval_s: float = 3.0
    status_refresh_s: float = 5.0
    frame: FrameConfig = field(default_factory=FrameConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    engines: dict[str, EngineConfig] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)

    def engine_of(self, model_key: str) -> EngineConfig:
        return self.engines[self.models[model_key].engine]

    def models_on_engine(self, engine_key: str) -> list[ModelConfig]:
        return [m for m in self.models.values() if m.engine == engine_key]


def _resolve_path(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path)
    env_path = os.environ.get("VQA_CONFIG")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | None = None) -> AppConfig:
    cfg_path = _resolve_path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"Config dosyası bulunamadı: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Config dosyası geçersiz YAML: {cfg_path}\n{e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"Config dosyası bir eşleme (mapping) olmalı: {cfg_path}")

    mode = os.environ.get("VQA_MODE") or raw.get("mode", "dual")
    if mode not in VALID_MODES:
        raise ConfigError(f"Geçersiz mode: '{mode}'. Geçerli değerler: {', '.join(VALID_MODES)}")

    engines_raw = raw.get("engines")
    if not isinstance(engines_raw, dict) or not engines_raw:
        raise ConfigError("config.yaml 'engines' altında en az bir motor tanımlamalı.")
    engines: dict[str, EngineConfig] = {}
    for key, e in engines_raw.items():
        if not isinstance(e, dict) or not e.get("base_url"):
            raise ConfigError(f"engines.{key} 'base_url' alanı içermeli.")
        engines[key] = EngineConfig(key=key, base_url=str(e["base_url"]).rstrip("/"))
    urls = [e.base_url for e in engines.values()]
    if len(set(urls)) != len(urls):
        raise ConfigError("Engine base_url değerleri benzersiz olmalı.")

    models_raw = raw.get("models")
    if not isinstance(models_raw, dict) or not models_raw:
        raise ConfigError("config.yaml 'models' altında en az bir model tanımlamalı.")
    models: dict[str, ModelConfig] = {}
    for key, m in models_raw.items():
        if not isinstance(m, dict):
            raise ConfigError(f"models.{key} bir eşleme olmalı.")
        missing = [f for f in ("display_name", "engine", "hf_id", "container") if not m.get(f)]
        if missing:
            raise ConfigError(f"models.{key} eksik alan(lar): {', '.join(missing)}")
        if m["engine"] not in engines:
            raise ConfigError(
                f"models.{key} bilinmeyen motora işaret ediyor: '{m['engine']}' "
                f"(mevcut: {', '.join(engines)})"
            )
        extra_dl = m.get("extra_downloads") or []
        if not isinstance(extra_dl, list) or not all(isinstance(d, str) and d for d in extra_dl):
            raise ConfigError(f"models.{key}.extra_downloads bir metin listesi olmalı.")
        models[key] = ModelConfig(
            key=key,
            display_name=str(m["display_name"]),
            engine=str(m["engine"]),
            hf_id=str(m["hf_id"]),
            container=str(m["container"]),
            serve_extra=str(m.get("serve_extra", "")),
            extra_downloads=tuple(extra_dl),
        )

    containers = [m.container for m in models.values()]
    if len(set(containers)) != len(containers):
        raise ConfigError("Container adları benzersiz olmalı.")
    hf_per_engine = [(m.engine, m.hf_id) for m in models.values()]
    if len(set(hf_per_engine)) != len(hf_per_engine):
        raise ConfigError("Aynı motorda aynı hf_id iki kez tanımlanamaz.")

    default_model = raw.get("default_model") or next(iter(models))
    if default_model not in models:
        raise ConfigError(
            f"default_model '{default_model}' models içinde yok "
            f"(mevcut: {', '.join(models)})."
        )

    frame_raw = raw.get("frame") or {}
    request_raw = raw.get("request") or {}
    try:
        frame = FrameConfig(
            max_dim=int(frame_raw.get("max_dim", 512)),
            jpeg_quality=int(frame_raw.get("jpeg_quality", 85)),
        )
        request = RequestConfig(
            read_timeout_s=float(request_raw.get("read_timeout_s", 120)),
            max_tokens=int(request_raw.get("max_tokens", 256)),
            temperature=float(request_raw.get("temperature", 0.2)),
        )
        cfg = AppConfig(
            mode=mode,
            default_model=default_model,
            switch_timeout_s=int(raw.get("switch_timeout_s", 300)),
            health_poll_interval_s=float(raw.get("health_poll_interval_s", 3)),
            status_refresh_s=float(raw.get("status_refresh_s", 5)),
            frame=frame,
            request=request,
            engines=engines,
            models=models,
        )
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Config sayısal alanlarından biri geçersiz: {e}") from e

    if cfg.frame.max_dim < 64:
        raise ConfigError("frame.max_dim en az 64 olmalı.")
    if not (1 <= cfg.frame.jpeg_quality <= 100):
        raise ConfigError("frame.jpeg_quality 1-100 aralığında olmalı.")
    if cfg.switch_timeout_s < 10:
        raise ConfigError("switch_timeout_s en az 10 olmalı.")
    if cfg.health_poll_interval_s <= 0 or cfg.status_refresh_s <= 0:
        raise ConfigError("health_poll_interval_s ve status_refresh_s pozitif olmalı.")
    return cfg
