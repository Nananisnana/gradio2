import textwrap

import pytest

from app.config import ConfigError, load_config

VALID = textwrap.dedent("""
    mode: dual
    default_model: smol
    engines:
      vllm:
        base_url: "http://localhost:8000/v1"
      sglang:
        base_url: "http://localhost:30000/v1"
    models:
      smol:
        display_name: "SmolVLM2-256M (vLLM)"
        engine: vllm
        hf_id: "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        container: "vllm-smol"
      intern:
        display_name: "InternVL2.5-1B (vLLM)"
        engine: vllm
        hf_id: "OpenGVLab/InternVL2_5-1B"
        container: "vllm-intern"
        serve_extra: "--trust-remote-code"
      llava:
        display_name: "LLaVA-OV-0.5B (SGLang)"
        engine: sglang
        hf_id: "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        container: "sglang-llava"
        extra_downloads:
          - "google/siglip-so400m-patch14-384"
""")


def write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_valid_config(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.mode == "dual"
    assert list(cfg.models) == ["smol", "intern", "llava"]
    assert cfg.models["intern"].serve_extra == "--trust-remote-code"
    assert cfg.engine_of("llava").base_url == "http://localhost:30000/v1"
    assert [m.key for m in cfg.models_on_engine("vllm")] == ["smol", "intern"]
    assert cfg.models["llava"].extra_downloads == ("google/siglip-so400m-patch14-384",)
    assert cfg.models["smol"].extra_downloads == ()
    assert cfg.models["smol"].system_prompt == ""   # varsayilan: genel sozlesme kullanilir


def test_invalid_extra_downloads(tmp_path):
    # VALID, textwrap.dedent'ten gecmis halidir: liste elemani 6 bosluk girintili
    target = 'extra_downloads:\n      - "google/siglip-so400m-patch14-384"'
    assert target in VALID  # girinti degisirse test sessizce no-op olmasin
    bad = VALID.replace(target, 'extra_downloads: "tek-metin-liste-degil"')
    with pytest.raises(ConfigError, match="extra_downloads"):
        load_config(write(tmp_path, bad))


def test_env_mode_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VQA_MODE", "mock")
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.mode == "mock"


def test_invalid_mode(tmp_path):
    with pytest.raises(ConfigError, match="mode"):
        load_config(write(tmp_path, VALID.replace("mode: dual", "mode: banana")))


def test_missing_model_field(tmp_path):
    broken = VALID.replace('container: "vllm-smol"', "")
    with pytest.raises(ConfigError, match="container"):
        load_config(write(tmp_path, broken))


def test_duplicate_container(tmp_path):
    dup = VALID.replace('container: "sglang-llava"', 'container: "vllm-smol"')
    with pytest.raises(ConfigError, match="benzersiz"):
        load_config(write(tmp_path, dup))


def test_unknown_engine_ref(tmp_path):
    bad = VALID.replace("engine: sglang", "engine: tensorrt")
    with pytest.raises(ConfigError, match="bilinmeyen motora"):
        load_config(write(tmp_path, bad))


def test_unknown_default_model(tmp_path):
    bad = VALID.replace("default_model: smol", "default_model: yok")
    with pytest.raises(ConfigError, match="default_model"):
        load_config(write(tmp_path, bad))


def test_duplicate_hf_on_same_engine(tmp_path):
    bad = VALID.replace(
        'hf_id: "OpenGVLab/InternVL2_5-1B"',
        'hf_id: "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"',
    )
    with pytest.raises(ConfigError, match="hf_id"):
        load_config(write(tmp_path, bad))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="bulunamadı"):
        load_config(str(tmp_path / "yok.yaml"))
