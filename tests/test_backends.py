import subprocess

import pytest

from app.backends import DockerController, ModelManager, ModelStatus
from app.config import (
    AppConfig,
    EngineConfig,
    FrameConfig,
    ModelConfig,
    RequestConfig,
)
from app.vqa_client import MockVqaClient


def make_cfg(mode="switch"):
    engines = {
        "vllm": EngineConfig("vllm", "http://localhost:8000/v1"),
        "sglang": EngineConfig("sglang", "http://localhost:30000/v1"),
    }
    models = {
        "smol": ModelConfig("smol", "SmolVLM", "vllm", "hf/smol", "vllm-smol"),
        "intern": ModelConfig("intern", "InternVL", "vllm", "hf/intern", "vllm-intern"),
        "llava": ModelConfig("llava", "LLaVA", "sglang", "hf/llava", "sglang-llava"),
    }
    return AppConfig(
        mode=mode, default_model="smol",
        switch_timeout_s=10, health_poll_interval_s=0.01, status_refresh_s=5,
        frame=FrameConfig(), request=RequestConfig(), engines=engines, models=models,
    )


class FakeDocker:
    """DockerController arayüzünü taklit eder; çağrıları kaydeder."""

    def __init__(self, running=None):
        self.running = set(running or [])
        self.calls = []
        self.fail_start = False

    def available(self):
        return True

    def start(self, c):
        self.calls.append(("start", c))
        if self.fail_start:
            raise RuntimeError("start hatasi")
        self.running.add(c)

    def stop(self, c):
        self.calls.append(("stop", c))
        self.running.discard(c)

    def container_state(self, c):
        return "running" if c in self.running else "exited"


def wire_served(mgr, cfg, fake, healthy=True):
    """served_model'i sahtele: motorda çalışan container'ın hf_id'sini döndürür."""

    def served(engine_key, timeout_s=2.0):
        if not healthy:
            return None
        for m in cfg.models.values():
            if m.engine == engine_key and m.container in fake.running:
                return m.hf_id
        return None

    mgr.served_model = served


def drain(gen):
    return list(gen)


def test_switch_mode_stops_all_others():
    cfg = make_cfg("switch")
    fake = FakeDocker(running={"sglang-llava"})
    mgr = ModelManager(cfg, docker=fake)
    wire_served(mgr, cfg, fake)

    events = drain(mgr.activate("smol"))
    assert ("stop", "sglang-llava") in fake.calls          # diğer MOTOR da durdu
    assert ("start", "vllm-smol") in fake.calls
    assert events[-1].done and events[-1].ok


def test_dual_mode_stops_only_same_engine():
    cfg = make_cfg("dual")
    fake = FakeDocker(running={"vllm-smol", "sglang-llava"})
    mgr = ModelManager(cfg, docker=fake)
    wire_served(mgr, cfg, fake)

    events = drain(mgr.activate("intern"))
    assert ("stop", "vllm-smol") in fake.calls             # aynı motor durdu
    assert ("stop", "sglang-llava") not in fake.calls      # diğer motor DOKUNULMADI
    assert ("start", "vllm-intern") in fake.calls
    assert events[-1].done and events[-1].ok
    assert "sglang-llava" in fake.running                  # SGLang slotu hâlâ ayakta


def test_activate_already_active_is_noop():
    cfg = make_cfg("dual")
    fake = FakeDocker(running={"vllm-smol"})
    mgr = ModelManager(cfg, docker=fake)
    wire_served(mgr, cfg, fake)

    events = drain(mgr.activate("smol"))
    assert fake.calls == []                                # hiç docker çağrısı yok
    assert events[-1].done and events[-1].ok


def test_activate_timeout():
    cfg = make_cfg("switch")
    fake = FakeDocker()
    mgr = ModelManager(cfg, docker=fake)
    wire_served(mgr, cfg, fake, healthy=False)             # asla servis etmez

    events = drain(mgr.activate("smol"))
    assert events[-1].done and not events[-1].ok
    assert "Zaman aşımı" in events[-1].message


def test_activate_docker_start_failure():
    cfg = make_cfg("switch")
    fake = FakeDocker()
    fake.fail_start = True
    mgr = ModelManager(cfg, docker=fake)
    wire_served(mgr, cfg, fake, healthy=False)

    events = drain(mgr.activate("smol"))
    assert events[-1].done and not events[-1].ok
    assert not mgr.switching                               # kilit her durumda bırakılmalı


def test_activate_crash_detected():
    cfg = make_cfg("switch")
    fake = FakeDocker()
    orig_start = fake.start

    def crashy_start(c):
        orig_start(c)
        fake.running.discard(c)                            # başlar başlamaz ölür

    fake.start = crashy_start
    mgr = ModelManager(cfg, docker=fake)
    wire_served(mgr, cfg, fake, healthy=False)

    events = drain(mgr.activate("smol"))
    assert events[-1].done and not events[-1].ok
    assert "crash" in events[-1].message


def test_statuses_active_starting_stopped():
    cfg = make_cfg("dual")
    fake = FakeDocker(running={"vllm-smol", "sglang-llava"})
    mgr = ModelManager(cfg, docker=fake)

    def served(engine_key, timeout_s=2.0):
        return "hf/smol" if engine_key == "vllm" else None  # sglang: ayakta ama hazır değil

    mgr.served_model = served
    st = mgr.statuses()
    assert st["smol"] == ModelStatus.ACTIVE
    assert st["intern"] == ModelStatus.STOPPED
    assert st["llava"] == ModelStatus.STARTING


def test_mock_mode_statuses_and_client():
    cfg = make_cfg(mode="mock")
    mgr = ModelManager(cfg, docker=FakeDocker())
    assert all(s == ModelStatus.ACTIVE for s in mgr.statuses().values())
    assert isinstance(mgr.client_for("smol"), MockVqaClient)


def test_docker_controller_uses_runner():
    calls = []

    def runner(cmd, timeout):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    dc = DockerController(runner=runner)
    assert dc.container_state("x") == "running"
    assert calls[0][:2] == ["docker", "inspect"]
