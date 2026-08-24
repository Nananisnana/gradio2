"""Model yaşam döngüsü: motor sağlık kontrolü, model durum rozetleri,
model aktivasyonu (docker start/stop orkestrasyonu).

Model Registry / Engine Registry ayrımı:
- Motor başına aynı anda TEK model servis edilir; her modelin kendi container'ı
  vardır, aynı motorun modelleri portu paylaşır.
- `dual` modda motor başına bir slot aktif kalabilir (vLLM'de bir model + SGLang'de
  bir model); aynı motor içinde model değişimi o motorun container'ını değiştirir.
- `switch` modda toplamda tek container ayakta kalır.

Sözleşme (jetson/setup.sh ile paylaşılan):
- Container'lar sabit adlarla önceden create edilmiştir; bu modül yalnız
  `docker start / stop -t 20 / inspect / version` çağırır (sudo'suz).
- Hazır olma tanımı: GET {base_url}/models -> HTTP 200; aktiflik tanımı: bu
  cevaptaki served-id, modelin hf_id'siyle eşleşir (yanlış modele istek gitmez).
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

import httpx

from .config import AppConfig
from .ui_text import TR
from .vqa_client import MockVqaClient, OpenAIVqaClient, VqaClient


class ModelStatus(str, Enum):
    ACTIVE = "active"        # motoru sağlıklı VE bu modeli servis ediyor
    STARTING = "starting"    # container ayakta ama henüz servis etmiyor
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class SwitchEvent:
    message: str
    done: bool = False
    ok: bool = False


# Testlerde sahte runner enjekte edilebilsin diye subprocess çağrısı tek noktada
Runner = Callable[[list[str], float], subprocess.CompletedProcess]


def _default_runner(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


_AVAILABLE_RETRY_TTL_S = 30.0


class DockerController:
    def __init__(self, docker_bin: str = "docker", runner: Runner = _default_runner):
        self._bin = docker_bin
        self._run = runner
        self._available: bool | None = None
        self._failed_at: float = 0.0

    def available(self) -> bool:
        # olumlu sonuc kalici cache'lenir; olumsuz sonuc TTL ile yeniden denenir
        # (docker daemon'i sonradan ayaga kalkabilir)
        if self._available:
            return True
        if time.monotonic() - self._failed_at < _AVAILABLE_RETRY_TTL_S:
            return False
        result = False
        if shutil.which(self._bin) is not None:
            try:
                cp = self._run([self._bin, "version", "--format", "{{.Server.Version}}"], 10)
                result = cp.returncode == 0
            except Exception:
                result = False
        if result:
            self._available = True
        else:
            self._failed_at = time.monotonic()
        return result

    def start(self, container: str) -> None:
        cp = self._run([self._bin, "start", container], 30)
        if cp.returncode != 0:
            raise RuntimeError((cp.stderr or cp.stdout or "").strip())

    def stop(self, container: str) -> None:
        cp = self._run([self._bin, "stop", "-t", "20", container], 45)
        if cp.returncode != 0:
            raise RuntimeError((cp.stderr or cp.stdout or "").strip())

    def container_state(self, container: str) -> str:
        try:
            cp = self._run(
                [self._bin, "inspect", "-f", "{{.State.Status}}", container], 10
            )
        except Exception:
            return "unknown"
        if cp.returncode != 0:
            return "missing"
        return (cp.stdout or "").strip() or "unknown"


class ModelManager:
    def __init__(self, cfg: AppConfig, docker: DockerController | None = None):
        self._cfg = cfg
        self._docker = docker or DockerController()
        self._lock = threading.Lock()
        self._clients: dict[str, VqaClient] = {}
        self._cached_statuses: dict[str, ModelStatus] = {
            key: ModelStatus.UNKNOWN for key in cfg.models
        }

    @property
    def switching(self) -> bool:
        return self._lock.locked()

    def served_model(self, engine_key: str, timeout_s: float = 2.0) -> str | None:
        """Motor sağlıklıysa şu an servis ettiği model id'sini döndürür, değilse None."""
        base_url = self._cfg.engines[engine_key].base_url
        try:
            r = httpx.get(f"{base_url}/models", timeout=timeout_s)
            if r.status_code != 200:
                return None
            data = r.json().get("data") or []
            return data[0].get("id") if data else None
        except Exception:
            return None

    def is_active(self, model_key: str) -> bool:
        model = self._cfg.models[model_key]
        return self.served_model(model.engine) == model.hf_id

    def statuses(self) -> dict[str, ModelStatus]:
        if self._cfg.mode == "mock":
            return {key: ModelStatus.ACTIVE for key in self._cfg.models}
        if self.switching:
            return dict(self._cached_statuses)
        served = {ek: self.served_model(ek) for ek in self._cfg.engines}
        result: dict[str, ModelStatus] = {}
        for key, model in self._cfg.models.items():
            if served.get(model.engine) == model.hf_id:
                result[key] = ModelStatus.ACTIVE
            elif (
                self._docker.available()
                and self._docker.container_state(model.container) == "running"
            ):
                result[key] = ModelStatus.STARTING
            else:
                result[key] = ModelStatus.STOPPED
        # bu hesap sürerken bir aktivasyon başlamış olabilir; onun tohumladığı
        # cache'i bayat değerlerle ezme
        if not self.switching:
            self._cached_statuses = dict(result)
        return result

    def client_for(self, model_key: str) -> VqaClient:
        if self._cfg.mode == "mock":
            return MockVqaClient()
        if model_key not in self._clients:
            model = self._cfg.models[model_key]
            engine = self._cfg.engines[model.engine]
            self._clients[model_key] = OpenAIVqaClient(engine, model, self._cfg.request)
        return self._clients[model_key]

    def activate(self, model_key: str) -> Iterator[SwitchEvent]:
        """Hedef modeli ayağa kaldırır.

        dual  -> yalnız AYNI MOTORUN diğer container'ları durdurulur
                 (diğer motorun slotu dokunulmaz kalır)
        switch-> TÜM model container'ları durdurulur, yalnız hedef kalır

        Asla exception fırlatmaz — her sonuç bir SwitchEvent olarak akar.
        """
        target = self._cfg.models[model_key]
        if not self._lock.acquire(blocking=False):
            yield SwitchEvent(TR["switch_already"], done=True, ok=False)
            return
        try:
            if self.served_model(target.engine) == target.hf_id:
                yield SwitchEvent(
                    TR["switch_ready"].format(name=target.display_name), done=True, ok=True
                )
                return
            if not self._docker.available():
                yield SwitchEvent(TR["switch_no_docker"], done=True, ok=False)
                return

            if self._cfg.mode == "switch":
                to_stop = [m for k, m in self._cfg.models.items() if k != model_key]
            else:  # dual: yalnız aynı motorun diğer modelleri
                to_stop = [
                    m for m in self._cfg.models_on_engine(target.engine)
                    if m.key != model_key
                ]

            self._cached_statuses = dict(self._cached_statuses)
            self._cached_statuses[model_key] = ModelStatus.STARTING
            for m in to_stop:
                self._cached_statuses[m.key] = ModelStatus.STOPPED

            try:
                for m in to_stop:
                    if self._docker.container_state(m.container) == "running":
                        yield SwitchEvent(
                            TR["switch_stopping"].format(name=m.display_name)
                        )
                        self._docker.stop(m.container)

                if self._docker.container_state(target.container) != "running":
                    yield SwitchEvent(
                        TR["switch_starting"].format(name=target.display_name)
                    )
                    self._docker.start(target.container)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                yield SwitchEvent(
                    TR["switch_docker_error"].format(detail=str(e)[:200]),
                    done=True, ok=False,
                )
                return

            timeout = self._cfg.switch_timeout_s
            interval = self._cfg.health_poll_interval_s
            start_t = time.monotonic()
            while True:
                elapsed = int(time.monotonic() - start_t)
                if self.served_model(target.engine) == target.hf_id:
                    self._cached_statuses[model_key] = ModelStatus.ACTIVE
                    yield SwitchEvent(
                        TR["switch_ready"].format(name=target.display_name),
                        done=True, ok=True,
                    )
                    return
                if self._docker.container_state(target.container) != "running":
                    self._cached_statuses[model_key] = ModelStatus.STOPPED
                    yield SwitchEvent(
                        TR["switch_crashed"].format(
                            name=target.display_name, container=target.container
                        ),
                        done=True, ok=False,
                    )
                    return
                if elapsed >= timeout:
                    self._cached_statuses[model_key] = ModelStatus.STARTING
                    yield SwitchEvent(
                        TR["switch_timeout"].format(
                            name=target.display_name,
                            timeout=timeout,
                            container=target.container,
                        ),
                        done=True, ok=False,
                    )
                    return
                yield SwitchEvent(
                    TR["switch_loading"].format(elapsed=elapsed, timeout=timeout)
                )
                time.sleep(interval)
        finally:
            self._lock.release()
