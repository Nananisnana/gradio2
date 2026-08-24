"""OpenAI-uyumlu VQA istemcisi + Türkçe hata eşleme + mock istemci.

Aynı çağrı şekli her motora gider; fark yalnız engine base_url'ü ve model hf_id'si.
Modelin gerçekten aktif olduğunu (served-id eşleşmesi) ModelManager istek öncesi
doğrular — buradaki istemci hf_id'yi doğrudan gönderir.

ask() bir VqaResult döndürür: sistem prompt'undaki grounding sözleşmesi gereği
model JSON {"answer", "boxes"} üretir; parse_grounding_result bozuk çıktıda
tüm metni cevap sayar (boxes=[]) — kutulama her zaman opsiyonel kalır.
"""
from __future__ import annotations

import time
from typing import Protocol

import httpx
import openai
from openai import OpenAI

from .config import EngineConfig, ModelConfig, RequestConfig
from .grounding import SYSTEM_PROMPT, Box, VqaResult, parse_grounding_result
from .ui_text import TR


class VqaError(Exception):
    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class VqaClient(Protocol):
    def ask(self, prompt: str, image_data_uri: str) -> VqaResult: ...


class OpenAIVqaClient:
    def __init__(self, engine: EngineConfig, model: ModelConfig, req: RequestConfig):
        self._model = model
        self._req = req
        self._client = OpenAI(
            base_url=engine.base_url,
            api_key="EMPTY",
            timeout=httpx.Timeout(
                connect=5.0, read=req.read_timeout_s, write=10.0, pool=5.0
            ),
            max_retries=0,  # 8GB paylaşımlı cihazda yinelenen istek OOM riski
        )

    def ask(self, prompt: str, image_data_uri: str) -> VqaResult:
        try:
            resp = self._client.chat.completions.create(
                model=self._model.hf_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                        ],
                    },
                ],
                max_tokens=self._req.max_tokens,
                temperature=self._req.temperature,
            )
        except openai.APITimeoutError as e:
            raise VqaError(TR["err_timeout"], str(e)) from e
        except openai.APIConnectionError as e:
            raise VqaError(TR["err_conn"], str(e)) from e
        except openai.APIStatusError as e:
            body = ""
            try:
                body = e.response.text[:200]
            except Exception:
                pass
            if e.status_code == 404:
                raise VqaError(
                    TR["err_model_404"].format(model=self._model.hf_id), body
                ) from e
            if e.status_code == 400:
                raise VqaError(TR["err_bad_request"].format(detail=body), body) from e
            raise VqaError(
                TR["err_server"].format(code=e.status_code, container=self._model.container),
                body,
            ) from e
        except Exception as e:
            raise VqaError(TR["err_unexpected"].format(detail=str(e)[:200]), str(e)) from e

        if not resp.choices:
            raise VqaError(TR["err_empty_answer"])
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            raise VqaError(TR["err_empty_answer"])
        return parse_grounding_result(raw)


# mock'ta kutu üretimini tetikleyen anahtar kelimeler (lokalizasyon imasi)
_MOCK_BOX_KEYWORDS = ("göster", "goster", "var mı", "var mi", "nerede", "kutula")


class MockVqaClient:
    def ask(self, prompt: str, image_data_uri: str) -> VqaResult:
        time.sleep(1.0)
        answer = TR["mock_answer"].format(prompt=prompt, kb=len(image_data_uri) // 1024)
        boxes: list[Box] = []
        if any(k in prompt.casefold() for k in _MOCK_BOX_KEYWORDS):
            boxes = [
                Box(label="mock-nesne", x1=0.15, y1=0.20, x2=0.55, y2=0.75),
                Box(label="mock-nesne", x1=0.60, y1=0.30, x2=0.90, y2=0.65),
            ]
        return VqaResult(answer=answer, boxes=boxes, raw_response="(mock)")
