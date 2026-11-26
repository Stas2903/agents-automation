"""Optional live adapter for any OpenAI-compatible chat endpoint.

Configured from ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``OPENAI_MODEL`` and ``OPENAI_FAMILY``.
``OPENAI_FAMILY`` names the model family actually served, so the eval harness can refuse a
judge from the same family as the interviewer.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any

from agents_automation.llm.seam import Message, TransportError


class OpenAICompatibleChat:
    def __init__(
        self,
        *,
        model: str | None = None,
        family: str | None = None,
        client: Any = None,
    ) -> None:
        resolved_model = model or os.environ.get("OPENAI_MODEL")
        if not resolved_model:
            raise RuntimeError("OPENAI_MODEL is not set; live mode is unavailable")
        self.model = resolved_model
        self._family = family or os.environ.get("OPENAI_FAMILY") or "openai"
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set; live mode is unavailable")
            sdk = importlib.import_module("openai")
            client = sdk.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        self._client = client

    @property
    def family(self) -> str:
        return self._family

    def complete(self, messages: Sequence[Message], *, timeout_s: float) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=payload, timeout=timeout_s
            )
            content = response.choices[0].message.content
        except Exception as exc:  # any SDK or response-shape failure counts as transport
            raise TransportError(f"openai-compatible call failed: {type(exc).__name__}") from exc
        return content or ""


class OpenAICompatibleEmbedder:
    """Embeddings from any OpenAI-compatible endpoint (``OPENAI_EMBED_MODEL``). Place it with
    the clearance of where that endpoint runs (see ``llm.boundary``)."""

    def __init__(self, *, model: str | None = None, client: Any = None) -> None:
        resolved_model = model or os.environ.get("OPENAI_EMBED_MODEL")
        if not resolved_model:
            raise RuntimeError("OPENAI_EMBED_MODEL is not set; live embeddings are unavailable")
        self._model = resolved_model
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set; live mode is unavailable")
            sdk = importlib.import_module("openai")
            client = sdk.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str], *, timeout_s: float) -> list[tuple[float, ...]]:
        try:
            response = self._client.embeddings.create(
                model=self._model, input=list(texts), timeout=timeout_s
            )
            vectors = [tuple(float(v) for v in item.embedding) for item in response.data]
        except Exception as exc:  # any SDK or response-shape failure counts as transport
            raise TransportError(f"embedding call failed: {type(exc).__name__}") from exc
        if len(vectors) != len(texts):
            raise TransportError("embedding call returned a different number of vectors")
        return vectors
