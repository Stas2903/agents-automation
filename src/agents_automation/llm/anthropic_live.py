"""Optional live adapter for Anthropic models. Needs the ``live`` extra and ``ANTHROPIC_API_KEY``.

The SDK is imported lazily, so the package (and every test) works without it installed.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any

from agents_automation.llm.seam import Message, TransportError

DEFAULT_MODEL = "claude-sonnet-5-5"


class AnthropicChat:
    family = "anthropic"

    def __init__(
        self, *, model: str | None = None, max_tokens: int = 2048, client: Any = None
    ) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.max_tokens = max_tokens
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set; live mode is unavailable")
            sdk = importlib.import_module("anthropic")
            client = sdk.Anthropic()
        self._client = client

    def complete(self, messages: Sequence[Message], *, timeout_s: float) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=turns,
                timeout=timeout_s,
            )
            texts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
        except Exception as exc:  # any SDK or response-shape failure counts as transport
            raise TransportError(f"anthropic call failed: {type(exc).__name__}") from exc
        return "".join(texts)
