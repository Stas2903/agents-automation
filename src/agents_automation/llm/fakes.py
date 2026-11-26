"""Deterministic fakes for the model seam. Used by the demo, every test and CI."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence

from agents_automation.llm.seam import Message
from agents_automation.llm.text import content_tokens


class ScriptExhaustedError(RuntimeError):
    """A scripted model was called more times than its script allows."""


class ScriptedModel:
    """Returns pre-written replies in order. An ``Exception`` entry is raised instead of returned.

    Every call is recorded so tests can assert exactly what the model was shown.
    """

    def __init__(self, family: str, script: Sequence[str | Exception]) -> None:
        self._family = family
        self._script = list(script)
        self.calls: list[tuple[Message, ...]] = []

    @property
    def family(self) -> str:
        return self._family

    def complete(self, messages: Sequence[Message], *, timeout_s: float) -> str:
        if len(self.calls) >= len(self._script):
            raise ScriptExhaustedError(
                f"scripted model ({self._family}) called {len(self.calls) + 1} times, "
                f"script has {len(self._script)} entries"
            )
        item = self._script[len(self.calls)]
        self.calls.append(tuple(messages))
        if isinstance(item, Exception):
            raise item
        return item


class HashingEmbedder:
    """A deterministic stand-in for an embedding model: hashed bag of content tokens, L2
    normalised. Texts that share words get a positive cosine; unrelated texts get about 0."""

    def __init__(self, dim: int = 64, model_id: str = "hashing-64-v1") -> None:
        self.dim = dim
        self._model_id = model_id
        self.calls: list[tuple[str, ...]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, texts: Sequence[str], *, timeout_s: float) -> list[tuple[float, ...]]:
        self.calls.append(tuple(texts))
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dim
        for tok in content_tokens(text):
            digest = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return tuple(v / norm for v in vec) if norm else tuple(vec)


class RuleModel:
    """Computes its reply from the messages with a pure function (a deterministic stand-in)."""

    def __init__(self, family: str, rule: Callable[[Sequence[Message]], str]) -> None:
        self._family = family
        self._rule = rule
        self.calls: list[tuple[Message, ...]] = []

    @property
    def family(self) -> str:
        return self._family

    def complete(self, messages: Sequence[Message], *, timeout_s: float) -> str:
        self.calls.append(tuple(messages))
        return self._rule(messages)
