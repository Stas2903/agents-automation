"""The model seam: a tiny protocol every real or fake model implements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class TransportError(Exception):
    """The model could not be reached or did not answer in time.

    Callers must treat this differently from a bad answer: a transport failure says nothing
    about the content (the eval harness turns it into VOID, never into FAIL).
    """


@runtime_checkable
class ChatModel(Protocol):
    """A chat model. ``family`` names the model family (used to enforce cross-family judging)."""

    @property
    def family(self) -> str: ...

    def complete(self, messages: Sequence[Message], *, timeout_s: float) -> str: ...


Vector = tuple[float, ...]


@runtime_checkable
class Embedder(Protocol):
    """An embedding model. ``model_id`` is stored with every vector, so a model change is
    detected and the affected chunks are embedded again."""

    @property
    def model_id(self) -> str: ...

    def embed(self, texts: Sequence[str], *, timeout_s: float) -> list[Vector]: ...
