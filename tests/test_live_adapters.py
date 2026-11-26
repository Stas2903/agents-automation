"""Live adapters, tested with fake clients: no SDK install and no network needed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agents_automation.llm import ChatModel, Message, TransportError
from agents_automation.llm.anthropic_live import AnthropicChat
from agents_automation.llm.openai_live import OpenAICompatibleChat

MSGS = (Message("system", "S1"), Message("system", "S2"), Message("user", "U"))


class FakeAnthropic:
    def __init__(self, fail: bool = False) -> None:
        self.kwargs: dict[str, Any] = {}
        self.fail = fail
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.fail:
            raise ConnectionError("network down")
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Hel"),
                SimpleNamespace(type="text", text="lo"),
            ]
        )


class FakeOpenAI:
    def __init__(self, fail: bool = False, content: str | None = "Hi") -> None:
        self.kwargs: dict[str, Any] = {}
        self.fail = fail
        self.content = content
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.fail:
            raise TimeoutError("slow")
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_anthropic_adapter_maps_messages_and_joins_text() -> None:
    client = FakeAnthropic()
    model = AnthropicChat(model="m-1", client=client)
    assert isinstance(model, ChatModel)
    assert model.family == "anthropic"
    assert model.complete(MSGS, timeout_s=7) == "Hello"
    assert client.kwargs["system"] == "S1\n\nS2"
    assert client.kwargs["messages"] == [{"role": "user", "content": "U"}]
    assert (client.kwargs["model"], client.kwargs["timeout"]) == ("m-1", 7)


def test_anthropic_failures_become_transport_errors() -> None:
    with pytest.raises(TransportError, match="ConnectionError"):
        AnthropicChat(client=FakeAnthropic(fail=True)).complete(MSGS, timeout_s=1)


def test_anthropic_requires_a_key_without_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicChat()


def test_openai_adapter_passes_messages_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_FAMILY", "some-family")
    client = FakeOpenAI()
    model = OpenAICompatibleChat(model="m-2", client=client)
    assert model.family == "some-family"
    assert model.complete(MSGS, timeout_s=3) == "Hi"
    assert client.kwargs["messages"][0] == {"role": "system", "content": "S1"}
    assert len(client.kwargs["messages"]) == 3


def test_openai_empty_content_is_an_empty_string() -> None:
    model = OpenAICompatibleChat(model="m", family="f", client=FakeOpenAI(content=None))
    assert model.complete(MSGS, timeout_s=1) == ""


def test_openai_failures_become_transport_errors() -> None:
    model = OpenAICompatibleChat(model="m", family="f", client=FakeOpenAI(fail=True))
    with pytest.raises(TransportError, match="TimeoutError"):
        model.complete(MSGS, timeout_s=1)


def test_openai_requires_a_model_and_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        OpenAICompatibleChat()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICompatibleChat(model="m")


class FakeEmbeddings:
    def __init__(self, fail: bool = False, extra: int = 0) -> None:
        self.kwargs: dict[str, Any] = {}
        self.fail = fail
        self.extra = extra
        self.embeddings = self

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.fail:
            raise ConnectionError("network down")
        n = len(kwargs["input"]) + self.extra
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, i]) for i in range(n)])


def test_the_embedder_sends_all_texts_in_one_call_and_returns_tuples() -> None:
    from agents_automation.llm import Embedder
    from agents_automation.llm.openai_live import OpenAICompatibleEmbedder

    client = FakeEmbeddings()
    embedder = OpenAICompatibleEmbedder(model="embed-x", client=client)
    assert isinstance(embedder, Embedder)
    assert embedder.model_id == "embed-x"
    assert embedder.embed(["a", "b"], timeout_s=7) == [(0.5, 0.0), (0.5, 1.0)]
    assert client.kwargs == {"model": "embed-x", "input": ["a", "b"], "timeout": 7}


@pytest.mark.parametrize("client", [FakeEmbeddings(fail=True), FakeEmbeddings(extra=1)])
def test_embedder_failures_are_transport_errors(client: Any) -> None:
    from agents_automation.llm.openai_live import OpenAICompatibleEmbedder

    with pytest.raises(TransportError):
        OpenAICompatibleEmbedder(model="embed-x", client=client).embed(["a"], timeout_s=1)


def test_the_embedder_needs_its_model_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents_automation.llm.openai_live import OpenAICompatibleEmbedder

    monkeypatch.delenv("OPENAI_EMBED_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_EMBED_MODEL"):
        OpenAICompatibleEmbedder()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICompatibleEmbedder(model="embed-x")
