"""Shared fixtures. Outbound network use is blocked for every test: a live call fails loudly."""

from __future__ import annotations

import socket

import pytest

from agents_automation.pipeline.planner import Request
from agents_automation.pipeline.tools import InMemoryStore


class NetworkBlockedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def guard(*_args: object, **_kwargs: object) -> None:
        raise NetworkBlockedError("tests must not touch the network")

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture
def store() -> InMemoryStore:
    s = InMemoryStore()
    s.create_session("s-1", frozenset({"u-1"}))
    s.create_session("s-2", frozenset({"u-2"}))
    return s


@pytest.fixture
def turn_request() -> Request:
    return Request(
        surface="interview_chat", intent="interview_turn", session_id="s-1", user_id="u-1"
    )
