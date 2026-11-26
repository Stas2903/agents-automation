"""Run a whole interview session: user turns -> interviewer turns -> finalize."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from agents_automation.finalize import (
    FinalizeResult,
    Screener,
    Validator,
    finalize,
)
from agents_automation.injection.screen import screen
from agents_automation.interview.turn import Interviewer, TurnResult
from agents_automation.interview.validator import Repairer, validate_with_repair
from agents_automation.llm import ChatModel, Message
from agents_automation.pipeline.planner import Request


@dataclass(frozen=True)
class Exchange:
    user: str
    result: TurnResult


@dataclass(frozen=True)
class SessionTranscript:
    exchanges: tuple[Exchange, ...]
    final: FinalizeResult


# Returns the next user message given the exchanges so far, or None when the user is done.
NextUserMessage = Callable[[Sequence[Exchange]], str | None]


def scripted_user(turns: Sequence[str]) -> NextUserMessage:
    def next_message(done: Sequence[Exchange]) -> str | None:
        return turns[len(done)] if len(done) < len(turns) else None

    return next_message


def run_session(
    *,
    session_id: str,
    user_id: str,
    next_user_message: NextUserMessage,
    interviewer: Interviewer,
    repairer: Repairer,
    judge_model: ChatModel | None,
    outage_turns: Collection[int] = (),
    max_turns: int = 12,
    screener: Screener = screen,
    validator: Validator = validate_with_repair,
) -> SessionTranscript:
    """``outage_turns`` are 1-based turn numbers during which every store write fails."""
    store = interviewer.store
    if session_id not in store.sessions:
        store.create_session(session_id, frozenset({user_id}))
    turn_request = Request("interview_chat", "interview_turn", session_id, user_id)
    exchanges: list[Exchange] = []
    history: list[Message] = []
    while len(exchanges) < max_turns:
        text = next_user_message(exchanges)
        if text is None:
            break
        store.fail_writes = (len(exchanges) + 1) in outage_turns
        try:
            result = interviewer.run_turn(turn_request, history, text)
        finally:
            store.fail_writes = False
        exchanges.append(Exchange(text, result))
        history += [Message("user", text), Message("assistant", result.reply)]

    final = finalize(
        Request("interview_chat", "finalize", session_id, user_id),
        store,
        repairer=repairer,
        judge_model=judge_model,
        screener=screener,
        validator=validator,
    )
    return SessionTranscript(tuple(exchanges), final)
