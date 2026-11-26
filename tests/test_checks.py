"""The interviewer's reply checks (honesty > promise > repetition), the one-corrective-per-turn
rule, the code fallbacks, and the turn budget that can skip the corrective call. Every row
is synthetic."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from agents_automation.budget import FakeClock
from agents_automation.interview.checks import (
    HONEST_NO_PROMISE,
    PROMISE_NOTE,
    REPETITION_NOTE,
    detect_promises,
    first_finding,
    repeated_questions,
)
from agents_automation.interview.honesty import HONEST_NOT_SAVED
from agents_automation.interview.turn import CORRECTIVE_NEEDS_S, Interviewer, TurnResult
from agents_automation.llm import Message, TransportError
from agents_automation.llm.fakes import ScriptedModel
from agents_automation.pipeline.planner import Request
from agents_automation.pipeline.tools import InMemoryStore

from .helpers import linear, turn_json

PROMISES = [
    "I'll email you the summary tonight.",
    "I will send the step list to your manager.",
    "We'll remind you on Monday.",
    "I am going to schedule a follow-up call.",
    "I'll also share this with finance.",
    "Sure, I'll let you know when it is approved.",
    "I’ll get back to you tomorrow.",
    "I will follow up with the clerk.",
    "I'll set up a meeting with IT.",
    "If you like, I'll forward it to the team.",  # an offer is still a promise it cannot keep
    "No problem, I'll book the room.",
    "Thanks, I'll ping the reviewer.",
]

NOT_PROMISES = [
    "I won't send anything anywhere.",
    "I will not email anyone.",
    "I can't promise I'll send it.",
    "I don't think I'll remind anyone.",
    "Shall I send it to finance?",  # a question
    "Should I let you know when it is approved?",
    "You said the clerk will send the invoice.",  # someone else's action
    "Then finance will notify the employee.",
    "So you'll email the receipt to the manager.",
    'The template says "We will send a confirmation".',  # quoted
    "I'll send it for review once we are done.",  # handing over the draft is real
    "I'll pass the draft on for approval.",
    "I'll send the draft to an administrator.",
    "Who sends the invoice to the customer?",
    "I'll ask one more question about the receipts.",
]


@pytest.mark.parametrize("text", PROMISES)
def test_promises_of_actions_outside_the_conversation_are_found(text: str) -> None:
    assert detect_promises(text) == [text]


@pytest.mark.parametrize("text", NOT_PROMISES)
def test_everything_else_is_not_a_promise(text: str) -> None:
    assert detect_promises(text) == []


def test_tables_do_not_overlap() -> None:
    assert not set(PROMISES) & set(NOT_PROMISES)


def test_only_the_promise_sentence_is_reported() -> None:
    reply = "Thanks, that helps. I'll email you the summary. Who approves the expense?"
    assert detect_promises(reply) == ["I'll email you the summary."]


@pytest.mark.parametrize(
    ("reply", "previous", "repeated"),
    [
        ("Who approves the expense?", "Thanks. Who approves the expense?", True),
        ("Thanks! So who approves the expense?", "Who approves the expense?", True),
        ("Who pays the expense?", "Who approves the expense?", False),
        ("Who approves the expense?", None, False),
        ("Who approves the expense?", "Noted, the manager approves.", False),
        ("Noted.", "Who approves the expense?", False),
    ],
)
def test_repeated_questions(reply: str, previous: str | None, repeated: bool) -> None:
    assert repeated_questions(reply, previous) is repeated


def test_the_checks_fire_in_priority_order_and_only_once() -> None:
    reply = "I've saved it. I'll email you a copy. Who approves the expense?"
    previous = "Who approves the expense?"
    finding = first_finding(reply, unsupported_claims=["I've saved it."], previous_reply=previous)
    assert finding is not None
    assert finding.check == "honesty"
    finding = first_finding(reply, unsupported_claims=[], previous_reply=previous)
    assert finding is not None
    assert finding.check == "promise"
    finding = first_finding(
        "Who approves the expense?", unsupported_claims=[], previous_reply=previous
    )
    assert finding is not None
    assert finding.check == "repetition"
    assert first_finding("Who pays?", unsupported_claims=[], previous_reply=previous) is None


# --- wired into the turn --------------------------------------------------------------

PREVIOUS = "Thanks. Who approves the expense?"


class _SlowModel(ScriptedModel):
    """Each call takes ``seconds`` on the fake clock."""

    def __init__(self, clock: FakeClock, seconds: float, script: Sequence[str | Exception]) -> None:
        super().__init__("test-family", script)
        self.clock, self.seconds = clock, seconds

    def complete(self, messages: Sequence[Message], *, timeout_s: float) -> str:
        self.clock.advance(self.seconds)
        return super().complete(messages, timeout_s=timeout_s)


def _turn(
    store: InMemoryStore,
    request: Request,
    *script: str | Exception,
    history: Sequence[Message] = (),
    interviewer: Interviewer | None = None,
) -> tuple[ScriptedModel, TurnResult]:
    model = ScriptedModel("test-family", list(script))
    result = (interviewer or Interviewer(model, store)).run_turn(request, list(history), "ok")
    return model, result


def test_a_promise_gets_one_correction_and_the_corrected_reply_is_used(
    store: InMemoryStore, turn_request: Request
) -> None:
    model, result = _turn(
        store,
        turn_request,
        turn_json("Got it. I'll email you a summary. Who pays the expense?"),
        turn_json("Got it. Who pays the expense?"),
    )
    assert (result.corrective, result.reply, result.model_calls) == (
        "promise",
        "Got it. Who pays the expense?",
        2,
    )
    assert result.promise_removed is False
    assert result.guard_fired is False
    assert model.calls[1][-1] == Message("user", PROMISE_NOTE)


def test_a_promise_that_survives_the_correction_is_removed_by_code(
    store: InMemoryStore, turn_request: Request
) -> None:
    stubborn = turn_json("I'll email you a summary. Who pays the expense?")
    _, result = _turn(store, turn_request, stubborn, stubborn)
    assert result.promise_removed is True
    assert result.reply == f"Who pays the expense? {HONEST_NO_PROMISE}"


def test_a_repeated_question_gets_one_correction(
    store: InMemoryStore, turn_request: Request
) -> None:
    history = [Message("user", "hi"), Message("assistant", PREVIOUS)]
    model, result = _turn(
        store,
        turn_request,
        turn_json("Who approves the expense?"),
        turn_json("And who pays it in the end?"),
        history=history,
    )
    assert (result.corrective, result.reply) == ("repetition", "And who pays it in the end?")
    assert model.calls[1][-1] == Message("user", REPETITION_NOTE)


def test_a_repeated_question_that_survives_is_only_reported(
    store: InMemoryStore, turn_request: Request
) -> None:
    history = [Message("user", "hi"), Message("assistant", PREVIOUS)]
    again = turn_json("Who approves the expense?")
    _, result = _turn(store, turn_request, again, again, history=history)
    assert (result.corrective, result.reply, result.model_calls) == (
        "repetition",
        "Who approves the expense?",
        2,
    )


def test_honesty_outranks_a_promise_and_there_is_still_only_one_correction(
    store: InMemoryStore, turn_request: Request
) -> None:
    both = turn_json("I've saved the draft. I'll email you a copy.")
    model, result = _turn(store, turn_request, both, both)
    assert result.corrective == "honesty"
    assert result.model_calls == 2
    assert len(model.calls) == 2
    # both code fallbacks then apply: the promise and the claim are gone
    assert result.reply == f"{HONEST_NO_PROMISE} {HONEST_NOT_SAVED}"
    assert result.promise_removed is True
    assert result.fallback_used is True


def test_a_correction_that_introduces_a_promise_is_still_cleaned(
    store: InMemoryStore, turn_request: Request
) -> None:
    _, result = _turn(
        store,
        turn_request,
        turn_json("I've saved the draft."),
        turn_json("Sorry. I'll send you the draft by email. Who pays?"),
    )
    assert result.reply == f"Sorry. Who pays? {HONEST_NO_PROMISE}"


def test_a_corrective_outage_falls_back_to_the_code_fallback(
    store: InMemoryStore, turn_request: Request
) -> None:
    _, result = _turn(
        store, turn_request, turn_json("I'll email you the summary."), TransportError("down")
    )
    assert result.reply == HONEST_NO_PROMISE
    assert result.model_calls == 2


def test_a_spent_turn_budget_skips_the_corrective_and_the_fallbacks_still_apply(
    store: InMemoryStore, turn_request: Request
) -> None:
    clock = FakeClock()
    model = _SlowModel(clock, 50.0, [turn_json("I've saved it. I'll email you a copy.")])
    interviewer = Interviewer(model, store, budget_s=60.0, clock=clock)
    result = interviewer.run_turn(turn_request, [], "ok")
    assert result.model_calls == 1  # the corrective was not made
    assert len(model.calls) == 1
    assert [str(s) for s in result.skips] == [
        f"corrective skipped: needs {CORRECTIVE_NEEDS_S:g}s, 10s left in this turn"
    ]
    assert result.reply == f"{HONEST_NO_PROMISE} {HONEST_NOT_SAVED}"
    assert result.corrective == "honesty"


def test_with_time_left_the_corrective_runs(store: InMemoryStore, turn_request: Request) -> None:
    clock = FakeClock()
    script = [turn_json("I'll email you a copy."), turn_json("Who pays?")]
    model = _SlowModel(clock, 10.0, script)
    result = Interviewer(model, store, budget_s=60.0, clock=clock).run_turn(turn_request, [], "ok")
    assert (result.reply, result.model_calls, result.skips) == ("Who pays?", 2, ())


def test_a_clean_reply_needs_no_check_and_no_extra_call(
    store: InMemoryStore, turn_request: Request
) -> None:
    model, result = _turn(store, turn_request, turn_json("Who pays the expense?", save=linear()))
    assert (result.corrective, result.model_calls) == (None, 1)
    assert len(model.calls) == 1
