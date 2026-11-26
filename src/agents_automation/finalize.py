"""The finalize chain: read the saved draft -> validate/repair -> screen -> mint status -> submit.

Status precedence: invalid > held_for_review > delivered_flagged > delivered. What the
employee is told is chosen by code from the status, so a held record is always announced as
needing human review, whatever the model would have said.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from agents_automation.injection.screen import Screening, screen
from agents_automation.interview.validator import (
    Issue,
    Repairer,
    RepairResult,
    validate_with_repair,
)
from agents_automation.llm import ChatModel
from agents_automation.models import Status, StepList
from agents_automation.pipeline.executor import StepStatus, execute
from agents_automation.pipeline.planner import Request, make_plan
from agents_automation.pipeline.policy import ALLOWED, AuthTable, authorize
from agents_automation.pipeline.router import route
from agents_automation.pipeline.seal import bind_values, seal
from agents_automation.pipeline.tools import InMemoryStore

MESSAGE_SENT = "Thank you. Your workflow draft has been sent to an administrator for approval."
MESSAGE_HELD = (
    "Thank you. Before this draft can go any further it needs a human review, "
    "so it has been passed to a reviewer."
)
MESSAGE_NOT_SUBMITTED = "I could not send the draft just now, so nothing has been submitted yet."
MESSAGE_NOTHING_SAVED = "There is no saved draft yet, so there is nothing to send."

Screener = Callable[[str, ChatModel | None], Screening]
Validator = Callable[[StepList, Repairer], RepairResult]


class FinalizeRefusedError(Exception):
    pass


@dataclass(frozen=True)
class FinalizeResult:
    status: Status
    step_list: StepList | None
    issues: tuple[Issue, ...]
    repair_rounds: int
    screening: Screening | None
    submitted: bool
    queue: str | None
    display_text: str  # masked rendering: what a reviewer sees first (empty when invalid)
    user_message: str
    transport_failed: bool = False  # a model call needed here could not be made


def _invalid_message(issues: tuple[Issue, ...]) -> str:
    listed = "; ".join(i.message for i in issues[:3])
    return f"The step list is not complete yet ({listed}). Let's fill in the gaps."


def _invalid(
    message: str,
    sl: StepList | None,
    issues: tuple[Issue, ...],
    rounds: int,
    transport_failed: bool = False,
) -> FinalizeResult:
    return FinalizeResult(
        status=Status.INVALID,
        step_list=sl,
        issues=issues,
        repair_rounds=rounds,
        screening=None,
        submitted=False,
        queue=None,
        display_text="",
        user_message=message,
        transport_failed=transport_failed,
    )


def finalize(
    request: Request,
    store: InMemoryStore,
    *,
    repairer: Repairer,
    judge_model: ChatModel | None,
    screener: Screener = screen,
    validator: Validator = validate_with_repair,
    table: AuthTable = ALLOWED,
) -> FinalizeResult:
    sealed = seal(make_plan(route(request.surface, "interviewer"), request))
    decision = authorize(sealed, table)
    if not decision.allowed:
        raise FinalizeRefusedError(decision.reason)
    tools = store.tools()

    read = execute(sealed, tools, only={"read"}, table=table).outcome("read")
    raw = read.output.get("draft", "") if read.status is StepStatus.OK else ""
    revision = read.output.get("revision", "")
    already_held = read.output.get("held") == "true"
    if not raw:
        return _invalid(MESSAGE_NOTHING_SAVED, None, (), 0)
    try:
        saved = StepList.model_validate_json(raw)
    except ValidationError:
        return _invalid(MESSAGE_NOTHING_SAVED, None, (), 0)

    repaired = validator(saved, repairer)
    if repaired.issues:
        # Hand back the saved list (what the store holds), not an unscreened repair candidate.
        return _invalid(
            _invalid_message(repaired.issues),
            saved,
            repaired.issues,
            repaired.rounds,
            repaired.transport_failed,
        )
    final = repaired.step_list

    # Screen the list that will actually ship, i.e. after any repair.
    screening = screener(final.render_text(), judge_model)
    # A held session stays held whatever this screening says; the result is still logged.
    status = Status.HELD_FOR_REVIEW if already_held else screening.status

    submitted, queue = True, None
    if final != saved:
        filled = bind_values(sealed, "save", {"content": final.model_dump_json()})
        saved_ok = execute(filled, tools, only={"save"}, table=table).outcome("save")
        submitted = saved_ok.status is StepStatus.OK
        revision = saved_ok.output.get("revision", "")
    if submitted:
        # Bind the submission to the revision that was screened.
        filled = bind_values(sealed, "submit", {"status": status.value, "revision": revision})
        sent = execute(filled, tools, only={"submit"}, table=table).outcome("submit")
        submitted = sent.status is StepStatus.OK
        queue = sent.output.get("queue")

    if not submitted:
        message = MESSAGE_NOT_SUBMITTED
    elif status is Status.HELD_FOR_REVIEW:
        message = MESSAGE_HELD
    else:
        message = MESSAGE_SENT
    return FinalizeResult(
        status=status,
        step_list=final,
        issues=(),
        repair_rounds=repaired.rounds,
        screening=screening,
        submitted=submitted,
        queue=queue,
        display_text=screening.masked_text,
        user_message=message,
    )
