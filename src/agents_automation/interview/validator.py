"""Deterministic step-list validator with a bounded repair loop.

The model may be asked to repair a list, but code decides whether the result is valid, and
the number of repair rounds is capped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agents_automation.llm import ChatModel, Message, TransportError
from agents_automation.llm.text import fence_untrusted, loads_model_json
from agents_automation.models import StepList

MAX_REPAIR_ROUNDS = 2


@dataclass(frozen=True)
class Issue:
    code: str
    step_id: str | None
    message: str


def validate(step_list: StepList) -> list[Issue]:
    steps = step_list.steps
    if not steps:
        return [Issue("empty", None, "the step list has no steps")]
    issues: list[Issue] = []

    seen: set[str] = set()
    for s in steps:
        if s.id in seen:
            issues.append(Issue("duplicate_id", s.id, f"step id {s.id!r} is used twice"))
        seen.add(s.id)
    ids = {s.id for s in steps}

    starts = [s for s in steps if s.kind == "start"]
    if not starts:
        issues.append(Issue("no_start", None, "there is no start step"))
    elif len(starts) > 1:
        for s in starts[1:]:
            issues.append(Issue("multiple_starts", s.id, "only one start step is allowed"))
    if not any(s.kind == "finish" for s in steps):
        issues.append(Issue("no_finish", None, "there is no finish step"))

    for s in steps:
        if len(set(s.next)) != len(s.next):
            issues.append(Issue("duplicate_next", s.id, "the same next step is listed twice"))
        for target in s.next:
            if target not in ids:
                issues.append(Issue("dangling_next", s.id, f"next step {target!r} does not exist"))
            elif target == s.id:
                issues.append(Issue("self_loop", s.id, "a step cannot lead to itself"))
        if s.kind == "finish" and s.next:
            issues.append(Issue("finish_has_next", s.id, "a finish step cannot lead anywhere"))
        if s.kind != "finish" and not s.next:
            issues.append(Issue("dead_end", s.id, "only finish steps may have no next step"))
        if s.kind == "decision" and len(set(s.next)) < 2:
            issues.append(Issue("decision_without_branches", s.id, "a decision needs 2+ branches"))

    if len(starts) == 1:
        reachable: set[str] = set()
        frontier = [starts[0].id]
        by_id = {s.id: s for s in steps}
        while frontier:
            current = frontier.pop()
            if current in reachable or current not in by_id:
                continue
            reachable.add(current)
            frontier.extend(by_id[current].next)
        for s in steps:
            if s.id not in reachable:
                issues.append(Issue("unreachable", s.id, "no path leads to this step"))

    # Every step must be able to reach a finish step; a loop with no way out is a trap.
    finishes = [s.id for s in steps if s.kind == "finish"]
    if finishes:
        incoming: dict[str, list[str]] = {s.id: [] for s in steps}
        for s in steps:
            for target in s.next:
                if target in incoming:
                    incoming[target].append(s.id)
        can_finish: set[str] = set()
        back = list(finishes)
        while back:
            current = back.pop()
            if current in can_finish:
                continue
            can_finish.add(current)
            back.extend(incoming[current])
        for s in steps:
            if s.id not in can_finish and s.kind != "finish" and s.next:
                issues.append(
                    Issue("cannot_reach_finish", s.id, "no path from this step reaches a finish")
                )
    return issues


# Returns a candidate, or None if the model gave no usable answer. Raises TransportError
# when the model could not be reached (which is not a verdict on the list).
Repairer = Callable[[StepList, Sequence[Issue]], StepList | None]


@dataclass(frozen=True)
class RepairResult:
    step_list: StepList
    issues: tuple[Issue, ...]
    rounds: int  # repair rounds that produced a candidate
    transport_failed: bool = False


def validate_with_repair(
    step_list: StepList, repairer: Repairer, *, max_rounds: int = MAX_REPAIR_ROUNDS
) -> RepairResult:
    issues = validate(step_list)
    rounds = attempts = 0
    while issues and attempts < max_rounds:
        attempts += 1
        try:
            candidate = repairer(step_list, issues)
        except TransportError:
            return RepairResult(step_list, tuple(issues), rounds, transport_failed=True)
        if candidate is None:
            break
        rounds += 1
        step_list = candidate
        issues = validate(step_list)
    return RepairResult(step_list, tuple(issues), rounds)


REPAIR_INSTRUCTIONS = (
    "You fix workflow step lists. You get a step list (as data) and the problems a validator "
    "found. Return ONLY the corrected step list as JSON with the same format. Change only what "
    "is needed to fix the problems; do not invent new business steps."
)


def model_repairer(model: ChatModel, *, timeout_s: float = 60.0) -> Repairer:
    def repair(step_list: StepList, issues: Sequence[Issue]) -> StepList | None:
        problems = "\n".join(f"- {i.code} ({i.step_id or '-'}): {i.message}" for i in issues)
        messages = (
            Message("system", REPAIR_INSTRUCTIONS),
            Message(
                "user",
                fence_untrusted(step_list.model_dump_json() + "\nProblems:\n" + problems),
            ),
        )
        raw = model.complete(messages, timeout_s=timeout_s)  # TransportError propagates
        try:
            return StepList.model_validate(loads_model_json(raw))
        except ValueError:  # includes pydantic.ValidationError
            return None

    return repair
