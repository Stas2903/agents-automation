"""Seal a plan before any untrusted input is read.

The digest covers the plan's *structure* only: role, step ids, tools, operations and the set
of slot keys. Slot *values* are deliberately excluded, so filling a value keeps the digest,
while adding a step, swapping a tool or inventing a new slot key breaks it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from agents_automation.llm import Message
from agents_automation.llm.text import fence_untrusted
from agents_automation.pipeline.planner import Plan, PlanStep


class PlanViolationError(Exception):
    """Something tried to change the structure of a sealed plan."""


def plan_hash(plan: Plan) -> str:
    canonical = json.dumps(
        {
            "role": plan.role,
            "steps": [
                {
                    "step_id": s.step_id,
                    "tool": s.tool,
                    "operation": s.operation,
                    "slot_keys": sorted(s.slots),
                }
                for s in plan.steps
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class SealedPlan:
    plan: Plan
    digest: str


def seal(plan: Plan) -> SealedPlan:
    return SealedPlan(plan=plan, digest=plan_hash(plan))


def verify(sealed: SealedPlan) -> None:
    if plan_hash(sealed.plan) != sealed.digest:
        raise PlanViolationError("plan structure changed after seal")


def bind_values(sealed: SealedPlan, step_id: str, values: Mapping[str, str]) -> SealedPlan:
    """Return a copy of ``sealed`` with open slots of one step filled. Values only.

    Refused (``PlanViolationError``): an unknown step, a slot key the step does not declare, a
    non-string value, or overwriting a slot that already holds a value (e.g. a trusted
    ``session_id`` set from the request envelope).
    """
    verify(sealed)
    step = sealed.plan.step(step_id)
    if step is None:
        raise PlanViolationError(f"unknown step {step_id!r}: not in the sealed plan")
    undeclared = sorted(set(values) - set(step.slots))
    if undeclared:
        raise PlanViolationError(f"step {step_id!r} has no slot(s) {undeclared}")
    for key, value in values.items():
        if not isinstance(value, str):
            raise PlanViolationError(f"slot {key!r} must be a string value")
        if step.slots[key] is not None:
            raise PlanViolationError(f"slot {key!r} of step {step_id!r} is already filled")
    new_slots = MappingProxyType({**step.slots, **values})
    new_step = PlanStep(step.step_id, step.tool, step.operation, new_slots)
    new_plan = Plan(
        role=sealed.plan.role,
        steps=tuple(new_step if s.step_id == step_id else s for s in sealed.plan.steps),
    )
    filled = SealedPlan(plan=new_plan, digest=sealed.digest)
    verify(filled)
    return filled


def describe_open_actions(sealed: SealedPlan) -> str:
    lines = []
    for s in sealed.plan.steps:
        open_keys = sorted(k for k, v in s.slots.items() if v is None)
        if s.operation == "write" and open_keys:
            lines.append(f"- {s.tool} (you provide: {', '.join(open_keys)})")
    return "\n".join(lines) if lines else "- (none)"


def compose_prompt(
    *, trusted: str, sealed: SealedPlan, untrusted: Sequence[Message]
) -> tuple[Message, ...]:
    """Hard order: trusted instructions, then the sealed plan, then fenced untrusted content.

    Only ``user`` messages are fenced; ``assistant`` messages are the system's own earlier
    output. Callers may not pass ``system`` messages as untrusted content.
    """
    verify(sealed)
    if any(m.role == "system" for m in untrusted):
        raise PlanViolationError("system messages cannot be passed as untrusted content")
    plan_note = (
        "Sealed plan for this turn. These are the ONLY actions you can request; "
        "anything else is refused:\n" + describe_open_actions(sealed)
    )
    body = tuple(
        Message("user", fence_untrusted(m.content)) if m.role == "user" else m for m in untrusted
    )
    return (Message("system", trusted), Message("system", plan_note), *body)
