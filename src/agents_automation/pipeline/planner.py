"""Rule-side planning from the trusted request envelope.

Plan *structure* (which tools, in which order, with which slot keys) comes from a fixed
template catalog. ``make_plan`` takes only the trusted :class:`Request`; it has no parameter
through which conversation text could reach it. Slots that the request can fill are filled
here; the rest stay ``None`` and may receive *values* later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

Operation = Literal["read", "write"]


@dataclass(frozen=True)
class Request:
    """The trusted envelope: set by the calling platform, never by the conversation."""

    surface: str
    intent: str
    session_id: str
    user_id: str


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    tool: str
    operation: Operation
    slots: Mapping[str, str | None]


@dataclass(frozen=True)
class Plan:
    role: str
    steps: tuple[PlanStep, ...]

    def step(self, step_id: str) -> PlanStep | None:
        return next((s for s in self.steps if s.step_id == step_id), None)


@dataclass(frozen=True)
class _StepTemplate:
    step_id: str
    tool: str
    operation: Operation
    # slot key -> name of the Request field that fills it, or None for a value filled later
    slots: Mapping[str, str | None]


def _t(step_id: str, tool: str, op: Operation, **slots: str | None) -> _StepTemplate:
    return _StepTemplate(step_id, tool, op, MappingProxyType(dict(slots)))


TEMPLATES: Mapping[tuple[str, str], tuple[_StepTemplate, ...]] = MappingProxyType(
    {
        ("interviewer", "interview_turn"): (
            _t("read", "fetch_draft", "read", session_id="session_id", viewer_id="user_id"),
            _t(
                "save",
                "save_draft",
                "write",
                session_id="session_id",
                viewer_id="user_id",
                content=None,
            ),
        ),
        ("interviewer", "finalize"): (
            _t("read", "fetch_draft", "read", session_id="session_id", viewer_id="user_id"),
            _t(
                "save",
                "save_draft",
                "write",
                session_id="session_id",
                viewer_id="user_id",
                content=None,
            ),
            _t(
                "submit",
                "submit_for_review",
                "write",
                session_id="session_id",
                viewer_id="user_id",
                status=None,
                revision=None,  # the revision that was screened; a stale one is refused
            ),
        ),
        ("knowledge", "ask"): (
            _t("search", "search_docs", "read", viewer_id="user_id", query=None),
        ),
        ("summarizer", "summarize"): (
            _t(
                "read",
                "read_chat",
                "read",
                viewer_id="user_id",
                channel=None,
                start=None,
                end=None,
            ),
        ),
        ("reviewer", "review"): (
            _t("read", "fetch_draft", "read", session_id="session_id", viewer_id="user_id"),
            _t("approve", "approve", "write", session_id="session_id", viewer_id="user_id"),
        ),
    }
)


class PlanningError(Exception):
    pass


def make_plan(role: str, request: Request) -> Plan:
    template = TEMPLATES.get((role, request.intent))
    if template is None:
        raise PlanningError(f"no plan template for role={role!r} intent={request.intent!r}")
    steps = []
    for t in template:
        slots: dict[str, str | None] = {}
        for key, field in t.slots.items():
            slots[key] = None if field is None else str(getattr(request, field))
        steps.append(PlanStep(t.step_id, t.tool, t.operation, MappingProxyType(slots)))
    return Plan(role=role, steps=tuple(steps))
