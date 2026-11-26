"""Authorization: a fixed (role, tool, operation) allow-set. Anything not listed is denied."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from agents_automation.pipeline.seal import PlanViolationError, SealedPlan, verify

AuthTable = frozenset[tuple[str, str, str]]

ALLOWED: AuthTable = frozenset(
    {
        ("interviewer", "fetch_draft", "read"),
        ("interviewer", "save_draft", "write"),
        ("interviewer", "submit_for_review", "write"),
        ("reviewer", "fetch_draft", "read"),
        ("reviewer", "approve", "write"),
        ("knowledge", "search_docs", "read"),
        ("summarizer", "read_chat", "read"),
    }
)


def may_run(role: str, tool: str, operation: str, table: AuthTable = ALLOWED) -> bool:
    return (role, tool, operation) in table


@dataclass(frozen=True)
class AccessVerdict:
    allowed: bool
    reason: str


def authorize(sealed: SealedPlan, table: AuthTable = ALLOWED) -> AccessVerdict:
    try:
        verify(sealed)
    except PlanViolationError as exc:
        return AccessVerdict(False, str(exc))
    plan = sealed.plan
    if not plan.steps:
        return AccessVerdict(False, "empty plan: denied")
    for s in plan.steps:
        if not may_run(plan.role, s.tool, s.operation, table):
            return AccessVerdict(
                False, f"{plan.role!r} may not {s.operation} with {s.tool!r} (step {s.step_id!r})"
            )
    return AccessVerdict(True, "all steps permitted")


def visible_to(scope: Collection[str] | None, viewer_ids: Collection[str]) -> bool:
    """A record is visible only if EVERY viewer is inside its access scope.

    No scope means nobody: missing metadata is a deny, never an allow.
    """
    if not scope or not viewer_ids:
        return False
    return set(viewer_ids) <= set(scope)
