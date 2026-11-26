"""Small builders shared by tests."""

from __future__ import annotations

import json
from typing import Any

from agents_automation.models import Step, StepList


def step(
    sid: str, kind: str = "task", nxt: tuple[str, ...] = (), title: str = "", actor: str = "Clerk"
) -> Step:
    return Step.model_validate(
        {"id": sid, "title": title or f"Step {sid}", "actor": actor, "kind": kind, "next": nxt}
    )


def linear(name: str = "Expense claim") -> StepList:
    return StepList(
        name=name,
        steps=(
            step("s1", "start", ("s2",), "Employee submits the expense form", "Employee"),
            step("s2", "task", ("s3",), "Manager checks the receipt", "Manager"),
            step("s3", "finish", (), "Finance pays the expense", "Finance"),
        ),
    )


def turn_json(reply: str, *, save: StepList | None = None, actions: Any = None) -> str:
    acts: Any = actions if actions is not None else []
    if save is not None:
        acts = [{"tool": "save_draft", "step_list": save.model_dump(mode="json")}]
    return json.dumps({"reply": reply, "actions": acts})
