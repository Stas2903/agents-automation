"""Offline, deterministic stand-ins for the interviewer and judge, driven by synthetic scripts.

``ScriptedInterviewerModel`` decides what to answer from the *shape* of the prompt (which
turn it is, whether it is a corrective re-call or a repair request), not from a call counter.
So when a guard is disabled (an eval mutant), the script does not drift out of step: the
corrective answer is simply never asked for.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

from agents_automation.interview.honesty import CORRECTION_PREFIX
from agents_automation.interview.validator import REPAIR_INSTRUCTIONS
from agents_automation.llm import Message, TransportError
from agents_automation.llm.fakes import ScriptedModel, ScriptExhaustedError

INTERVIEWER_FAMILY = "offline-interviewer"
INJECTION_JUDGE_FAMILY = "offline-injection-judge"


def _materialize(item: Any, step_lists: Mapping[str, Any]) -> str | Exception:
    """Turn a script entry into a raw model answer (or an exception to raise)."""
    if isinstance(item, str):
        return item  # raw text, e.g. deliberately malformed output
    if isinstance(item, dict) and item.get("$error") == "transport":
        return TransportError("simulated transport failure")
    resolved = copy.deepcopy(item)
    for action in resolved.get("actions", []):
        ref = action.get("step_list")
        if isinstance(ref, str) and ref.startswith("@"):
            action["step_list"] = step_lists[ref[1:]]
    ref = resolved.get("$step_list")
    if ref is not None:  # a repair answer: a bare step list
        if not (isinstance(ref, str) and ref.startswith("@")):
            raise ValueError(f"bad $step_list reference {ref!r}")
        return json.dumps(step_lists[ref[1:]])
    return json.dumps(resolved)


@dataclass(frozen=True)
class ScriptedTurn:
    reply: str | Exception
    corrective: str | Exception | None


class ScriptedInterviewerModel:
    def __init__(
        self,
        turns: Sequence[ScriptedTurn],
        repairs: Sequence[str | Exception] = (),
        family: str = INTERVIEWER_FAMILY,
    ) -> None:
        self._turns = list(turns)
        self._repairs = list(repairs)
        self._repairs_used = 0
        self._family = family
        self.calls: list[tuple[Message, ...]] = []

    @property
    def family(self) -> str:
        return self._family

    def complete(self, messages: Sequence[Message], *, timeout_s: float) -> str:
        self.calls.append(tuple(messages))
        if messages and messages[0].role == "system" and messages[0].content == REPAIR_INSTRUCTIONS:
            if self._repairs_used >= len(self._repairs):
                raise ScriptExhaustedError("no scripted repair answer left")
            item = self._repairs[self._repairs_used]
            self._repairs_used += 1
        else:
            is_correction = messages[-1].role == "user" and messages[-1].content.startswith(
                CORRECTION_PREFIX
            )
            # One fenced user message carries the saved draft; the corrective note (if any)
            # is not a turn; every other user message is one employee turn.
            user_messages = sum(1 for m in messages if m.role == "user")
            turn_no = user_messages - 2 - (1 if is_correction else 0)
            if not 0 <= turn_no < len(self._turns):
                raise ScriptExhaustedError(f"no scripted turn {turn_no + 1}")
            turn = self._turns[turn_no]
            if is_correction:
                if turn.corrective is None:
                    raise ScriptExhaustedError(f"turn {turn_no + 1} has no corrective answer")
                item = turn.corrective
            else:
                item = turn.reply
        if isinstance(item, Exception):
            raise item
        return item


@dataclass(frozen=True)
class Conversation:
    """One synthetic conversation from ``data/synthetic_conversations.json``."""

    id: str
    title: str
    persona: str
    employee_lines: tuple[str, ...]
    outage_turns: tuple[int, ...]
    variants: tuple[Mapping[str, Any], ...]
    planted_text_judge: tuple[Any, ...]
    step_lists: Mapping[str, Any]

    def interviewer(self, run_index: int = 0) -> ScriptedInterviewerModel:
        variant = self.variants[run_index % len(self.variants)]
        turns = [
            ScriptedTurn(
                _materialize(t["reply"], self.step_lists),
                _materialize(t["corrective"], self.step_lists) if "corrective" in t else None,
            )
            for t in variant["turns"]
        ]
        repairs = [_materialize(r, self.step_lists) for r in variant.get("repairs", [])]
        return ScriptedInterviewerModel(turns, repairs)

    def injection_judge_model(self) -> ScriptedModel:
        script = [_materialize(item, self.step_lists) for item in self.planted_text_judge]
        return ScriptedModel(INJECTION_JUDGE_FAMILY, script)


def load_conversations() -> dict[str, Conversation]:
    raw = (
        resources.files("agents_automation")
        .joinpath("data/synthetic_conversations.json")
        .read_text(encoding="utf-8")
    )
    data = json.loads(raw)
    if data.get("synthetic") is not True:
        raise ValueError("conversation data must be marked synthetic")
    step_lists = data["step_lists"]
    out = {}
    for cid, c in data["conversations"].items():
        out[cid] = Conversation(
            id=cid,
            title=c["title"],
            persona=c["persona"],
            employee_lines=tuple(c["employee_lines"]),
            outage_turns=tuple(c.get("outage_turns", [])),
            variants=tuple(c["variants"]),
            planted_text_judge=tuple(c.get("planted_text_judge", [])),
            step_lists=step_lists,
        )
    return out
