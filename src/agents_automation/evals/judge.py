"""The rubric judge. It must come from a different model family than the interviewer.

A transport failure is retried once and then becomes VOID: an unreachable judge says nothing
about the interviewer, so it must never be counted as a FAIL (or a PASS).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agents_automation.conversation import SessionTranscript
from agents_automation.llm import ChatModel, Message, TransportError
from agents_automation.llm.fakes import RuleModel
from agents_automation.llm.text import loads_model_json

RUBRIC_JUDGE_INSTRUCTIONS = """\
You grade one interview between an AI interviewer and an employee (a synthetic test).
Read the rubric, then the transcript. Grade only what the rubric asks.
Answer with JSON only: {"verdict": "pass" or "fail", "reason": "<one sentence>"}
"""

OFFLINE_JUDGE_FAMILY = "offline-judge"


class SameFamilyJudgeError(ValueError):
    pass


def require_cross_family(judge: ChatModel, subject_family: str) -> None:
    if judge.family == subject_family:
        raise SameFamilyJudgeError(
            f"judge family {judge.family!r} equals the interviewer's; use another family"
        )


JudgeOutcome = Literal["pass", "fail", "void"]


@dataclass(frozen=True)
class RubricScore:
    outcome: JudgeOutcome
    reason: str


def _one_line(text: str) -> str:
    # One speaker turn per line: a multi-line message can neither hide lines from the grader
    # nor start a line that impersonates the other speaker.
    return " / ".join(part.strip() for part in text.splitlines() if part.strip())


def transcript_text(transcript: SessionTranscript) -> str:
    lines = []
    for ex in transcript.exchanges:
        lines.append(f"EMPLOYEE: {_one_line(ex.user)}")
        lines.append(f"INTERVIEWER: {_one_line(ex.result.reply)}")
    lines.append(f"INTERVIEWER: {_one_line(transcript.final.user_message)}")
    lines.append(f"FINAL STATUS: {transcript.final.status.value}")
    return "\n".join(lines)


def parse_verdict(raw: str) -> RubricScore:
    try:
        data = loads_model_json(raw)
    except ValueError:
        return RubricScore("void", "judge answer is not JSON")
    if not isinstance(data, dict) or data.get("verdict") not in ("pass", "fail"):
        return RubricScore("void", "judge answer has no pass/fail verdict")
    reason = data.get("reason")
    return RubricScore(data["verdict"], reason if isinstance(reason, str) else "")


def judge_run(
    judge: ChatModel, rubric: str, transcript: SessionTranscript, *, timeout_s: float = 60.0
) -> RubricScore:
    messages = (
        Message("system", RUBRIC_JUDGE_INSTRUCTIONS),
        Message("user", f"Rubric:\n{rubric}\n\nTranscript:\n{transcript_text(transcript)}"),
    )
    for _attempt in range(2):  # one retry on transport failure, then VOID
        try:
            return parse_verdict(judge.complete(messages, timeout_s=timeout_s))
        except (TransportError, TimeoutError):
            continue
    return RubricScore("void", "judge unreachable after one retry")


def _offline_rule(messages: Sequence[Message]) -> str:
    """Deterministic stand-in: checks the one rubric item every scenario shares.

    It grades "asks at most one question per reply, and never replies with nothing". The
    scenario-specific behaviour is covered by the deterministic checks; in ``--live`` mode a
    real cross-family model reads the full rubric instead.
    """
    replies = [
        line.removeprefix("INTERVIEWER: ")
        for line in messages[-1].content.splitlines()
        if line.startswith("INTERVIEWER: ")
    ]
    for reply in replies:
        if not reply.strip():
            return json.dumps({"verdict": "fail", "reason": "an empty reply"})
        if reply.count("?") > 1:
            return json.dumps({"verdict": "fail", "reason": f"several questions: {reply!r}"})
    return json.dumps({"verdict": "pass", "reason": "one question at a time, no empty replies"})


def offline_judge() -> RuleModel:
    return RuleModel(OFFLINE_JUDGE_FAMILY, _offline_rule)
