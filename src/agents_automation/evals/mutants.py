"""Mutants: the real system with one guard switched off. The harness must catch every one.

A harness that stays green when a guard is removed is measuring nothing. ``--mutant NAME``
runs the scenarios against a mutant and succeeds only if the hard gate FAILS.

``SYSTEM`` is the unmutated configuration; each mutant replaces exactly one of its parts.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, replace

from agents_automation.directory import Directory
from agents_automation.finalize import Screener, Validator
from agents_automation.injection.screen import Screening, screen
from agents_automation.interview.checks import detect_promises
from agents_automation.interview.honesty import Claim, detect_save_claims
from agents_automation.interview.turn import ClaimDetector, PromiseDetector
from agents_automation.interview.validator import Repairer, RepairResult, validate_with_repair
from agents_automation.knowledge.role import AnswerParser, parse_answer
from agents_automation.llm import ChatModel
from agents_automation.llm.boundary import Placement, Sensitivity
from agents_automation.llm.text import loads_model_json
from agents_automation.models import Status, StepList
from agents_automation.summarizer.chat import TeamChat
from agents_automation.summarizer.periods import PeriodRequest
from agents_automation.summarizer.role import CitationJudge, make_citation_judge
from agents_automation.summarizer.summaries import parse_cited_text


def _same_directory(directory: Directory) -> Directory:
    return directory


def _same_placement(placement: Placement) -> Placement:
    return placement


@dataclass(frozen=True)
class SystemConfig:
    name: str
    description: str
    detector: ClaimDetector = detect_save_claims
    screener: Screener = screen
    validator: Validator = validate_with_repair
    promise_detector: PromiseDetector = detect_promises
    answer_parser: AnswerParser = parse_answer
    judge_factory: Callable[[TeamChat, PeriodRequest], CitationJudge] = make_citation_judge
    directory_view: Callable[[Directory], Directory] = _same_directory
    clearance: Callable[[Placement], Placement] = _same_placement


SYSTEM = SystemConfig("system", "the real system, every guard on")


def _no_claims(text: str, *, earlier_saves_possible: bool = True) -> list[Claim]:
    return []


def _no_screen(text: str, _judge: ChatModel | None) -> Screening:
    return Screening(Status.DELIVERED, "prefilter", (), None, text, "mutant: screening disabled")


def _no_validation(step_list: StepList, _repairer: Repairer) -> RepairResult:
    return RepairResult(step_list, (), 0)


def _no_promises(text: str) -> list[str]:
    return []


def _any_citation(raw: str, _allowed: Collection[str]) -> tuple[str, tuple[str, ...]] | None:
    """Accepts any answer with a citation list, whether or not it was given those passages."""
    try:
        data = loads_model_json(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    answer, citations = data.get("answer"), data.get("citations")
    if isinstance(answer, str) and answer.strip() and isinstance(citations, list):
        return answer.strip(), tuple(str(c) for c in citations)
    return None


def _unchecked_judge(_chat: TeamChat, _period: PeriodRequest) -> CitationJudge:
    def judge(raw: str, _material: Collection[str]) -> tuple[str, tuple[str, ...]] | None:
        return parse_cited_text(raw)

    return judge


def all_groups(directory: Directory) -> Directory:
    """Everyone belongs to every group: access control is effectively off."""
    groups = frozenset().union(*directory.members.values())
    return Directory({user: groups for user in directory.members})


def cleared_for_everything(placement: Placement) -> Placement:
    return Placement(placement.name, Sensitivity.CONFIDENTIAL)


MUTANTS: dict[str, SystemConfig] = {
    m.name: m
    for m in (
        replace(
            SYSTEM,
            name="claims-without-saving",
            description="the honesty guard never sees a claim, so false 'saved' replies go out",
            detector=_no_claims,
        ),
        replace(
            SYSTEM,
            name="no-injection-screen",
            description="every finished list is delivered without screening",
            screener=_no_screen,
        ),
        replace(
            SYSTEM,
            name="skip-validation",
            description="every saved list counts as valid, so broken lists ship unrepaired",
            validator=_no_validation,
        ),
        replace(
            SYSTEM,
            name="acl-off",
            description="every reader belongs to every group, so restricted documents and "
            "private channels are open",
            directory_view=all_groups,
        ),
        replace(
            SYSTEM,
            name="boundary-off",
            description="every model counts as cleared for confidential data",
            clearance=cleared_for_everything,
        ),
        replace(
            SYSTEM,
            name="ungrounded-answers",
            description="answers may cite passages they were never given",
            answer_parser=_any_citation,
        ),
        replace(
            SYSTEM,
            name="summary-citations-unchecked",
            description="composed summaries are not checked against the requested period",
            judge_factory=_unchecked_judge,
        ),
        replace(
            SYSTEM,
            name="promises-unchecked",
            description="promises of actions the interviewer cannot take go out unchecked",
            promise_detector=_no_promises,
        ),
    )
}
