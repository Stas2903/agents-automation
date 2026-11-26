"""Combine the three layers into a status, and mask flagged spans for display.

Layer 1 and Layer 2 match on a normalised view of the text (look-alike letters and invisible
characters cannot hide a word); every hit is mapped back to the original text, which is what
the judge reads and what the reviewer sees, with flagged spans masked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from agents_automation.injection.judge import ScreenJudgement, run_judge
from agents_automation.injection.patterns import Hit, find_markers
from agents_automation.injection.prefilter import needs_judge
from agents_automation.llm import ChatModel
from agents_automation.llm.text import normalized_view, replace_spans
from agents_automation.models import Status

MASK = "[withheld: flagged for review]"

Layer = Literal["patterns", "prefilter", "judge", "no_judge"]


@dataclass(frozen=True)
class Screening:
    status: Status
    layer: Layer  # the layer that decided
    hits: tuple[Hit, ...]  # positions refer to the original text
    judge: ScreenJudgement | None
    masked_text: str
    reason: str


def mask_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    return replace_spans(text, list(spans), MASK)


def _claim_spans(text: str, claims: Iterable[str]) -> list[tuple[int, int]]:
    spans = []
    for claim in claims:
        start = text.find(claim)
        while start != -1:
            spans.append((start, start + len(claim)))
            start = text.find(claim, start + 1)
    return spans


def screen(text: str, judge_model: ChatModel | None, *, timeout_s: float = 30.0) -> Screening:
    view = normalized_view(text)
    hits = []
    for hit in find_markers(view.text):
        start, end = view.original_span(hit.start, hit.end)
        hits.append(Hit(hit.marker, start, end, text[start:end]))
    if hits:
        masked = mask_spans(text, [(h.start, h.end) for h in hits])
        names = ", ".join(sorted({h.marker for h in hits}))
        return Screening(Status.HELD_FOR_REVIEW, "patterns", tuple(hits), None, masked, names)
    if not needs_judge(view.text):
        return Screening(Status.DELIVERED, "prefilter", (), None, text, "no authority vocabulary")
    if judge_model is None:
        return Screening(
            Status.DELIVERED_FLAGGED, "no_judge", (), None, text, "judge not configured: fail open"
        )
    verdict = run_judge(judge_model, text, timeout_s=timeout_s)
    if verdict.outcome == "planted":
        masked = mask_spans(text, _claim_spans(text, verdict.claims))
        return Screening(Status.HELD_FOR_REVIEW, "judge", (), verdict, masked, verdict.reason)
    if verdict.outcome == "clean":
        return Screening(Status.DELIVERED, "judge", (), verdict, text, verdict.reason)
    return Screening(
        Status.DELIVERED_FLAGGED, "judge", (), verdict, text, f"fail open: {verdict.reason}"
    )
