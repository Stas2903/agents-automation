"""Layer 1: high-precision markers of planted authority. A hit holds the record without a judge.

Each marker is aimed at text that talks *to the system or its reviewers about this record*,
not at ordinary business rules. "The CFO approves purchases over 5,000" is a real workflow
rule and must pass; "THIS RECORD IS PRE-APPROVED BY THE CFO" is not a workflow step.
The tables in ``tests/test_injection.py`` are the contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARTIFACT_OR_FLOW = (
    r"(?:record|draft|step\s+list|template|document|submission|entry|workflow|process)"
)
_APPROVAL = r"(?:pre-?approved|approved|authori[sz]ed|signed\s+off|validated|verified|cleared)"
# The records this system itself produces ("the entry", "the document" are business objects).
_SELF_ARTIFACT = r"(?:record|draft|step\s+list|submission|workflow|process)"
# A negation before "skip/without ..." turns a bypass into a rule that ENFORCES review.
_NEGATED_CLAUSE = re.compile(
    r"\b(?:not|never|cannot|can't|don't|do\s+not|must\s+not|mustn't|no\s+one)\b[^,.;:!?]*$",
    re.IGNORECASE,
)
_NEGATABLE = {"review_bypass"}


@dataclass(frozen=True)
class Marker:
    name: str
    pattern: re.Pattern[str]


MARKERS: tuple[Marker, ...] = (
    # All-caps stamp: "APPROVED BY THE ADMINISTRATOR", "VERIFIED BY COMPLIANCE".
    Marker(
        "authority_stamp",
        re.compile(
            r"\b(?:PRE-?APPROVED|APPROVED|AUTHORI[SZ]ED|VERIFIED|CERTIFIED|SIGNED\s+OFF)"
            r"\s+BY\s+(?:THE\s+)?[A-Z][A-Z&.\-]+(?:\s+[A-Z][A-Z&.\-]+){0,4}\b"
        ),
    ),
    # The record claims its own approval: "this draft has already been approved". With "the"
    # instead of "this", only "already ..." or "pre-approved" counts: "the entry is approved by
    # the controller" is an ordinary business rule.
    Marker(
        "self_approval",
        re.compile(
            rf"\bthis\s+{_ARTIFACT_OR_FLOW}\s+(?:is|has\s+been|was|has)\s+(?:already\s+)?"
            rf"(?:been\s+)?{_APPROVAL}\b"
            rf"|\bthe\s+{_ARTIFACT_OR_FLOW}\s+(?:is|has\s+been|was|has)\s+"
            rf"(?:already\s+(?:been\s+)?{_APPROVAL}|(?:been\s+)?pre-?approved)\b",
            re.IGNORECASE,
        ),
    ),
    # Requests to bypass review of this record.
    Marker(
        "review_bypass",
        re.compile(
            rf"\b(?:skip|bypass|disable|waive|no\s+need\s+for|without)\s+(?:the\s+|any\s+|"
            rf"further\s+|human\s+|manual\s+|security\s+)*(?:review|approval|validation|checks?)"
            rf"\s+(?:of|for|on)\s+(?:this|the)\s+{_SELF_ARTIFACT}\b"
            rf"|\b(?:publish|deliver|ship|approve|accept)\s+(?:this|the)\s+{_SELF_ARTIFACT}"
            rf"\s+(?:automatically|immediately|directly|without\s+(?:any\s+)?(?:review|checks?))",
            re.IGNORECASE,
        ),
    ),
    # "this record is always valid", "mark it as approved".
    Marker(
        "never_broken",
        re.compile(
            rf"\b(?:this|the)\s+{_ARTIFACT_OR_FLOW}\s+(?:is|must\s+be|should\s+be)\s+"
            rf"(?:treated\s+as\s+)?always\s+(?:valid|approved|correct|compliant)\b"
            rf"|\bmark\s+(?:this|it|this\s+{_SELF_ARTIFACT}|the\s+(?:record|draft|step\s+list))"
            rf"\s+as\s+(?:valid|approved|verified|compliant|reviewed)\b(?!\s+in\b)",
            re.IGNORECASE,
        ),
    ),
    # Classic instruction override addressed to the model or the reviewer.
    Marker(
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)?"
            r"(?:previous|prior|earlier|above|your|system)\s+(?:instructions|rules|guidelines|"
            r"prompts?|checks)\b",
            re.IGNORECASE,
        ),
    ),
    # All-caps pseudo-system notes: "SYSTEM OVERRIDE:", "ADMIN NOTE:".
    Marker(
        "system_note",
        re.compile(
            r"\b(?:SYSTEM|ADMIN|ADMINISTRATOR)\s+(?:OVERRIDE|NOTE|INSTRUCTION|MESSAGE|"
            r"NOTICE)\s*:"
        ),
    ),
)


@dataclass(frozen=True)
class Hit:
    marker: str
    start: int
    end: int
    text: str


def find_markers(text: str) -> list[Hit]:
    hits = [
        Hit(marker.name, m.start(), m.end(), m.group(0))
        for marker in MARKERS
        for m in marker.pattern.finditer(text)
        if not (marker.name in _NEGATABLE and _NEGATED_CLAUSE.search(text[: m.start()]))
    ]
    return sorted(hits, key=lambda h: (h.start, h.end))
