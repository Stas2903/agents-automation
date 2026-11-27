"""Deterministic stand-ins for the models of the knowledge and summarizer roles and the intent
classifier. They read the same prompts a real model would get and answer in the same JSON, so
the demo, the tests and CI exercise the real code paths offline.

They are deliberately simple: the extractive answerer only answers when one source sentence,
read under its section heading, covers most of the question's words, and declines otherwise,
like a careful model would.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from agents_automation.llm import Message
from agents_automation.llm.fakes import RuleModel
from agents_automation.llm.text import content_tokens

OFFLINE_FAMILY = "offline-roles"
COVERAGE_NEEDED = 0.6

_PASSAGE = re.compile(r"^\[(?P<id>[^\]\s]+)\] (?P<heading>[^:]*): (?P<text>.*)$", re.MULTILINE)
_QUESTION = re.compile(r"^Question: (?P<q>.*)$", re.MULTILINE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _matches(token: str, words: set[str]) -> bool:
    if len(token) < 4:
        return token in words
    return any(w.startswith(token[:4]) for w in words)


def coverage(question: str, sentence: str) -> float:
    wanted = content_tokens(question)
    if not wanted:
        return 0.0
    words = set(content_tokens(sentence))
    return sum(_matches(t, words) for t in wanted) / len(wanted)


def _all_text(messages: Sequence[Message]) -> str:
    return "\n".join(m.content for m in messages if m.role == "user")


def _extractive_answer(messages: Sequence[Message]) -> str:
    text = _all_text(messages)
    q = _QUESTION.search(text)
    question = q.group("q") if q else ""
    # A sentence is read under its heading path, as a reader would ("Salary bands > Bands"
    # answers "What are the salary bands?"). Ties go to the section whose own heading matches,
    # then to the sentence that matches by itself.
    best: tuple[tuple[float, float, float], str, str] | None = None
    for m in _PASSAGE.finditer(text):
        heading = m.group("heading")
        section = heading.rsplit(">", 1)[-1]
        for sentence in _SENTENCES.split(m.group("text")):
            score = (
                coverage(question, f"{heading} {sentence}"),
                coverage(question, section),
                coverage(question, sentence),
            )
            if best is None or score > best[0]:
                best = (score, sentence.strip(), m.group("id"))
    if best is None or best[0][0] < COVERAGE_NEEDED:
        return json.dumps({"answer": None, "citations": []})
    return json.dumps({"answer": best[1], "citations": [best[2]]})


def extractive_answerer(family: str = OFFLINE_FAMILY) -> RuleModel:
    return RuleModel(family, _extractive_answer)


def _coverage_rerank(messages: Sequence[Message]) -> str:
    text = _all_text(messages)
    q = _QUESTION.search(text)
    question = q.group("q") if q else ""
    passages = [(m.group("id"), m.group("text")) for m in _PASSAGE.finditer(text)]
    ranked = sorted(passages, key=lambda p: (-coverage(question, p[1]), p[0]))
    return json.dumps({"order": [pid for pid, _ in ranked]})


def coverage_reranker(family: str = OFFLINE_FAMILY) -> RuleModel:
    return RuleModel(family, _coverage_rerank)


_CHAT_LINE = re.compile(r"^\[(?P<id>m\d+)\] (?P<author>\S+) (?P<day>\S+): (?P<text>.*)$", re.M)
_SUMMARY_LINE = re.compile(
    r"^(?P<label>.+?) summary \(cites (?P<cites>[^)]*)\): (?P<text>.*)$", re.M
)
_DECISION = re.compile(r"\b(?:decision|decided|agreed|approved|fixed|will)\b", re.IGNORECASE)


def _sentence(text: str) -> str:
    text = text.removeprefix("Decision: ")
    return text[:1].upper() + text[1:]


def _write_summary(messages: Sequence[Message]) -> str:
    lines = list(_CHAT_LINE.finditer(_all_text(messages)))
    if not lines:
        return json.dumps({"summary": "", "cites": []})
    chosen = [m for m in lines if _DECISION.search(m.group("text"))] or lines[:1]
    text = " ".join(_sentence(m.group("text")) for m in chosen)
    return json.dumps({"summary": text, "cites": [m.group("id") for m in chosen]})


def summary_writer(family: str = OFFLINE_FAMILY) -> RuleModel:
    """Keeps the decisions of a period (or its first message when nothing was decided)."""
    return RuleModel(family, _write_summary)


def _compose(messages: Sequence[Message]) -> str:
    text = _all_text(messages)
    parts: list[str] = []
    cites: list[str] = []
    for m in _SUMMARY_LINE.finditer(text):
        parts.append(m.group("text"))
        cites.extend(c.strip() for c in m.group("cites").split(",") if c.strip())
    for m in _CHAT_LINE.finditer(text):
        if _DECISION.search(m.group("text")):
            parts.append(_sentence(m.group("text")))
            cites.append(m.group("id"))
    return json.dumps({"summary": " ".join(parts), "cites": list(dict.fromkeys(cites))})


def summary_composer(family: str = OFFLINE_FAMILY) -> RuleModel:
    return RuleModel(family, _compose)
