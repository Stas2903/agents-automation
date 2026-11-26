"""The interviewer's reply checks, in priority order, with at most one corrective per turn.

1. honesty: a save claim that the ledger does not support (``honesty.py``);
2. promise: a promise to act outside the conversation ("I'll email you the summary"); the
   interviewer has no tool for that, so the promise cannot be kept;
3. repetition: the same question as in the interviewer's previous message.

Only the first check that fires asks the model to correct itself, so a turn costs at most
one extra call. Code fallbacks apply afterwards regardless: unsupported save claims and
promises are removed and replaced by an honest note; a repeated question is only reported.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from agents_automation.interview.honesty import CORRECTION_PREFIX
from agents_automation.llm.text import content_tokens, normalize_text

PROMISE_NOTE = (
    CORRECTION_PREFIX + " Your reply promises an action outside this conversation (sending, "
    "reminding, scheduling or similar), but you have no tool for that. Rewrite the reply "
    "without that promise. Answer with the same JSON format."
)
REPETITION_NOTE = (
    CORRECTION_PREFIX + " Your reply asks the same question as your previous message. Ask about "
    "a different part of the workflow instead. Answer with the same JSON format."
)
HONEST_NO_PROMISE = "Note: I can't send, remind or schedule anything myself."

_APOS = "['’]"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTED = re.compile('"[^"]*"|“[^”]*”|«[^»]*»')
_ACTION = (
    r"(?:send|e-?mail|forward|remind|call|ring|phone|notify|share|book|schedule|ping|"
    r"message|text|follow\s+up|get\s+back\s+to|let\s+(?:you|them|him|her|the\s+\w+)\s+know|"
    r"set\s+up\s+(?:a|the)\s+(?:meeting|call|reminder))"
)
_ACTION_ING = (
    r"(?:sending|e-?mailing|forwarding|reminding|calling|notifying|sharing|booking|"
    r"scheduling|pinging|messaging|texting|following\s+up)"
)
_ADV = (
    r"(?:also|then|just|really|gladly|happily|personally|definitely|certainly|surely|quickly|"
    r"go\s+ahead\s+and)"
)
_PROMISE = re.compile(
    rf"\b(?:i|we)(?:\s*{_APOS}ll|\s+will|\s+shall|\s*{_APOS}(?:m|re)\s+(?:going\s+to|gonna)|"
    rf"\s+(?:am|are)\s+(?:going\s+to|gonna)|\s*{_APOS}(?:m|re)\s+happy\s+to|"
    rf"\s+(?:am|are)\s+happy\s+to|\s*{_APOS}(?:d|ll)\s+be\s+(?:happy|sure)\s+to|"
    r"\s+will\s+be\s+(?:happy|sure)\s+to)\s+"
    rf"(?:{_ADV}\s+)*(?:(?:make|be)\s+sure\s+to\s+(?:{_ADV}\s+)*)?"
    rf"(?:be\s+{_ACTION_ING}|{_ACTION})\b",
    re.IGNORECASE,
)
# Present-tense informing, not a promise: "I'm happy to let you know that the draft is
# complete", "I'm happy to share what I have", "I'm happy to call this step the approval step".
_INFORMING = re.compile(
    rf"(?:i|we)\s*(?:{_APOS}m|am|{_APOS}re|are)\s+happy\s+to\s+"
    # "that" must open a clause about the present ("... that the draft is complete"); "share
    # that file with you tomorrow" stays a promise
    r"(?:(?:let\s+you\s+know|share)\s+that\s+"
    r"(?:the|this|it|we|i|you|our|your|there|everything)\b"
    r"|share\s+what\s+(?:i|we)\s+(?:have|know|found|see)\b"
    # naming, not calling someone: "call this step the approval step"
    r"|call\s+(?:this|that|it)\s+(?:the|a|an|step|stage|part|phase)\b)",
    re.IGNORECASE,
)
_FUTURE_TAIL = re.compile(
    r"\b(?:tomorrow|tonight|later|soon|this\s+(?:morning|afternoon|evening|week)|next\s+\w+|"
    r"the\s+moment|the\s+next\s+time|as\s+soon\s+as|by\s+e-?mail|with\s+you)\b"
    rf"|\band\s+(?:{_ADV}\s+)*{_ACTION}\b",
    re.IGNORECASE,
)
# What the interviewer CAN do: hand the finished draft over for review (finalize does that).
_HAND_OVER = re.compile(
    r"^\s+(?:it|this|that|the\s+(?:\w+\s+){0,2}?draft)\s+(?:on\s+)?(?:for|to)\s+"
    r"(?:review|approval|(?:an?\s+|the\s+)?(?:administrator|reviewer))\b",
    re.IGNORECASE,
)
# A negation only cancels the promise when it governs it directly: "I can't promise I'll ...",
# "I don't think I'll ...". A negation in an idiom earlier on ("No worries, I'll email you",
# "Not a problem, I'll send it") does not.
_NEGATED_INTENT = re.compile(
    rf"\b(?:(?:can{_APOS}t|cannot|can\s+not|won{_APOS}t|don{_APOS}t|do\s+not|doesn{_APOS}t|never)"
    r"\s+(?:promise|guarantee|say|said|think|know|expect|plan|intend)"
    rf"|(?:am|{_APOS}m)\s+not\s+sure)"
    r"(?:\s+(?:you|them|him|her|anyone))?(?:\s+for\s+sure)?(?:\s+(?:that|if|whether))?\s*$",
    re.IGNORECASE,
)
# A real question asks whether to act ("Shall I say I'll remind you?"); "I'll email you, OK?"
# is a promise with a tag.
_INVERTED_QUESTION = re.compile(
    r"^\s*(?:(?:so|okay|ok|and|then|well|now)\s*,?\s+){0,2}"
    r"(?:(?:shall|should|will|would|can|could|may|do|does|did)\s+(?:i|we|you)\b"
    r"|(?:want|like)\s+me\s+to\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    check: str  # "honesty" | "promise" | "repetition"
    sentences: tuple[str, ...]


def detect_promises(text: str) -> list[str]:
    """The sentences in which the interviewer promises an action it cannot take."""
    found = []
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = raw.strip()
        if not sentence:
            continue
        view = _QUOTED.sub(" ", normalize_text(sentence))
        if view.rstrip().endswith("?") and _INVERTED_QUESTION.match(view):
            continue
        for m in _PROMISE.finditer(view):
            if _NEGATED_INTENT.search(view[: m.start()]):
                continue
            informing = _INFORMING.match(view[m.start() :])
            if informing and _FUTURE_TAIL.search(view[m.start() :]):
                informing = None  # "... share that this afternoon": still a promise
            if _HAND_OVER.match(view[m.end() :]) or informing:
                continue
            found.append(sentence)
            break
    return found


def remove_sentences(text: str, sentences: list[str]) -> str:
    drop = set(sentences)
    kept = (s.strip() for s in _SENTENCE_SPLIT.split(text))
    return " ".join(s for s in kept if s and s not in drop)


def _questions(text: str) -> list[frozenset[str]]:
    out = []
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = raw.strip()
        if sentence.endswith("?"):
            tokens = frozenset(content_tokens(normalize_text(sentence)))
            if tokens:
                out.append(tokens)
    return out


def repeated_questions(reply: str, previous_reply: str | None, *, threshold: float = 0.8) -> bool:
    """True if a question in ``reply`` is (nearly) the same as one in ``previous_reply``."""
    if not previous_reply:
        return False
    before = _questions(previous_reply)
    for q in _questions(reply):
        for p in before:
            if len(q & p) / len(q | p) >= threshold:
                return True
    return False


def first_finding(
    reply: str,
    *,
    unsupported_claims: list[str],
    previous_reply: str | None,
    promise_detector: Callable[[str], list[str]] = detect_promises,
) -> Finding | None:
    if unsupported_claims:
        return Finding("honesty", tuple(unsupported_claims))
    promises = promise_detector(reply)
    if promises:
        return Finding("promise", tuple(promises))
    if repeated_questions(reply, previous_reply):
        return Finding("repetition", ())
    return None
