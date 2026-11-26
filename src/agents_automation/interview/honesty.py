"""The honesty guard.

The interviewer's *actions* travel in a structured JSON side-output, so the model cannot
"call save" by writing prose. It can still *claim* in prose that it saved something. This
module detects such claims (pure regex, English) so the turn loop can compare them with the
ledger of what actually happened.

The MUST-FIRE / MUST-NOT-FIRE tables in ``tests/test_honesty_detector.py`` are the contract.
The detector is deliberately narrow: it only fires on the interviewer asserting that the
*draft* (or "it", "everything" ...) was persisted, and the guards below suppress negations,
intent, questions, quoted text, reported speech, earlier saves, and descriptions of the
employee's own systems.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents_automation.llm.text import normalize_text

STRONG_VERBS = r"saved|recorded|stored|logged"
WEAK_VERBS = r"added|updated|captured|noted|written\s+down|put|included"
_APOS = "['’]"
_ADVERBS = rf"(?:\s+have|{_APOS}ve|\s+(?:just|now|already|also|\w+ly))*"

# Where a "weak" verb counts as persistence ("I've added it to your draft").
_DRAFT_TARGET = (
    r"(?:the|your|our|my)\s+(?:draft|process(?:\s+draft)?|step\s+list|list|notes|"
    r"process\s+map|workflow(?:\s+draft)?)"
)

# 1) First person: "I've saved", "I have successfully recorded", "I've got everything saved".
_FIRST_PERSON_STRONG = re.compile(
    rf"\b(?:i|we){_ADVERBS}\s+(?:got\s+(?:\w+\s+){{0,3}})?(?:{STRONG_VERBS})\b",
    re.IGNORECASE,
)
_FIRST_PERSON_COORDINATED = re.compile(
    rf"\b(?:i|we)\b[^.!?,;]{{0,60}}?\band\s+(?:have\s+|{_APOS}ve\s+|also\s+|then\s+)*"
    rf"(?:{STRONG_VERBS})\b",
    re.IGNORECASE,
)
_FIRST_PERSON_WEAK = re.compile(
    rf"\b(?:i|we){_ADVERBS}\s+(?:{WEAK_VERBS})\b"
    rf"(?:(?:\s+\w+){{0,6}}?\s+(?:to|in|into|on))?\s+{_DRAFT_TARGET}",
    re.IGNORECASE,
)

# 2) Passive with a subject that refers to the conversation's own artifact
#    ("Your workflow draft is saved", "All six steps are saved", "Everything you told me is").
_ARTIFACT_NOUN = r"(?:draft|process|step\s+list|steps|answers|changes|workflow)"
_SUBJECT = (
    r"it|this|that|all\s+of\s+(?:it|this|that)|"
    r"everything(?:\s+(?:you|we)\s+(?:told\s+me|said|discussed|covered|described))?|"
    rf"(?:the|your|our)\s+(?:\w+\s+){{0,2}}?{_ARTIFACT_NOUN}|all\s+(?:\w+\s+)?steps"
)
_PASSIVE = re.compile(
    rf"\b(?P<subject>{_SUBJECT})"
    rf"(?:\s+(?:is|are|has\s+been|have\s+been|was|were)|{_APOS}s(?:\s+been)?)"
    rf"(?:\s+(?:now|already|all|\w+ly))*\s+(?:{STRONG_VERBS}|updated|captured)\b",
    re.IGNORECASE,
)

# 3) Elliptic one-liners: "Saved!", "Done, saved.", "Draft saved.", "Saved the draft with ..."
_DASHES = "—–-"
_LEAD = (
    r"^(?:[-*•]\s*)?"
    rf"(?:(?:done|ok(?:ay)?|great|perfect|all\s+set|got\s+it)\s*[,:;.!{_DASHES}]*\s*)?"
)
_ELLIPTIC = re.compile(
    rf"{_LEAD}(?:all\s+|now\s+)?(?:{STRONG_VERBS})(?:\s+(?:it|that|this))?\s*[.!]*$"
    rf"|{_LEAD}(?:(?:the|your)\s+(?:\w+\s+)?)?(?:draft|steps|changes)\s+"
    rf"(?:{STRONG_VERBS}|updated)\b"
    rf"|{_LEAD}(?:{STRONG_VERBS})\s+(?:the|your|it|them|all|everything)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTED = re.compile("\"[^\"]*\"|“[^”]*”|(?<![\\w])'[^'\\n]+'(?![\\w])|«[^»]*»")
_CLAUSE_BREAK = re.compile(r"[,;:—–]|\s-\s|\bbut\b|\band\s+then\b", re.IGNORECASE)
# "..., right?" turns the whole sentence into a question asking for confirmation.
_TAG_QUESTION = re.compile(
    rf"^\s*(?:right|correct|yes|no|ok(?:ay)?|is\s+it|isn{_APOS}t\s+it|has\s+it|"
    rf"hasn{_APOS}t\s+it)\s*\?$",
    re.IGNORECASE,
)

_NEGATION = re.compile(
    rf"\b(?:not|no|none|never|nothing|neither|nor|without|cannot)\b|n{_APOS}t", re.IGNORECASE
)
_HARMLESS_NEGATION = re.compile(r"\bno\s+(?:problem|worries)\b", re.IGNORECASE)
# Intent scopes over the verb only when it sits right before the claim ("I will make sure it
# is saved"), never when it opens the sentence ("I can confirm I've saved it" is a claim).
_MODAL = (
    r"will|would|shall|should|could|can|may|might|must|going\s+to|want\s+to|about\s+to|"
    rf"need\s+to|needs\s+to|have\s+to|plan\s+to|try\s+to|let\s+me|ready\s+to|{_APOS}ll|{_APOS}d"
)
# "can/could confirm" asserts; "will/need to/let me confirm" is still to happen.
_FUTURE = (
    r"will|shall|should|must|going\s+to|want\s+to|about\s+to|need\s+to|needs\s+to|have\s+to|"
    rf"plan\s+to|try\s+to|let\s+me|{_APOS}ll"
)
_INTENT = re.compile(
    rf"(?:\b(?:{_MODAL}|to\s+be)|(?:\b(?:{_FUTURE})\s+(?:make\s+sure|ensure|confirm|check|"
    rf"verify)(?:\s+that)?))\s*$",
    re.IGNORECASE,
)
_SUBORDINATOR = re.compile(
    r"^\s*(?:if|once|when|whenever|after|before|until|unless|as\s+soon\s+as)\b", re.IGNORECASE
)
# A sentence that opens with a condition is hypothetical as a whole ("If you confirm, it's
# saved."), while a past temporal clause is not ("When you mentioned it, I added it ...").
_CONDITIONAL_SENTENCE = re.compile(
    r"^\s*(?:if|once|unless|until|before|as\s+soon\s+as|provided\s+that)\b", re.IGNORECASE
)
# Reported speech ("the clerk mentioned we stored ..."), but not "as you said, I've saved it".
_REPORTED = re.compile(
    r"(?<!\bas\s)\b(?!i\b)\w+\s+(?:said|say|says|mentioned|told\s+me|wrote|explained|think|"
    r"thinks|thought|believe|believes|assume|assumed)\b(?:\s+that)?",
    re.IGNORECASE,
)
# A save that happened in an earlier turn cannot be checked against this turn's ledger.
# Only applied when a draft from an earlier turn actually exists (see detect_save_claims).
# "before" is left out on purpose: it is usually a conjunction ("I've saved it before you go").
# "Earlier than" is a comparison inside a claim ("I saved it earlier than planned").
_EARLIER = re.compile(
    r"^\s+(?:\w+\s+){0,3}?(?:earlier|previously|last\s+(?:time|turn)|yesterday)\b"
    r"(?!\s+than\b)",
    re.IGNORECASE,
)
# Places and purposes that ARE the system's own storage: a claim about them is still a claim.
# "the accounting system" stays the employee's (only a few neutral adjectives are allowed
# after "the"/"this"), while "our <anything> database" is the system's own.
_STORE_NOUN = r"(?:(?:system|platform|database|server|record)s?|cloud)"
_OWN_STORAGE = re.compile(
    r"^(?:you|us|here|later|next\s+time|review|"
    r"your\s+(?:account|profile|workspace|name|draft|notes|list)|"
    rf"our\s+(?:\w+\s+)?(?:{_STORE_NOUN}|side|end)|"
    rf"(?:the|this)\s+(?:(?:own|internal|secure|safe)\s+)?{_STORE_NOUN}|"
    r"(?:the|this)\s+(?:chat|conversation))\b",
    re.IGNORECASE,
)
# Passive subjects that are the conversation's own artifact, so "...and then ..." after them
# is not a step of the employee's process.
_ARTIFACT_SUBJECT = re.compile(
    r"\b(?:draft|answers|changes)\b|^everything\s+(?:you|we)\b|^all\s+(?:\w+\s+)?steps\b",
    re.IGNORECASE,
)
# "...it is logged and then checked by finance": a step of the employee's process.
_PROCESS_CONTINUES = re.compile(r"^\s+(?:and\s+)?then\b", re.IGNORECASE)
# "we stored nothing": a negation right after the verb.
_NEGATED_OBJECT = re.compile(
    r"^\s+(?:nothing|none|never|no\s+(?!fewer\b|less\b)\w+)\b", re.IGNORECASE
)
# "...is recorded in the ERP / by finance / for seven years": the employee's own systems.
_ELSEWHERE = re.compile(
    r"^\s+(?:\w+ly\s+)?(?:in|into|on|to|by|from|at|with|for|until|under|via)\s+"
    r"(?P<target>[^.,;!?]*)",
    re.IGNORECASE,
)
_PROCESS_ADVERB = re.compile(
    r"^\s+(?:automatically|centrally|electronically|manually|digitally|separately|locally)\b",
    re.IGNORECASE,
)
_DRAFT_TARGET_RE = re.compile(rf"^{_DRAFT_TARGET}\b", re.IGNORECASE)
_SAVED_SENSE = re.compile(
    r"^\s+(?:you\s+|us\s+|me\s+)?(?:some\s+|a\s+lot\s+of\s+|a\s+few\s+|\d+\s+)?"
    r"(?:time|money|effort|hours?|days?|minutes?|steps?|trouble|a\s+step)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Claim:
    sentence: str
    phrase: str


def _clause_prefix(sentence: str, start: int) -> str:
    prefix = sentence[:start]
    parts = _CLAUSE_BREAK.split(prefix)
    return parts[-1] if parts else prefix


def _negated(clause: str, match: str) -> bool:
    if _NEGATION.search(match):
        return True
    # Only a negation close to the verb scopes over it ("No problem I've saved it" is a claim).
    near = " ".join(_HARMLESS_NEGATION.sub(" ", clause).split()[-4:])
    return _NEGATION.search(near) is not None


def _guarded(
    sentence: str, match: re.Match[str], *, passive: bool, earlier_saves_possible: bool
) -> bool:
    if _CONDITIONAL_SENTENCE.search(sentence):
        return True
    clause = _clause_prefix(sentence, match.start())
    if _negated(clause, match.group(0)):
        return True
    if _INTENT.search(clause):
        return True
    if _SUBORDINATOR.search(clause):
        return True
    if _REPORTED.search(clause):
        return True
    tail = sentence[match.end() :]
    if _SAVED_SENSE.match(tail) or _NEGATED_OBJECT.match(tail):
        return True
    if earlier_saves_possible and _EARLIER.match(tail):
        return True
    if passive:
        # "...and then ..." describes a process step, unless the subject is the conversation's
        # own artifact ("Your draft is saved and then it goes to review" is still a claim).
        about_artifact = _ARTIFACT_SUBJECT.search(match.group("subject"))
        if _PROCESS_ADVERB.match(tail) or (_PROCESS_CONTINUES.match(tail) and not about_artifact):
            return True
        elsewhere = _ELSEWHERE.match(tail)
        if elsewhere:
            target = elsewhere.group("target").strip()
            if not (_DRAFT_TARGET_RE.match(target) or _OWN_STORAGE.match(target)):
                return True
    return False


def _statement_part(sentence: str) -> str | None:
    """The part of a sentence that asserts something: a question asserts nothing, but in
    "I've saved the draft, who approves it next?" the first clause still does."""
    if not sentence.endswith("?"):
        return sentence
    parts = _CLAUSE_BREAK.split(sentence)
    if len(parts) < 2 or _TAG_QUESTION.match(parts[-1]):
        return None
    head = sentence[: len(sentence) - len(parts[-1])].rstrip(" ,;:-—–")
    return head or None


def detect_save_claims(text: str, *, earlier_saves_possible: bool = True) -> list[Claim]:
    """Return the sentences in which the speaker claims to have persisted the draft.

    ``earlier_saves_possible`` says whether a draft from an earlier turn exists; only then is
    "the draft we saved earlier" accepted as a reference instead of a claim.
    """
    claims: list[Claim] = []
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = raw.strip()
        normalized = normalize_text(sentence)
        statement = _statement_part(normalized) if normalized else None
        if not statement:
            continue
        cleaned = _QUOTED.sub(" ", statement)
        found: re.Match[str] | None = None
        for pattern, passive in (
            (_FIRST_PERSON_STRONG, False),
            (_FIRST_PERSON_COORDINATED, False),
            (_FIRST_PERSON_WEAK, False),
            (_PASSIVE, True),
        ):
            for m in pattern.finditer(cleaned):
                guarded = _guarded(
                    cleaned, m, passive=passive, earlier_saves_possible=earlier_saves_possible
                )
                if not guarded:
                    found = m
                    break
            if found:
                break
        if found is None:
            found = _ELLIPTIC.match(cleaned.strip())
        if found is not None:
            claims.append(Claim(sentence=sentence, phrase=found.group(0).strip()))
    return claims


def remove_claims(text: str, claims: list[Claim]) -> str:
    """Drop the sentences that carry unsupported claims, keeping the rest of the reply."""
    claimed = {c.sentence for c in claims}
    sentences = (s.strip() for s in _SENTENCE_SPLIT.split(text))
    return " ".join(s for s in sentences if s and s not in claimed)


# Sent as an unfenced user message (employee text is always fenced, so it cannot fake this).
CORRECTION_PREFIX = "[system note, not from the employee] Correction required."
CORRECTIVE_NOTE = (
    CORRECTION_PREFIX + " Your reply says the draft was saved, but no save succeeded in this "
    "turn{reason}. Either include a save_draft action with the step list, or rewrite the reply "
    "so it does not claim that anything was saved. Answer with the same JSON format."
)

HONEST_NOT_SAVED = "Note: I have not saved anything in this turn."
HONEST_SAVE_FAILED = "Note: I tried to save the draft, but the save did not go through."
