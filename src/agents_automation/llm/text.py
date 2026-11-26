"""Small text helpers shared by every caller that talks to a model or screens model output.

Keyword screens match on a *normalised view* of the text, so spelling tricks (look-alike
letters, invisible characters, full-width forms) cannot hide a word. The original text is
never rewritten: every match in the view maps back to a span of the original, which is what
gets masked or neutralised. A Cyrillic name therefore reaches the model and the reviewer
exactly as it was typed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Latin look-alikes (Cyrillic, Greek, IPA) used to dodge keyword filters. The table is built
# from code points so the source file stays ASCII.
_CONFUSABLE_PAIRS = {
    0x0410: "A", 0x0412: "B", 0x0415: "E", 0x041A: "K", 0x041C: "M", 0x041D: "H",
    0x041E: "O", 0x0420: "P", 0x0421: "C", 0x0422: "T", 0x0425: "X", 0x0430: "a",
    0x0435: "e", 0x043E: "o", 0x0440: "p", 0x0441: "c", 0x0443: "y", 0x0445: "x",
    0x0405: "S", 0x0406: "I", 0x0408: "J", 0x0455: "s", 0x0456: "i", 0x0458: "j",
    0x0474: "V", 0x0475: "v", 0x0501: "d", 0x051B: "q", 0x051D: "w",
    0x0391: "A", 0x0392: "B", 0x0395: "E", 0x0396: "Z", 0x0397: "H", 0x0399: "I",
    0x039A: "K", 0x039C: "M", 0x039D: "N", 0x039F: "O", 0x03A1: "P", 0x03A4: "T",
    0x03A5: "Y", 0x03A7: "X", 0x03B1: "a", 0x03B9: "i", 0x03BA: "k", 0x03BD: "v",
    0x03BF: "o", 0x03C1: "p", 0x03C5: "u", 0x03F2: "c",
    0x0251: "a", 0x0261: "g", 0x026A: "I",
    0x057D: "u", 0x0585: "o",  # Armenian
    # Latin small capitals fold to UPPER case, so a small-caps stamp still reads as a stamp.
    0x1D00: "A", 0x0299: "B", 0x1D04: "C", 0x1D05: "D", 0x1D07: "E", 0x0262: "G",
    0x029C: "H", 0x1D0A: "J", 0x1D0B: "K", 0x029F: "L", 0x1D0D: "M", 0x0274: "N",
    0x1D0F: "O", 0x1D18: "P", 0x0280: "R", 0xA731: "S", 0x1D1B: "T", 0x1D1C: "U",
    0x1D20: "V", 0x1D21: "W", 0x028F: "Y", 0x1D22: "Z",
    # Cherokee letters that look like Latin capitals.
    0x13AA: "A", 0x13F4: "B", 0x13DF: "C", 0x13A0: "D", 0x13AC: "E", 0x13BB: "H",
    0x13AB: "J", 0x13E6: "K", 0x13DE: "L", 0x13B7: "M", 0x13BE: "O", 0x13E2: "P",
    0x13A1: "R", 0x13DA: "S", 0x13A2: "T", 0x13D9: "V", 0x13D4: "W", 0x13C3: "Z",
}  # fmt: skip
_CONFUSABLES = {chr(cp): latin for cp, latin in _CONFUSABLE_PAIRS.items()}
_INVISIBLE = {"Cf", "Mn", "Me"}  # format chars, combining marks, variation selectors
# Hangul fillers and the Braille blank render as empty space, but Unicode calls them letters.
_BLANK_LETTERS = {chr(cp) for cp in (0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800)}


def _is_lower_cherokee(ch: str) -> bool:
    cp = ord(ch)
    return 0xAB70 <= cp <= 0xABBF or 0x13F8 <= cp <= 0x13FD


def _fold_char(ch: str) -> str:
    if ch in _BLANK_LETTERS:
        return ""
    if _is_lower_cherokee(ch):
        ch = ch.upper()  # lower-case Cherokee renders like Latin small capitals
    if ch in _CONFUSABLES:
        # looked up before NFKC, which can map an entry away (U+03F2 becomes U+03C2)
        return _CONFUSABLES[ch]
    out = []
    for part in unicodedata.normalize("NFKC", ch):
        if unicodedata.category(part) in _INVISIBLE:
            continue
        out.append(_CONFUSABLES.get(part, part))
    return "".join(out)


@dataclass(frozen=True)
class NormalizedView:
    """A normalised copy of ``original`` plus, for every character of it, the index of the
    original character it came from."""

    original: str
    text: str
    origin: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a non-empty span of ``text`` back to the covering span of ``original``,
        including any invisible characters dropped right after its last character."""
        first = self.origin[start]
        stop = self.origin[end - 1] + 1
        while stop < len(self.original) and not _fold_char(self.original[stop]):
            stop += 1
        return first, stop


def normalized_view(text: str) -> NormalizedView:
    chars: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        folded = _fold_char(ch)
        chars.append(folded)
        origin.extend([i] * len(folded))
    return NormalizedView(text, "".join(chars), tuple(origin))


def normalize_text(text: str) -> str:
    return normalized_view(text).text


def replace_spans(text: str, spans: list[tuple[int, int]], replacement: str) -> str:
    """Replace each span (merged when they overlap) with ``replacement``."""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out, cursor = [], 0
    for start, end in merged:
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# Neutralise every tag OPENER, whatever follows it: a closing ">" may come after a newline,
# many spaces or long attributes. The match also takes the rest of the tag up to a ">" on the
# same line within 200 characters, so text between "<untrusted" and a later ">" on that line
# is replaced too; text on other lines is never touched. Look-alike "<" and "/" count.
_LT = "<" + chr(0x2039) + chr(0x1438)  # "<", single left angle quote, Canadian syllabics PA
_SLASH = "/" + chr(92) + chr(0x2215)  # "/", backslash, division slash
_FENCE_TAG = re.compile(
    rf"[{re.escape(_LT)}]\s*[{re.escape(_SLASH)}]*\s*" + r"untrusted\b(?:[^<>\n]{0,200}>)?",
    re.IGNORECASE,
)
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def fence_untrusted(text: str) -> str:
    """Wrap untrusted text in <untrusted> tags. Any fence tag inside it (any case, spacing,
    attributes, look-alike letters or invisible characters) is neutralised first, so the text
    cannot close its own fence early. Everything else passes through unchanged."""
    view = normalized_view(text)
    spans = [view.original_span(m.start(), m.end()) for m in _FENCE_TAG.finditer(view.text)]
    return "<untrusted>\n" + replace_spans(text, spans, "[tag removed]") + "\n</untrusted>"


def loads_model_json(raw: str) -> Any:
    """Parse a model's JSON answer, tolerating a surrounding ``` or ```json code fence.

    Raises ``ValueError`` (``json.JSONDecodeError``) when the answer is not JSON.
    """
    return json.loads(_CODE_FENCE.sub("", raw))


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORD_TEXT = (
    "a an and are as at be by can do does for from has have how i if in is it its me my of on "
    "or our so that the their them then there this to us was we what when where which who why "
    "will with you your"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def content_tokens(text: str) -> list[str]:
    """Lower-case word tokens without stopwords, with a crude plural fold ("receipts")."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out
