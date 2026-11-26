"""Structural chunking: split a markdown document at its headings, then pack whole sentences
into chunks of about ``target_chars``. A sentence is never cut unless it alone is longer than
the target, and every chunk remembers its heading path ("Expense policy > Receipts")."""

from __future__ import annotations

import re
from dataclasses import dataclass

CHUNKER_VERSION = "headings-sentences-v1"
TARGET_CHARS = 400

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    heading: str
    text: str


def _split_long(sentence: str, target: int) -> list[str]:
    words, parts, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > target:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _pack(sentences: list[str], target: int) -> list[str]:
    chunks, current = [], ""
    for sentence in sentences:
        pieces = _split_long(sentence, target) if len(sentence) > target else [sentence]
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > target:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_document(markdown: str, *, target_chars: int = TARGET_CHARS) -> list[Chunk]:
    if target_chars < 20:
        raise ValueError("target_chars must be at least 20")
    path: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    lines: list[str] = []

    def flush() -> None:
        text = " ".join(" ".join(lines).split())
        if text:
            sections.append((" > ".join(title for _, title in path), _SENTENCE_END.split(text)))
        lines.clear()

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = [(lvl, title) for lvl, title in path if lvl < level]
            path.append((level, m.group(2)))
        else:
            lines.append(line)
    flush()
    return [
        Chunk(heading, text)
        for heading, sentences in sections
        for text in _pack(sentences, target_chars)
    ]
