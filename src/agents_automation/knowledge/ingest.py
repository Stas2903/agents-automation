"""Idempotent ingest: chunk, screen, embed only what changed, upsert.

* A chunk whose text, embedding model and chunker version are all unchanged is skipped
  without calling the embedder, so re-running ingest over the same corpus costs nothing.
* A document is refused (nothing sent) when the embedder is not cleared for its label, and
  anything indexed for it before is removed, so a stricter relabel can never leave the old,
  looser copy searchable.
* A batch that lists a document twice is refused up front (the outcome would otherwise
  depend on which embed call fails).
* Access only ever narrows before anything is embedded: a first pass removes refused
  documents, applies narrowing relabels (a stricter label, a subset of the groups) to the
  stored copy, and removes the stored copy of any document whose access widens or moves.
  Then each document is embedded and written in one transaction, so a failure anywhere in a
  batch never leaves an old copy readable by more people than the new access allows.
* A chunk whose text matches the injection screen's pattern layer (on the normalised view,
  heading and text together) is quarantined; when a match spans chunks, the whole document
  is. Subtler text is NOT screened here: it reaches the answer prompt inside the fence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources

from agents_automation.injection.patterns import find_markers
from agents_automation.knowledge.chunker import CHUNKER_VERSION, Chunk, chunk_document
from agents_automation.knowledge.store import ChunkVersion, KnowledgeStore, NewChunk
from agents_automation.llm.boundary import (
    BoundaryError,
    PlacedEmbedder,
    Sensitivity,
    require_clearance,
)
from agents_automation.llm.text import normalize_text


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    sensitivity: Sensitivity
    groups: tuple[str, ...]
    body: str


@dataclass
class IngestReport:
    embedded: int = 0
    unchanged: int = 0
    removed: int = 0
    embed_calls: int = 0
    quarantined: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)


def _hash(heading: str, text: str) -> str:
    return hashlib.sha256(f"{heading}\n{text}".encode()).hexdigest()


def ingest(
    store: KnowledgeStore,
    documents: Sequence[Document],
    embedder: PlacedEmbedder,
    *,
    timeout_s: float = 30.0,
) -> IngestReport:
    ids = [d.doc_id for d in documents]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        # with two entries the outcome would depend on which embed fails: refuse the batch
        raise ValueError(f"documents listed more than once: {', '.join(duplicates)}")
    report = IngestReport()
    model_id = embedder.embedder.model_id
    # Pass 1, before any embed call: make every document's stored access safe, so that a
    # failure later in the batch can never leave an old, looser copy behind.
    todo_docs: list[Document] = []
    for doc in documents:
        try:
            require_clearance(embedder.placement, doc.sensitivity)
        except BoundaryError as exc:
            # not cleared for the new label: nothing is sent, and the old copy is removed
            report.refused.append(f"{doc.doc_id}: {exc}")
            report.removed += store.forget_document(doc.doc_id)
            continue
        stored = store.doc_row(doc.doc_id)
        if stored is not None and stored != (doc.sensitivity, frozenset(doc.groups)):
            old_label, old_groups = stored
            if doc.sensitivity >= old_label and set(doc.groups) <= old_groups:
                # narrowing only (groups are not hierarchical: a subset of the old
                # groups, nothing new): safe to apply to the old text right away
                store.set_metadata(doc.doc_id, doc.title, doc.sensitivity, doc.groups)
            else:
                # widening: the old text must never be readable under the new access
                report.removed += store.forget_document(doc.doc_id)
        todo_docs.append(doc)
    # Pass 2: chunk, screen, embed what changed, write each document in one transaction.
    for doc in todo_docs:
        chunks = chunk_document(doc.body)
        planted = {i for i, c in enumerate(chunks) if _planted(_plain(c.heading, c.text))}
        # the remaining chunks in order, as one string (with and without their headings):
        # catches a marker cut across chunks or across a heading
        kept = [c for i, c in enumerate(chunks) if i not in planted]
        if _planted(" ".join(c.text for c in kept)) or _planted(_as_read(kept)):
            # a marker split across chunks: no chunk of this document can be trusted
            report.quarantined.append(f"{doc.doc_id} (planted text spans chunks)")
            report.removed += store.forget_document(doc.doc_id)
            continue
        report.quarantined.extend(f"{doc.doc_id}#{i}" for i in sorted(planted))
        known = store.versions(doc.doc_id)
        keep: list[int] = []
        todo: list[tuple[int, str, str, ChunkVersion]] = []
        for ord_, chunk in enumerate(chunks):
            if ord_ in planted:
                continue
            version = ChunkVersion(_hash(chunk.heading, chunk.text), model_id, CHUNKER_VERSION)
            if known.get(ord_) == version:
                report.unchanged += 1
                keep.append(ord_)
            else:
                todo.append((ord_, chunk.heading, chunk.text, version))
        new_chunks: list[NewChunk] = []
        if todo:
            # embed first: if this fails, nothing of the document has changed in the store
            vectors = embedder.embedder.embed([t for _, _, t, _ in todo], timeout_s=timeout_s)
            report.embed_calls += 1
            if len(vectors) != len(todo):
                raise ValueError("embedder returned a different number of vectors")
            new_chunks = [
                NewChunk(ord_, heading, text, version, vector)
                for (ord_, heading, text, version), vector in zip(todo, vectors, strict=True)
            ]
        report.removed += store.replace_document(
            doc.doc_id, doc.title, doc.sensitivity, doc.groups, keep=keep, new_chunks=new_chunks
        )
        report.embedded += len(new_chunks)
    return report


def _plain(heading: str, text: str) -> str:
    return f"{heading.replace(' > ', ' ')} {text}"


def _as_read(chunks: Sequence[Chunk]) -> str:
    """The chunks as one text, the way the document reads: each heading appears once, where
    it is entered, so a phrase cut across headings and text is whole again."""
    parts: list[str] = []
    previous: list[str] = []
    for c in chunks:
        path = c.heading.split(" > ") if c.heading else []
        shared = 0
        while shared < min(len(path), len(previous)) and path[shared] == previous[shared]:
            shared += 1
        parts.extend(path[shared:])
        parts.append(c.text)
        previous = path
    return " ".join(parts)


def _planted(text: str) -> bool:
    """The pattern layer on the normalised view, as the injection screen applies it."""
    return bool(find_markers(normalize_text(text)))


def load_documents() -> list[Document]:
    """The synthetic office handbook shipped with the package."""
    raw = json.loads(
        resources.files("agents_automation")
        .joinpath("data/knowledge_docs.json")
        .read_text(encoding="utf-8")
    )
    if raw.get("synthetic") is not True:
        raise ValueError("knowledge_docs.json must be marked synthetic")
    return [
        Document(
            d["doc_id"],
            d["title"],
            Sensitivity.parse(d["sensitivity"]),
            tuple(d["groups"]),
            chr(10).join(d["body"]),
        )
        for d in raw["documents"]
    ]
