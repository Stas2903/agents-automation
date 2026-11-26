"""The document store: SQLite with an FTS5 keyword index and vectors stored as blobs.

Access control happens INSIDE the queries: both the keyword search and the vector scan only
ever see chunks of documents shared with one of the reader's groups. A document with no
groups is visible to nobody. Turning chunk ids back into text checks the same rule again.
"""

from __future__ import annotations

import sqlite3
from array import array
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from agents_automation.llm.boundary import Sensitivity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    sensitivity INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS doc_groups (
    doc_id TEXT NOT NULL REFERENCES docs(doc_id),
    grp    TEXT NOT NULL,
    PRIMARY KEY (doc_id, grp)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES docs(doc_id),
    ord             INTEGER NOT NULL,
    heading         TEXT NOT NULL,
    text            TEXT NOT NULL,
    text_hash       TEXT NOT NULL,
    embed_model     TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    vector          BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED, heading, text, tokenize = 'porter unicode61'
);
"""


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str
    sensitivity: Sensitivity


@dataclass(frozen=True)
class ChunkVersion:
    text_hash: str
    embed_model: str
    chunker_version: str


@dataclass(frozen=True)
class NewChunk:
    ord_: int
    heading: str
    text: str
    version: ChunkVersion
    vector: Sequence[float]


def _pack(vector: Sequence[float]) -> bytes:
    return array("d", vector).tobytes()


def _unpack(blob: bytes) -> tuple[float, ...]:
    values = array("d")
    values.frombytes(blob)
    return tuple(values)


class KnowledgeStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.executescript(_SCHEMA)

    # --- writes (ingest only) ---------------------------------------------------------
    # Each public write is ONE transaction, so a reader never sees a document half updated
    # (for example new, looser groups over old, stricter text).

    def versions(self, doc_id: str) -> dict[int, ChunkVersion]:
        rows = self.db.execute(
            "SELECT ord, text_hash, embed_model, chunker_version FROM chunks WHERE doc_id = ?",
            (doc_id,),
        )
        return {ord_: ChunkVersion(h, m, v) for ord_, h, m, v in rows}

    def _delete_chunk(self, chunk_id: str) -> bool:
        self.db.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk_id,))
        return self.db.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,)).rowcount > 0

    def replace_document(
        self,
        doc_id: str,
        title: str,
        sensitivity: Sensitivity,
        groups: Collection[str],
        *,
        keep: Collection[int],
        new_chunks: Sequence[NewChunk],
    ) -> int:
        """Write the document's label and groups, keep the chunks at ``keep``, write
        ``new_chunks`` and delete every other chunk of it. Returns how many were deleted."""
        wanted = set(keep) | {c.ord_ for c in new_chunks}
        with self.db:
            self.db.execute(
                "INSERT INTO docs VALUES (?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET "
                "title = excluded.title, sensitivity = excluded.sensitivity",
                (doc_id, title, int(sensitivity)),
            )
            self.db.execute("DELETE FROM doc_groups WHERE doc_id = ?", (doc_id,))
            self.db.executemany(
                "INSERT INTO doc_groups VALUES (?, ?)", [(doc_id, g) for g in sorted(set(groups))]
            )
            stale = [
                cid
                for cid, ord_ in self.db.execute(
                    "SELECT chunk_id, ord FROM chunks WHERE doc_id = ?", (doc_id,)
                ).fetchall()
                if ord_ not in wanted
            ]
            removed = sum(self._delete_chunk(cid) for cid in stale)
            for c in new_chunks:
                chunk_id = f"{doc_id}#{c.ord_}"
                self._delete_chunk(chunk_id)  # the old row AND its old full-text entry
                self.db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk_id,
                        doc_id,
                        c.ord_,
                        c.heading,
                        c.text,
                        c.version.text_hash,
                        c.version.embed_model,
                        c.version.chunker_version,
                        _pack(c.vector),
                    ),
                )
                self.db.execute(
                    "INSERT INTO chunk_fts VALUES (?, ?, ?)", (chunk_id, c.heading, c.text)
                )
        return removed

    def set_metadata(
        self, doc_id: str, title: str, sensitivity: Sensitivity, groups: Collection[str]
    ) -> None:
        """Write the label and groups only (used to narrow access before any embed call)."""
        with self.db:
            self.db.execute(
                "UPDATE docs SET title = ?, sensitivity = ? WHERE doc_id = ?",
                (title, int(sensitivity), doc_id),
            )
            self.db.execute("DELETE FROM doc_groups WHERE doc_id = ?", (doc_id,))
            self.db.executemany(
                "INSERT INTO doc_groups VALUES (?, ?)", [(doc_id, g) for g in sorted(set(groups))]
            )

    def forget_document(self, doc_id: str) -> int:
        """Remove the document entirely: its chunks, its groups and its row."""
        with self.db:
            ids = [
                r[0]
                for r in self.db.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
                ).fetchall()
            ]
            removed = sum(self._delete_chunk(cid) for cid in ids)
            self.db.execute("DELETE FROM doc_groups WHERE doc_id = ?", (doc_id,))
            self.db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
        return removed

    def doc_row(self, doc_id: str) -> tuple[Sensitivity, frozenset[str]] | None:
        """The stored label and groups of a document."""
        row = self.db.execute("SELECT sensitivity FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        groups = self.db.execute("SELECT grp FROM doc_groups WHERE doc_id = ?", (doc_id,))
        return Sensitivity(row[0]), frozenset(g for (g,) in groups)

    # --- reads (always scoped to the reader's groups) ----------------------------------

    @staticmethod
    def _allowed(groups: Collection[str]) -> tuple[str, list[str]]:
        marks = ", ".join("?" for _ in groups)
        return f"SELECT doc_id FROM doc_groups WHERE grp IN ({marks})", sorted(groups)

    def keyword_search(
        self, terms: Sequence[str], groups: Collection[str], limit: int
    ) -> list[str]:
        """Chunk ids by BM25 rank. ``terms`` are plain word tokens; each is quoted, so FTS
        query syntax in user text is never interpreted."""
        cleaned = [t.replace('"', "").strip() for t in terms]
        cleaned = [t for t in cleaned if t]
        if not cleaned or not groups:
            return []
        match = " OR ".join(f'"{t}"' for t in cleaned)
        allowed, params = self._allowed(groups)
        rows = self.db.execute(
            "SELECT f.chunk_id FROM chunk_fts AS f JOIN chunks AS c ON c.chunk_id = f.chunk_id "
            f"WHERE chunk_fts MATCH ? AND c.doc_id IN ({allowed}) "
            "ORDER BY bm25(chunk_fts), f.chunk_id LIMIT ?",
            (match, *params, limit),
        )
        return [r[0] for r in rows]

    def vectors(self, groups: Collection[str]) -> list[tuple[str, tuple[float, ...]]]:
        if not groups:
            return []
        allowed, params = self._allowed(groups)
        rows = self.db.execute(
            f"SELECT chunk_id, vector FROM chunks WHERE doc_id IN ({allowed}) ORDER BY chunk_id",
            params,
        )
        return [(cid, _unpack(blob)) for cid, blob in rows]

    def chunks(self, chunk_ids: Sequence[str], groups: Collection[str]) -> list[StoredChunk]:
        """The chunks, in the order asked, that the reader may see (others are dropped)."""
        if not chunk_ids or not groups:
            return []
        allowed, params = self._allowed(groups)
        marks = ", ".join("?" for _ in chunk_ids)
        rows = self.db.execute(
            "SELECT c.chunk_id, c.doc_id, d.title, c.heading, c.text, d.sensitivity "
            "FROM chunks AS c JOIN docs AS d ON d.doc_id = c.doc_id "
            f"WHERE c.chunk_id IN ({marks}) AND c.doc_id IN ({allowed})",
            (*chunk_ids, *params),
        )
        found = {r[0]: StoredChunk(r[0], r[1], r[2], r[3], r[4], Sensitivity(r[5])) for r in rows}
        return [found[c] for c in chunk_ids if c in found]

    def count_chunks(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
