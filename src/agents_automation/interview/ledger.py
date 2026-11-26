"""The ledger: what actually happened in a turn, as reported by the executor (not the model)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LedgerEntry:
    tool: str
    status: str  # an executor StepStatus value, or "refused" when the plan did not allow it
    detail: str | None = None


@dataclass
class Ledger:
    entries: list[LedgerEntry] = field(default_factory=list)

    def add(self, tool: str, status: str, detail: str | None = None) -> None:
        self.entries.append(LedgerEntry(tool, status, detail))

    @property
    def saved(self) -> bool:
        return any(e.tool == "save_draft" and e.status == "ok" for e in self.entries)

    @property
    def save_attempted(self) -> bool:
        return any(e.tool == "save_draft" for e in self.entries)
