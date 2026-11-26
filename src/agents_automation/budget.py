"""A per-turn time budget for optional model calls.

Required calls (the turn's one model call, a role's answer) keep their own timeout. Optional
calls (a reranker, a corrective re-call, a polishing step) ask the budget first; when there is
not enough time left they are skipped, and every skip is recorded so it can be reported
instead of disappearing silently. The clock is injectable, so tests control time exactly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock cannot go backwards")
        self._t += seconds


@dataclass(frozen=True)
class Skip:
    name: str
    reason: str

    def __str__(self) -> str:
        return f"{self.name} skipped: {self.reason}"


class TurnBudget:
    def __init__(self, seconds: float, clock: Clock | None = None) -> None:
        if math.isnan(seconds) or seconds < 0:
            raise ValueError("a budget cannot be negative or NaN")
        self.clock: Clock = clock or MonotonicClock()
        self.deadline = self.clock.now() + seconds
        self._skips: list[Skip] = []

    def remaining(self) -> float:
        return max(0.0, self.deadline - self.clock.now())

    def allow(self, name: str, *, needs_s: float) -> bool:
        """True if an optional call needing ``needs_s`` seconds still fits; otherwise the
        skip is recorded and False is returned."""
        left = self.remaining()
        if left >= needs_s:
            return True
        need = "more time than any turn has" if math.isinf(needs_s) else f"{needs_s:g}s"
        self._skips.append(Skip(name, f"needs {need}, {left:g}s left in this turn"))
        return False

    def record_skip(self, name: str, reason: str) -> None:
        """Record an optional step that was skipped for a reason other than time."""
        self._skips.append(Skip(name, reason))

    @property
    def skips(self) -> tuple[Skip, ...]:
        return tuple(self._skips)


def unlimited() -> TurnBudget:
    """A budget for callers that do not set one (every optional call fits)."""
    return TurnBudget(float("inf"))
