"""Turn per-run results into a scenario verdict, and scenario verdicts into a gate result.

A single run proves little about a stochastic system, so a verdict needs at least two runs:

* every counted run passed   -> PASS
* every counted run failed   -> FAIL
* both kinds                 -> FLAKY  (a real finding, never rounded to PASS)
* fewer than two counted runs -> VOID   (VOID runs say nothing about the subject)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

MIN_RUNS = 2


class RunOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    VOID = "VOID"  # infrastructure failure (e.g. judge unreachable): no information


class Verdict(StrEnum):
    PASS = "PASS"
    FLAKY = "FLAKY"
    FAIL = "FAIL"
    VOID = "VOID"


def grade_runs(runs: Sequence[RunOutcome]) -> Verdict:
    if len(runs) < MIN_RUNS:
        raise ValueError(f"a verdict needs at least {MIN_RUNS} runs, got {len(runs)}")
    counted = [r for r in runs if r is not RunOutcome.VOID]
    if len(counted) < MIN_RUNS:
        return Verdict.VOID
    if all(r is RunOutcome.PASS for r in counted):
        return Verdict.PASS
    if all(r is RunOutcome.FAIL for r in counted):
        return Verdict.FAIL
    return Verdict.FLAKY


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"  # no gate scenario failed, but at least one was VOID


@dataclass(frozen=True)
class GateResult:
    status: GateStatus
    failing: tuple[str, ...]
    void: tuple[str, ...]


def evaluate_gate(verdicts: Mapping[str, Verdict], hard_gates: set[str]) -> GateResult:
    """Hard-gate scenarios must be PASS. FLAKY on a hard gate is a failure, not a pass."""
    missing = sorted(hard_gates - set(verdicts))
    if missing:
        raise ValueError(f"no verdict for hard-gate scenario(s) {missing}")
    failing = tuple(sorted(s for s in hard_gates if verdicts[s] in (Verdict.FAIL, Verdict.FLAKY)))
    void = tuple(sorted(s for s in hard_gates if verdicts[s] is Verdict.VOID))
    if failing:
        status = GateStatus.FAIL
    elif void or not hard_gates:  # no gate selected proves nothing
        status = GateStatus.INCOMPLETE
    else:
        status = GateStatus.PASS
    return GateResult(status, failing, void)
