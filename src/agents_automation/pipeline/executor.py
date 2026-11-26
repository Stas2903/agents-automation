"""The executor: the untrusted zone of the pipeline.

* Authorization is re-checked on every step, not trusted from ``authorize`` earlier.
* Steps exchange plain string values only (the "values-only channel"); a tool's output can
  fill an *open* slot of a later step but can never overwrite a filled one or add steps.
* The first denial or failure halts the rest of the plan (fail-closed).
* Tool exceptions are logged by class name only, so record content never reaches logs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from agents_automation.pipeline.policy import ALLOWED, AuthTable, may_run
from agents_automation.pipeline.seal import PlanViolationError, SealedPlan, verify

logger = logging.getLogger(__name__)

Tool = Callable[[Mapping[str, str]], Mapping[str, str]]

RESERVED_PREFIX = "_"


class StepStatus(StrEnum):
    OK = "ok"
    DENIED = "denied"
    FAILED = "failed"
    SKIPPED = "skipped"  # not run because an earlier step halted the plan
    NOT_REQUESTED = "not_requested"  # excluded by ``only``


@dataclass(frozen=True)
class StepOutcome:
    step_id: str
    tool: str
    status: StepStatus
    output: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    error: str | None = None


@dataclass(frozen=True)
class StepsOutcome:
    outcomes: tuple[StepOutcome, ...]

    def outcome(self, step_id: str) -> StepOutcome:
        return next(o for o in self.outcomes if o.step_id == step_id)

    @property
    def ok(self) -> bool:
        return all(o.status in (StepStatus.OK, StepStatus.NOT_REQUESTED) for o in self.outcomes)


def execute(
    sealed: SealedPlan,
    tools: Mapping[str, Tool],
    *,
    only: Collection[str] | None = None,
    table: AuthTable = ALLOWED,
) -> StepsOutcome:
    verify(sealed)
    plan = sealed.plan
    if only is not None:
        unknown = sorted(set(only) - {s.step_id for s in plan.steps})
        if unknown:
            raise PlanViolationError(f"cannot run unknown step(s) {unknown}")

    outcomes: list[StepOutcome] = []
    accumulated: dict[str, str] = {}
    halted = False
    for step in plan.steps:
        if only is not None and step.step_id not in only:
            outcomes.append(StepOutcome(step.step_id, step.tool, StepStatus.NOT_REQUESTED))
            continue
        if halted:
            outcomes.append(StepOutcome(step.step_id, step.tool, StepStatus.SKIPPED))
            continue
        if not may_run(plan.role, step.tool, step.operation, table):
            outcomes.append(
                StepOutcome(step.step_id, step.tool, StepStatus.DENIED, error="not permitted")
            )
            halted = True
            continue
        tool = tools.get(step.tool)
        if tool is None:
            outcomes.append(
                StepOutcome(step.step_id, step.tool, StepStatus.DENIED, error="unknown tool")
            )
            halted = True
            continue

        inputs: dict[str, str] = {}
        missing: list[str] = []
        for key, value in step.slots.items():
            resolved = value if value is not None else accumulated.get(key)
            if resolved is None:
                missing.append(key)
            else:
                inputs[key] = resolved
        if missing:
            outcomes.append(
                StepOutcome(
                    step.step_id,
                    step.tool,
                    StepStatus.FAILED,
                    error=f"missing slot value(s): {sorted(missing)}",
                )
            )
            halted = True
            continue

        try:
            raw = tool(MappingProxyType(inputs))
        except Exception as exc:
            logger.warning("step %s (%s) failed: %s", step.step_id, step.tool, type(exc).__name__)
            outcomes.append(
                StepOutcome(step.step_id, step.tool, StepStatus.FAILED, error=type(exc).__name__)
            )
            halted = True
            continue

        output = {
            k: v
            for k, v in raw.items()
            if isinstance(k, str) and isinstance(v, str) and not k.startswith(RESERVED_PREFIX)
        }
        accumulated.update(output)
        outcomes.append(
            StepOutcome(step.step_id, step.tool, StepStatus.OK, MappingProxyType(output))
        )
    return StepsOutcome(tuple(outcomes))
