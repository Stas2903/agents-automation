"""Domain models shared by every layer: the step list the interviewer builds, and final statuses."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StepKind = Literal["start", "task", "decision", "finish"]


class Step(BaseModel):
    """One step of a workflow. ``next`` lists the ids of the steps that may follow it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    kind: StepKind
    next: tuple[str, ...] = ()


class StepList(BaseModel):
    """A named workflow: the artifact the interviewer produces and the reviewer approves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    steps: tuple[Step, ...]

    def render_text(self) -> str:
        """Plain-text rendering used for display and for injection screening."""
        lines = [self.name]
        for step in self.steps:
            arrow = f" -> {', '.join(step.next)}" if step.next else ""
            lines.append(f"[{step.id}] ({step.kind}) {step.actor}: {step.title}{arrow}")
        return "\n".join(lines)


class Status(StrEnum):
    """Content verdict of a finalized interview, in precedence order (first wins).

    Whether the record actually reached a queue is reported separately (``submitted``).
    """

    INVALID = "invalid"  # nothing saved, or the step list failed validation after repair
    HELD_FOR_REVIEW = "held_for_review"  # looks tampered with; a human must look first
    DELIVERED_FLAGGED = "delivered_flagged"  # shipped, but the screen could not finish
    DELIVERED = "delivered"
