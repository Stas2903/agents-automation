"""Data-boundary routing: a model only ever sees data it is cleared for.

Every piece of data carries a :class:`Sensitivity` label, and every model (or embedder) is
deployed at a :class:`Placement` with a clearance. A call takes the highest label among its
inputs and goes to the first model, in preference order, whose clearance covers it. If no
model is cleared, the call fails closed with :class:`BoundaryError` and nothing is sent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum

from agents_automation.llm.seam import ChatModel, Embedder, Message


class Sensitivity(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2

    @classmethod
    def parse(cls, label: str) -> Sensitivity:
        """Unknown or empty labels count as CONFIDENTIAL: bad metadata never widens access.
        (A label that is missing altogether is a load error in the data loaders.)"""
        try:
            return cls[label.strip().upper()]
        except KeyError:
            return cls.CONFIDENTIAL


def highest(labels: Iterable[Sensitivity]) -> Sensitivity:
    return max(labels, default=Sensitivity.PUBLIC)


class BoundaryError(Exception):
    """No model is cleared for the data; the call was not made."""


@dataclass(frozen=True)
class Placement:
    name: str  # e.g. "self-hosted", "external"
    clearance: Sensitivity


def require_clearance(placement: Placement, sensitivity: Sensitivity) -> None:
    if placement.clearance < sensitivity:
        raise BoundaryError(
            f"{placement.name} is cleared for {placement.clearance.name}, "
            f"data is {sensitivity.name}: not sent"
        )


@dataclass(frozen=True)
class PlacedModel:
    model: ChatModel
    placement: Placement


@dataclass(frozen=True)
class PlacedEmbedder:
    embedder: Embedder
    placement: Placement


class ModelRouter:
    """Routes each call to the first model (in preference order) cleared for its data."""

    def __init__(self, models: Sequence[PlacedModel]) -> None:
        self.models = tuple(models)

    def pick(self, sensitivity: Sensitivity) -> PlacedModel:
        for placed in self.models:
            if placed.placement.clearance >= sensitivity:
                return placed
        raise BoundaryError(f"no model is cleared for {sensitivity.name} data: not sent")

    def complete(
        self, messages: Sequence[Message], *, sensitivity: Sensitivity, timeout_s: float
    ) -> str:
        placed = self.pick(sensitivity)
        require_clearance(placed.placement, sensitivity)  # checked again at the call itself
        return placed.model.complete(messages, timeout_s=timeout_s)
