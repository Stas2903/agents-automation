"""Route a request surface to a role. Deterministic; an unknown surface is refused.

The surface comes from the trusted envelope and fixes which roles may answer there. A surface
with one role needs no choice. A surface with several roles needs a ``chosen`` role (picked
by the intent router), and that choice is only accepted if the surface allows it: text can
pick which allowed role answers, never what a role may do or which other roles exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

SURFACE_ROLES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "interview_chat": ("interviewer",),
        "review_console": ("reviewer",),
        "office_chat": ("knowledge", "summarizer", "interviewer"),
    }
)


class RoutingError(Exception):
    pass


def allowed_roles(surface: str) -> tuple[str, ...]:
    try:
        return SURFACE_ROLES[surface]
    except KeyError:
        raise RoutingError(f"unknown surface {surface!r}: refused") from None


def route(surface: str, chosen: str | None = None) -> str:
    roles = allowed_roles(surface)
    if chosen is None:
        if len(roles) != 1:
            raise RoutingError(f"surface {surface!r} needs a role choice: refused")
        return roles[0]
    if chosen not in roles:
        raise RoutingError(f"role {chosen!r} is not available on {surface!r}: refused")
    return chosen
