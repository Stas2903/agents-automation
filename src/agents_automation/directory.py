"""Who belongs to which groups. Comes from the trusted side (an identity provider in a real
deployment); an unknown user belongs to no group and therefore sees nothing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType


@dataclass(frozen=True)
class Directory:
    members: Mapping[str, frozenset[str]]

    def groups(self, user_id: str) -> frozenset[str]:
        return self.members.get(user_id, frozenset())


def load_directory() -> Directory:
    raw = json.loads(
        resources.files("agents_automation")
        .joinpath("data/directory.json")
        .read_text(encoding="utf-8")
    )
    if raw.get("synthetic") is not True:
        raise ValueError("directory.json must be marked synthetic")
    return Directory(
        MappingProxyType({user: frozenset(groups) for user, groups in raw["users"].items()})
    )
