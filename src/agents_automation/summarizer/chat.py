"""The team chat: channels, membership and messages, anchored to one team timezone.

A message belongs to the day on which it was sent in the team's timezone, not in UTC: a
message sent at 23:30 UTC by a team two hours ahead belongs to the next day. Messages that
carry planted instructions (the injection screen's pattern layer) are held out of every
summary and of the raw tail.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib import resources
from types import MappingProxyType

from agents_automation.injection.patterns import find_markers
from agents_automation.llm.boundary import Sensitivity
from agents_automation.llm.text import normalize_text


@dataclass(frozen=True)
class Channel:
    name: str
    member_groups: frozenset[str]
    sensitivity: Sensitivity

    def readable_by(self, groups: Collection[str]) -> bool:
        """Members only. A channel without member groups is readable by nobody."""
        return bool(self.member_groups & set(groups))


@dataclass(frozen=True)
class ChatPost:
    post_id: str
    channel: str
    author: str
    sent_at: datetime  # timezone-aware
    text: str


class ChannelAccessError(Exception):
    pass


@dataclass(frozen=True)
class TeamChat:
    tz: timezone
    channels: Mapping[str, Channel]
    messages: tuple[ChatPost, ...]  # held-out messages are not in here
    held_out: tuple[str, ...]
    summaries_written_through: date

    def local_date(self, message: ChatPost) -> date:
        return message.sent_at.astimezone(self.tz).date()

    def channel_for(self, name: str, groups: Collection[str]) -> Channel:
        channel = self.channels.get(name)
        if channel is None or not channel.readable_by(groups):
            # the same answer for "does not exist" and "not a member": no channel probing
            raise ChannelAccessError(f"#{name} is not a channel you are a member of")
        return channel

    def in_period(self, channel: str, start: date, end: date) -> list[ChatPost]:
        return [
            m for m in self.messages if m.channel == channel and start <= self.local_date(m) <= end
        ]

    def by_id(self) -> dict[str, ChatPost]:
        return {m.post_id: m for m in self.messages}


def _parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {raw!r}")
    return ts


def build_chat(raw: Mapping[str, object]) -> TeamChat:
    if raw.get("synthetic") is not True:
        raise ValueError("the team chat must be marked synthetic")
    offset = raw["timezone_offset_hours"]
    if not isinstance(offset, int | float):
        raise ValueError("timezone_offset_hours must be a number")
    tz = timezone(timedelta(hours=offset), "team time")
    channels_raw = raw["channels"]
    messages_raw = raw["messages"]
    if not isinstance(channels_raw, dict) or not isinstance(messages_raw, list):
        raise ValueError("channels must be an object and messages a list")
    channels = {
        name: Channel(name, frozenset(c["member_groups"]), Sensitivity.parse(c["sensitivity"]))
        for name, c in channels_raw.items()
    }
    kept: list[ChatPost] = []
    held: list[ChatPost] = []
    for m in messages_raw:
        msg = ChatPost(m["id"], m["channel"], m["author"], _parse_ts(m["sent_at"]), m["text"])
        if msg.channel not in channels:
            raise ValueError(f"message {msg.post_id} names unknown channel {msg.channel!r}")
        (held if find_markers(normalize_text(msg.text)) else kept).append(msg)
    return TeamChat(
        tz=tz,
        channels=MappingProxyType(channels),
        messages=tuple(sorted(kept, key=lambda m: (m.sent_at, m.post_id))),
        held_out=tuple(m.post_id for m in held),
        summaries_written_through=date.fromisoformat(str(raw["summaries_written_through"])),
    )


def load_team_chat() -> TeamChat:
    text = (
        resources.files("agents_automation")
        .joinpath("data/team_chat.json")
        .read_text(encoding="utf-8")
    )
    return build_chat(json.loads(text))
