"""The read side: cover a requested period with the coarsest summaries that fit inside it.

The span of the request decides the coarsest granularity allowed: 56 days or more may use
monthly summaries, 14 days or more weekly ones, anything shorter only daily ones. The reader
places every whole month that has a summary first, then whole (Monday to Sunday) weeks in the
gaps, then daily summaries, so a request that starts mid-month still uses the months inside
it. Raw messages are read only for days that no summary covers (for example the tail
after the last written summary). The result is a coverage list that names every source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from agents_automation.summarizer.chat import ChatPost, TeamChat
from agents_automation.summarizer.periods import (
    MONTH_SPAN_DAYS,
    WEEK_SPAN_DAYS,
    month_bounds,
    span_days,
    week_bounds,
)
from agents_automation.summarizer.summaries import Granularity, Summary, SummaryShelf

SegmentKind = Literal["month", "week", "day", "raw", "quiet"]


@dataclass(frozen=True)
class Segment:
    kind: SegmentKind
    start: date
    end: date
    summary: Summary | None = None
    messages: tuple[ChatPost, ...] = ()

    def label(self) -> str:
        span = self.start.isoformat() if self.start == self.end else f"{self.start}..{self.end}"
        return f"{self.kind} {span}"


def allowed_granularities(start: date, end: date) -> tuple[Granularity, ...]:
    days = span_days(start, end)
    if days >= MONTH_SPAN_DAYS:
        return ("month", "week", "day")
    if days >= WEEK_SPAN_DAYS:
        return ("week", "day")
    return ("day",)


def _aligned(
    shelf: SummaryShelf, channel: str, granularity: Granularity, start: date, end: date
) -> list[Segment]:
    """Summaries of ``granularity`` whose whole period lies inside [start, end], in order."""
    bounds = month_bounds if granularity == "month" else week_bounds
    found: list[Segment] = []
    day = start
    while day <= end:
        first, last = bounds(day)
        summary = shelf.get(channel, granularity, first)
        if first >= start and last <= end and summary is not None:
            found.append(Segment(granularity, first, last, summary))
        day = last + timedelta(days=1)
    return found


def _days(
    chat: TeamChat, shelf: SummaryShelf, channel: str, start: date, end: date
) -> list[Segment]:
    out: list[Segment] = []
    day = start
    while day <= end:
        summary = shelf.get(channel, "day", day)
        if summary is not None:
            out.append(Segment("day", day, day, summary))
        else:
            raw = tuple(chat.in_period(channel, day, day))
            out.append(Segment("raw" if raw else "quiet", day, day, None, raw))
        day += timedelta(days=1)
    return out


def _cover(
    chat: TeamChat,
    shelf: SummaryShelf,
    channel: str,
    start: date,
    end: date,
    levels: tuple[Granularity, ...],
) -> list[Segment]:
    if start > end:
        return []
    if not levels:
        return _days(chat, shelf, channel, start, end)
    segments: list[Segment] = []
    cursor = start
    for seg in _aligned(shelf, channel, levels[0], start, end):
        segments += _cover(chat, shelf, channel, cursor, seg.start - timedelta(days=1), levels[1:])
        segments.append(seg)
        cursor = seg.end + timedelta(days=1)
    segments += _cover(chat, shelf, channel, cursor, end, levels[1:])
    return segments


def cover(
    chat: TeamChat, shelf: SummaryShelf, channel: str, start: date, end: date
) -> list[Segment]:
    """Whole months first, then whole weeks in the gaps, then days, then raw messages."""
    coarse = tuple(g for g in allowed_granularities(start, end) if g != "day")
    return _merge(_cover(chat, shelf, channel, start, end, coarse))


def _merge(segments: list[Segment]) -> list[Segment]:
    """Join neighbouring days that have no summary into one raw (or quiet) segment."""
    out: list[Segment] = []
    for seg in segments:
        prev = out[-1] if out else None
        if (
            prev is not None
            and seg.summary is None
            and prev.summary is None
            and prev.end + timedelta(days=1) == seg.start
        ):
            messages = prev.messages + seg.messages
            out[-1] = Segment("raw" if messages else "quiet", prev.start, seg.end, None, messages)
        else:
            out.append(seg)
    return out


def material_ids(segments: list[Segment]) -> set[str]:
    """Every message id the composed summary may cite."""
    ids: set[str] = set()
    for seg in segments:
        if seg.summary is not None:
            ids.update(seg.summary.cites)
        ids.update(m.post_id for m in seg.messages)
    return ids
