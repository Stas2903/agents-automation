"""Calendar periods in the team's timezone, and the parsing of a requested period.

The request text is untrusted. It is parsed here into three plain values (channel, start,
end) that are bound to the sealed plan's open slots; nothing else from the text is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

MAX_SPAN_DAYS = 366
MONTH_SPAN_DAYS = 56  # periods this long are read from monthly summaries where possible
WEEK_SPAN_DAYS = 14

_MONTHS = {
    name: i
    for i, name in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}
_CHANNEL = re.compile(r"#([a-z][a-z0-9_-]{0,63})\b", re.IGNORECASE)
_RANGE = re.compile(
    r"\b(?:from\s+(\d{4}-\d{2}-\d{2})\s+(?:to|until|till|through)\s+(\d{4}-\d{2}-\d{2}|today)"
    r"|since\s+(\d{4}-\d{2}-\d{2}))\b",
    re.I,
)
# "since February": the named period runs until today
_SINCE = re.compile(r"\bsince\b", re.I)
# "I need it today": a deadline, not a period
_DEADLINE_TODAY = re.compile(
    r"\b(?:need|want)s?\s+(?:it|them|this|that)\s+(?:by\s+)?today\b"
    r"|\b(?:by|before)\s+(?:the\s+end\s+of\s+)?today\b",
    re.I,
)
# "may" is a month only when it is not the verb ("May I have a recap ...?")
_MONTH = re.compile(
    r"\b("
    + "|".join(m for m in _MONTHS if m != "may")
    # "may" is the verb in "May I ...?" and in "I may need it later"
    + r"|(?<!\bi\s)(?<!\bwe\s)(?<!\byou\s)(?<!\bthey\s)(?<!\bit\s)(?<!\bthis\s)(?<!\bthat\s)"
    r"(?<!\bthere\s)may"
    r"(?!\s+(?:i|we|you|they|be|have|not|need|take|help|want|also|still|well)\b))\b"
    r"(?:\s+(\d{4}))?",
    re.IGNORECASE,
)
_LAST = re.compile(r"\blast\s+(\d{1,3}|two|three|four)\s+(day|week)s?\b", re.IGNORECASE)
_RELATIVE = re.compile(
    r"\b(last\s+month|this\s+month|last\s+week|this\s+week|yesterday|today)\b", re.IGNORECASE
)
_CLOSING_TODAY = re.compile(r"\b(?:up\s+to|until|till|through)\s+today\b", re.IGNORECASE)
_WORDS = {"two": 2, "three": 3, "four": 4}


def month_bounds(day: date) -> tuple[date, date]:
    first = day.replace(day=1)
    nxt = (first + timedelta(days=32)).replace(day=1)
    return first, nxt - timedelta(days=1)


def week_bounds(day: date) -> tuple[date, date]:
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def span_days(start: date, end: date) -> int:
    return (end - start).days + 1


@dataclass(frozen=True)
class PeriodRequest:
    channel: str
    start: date
    end: date


class PeriodError(ValueError):
    pass


def parse_period_request(text: str, today: date, default_channel: str = "general") -> PeriodRequest:
    m = _CHANNEL.search(text)
    channel = m.group(1).lower() if m else default_channel

    if r := _RANGE.search(text):
        first, last_day, since = r.group(1), r.group(2), r.group(3)
        try:
            if since is not None:
                start, end = date.fromisoformat(since), today
            else:
                end = today if last_day.lower() == "today" else date.fromisoformat(last_day)
                start = date.fromisoformat(first)
        except ValueError as exc:
            raise PeriodError(f"not a valid date: {exc}") from None
    else:
        # "... up to today" only closes the period; it is not a second period
        # channel names ("#may-launch") and deadlines ("I need it today") are not periods
        text = _DEADLINE_TODAY.sub(" ", _CHANNEL.sub(" ", text))
        plain = _CLOSING_TODAY.sub(" ", text)
        closed_by_today = plain != text or _SINCE.search(text) is not None
        last, mo, rel = _LAST.search(plain), _MONTH.search(plain), _RELATIVE.search(plain)
        found = sorted(
            {
                " ".join(m.group(0).lower().split())
                for pattern in (_LAST, _MONTH, _RELATIVE)
                for m in pattern.finditer(plain)
            }
        )
        if len(found) > 1:
            # "yesterday's thread about the May release": a guess would silently summarize the
            # wrong period, so ask instead (explicit dates above always win)
            quoted = ", ".join(f'"{f}"' for f in found)
            raise PeriodError(f"that names more than one period ({quoted}); please give one")
        start, end = _period(last, mo, rel, today)
        if closed_by_today:  # "from February until today": February 1 to today
            end = today
    if start > today:
        raise PeriodError("the period is in the future")
    return checked(PeriodRequest(channel, start, min(end, today)))


def _period(
    last: re.Match[str] | None,
    mo: re.Match[str] | None,
    rel: re.Match[str] | None,
    today: date,
) -> tuple[date, date]:
    if last is not None:
        raw = last.group(1).lower()
        n = _WORDS.get(raw) or int(raw)
        if n < 1:
            raise PeriodError("a period needs at least one day")
        days = n * (7 if last.group(2).lower() == "week" else 1)
        return today - timedelta(days=days - 1), today
    if mo is not None:
        month = _MONTHS[mo.group(1).lower()]
        # a month without a year is the most recent one: in March, "December" is last year's
        year = int(mo.group(2)) if mo.group(2) else today.year - (month > today.month)
        return month_bounds(date(year, month, 1))
    if rel is not None:
        return _relative(" ".join(rel.group(1).lower().split()), today)
    return today - timedelta(days=6), today  # documented default: the last 7 days


def _relative(phrase: str, today: date) -> tuple[date, date]:
    if phrase == "today":
        return today, today
    if phrase == "yesterday":
        day = today - timedelta(days=1)
        return day, day
    if phrase == "this week":
        return week_bounds(today)[0], today
    if phrase == "last week":
        return week_bounds(week_bounds(today)[0] - timedelta(days=1))
    if phrase == "this month":
        return today.replace(day=1), today
    return month_bounds(today.replace(day=1) - timedelta(days=1))  # last month


def checked(request: PeriodRequest) -> PeriodRequest:
    if request.start > request.end:
        raise PeriodError("the period starts after it ends")
    if span_days(request.start, request.end) > MAX_SPAN_DAYS:
        raise PeriodError(f"periods longer than {MAX_SPAN_DAYS} days are not summarized")
    return request
