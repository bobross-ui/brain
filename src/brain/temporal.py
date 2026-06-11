import calendar
import re
from datetime import date, datetime, timedelta

from pydantic import BaseModel


class TimeRef(BaseModel):
    date: str | None = None
    start: str | None = None
    end: str | None = None
    raw_phrase: str
    anchor: str | None = None
    method: str


_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_PATTERN = "|".join(
    sorted((re.escape(month) for month in _MONTHS), key=len, reverse=True)
)
_DAY_MONTH_YEAR = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
    rf"(?P<month>{_MONTH_PATTERN})[,]?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_MONTH_DAY_YEAR = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+"
    rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")
_MONTH_YEAR = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})[,]?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_DAYS_AGO = re.compile(
    r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)\s+days?\s+ago\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _point_time_ref(
    value: date,
    match: re.Match[str],
    *,
    anchor: str | None,
    method: str,
) -> TimeRef:
    return TimeRef(
        date=value.isoformat(),
        raw_phrase=match.group(0),
        anchor=anchor,
        method=method,
    )


def _range_time_ref(
    start: date,
    end: date,
    match: re.Match[str],
    *,
    anchor: str | None,
    method: str,
) -> TimeRef:
    return TimeRef(
        start=start.isoformat(),
        end=end.isoformat(),
        raw_phrase=match.group(0),
        anchor=anchor,
        method=method,
    )


def _absolute_date(text: str, *, anchor: str | None) -> TimeRef | None:
    for pattern, method in (
        (_ISO_DATE, "absolute_iso"),
        (_DAY_MONTH_YEAR, "absolute_day_month_year"),
        (_MONTH_DAY_YEAR, "absolute_month_day_year"),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        try:
            value = date(
                int(match.group("year")),
                _month_number(match.group("month")),
                int(match.group("day")),
            )
        except ValueError:
            return None
        return _point_time_ref(value, match, anchor=anchor, method=method)

    match = _MONTH_YEAR.search(text)
    if match is None:
        return None
    year = int(match.group("year"))
    month = _month_number(match.group("month"))
    last_day = calendar.monthrange(year, month)[1]
    return _range_time_ref(
        date(year, month, 1),
        date(year, month, last_day),
        match,
        anchor=anchor,
        method="absolute_month",
    )


def _month_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    return _MONTHS[value.lower()]


def _anchor_date(anchor: str | None) -> date | None:
    if anchor is None:
        return None
    try:
        return datetime.fromisoformat(anchor.replace("Z", "+00:00")).date()
    except ValueError:
        resolved = _absolute_date(anchor, anchor=None)
        if resolved is None or resolved.date is None:
            return None
        return date.fromisoformat(resolved.date)


def resolve(phrase: str, anchor: str | None = None) -> TimeRef | None:
    text = str(phrase)
    anchored = _anchor_date(anchor)
    if anchored is not None:
        match = _DAYS_AGO.search(text)
        if match is not None:
            raw_count = match.group("count").lower()
            count = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS[raw_count]
            return _point_time_ref(
                anchored - timedelta(days=count),
                match,
                anchor=anchor,
                method="relative_days_ago",
            )

        match = re.search(r"\byesterday\b", text, re.IGNORECASE)
        if match is not None:
            return _point_time_ref(
                anchored - timedelta(days=1),
                match,
                anchor=anchor,
                method="relative_yesterday",
            )

        match = re.search(r"\blast week\b", text, re.IGNORECASE)
        if match is not None:
            return _range_time_ref(
                anchored - timedelta(days=7),
                anchored - timedelta(days=1),
                match,
                anchor=anchor,
                method="relative_last_week",
            )

        match = re.search(r"\blast weekend\b", text, re.IGNORECASE)
        if match is not None:
            previous_sunday = anchored - timedelta(days=anchored.weekday() + 1)
            return _range_time_ref(
                previous_sunday - timedelta(days=1),
                previous_sunday,
                match,
                anchor=anchor,
                method="relative_last_weekend",
            )

    return _absolute_date(text, anchor=anchor)


def query_date_filters(question: str) -> dict[str, str]:
    time_ref = resolve(question)
    if time_ref is None:
        return {}

    lowered = question.lower()
    phrase_start = lowered.find(time_ref.raw_phrase.lower())
    prefix = lowered[:phrase_start] if phrase_start >= 0 else lowered
    if time_ref.date is not None:
        if re.search(r"\bbefore\b", prefix):
            return {"event_before": time_ref.date}
        if re.search(r"\b(after|since)\b", prefix):
            return {"event_after": time_ref.date}
        return {"event_on": time_ref.date}

    if time_ref.start is not None and time_ref.end is not None:
        if re.search(r"\bbefore\b", prefix):
            return {"event_before": time_ref.start}
        if re.search(r"\b(after|since)\b", prefix):
            return {"event_after": time_ref.end}
        return {
            "event_after": time_ref.start,
            "event_before": time_ref.end,
        }
    return {}
