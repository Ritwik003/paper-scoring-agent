"""Work out calendar dates for a given week number.

We use the US-style week convention (matching the weekly calendar you're
working from): each week runs SUNDAY -> SATURDAY, and week 1 is the week
that contains January 1st.
"""
from datetime import date, timedelta


def _week1_start(year: int) -> date:
    """The Sunday that begins week 1 of a year (the week containing Jan 1)."""
    jan1 = date(year, 1, 1)
    days_since_sunday = (jan1.weekday() + 1) % 7
    return jan1 - timedelta(days=days_since_sunday)


def week_range(year: int, week: int) -> tuple[date, date]:
    """Return (sunday_start, saturday_end) for the given year and week number."""
    start = _week1_start(year) + timedelta(weeks=week - 1)
    end = start + timedelta(days=6)
    return start, end


def current_week(today: date | None = None) -> tuple[int, int]:
    """Return the (year, week_number) that `today` falls in."""
    today = today or date.today()
    if today >= _week1_start(today.year + 1):
        return today.year + 1, 1
    year = today.year
    start = _week1_start(year)
    if today < start:
        year -= 1
        start = _week1_start(year)
    week = (today - start).days // 7 + 1
    return year, week


def last_completed_week(today: date | None = None) -> tuple[int, int]:
    """Return (year, week) of the week just before today's week."""
    today = today or date.today()
    return current_week(today - timedelta(days=7))