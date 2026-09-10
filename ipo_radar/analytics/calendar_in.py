"""
Indian clearing calendar.

ASBA mandates settle through the banking/clearing system, so "three business
days" means three CLEARING days. Treating every weekday as a settlement day
silently unlocks capital early whenever a mid-week holiday lands in the
window - and unlocking early is the dangerous direction, because the optimiser
will happily commit money that has not actually arrived.

Holidays come from three places, in priority order:

  1. `data/holidays.json` - an operator override. Always wins.
  2. A curated seed table of variable-date festivals (lunar/solar, not
     computable from a formula).
  3. Rules that ARE computable for any year: the fixed-date national holidays
     and Good Friday via the Gregorian Easter algorithm.

The seed table is SEED DATA, not gospel. Exchange and clearing holidays are
published annually by NSE and RBI and occasionally move.

A year with no seed entry falls back to rules only, and that errs in the
UNSAFE direction: fewer known holidays means fewer skipped days, an EARLIER
assumed unlock, a shorter blocked window, and therefore an optimiser that
commits MORE capital than has actually cleared. (An earlier draft of this
docstring claimed the opposite. It was wrong: a shorter block makes
`CapitalTimeline.can_fit` succeed more often, not less.)

So an unseeded year is padded by UNSEEDED_YEAR_PAD_DAYS extra clearing days
and `holidays_seeded()` exposes the condition, letting callers disclose that
the settlement date is an estimate rather than a calendar lookup.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from ..config import DATA_DIR

log = logging.getLogger("ipo_radar.calendar")

HOLIDAY_OVERRIDE = DATA_DIR / "holidays.json"
_WARNED_YEARS: set[int] = set()      # warn once per year, not once per call

# Fixed-date national holidays observed by banks and clearing houses.
FIXED_HOLIDAYS = (
    (1, 26),    # Republic Day
    (5, 1),     # Maharashtra Day / Labour Day
    (8, 15),    # Independence Day
    (10, 2),    # Gandhi Jayanti
    (12, 25),   # Christmas
)

# Variable-date festivals. These follow lunar/lunisolar calendars and cannot
# be derived arithmetically, so they are tabulated. VERIFY against the annual
# NSE/RBI clearing-holiday circular before relying on a live settlement date.
# Extra clearing days added when a year has no curated holiday list. Roughly
# the number of variable-date closures a typical Indian year carries in any
# three-day window, and it errs late rather than early.
UNSEEDED_YEAR_PAD_DAYS = 1

SEED_HOLIDAYS: dict[int, tuple[str, ...]] = {
    2025: (
        "2025-02-26",  # Mahashivratri
        "2025-03-14",  # Holi
        "2025-03-31",  # Id-ul-Fitr (Ramzan Id)
        "2025-04-10",  # Mahavir Jayanti
        "2025-04-14",  # Dr Ambedkar Jayanti
        "2025-04-18",  # Good Friday
        "2025-06-07",  # Bakri Id
        "2025-07-06",  # Muharram
        "2025-08-27",  # Ganesh Chaturthi
        "2025-10-01",  # Dussehra
        "2025-10-21",  # Diwali Laxmi Pujan
        "2025-10-22",  # Diwali Balipratipada
        "2025-11-05",  # Guru Nanak Jayanti
    ),
    2026: (
        "2026-01-01",  # New Year (bank clearing)
        "2026-02-15",  # Mahashivratri
        "2026-03-04",  # Holi
        "2026-03-21",  # Id-ul-Fitr
        "2026-03-31",  # Mahavir Jayanti
        "2026-04-03",  # Good Friday
        "2026-04-14",  # Dr Ambedkar Jayanti
        "2026-05-28",  # Bakri Id
        "2026-06-26",  # Muharram
        "2026-09-15",  # Ganesh Chaturthi
        "2026-10-20",  # Dussehra
        "2026-11-08",  # Diwali Laxmi Pujan
        "2026-11-09",  # Diwali Balipratipada
        "2026-11-24",  # Guru Nanak Jayanti
    ),
    2027: (
        "2027-01-01",
        "2027-03-06",  # Mahashivratri
        "2027-03-23",  # Holi
        "2027-03-11",  # Id-ul-Fitr
        "2027-03-26",  # Good Friday
        "2027-04-14",
        "2027-05-17",  # Bakri Id
        "2027-06-16",  # Muharram
        "2027-09-04",  # Ganesh Chaturthi
        "2027-10-09",  # Dussehra
        "2027-10-29",  # Diwali
        "2027-11-13",  # Guru Nanak Jayanti
    ),
}


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lm = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lm) // 451
    month, day = divmod(h + lm - 7 * m + 114, 31)
    return date(year, month, day + 1)


def good_friday(year: int) -> date:
    return easter_sunday(year) - timedelta(days=2)


def _load_override() -> dict[int, set[date]]:
    """`data/holidays.json`: {"2026": ["2026-03-04", ...]} or a flat list."""
    if not HOLIDAY_OVERRIDE.exists():
        return {}
    try:
        raw = json.loads(HOLIDAY_OVERRIDE.read_text())
    except Exception as exc:
        log.warning("bad holidays.json: %s", exc)
        return {}
    out: dict[int, set[date]] = {}
    items = (raw.items() if isinstance(raw, dict)
             else [("*", raw)] if isinstance(raw, list) else [])
    for _key, values in items:
        for v in values or []:
            try:
                d = date.fromisoformat(str(v))
            except ValueError:
                continue
            out.setdefault(d.year, set()).add(d)
    return out


def holidays_for(year: int) -> frozenset[date]:
    """
    Every non-trading clearing day in `year`, weekends excluded.

    Deliberately NOT cached. A long-running daemon must pick up an ad-hoc
    clearing holiday the moment it is dropped into `data/holidays.json`;
    memoising this would freeze the calendar for the life of the process and
    silently keep settling against a stale one. Reading a few hundred bytes of
    JSON costs microseconds and is called a handful of times per cycle.
    """
    days: set[date] = set()
    for month, day in FIXED_HOLIDAYS:
        try:
            days.add(date(year, month, day))
        except ValueError:
            pass
    days.add(good_friday(year))
    seed = SEED_HOLIDAYS.get(year)
    if seed:
        for iso in seed:
            try:
                days.add(date.fromisoformat(iso))
            except ValueError:
                continue
    elif year not in _WARNED_YEARS:
        # Without the cache this would otherwise log on every single call.
        _WARNED_YEARS.add(year)
        log.warning("no seeded clearing holidays for %d - using fixed-date "
                    "rules only and padding settlement by %d clearing day(s)",
                    year, UNSEEDED_YEAR_PAD_DAYS)
    for d in _load_override().get(year, set()):
        days.add(d)
    # weekends are handled separately; keep the set to genuine weekday closures
    return frozenset(d for d in days if d.weekday() < 5)


def holidays_seeded(year: int) -> bool:
    """True when this year has a curated variable-date holiday list."""
    return year in SEED_HOLIDAYS


def is_clearing_day(d: date) -> bool:
    """A day on which banks settle: a weekday that is not a holiday."""
    return d.weekday() < 5 and d not in holidays_for(d.year)


def next_clearing_day(d: date) -> date:
    cur = d + timedelta(days=1)
    while not is_clearing_day(cur):
        cur += timedelta(days=1)
    return cur


def add_clearing_days(start: date, days: int) -> date:
    """
    Advance N clearing days, padding when the calendar is unknown.

    Without the pad an unseeded year silently under-counts holidays and
    releases capital early - the one direction that lets the optimiser commit
    money it does not have.
    """
    target = days
    if not holidays_seeded(start.year):
        target += UNSEEDED_YEAR_PAD_DAYS
    cur, moved = start, 0
    while moved < target:
        cur += timedelta(days=1)
        if is_clearing_day(cur):
            moved += 1
    return cur


def describe(year: int) -> list[str]:
    return sorted(d.isoformat() for d in holidays_for(year))
