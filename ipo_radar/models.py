"""Domain objects. Plain dataclasses so they round-trip through SQLite/JSON."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any


# ---------------------------------------------------------------- categories
QIB = "QIB"
NII = "NII"
SNII = "sNII"
BNII = "bNII"
RETAIL = "RETAIL"
EMPLOYEE = "EMPLOYEE"
SHAREHOLDER = "SHAREHOLDER"

CATEGORY_ORDER = [QIB, NII, BNII, SNII, RETAIL, EMPLOYEE, SHAREHOLDER]


@dataclass
class IPO:
    symbol: str
    name: str
    status: str = "Unknown"            # Active | Forthcoming | Closed | Listed
    board: str = "MAINBOARD"           # MAINBOARD | SME
    open_date: str | None = None       # ISO yyyy-mm-dd
    close_date: str | None = None
    listing_date: str | None = None
    price_low: float | None = None
    price_high: float | None = None
    lot_size: int | None = None
    issue_size_shares: float | None = None
    issue_size_cr: float | None = None
    fresh_issue_cr: float | None = None
    ofs_cr: float | None = None
    face_value: float | None = None
    registrar: str | None = None
    sector: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------- helpers
    @property
    def cap_price(self) -> float | None:
        return self.price_high or self.price_low

    @property
    def lot_value(self) -> float | None:
        if self.cap_price and self.lot_size:
            return self.cap_price * self.lot_size
        return None

    def min_lots_for(self, ticket: float) -> int | None:
        """Smallest lot count whose value clears `ticket` (sNII/bNII entry)."""
        lv = self.lot_value
        if not lv:
            return None
        return max(1, math.ceil(ticket / lv))

    def max_retail_lots(self, cap: float = 200_000.0) -> int | None:
        """Most lots a retail applicant may bid (bid value must stay < 2L)."""
        lv = self.lot_value
        if not lv:
            return None
        return max(1, int(cap // lv))

    def days(self) -> tuple[date | None, date | None]:
        def p(s: str | None) -> date | None:
            try:
                return date.fromisoformat(s) if s else None
            except ValueError:
                return None
        return p(self.open_date), p(self.close_date)

    def is_open(self, today: date | None = None) -> bool:
        today = today or date.today()
        o, c = self.days()
        return bool(o and c and o <= today <= c)

    def bid_progress(self, now: datetime | None = None) -> float:
        """0.0 at open bell to 1.0 at close. Drives subscription projection."""
        now = now or datetime.now()
        o, c = self.days()
        if not (o and c):
            return 0.0
        start = datetime.combine(o, datetime.min.time()).replace(hour=10)
        end = datetime.combine(c, datetime.min.time()).replace(hour=17)
        if now <= start:
            return 0.0
        if now >= end:
            return 1.0
        # only count 10:00-17:00 windows, not overnight dead time
        total = live = 0.0
        cur = o
        while cur <= c:
            d_start = datetime.combine(cur, datetime.min.time()).replace(hour=10)
            d_end = datetime.combine(cur, datetime.min.time()).replace(hour=17)
            total += 7.0
            if now >= d_end:
                live += 7.0
            elif now > d_start:
                live += (now - d_start).total_seconds() / 3600.0
            cur = date.fromordinal(cur.toordinal() + 1)
        return min(1.0, live / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CategoryBid:
    category: str
    offered: float | None
    bid: float | None
    times: float | None
    sr_no: str = ""

    @property
    def subscribed(self) -> float:
        if self.times is not None:
            return self.times
        if self.offered and self.bid:
            return self.bid / self.offered
        return 0.0


@dataclass
class SubscriptionSnapshot:
    symbol: str
    ts: str
    categories: dict[str, CategoryBid] = field(default_factory=dict)
    total_times: float | None = None
    source: str = "nse"

    def get(self, cat: str) -> CategoryBid | None:
        return self.categories.get(cat)

    def times(self, cat: str) -> float:
        c = self.categories.get(cat)
        return c.subscribed if c else 0.0

    # sNII and bNII are subdivisions of NII; counting them alongside their
    # parent would double the non-institutional book.
    TOP_LEVEL = (QIB, NII, RETAIL, EMPLOYEE, SHAREHOLDER)

    def total_subscription(self) -> float | None:
        """
        Overall subscription multiple.

        `total_times` comes from NSE's "Total" row, which is frequently
        absent - it reads None even on a fully subscribed book - so callers
        that branch on "is this undersubscribed" cannot rely on it. This
        derives the figure from shares bid over shares offered instead, and
        returns None only when there is genuinely nothing to compute from.
        """
        if isinstance(self.total_times, (int, float)) and self.total_times > 0:
            return float(self.total_times)
        offered = bid = 0.0
        for name in self.TOP_LEVEL:
            c = self.categories.get(name)
            if not c or not c.offered or c.bid is None:
                continue
            offered += float(c.offered)
            bid += float(c.bid)
        if offered <= 0:
            return None
        return bid / offered


@dataclass
class GMPSnapshot:
    symbol: str
    ts: str
    gmp: float | None = None
    price: float | None = None
    est_listing: float | None = None
    est_gain_pct: float | None = None
    source: str = "ipowatch"
    raw_name: str = ""


@dataclass
class Financials:
    """Restated consolidated figures from the RHP, in Rs crore."""
    fy: str = ""
    revenue: float | None = None
    ebitda: float | None = None
    pat: float | None = None
    net_worth: float | None = None
    total_debt: float | None = None
    eps: float | None = None
    ronw_pct: float | None = None
    roce_pct: float | None = None
    cfo: float | None = None


@dataclass
class AllotmentOdds:
    """Result of the allotment engine for one category."""
    category: str
    subscription_x: float
    lots_applied: int
    ticket_value: float
    p_single: float                # P(>=1 lot) for ONE application
    expected_lots: float           # E[lots allotted] per application
    p_any_multi: float             # P(>=1 lot) across N applications
    expected_lots_multi: float
    mechanism: str                 # "full" | "lottery" | "proportionate"
    max_allottees: float | None = None
    est_applications: float | None = None
    note: str = ""


@dataclass
class Verdict:
    symbol: str
    ts: str
    score: float
    grade: str                     # STRONG APPLY / APPLY / NEUTRAL / AVOID /
                                   # STRONG AVOID / SPECULATIVE FLIP
    stance_listing: str
    stance_longterm: str
    confidence: float
    factors: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    expected_listing_gain_pct: float | None = None
    listing_gain_range: tuple[float, float] | None = None
    p_positive_listing: float | None = None
    allotment: dict[str, AllotmentOdds] = field(default_factory=dict)
    projected_subscription: dict[str, float] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    # Bayesian posteriors per category (median / CI / confidence)
    posteriors: dict[str, Any] = field(default_factory=dict)
    # what the guardrail interceptor corrected or rejected
    guardrails: dict[str, Any] = field(default_factory=dict)
    # T-day clock, spike signals and the operational call
    tday: dict[str, Any] = field(default_factory=dict)
    # score weights actually used, after the bid-window ramp
    weights_used: dict[str, float] = field(default_factory=dict)
    # full valuation payload (P/E, EV/EBITDA, sector, basis)
    valuation: dict[str, Any] = field(default_factory=dict)
    # macro regime snapshot this verdict was scored against
    regime: dict[str, Any] = field(default_factory=dict)
    # anchor-book composition and lock-in overhang risk
    anchors: dict[str, Any] = field(default_factory=dict)
    # terminal veto state (Value Trap / speculative-flip exception), carried so
    # that any later re-grade cannot silently undo it
    veto: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["allotment"] = {k: asdict(v) for k, v in self.allotment.items()}
        return d
