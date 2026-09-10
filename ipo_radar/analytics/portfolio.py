"""
Capital allocation across IPOs, in time.

Two constraints govern this, and only one of them is about money.

The first is the obvious one: one PAN may submit exactly one application per
IPO (duplicates are rejected at the registrar), and every application blocks
cash via ASBA. So the question is which applications maximise the objective
per rupee committed.

The second is temporal, and it is the one a naive knapsack gets wrong. Blocked
cash is not spent — it is *unavailable for a while*. Under the T+3 listing
regime the mandate holds funds from the close date until roughly three
business days later. An issue closing on the 11th therefore sterilises capital
through the 16th, and if a far better issue closes on the 15th, spending
everything on the 11th means missing it entirely. The optimiser must model the
calendar, not just the budget.

Hence: applications carry an interval, a `CapitalTimeline` tracks how much is
committed on every day, and high-conviction issues claim their window BEFORE
mediocre earlier ones get to spend it. Capital is reserved only where the
windows genuinely overlap, so a good issue closing after everything else has
already unlocked costs the earlier applications nothing.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Iterable

from ..config import PortfolioConfig
from ..models import AllotmentOdds, IPO, RETAIL
from ..util import today_ist
from .calendar_in import add_clearing_days, is_clearing_day, next_clearing_day

log = logging.getLogger("ipo_radar.portfolio")


class Objective(str, Enum):
    MAXIMIZE_ROI = "maximize_roi"
    MAXIMIZE_ABSOLUTE_PROFIT = "maximize_absolute_profit"

    @classmethod
    def parse(cls, value: str | None) -> "Objective":
        try:
            return cls(str(value or "").strip().lower())
        except ValueError:
            return cls.MAXIMIZE_ROI


# ------------------------------------------------------------- calendar
def add_business_days(start: date, days: int) -> date:
    """
    Advance by N CLEARING days - weekends and Indian banking holidays.

    ASBA mandates settle through the clearing system, so a mid-week holiday
    genuinely pushes the release by a further 24-48 hours. Counting every
    weekday as a settlement day would unlock capital early on paper, and the
    optimiser would then commit money that has not actually arrived.
    """
    return add_clearing_days(start, days)


def asba_unlock_date(close_date: date, business_days: int = 3) -> date:
    """
    The day the sponsor bank releases the mandate (T+3 clearing days).

    Note this is the RELEASE date, not the date the money can be used again -
    see `capital_available_from`.
    """
    return add_clearing_days(close_date, business_days)


def capital_available_from(unlock: date) -> date:
    """
    The first day released capital can actually back a new bid.

    Sponsor banks (SBI, HDFC, ICICI) push ASBA revocations in an evening batch,
    typically between 18:00 and 23:59 IST on the release date. Money released
    on the evening of day T is therefore useless for an issue CLOSING on day T
    - the UPI mandate cut-off is 17:00, hours before the funds land. Treating
    it as same-day available would double-spend the balance on paper and
    produce a bid that fails at the bank.
    """
    return next_clearing_day(unlock)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ structures
@dataclass
class Application:
    symbol: str
    name: str
    category: str
    lots: int
    shares: int
    capital: float
    p_allot: float
    expected_shares: float
    expected_profit: float
    roi_on_blocked: float
    pan_index: int = 1
    rationale: str = ""
    score: float = 0.0
    close_date: str | None = None
    block_from: str | None = None
    unlock_on: str | None = None        # sponsor bank releases the mandate
    available_from: str | None = None   # first day it can back another bid
    days_blocked: int = 0
    reserved: bool = False          # claimed its window ahead of the queue

    def value(self, objective: Objective) -> float:
        return (self.expected_profit
                if objective is Objective.MAXIMIZE_ABSOLUTE_PROFIT
                else self.roi_on_blocked)


@dataclass
class CashFlowEvent:
    day: date
    committed: float = 0.0          # newly blocked on this day
    released: float = 0.0           # freed on this day
    outstanding: float = 0.0        # total blocked after the day's movements
    headroom: float = 0.0
    symbols_in: list[str] = field(default_factory=list)
    symbols_out: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"date": self.day.isoformat(),
                "committed": round(self.committed, 2),
                "released": round(self.released, 2),
                "outstanding": round(self.outstanding, 2),
                "headroom": round(self.headroom, 2),
                "in": self.symbols_in, "out": self.symbols_out}


@dataclass
class Allocation:
    applications: list[Application] = field(default_factory=list)
    capital_used: float = 0.0            # peak simultaneous commitment
    capital_available: float = 0.0
    expected_profit: float = 0.0
    skipped: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[CashFlowEvent] = field(default_factory=list)
    objective: str = Objective.MAXIMIZE_ROI.value
    reserved_symbols: list[str] = field(default_factory=list)

    @property
    def expected_roi_pct(self) -> float:
        return (self.expected_profit / self.capital_used * 100.0
                if self.capital_used else 0.0)

    @property
    def peak_blocked(self) -> float:
        return self.capital_used

    def timeline_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.timeline]


class CapitalTimeline:
    """
    Day-by-day ledger of committed capital.

    Feasibility is checked across an application's whole blocked window rather
    than against a single running total, which is what lets a later issue keep
    its claim on cash an earlier one would otherwise have sterilised.
    """

    def __init__(self, capital: float) -> None:
        self.capital = float(capital)
        self._blocked: dict[date, float] = defaultdict(float)
        self._in: dict[date, list[str]] = defaultdict(list)
        self._out: dict[date, list[str]] = defaultdict(list)

    @staticmethod
    def _days(start: date, end: date) -> Iterable[date]:
        """Blocked on [start, end) - funds are usable again ON the unlock day."""
        cur = start
        while cur < end:
            yield cur
            cur += timedelta(days=1)

    def can_fit(self, start: date, end: date, amount: float) -> bool:
        if amount <= 0:
            return False
        return all(self._blocked[d] + amount <= self.capital + 1e-6
                   for d in self._days(start, end))

    def headroom(self, start: date, end: date) -> float:
        used = max((self._blocked[d] for d in self._days(start, end)),
                   default=0.0)
        return max(0.0, self.capital - used)

    def commit(self, start: date, end: date, amount: float,
               symbol: str = "") -> None:
        for d in self._days(start, end):
            self._blocked[d] += amount
        if symbol:
            self._in[start].append(symbol)
            self._out[end].append(symbol)

    def peak(self) -> float:
        return max(self._blocked.values(), default=0.0)

    def events(self) -> list[CashFlowEvent]:
        days = sorted(set(self._blocked) | set(self._in) | set(self._out))
        out: list[CashFlowEvent] = []
        prev = 0.0
        for d in days:
            outstanding = self._blocked.get(d, 0.0)
            delta = outstanding - prev
            out.append(CashFlowEvent(
                day=d,
                committed=max(0.0, delta), released=max(0.0, -delta),
                outstanding=outstanding,
                headroom=max(0.0, self.capital - outstanding),
                symbols_in=sorted(self._in.get(d, [])),
                symbols_out=sorted(self._out.get(d, []))))
            prev = outstanding
        # closing event: the day the last mandate releases
        if days:
            last_end = max(days) + timedelta(days=1)
            if self._blocked.get(last_end, 0.0) == 0.0 and prev > 0:
                out.append(CashFlowEvent(
                    day=last_end, released=prev, outstanding=0.0,
                    headroom=self.capital,
                    symbols_out=sorted(self._out.get(last_end, []))))
        return out


# ------------------------------------------------------------- optimiser
class PortfolioOptimizer:
    # The score thresholds have ONE home: PortfolioConfig. Restating them here
    # as literal defaults is what let the T-Day alert floor sit at 50 while the
    # optimiser was raised to 61 - the engine passed the configured value, so
    # the stale literal stayed invisible until a NEUTRAL issue was refused
    # capital and pushed a CRITICAL "apply now" alert in the same breath.
    # None means "ask the config", resolved per call rather than frozen at
    # import so a reloaded config is picked up.
    def __init__(self, total_capital: float, num_pans: int = 1,
                 min_score: float | None = None,
                 min_expected_gain_pct: float = 2.0,
                 objective: str | Objective = Objective.MAXIMIZE_ROI,
                 lookahead_days: int = 14,
                 settlement_days: int = 3,
                 reserve_min_score: float | None = None,
                 today: date | None = None) -> None:
        defaults = PortfolioConfig()
        self.capital = total_capital
        self.pans = max(1, num_pans)
        self.min_score = (defaults.min_score if min_score is None
                          else min_score)
        self.min_gain = min_expected_gain_pct
        self.objective = (objective if isinstance(objective, Objective)
                          else Objective.parse(objective))
        self.lookahead_days = max(0, lookahead_days)
        self.settlement_days = max(0, settlement_days)
        self.reserve_min_score = (defaults.reserve_min_score
                                  if reserve_min_score is None
                                  else reserve_min_score)
        # IST, never the host clock. A UTC server rolls over at 05:30 IST, so
        # between 18:30 and 23:59 UTC `date.today()` still reports yesterday in
        # Indian terms - which would shift every block window by a full day and
        # let the optimiser fund an issue whose bidding had already closed.
        self.today = today or today_ist()

    # ----------------------------------------------------- plan building
    def _plans_for(self, c: dict[str, Any]) -> list[Application]:
        """Every affordable application plan for one IPO, richest last."""
        ipo: IPO = c["ipo"]
        gain = c.get("expected_gain_pct")
        score = float(c.get("score") or 0.0)
        close = _parse_date(ipo.close_date)
        if close is None or gain is None:
            return []
        block_from = max(close, self.today)
        unlock = asba_unlock_date(close, self.settlement_days)
        if unlock <= block_from:
            unlock = next_clearing_day(block_from)
        # the evening-release rule: unusable until the next clearing day
        available = capital_available_from(unlock)

        plans: list[Application] = []
        for _key, odds in (c.get("allotment") or {}).items():
            if not isinstance(odds, AllotmentOdds) or odds.mechanism == "n/a":
                continue
            if odds.ticket_value <= 0 or odds.ticket_value > self.capital:
                continue
            lot = ipo.lot_size or 0
            exp_shares = odds.expected_lots * lot
            profit = exp_shares * (ipo.cap_price or 0) * (gain / 100.0)
            roi = profit / odds.ticket_value * 100.0 if odds.ticket_value else 0.0
            app = Application(
                symbol=ipo.symbol, name=ipo.name, category=odds.category,
                lots=odds.lots_applied, shares=odds.lots_applied * lot,
                capital=odds.ticket_value, p_allot=odds.p_single,
                expected_shares=exp_shares, expected_profit=profit,
                roi_on_blocked=roi, score=score,
                rationale=f"{odds.mechanism}; {odds.note}",
                close_date=close.isoformat(),
                block_from=block_from.isoformat(),
                unlock_on=unlock.isoformat(),
                available_from=available.isoformat(),
                days_blocked=(available - block_from).days)
            if app.expected_profit > 0:
                plans.append(app)
        plans.sort(key=lambda a: a.capital)
        return plans

    # ------------------------------------------------------------ filters
    def _eligible(self, c: dict[str, Any], alloc: Allocation) -> bool:
        ipo: IPO = c["ipo"]
        close = _parse_date(ipo.close_date)
        if close is None:
            alloc.skipped.append({"symbol": ipo.symbol, "why": "no close date"})
            return False
        if close < self.today:
            return False                       # already shut, nothing to do
        if (close - self.today).days > self.lookahead_days:
            alloc.skipped.append({
                "symbol": ipo.symbol,
                "why": (f"closes in {(close - self.today).days}d, beyond the "
                        f"{self.lookahead_days}d planning horizon")})
            return False

        veto = c.get("veto") or {}
        if veto.get("value_trap") and not veto.get("speculative_flip"):
            alloc.skipped.append({
                "symbol": ipo.symbol,
                "why": (veto.get("reason")
                        or "Value Trap: earns below its cost of capital")})
            return False

        gain = c.get("expected_gain_pct")
        score = float(c.get("score") or 0.0)
        if gain is None:
            alloc.skipped.append({"symbol": ipo.symbol,
                                  "why": "no listing estimate available"})
            return False
        if score < self.min_score:
            alloc.skipped.append({"symbol": ipo.symbol,
                                  "why": f"score {score:.0f} below cutoff "
                                         f"{self.min_score:.0f}"})
            return False
        if gain < self.min_gain:
            alloc.skipped.append({"symbol": ipo.symbol,
                                  "why": f"expected listing gain {gain:+.1f}% "
                                         "too thin to justify blocking cash"})
            return False
        return True

    # ------------------------------------------------------- upgrade pass
    def _upgrade_for_absolute_profit(self, alloc: Allocation,
                                     by_symbol: dict[str, list[Application]],
                                     timeline: CapitalTimeline) -> None:
        """
        Convert idle capital into rupees.

        With the selection already made on density, whatever capital remains
        free is earning nothing. This walks the chosen applications and swaps
        each to the largest plan for the same issue that still fits its own
        blocked window, taking the extra profit. Richest upgrade first, so the
        scarce headroom goes where it buys the most.
        """
        while True:
            best_gain, best_idx, best_plan = 0.0, None, None
            for i, app in enumerate(alloc.applications):
                for plan in by_symbol.get(app.symbol, []):
                    if plan.capital <= app.capital:
                        continue
                    gain = plan.expected_profit - app.expected_profit
                    if gain <= best_gain:
                        continue
                    start = _parse_date(app.block_from)
                    end = _parse_date(app.available_from or app.unlock_on)
                    if start is None or end is None:
                        continue
                    # only the DELTA needs to fit; the original is committed
                    if not timeline.can_fit(start, end, plan.capital - app.capital):
                        continue
                    best_gain, best_idx, best_plan = gain, i, plan
            if best_idx is None or best_plan is None:
                return
            old = alloc.applications[best_idx]
            start = _parse_date(old.block_from)
            end = _parse_date(old.available_from or old.unlock_on)
            timeline.commit(start, end, best_plan.capital - old.capital)
            upgraded = Application(**{**best_plan.__dict__,
                                      "pan_index": old.pan_index,
                                      "reserved": old.reserved})
            alloc.applications[best_idx] = upgraded
            alloc.expected_profit += best_gain

    # --------------------------------------------------------- allocation
    def optimise(self, candidates: list[dict[str, Any]]) -> Allocation:
        alloc = Allocation(capital_available=self.capital,
                           objective=self.objective.value)
        by_symbol: dict[str, list[Application]] = {}
        for c in candidates:
            if not self._eligible(c, alloc):
                continue
            plans = self._plans_for(c)
            if not plans:
                alloc.skipped.append({"symbol": c["ipo"].symbol,
                                      "why": "no affordable application plan"})
                continue
            by_symbol[c["ipo"].symbol] = plans

        # Selection ALWAYS ranks by profit density, in both modes. Ranking by
        # raw rupees is the classic knapsack error: a big sNII ticket wins the
        # comparison, sterilises the capital, and crowds out several cheaper
        # applications worth more in total. Measured on live data it produced
        # Rs 9,445 of expected profit against Rs 12,853 for density ranking -
        # the absolute-profit objective was losing absolute profit.
        #
        # The objective instead decides what happens to LEFTOVER capital:
        #   maximize_roi              - stop; idle cash beats a poor return
        #   maximize_absolute_profit  - upgrade to larger tickets while any
        #                               capital would otherwise sit unused
        options: list[Application] = [
            min(plans, key=lambda a: -a.roi_on_blocked)
            for plans in by_symbol.values()]

        timeline = CapitalTimeline(self.capital)

        # Pass 1 - conviction first. High-scoring issues claim their window
        # before anything else can sterilise the cash, which is the whole
        # point: a 70-scorer closing on the 15th must not lose out to a
        # 55-scorer closing on the 11th whose funds are still locked up.
        reserved = sorted((o for o in options if o.score >= self.reserve_min_score),
                          key=lambda a: (-a.score, a.close_date or ""))
        rest = [o for o in options if o.score < self.reserve_min_score]

        for pool, is_reserved in ((reserved, True), (rest, False)):
            if not is_reserved:
                # Pass 2 - everything else, ranked by the chosen objective.
                pool = sorted(pool, key=lambda a: a.value(self.objective),
                              reverse=True)
            for pan in range(1, self.pans + 1):
                for opt in pool:
                    if pan > 1 and opt.p_allot >= 0.999:
                        # extra PANs only help under a lottery; in the
                        # proportionate regime they just split the same money
                        continue
                    start = _parse_date(opt.block_from)
                    end = _parse_date(opt.available_from or opt.unlock_on)
                    if start is None or end is None:
                        continue
                    if not timeline.can_fit(start, end, opt.capital):
                        if pan == 1:
                            alloc.skipped.append({
                                "symbol": opt.symbol,
                                "why": (f"no capital free between "
                                        f"{opt.block_from} and "
                                        f"{opt.available_from} "
                                        f"- reserved for higher-conviction "
                                        f"issues in the same window")})
                        continue
                    timeline.commit(start, end, opt.capital, opt.symbol)
                    app = Application(**{**opt.__dict__, "pan_index": pan,
                                         "reserved": is_reserved})
                    alloc.applications.append(app)
                    alloc.expected_profit += app.expected_profit
                    if is_reserved and app.symbol not in alloc.reserved_symbols:
                        alloc.reserved_symbols.append(app.symbol)

        if self.objective is Objective.MAXIMIZE_ABSOLUTE_PROFIT:
            self._upgrade_for_absolute_profit(alloc, by_symbol, timeline)

        alloc.timeline = timeline.events()
        # Peak simultaneous commitment, not the sum: cash released before the
        # next application is the same cash used twice, and reporting the sum
        # would overstate what you actually need in the account.
        alloc.capital_used = timeline.peak()
        alloc.applications.sort(
            key=lambda a: (a.close_date or "", a.pan_index,
                           -a.value(self.objective)))
        return alloc
