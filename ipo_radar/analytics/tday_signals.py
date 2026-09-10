"""
T-Day execution signals: the last two and a half hours that actually matter.

Everything else in this package informs a decision. This module makes one,
against a clock. The operative facts on an Indian mainboard IPO's closing day:

  * Bidding runs 10:00-17:00 IST. The UPI mandate must be ACCEPTED in your
    banking app by 17:00 IST - NSE publishes the exact cut-off per issue and
    we parse it rather than assume it.
  * Accepting the mandate is a separate act from placing the bid, it happens
    in a different app, and mandates can take several minutes to arrive.
    A bid placed at 16:55 with an unaccepted mandate is not an application.
  * QIB demand arrives last, so the information you most want is the
    information that shows up latest. Between 14:00 and 16:30 you have both
    a meaningful book and enough runway to act on it. That is the window.
  * Cut-off price is available ONLY to Retail Individual Investors and
    eligible employees. NII/HNI bids must name a price - and naming anything
    below the cap risks being excluded from allotment entirely if the issue
    prices at the cap, which oversubscribed issues almost always do.
"""
from __future__ import annotations

import logging
import math
import re
import statistics
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any
from ..config import PortfolioConfig
from ..models import AllotmentOdds, BNII, IPO, QIB, RETAIL, SNII
from ..util import IST
from ..store import Store
from ..util import clamp

log = logging.getLogger("ipo_radar.tday")

BID_OPEN_HOUR = 10
BID_CLOSE_HOUR = 17                 # 17:00 IST, and the UPI mandate deadline
DECISION_WINDOW_START = time(14, 0)


def to_ist(value: datetime | str) -> datetime:
    """
    Interpret a stored timestamp correctly.

    Subscription rows are written with `datetime.now().isoformat()` - naive
    LOCAL time. Forcibly stamping those with tzinfo=IST is a no-op on a host
    already in IST and silently wrong everywhere else, which would shift every
    velocity window by the UTC offset and corrupt spike detection. A naive
    value is therefore localised (astimezone() on a naive datetime assumes
    host-local, which is exactly how it was written) and then converted.
    """
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.astimezone()          # attach the host's zone, do not shift
    return dt.astimezone(IST)
FINAL_CALL_START = time(16, 30)
SPIKE_WINDOW_MIN = 120              # the final two hours
SPIKE_SIGMA = 2.0
# Placing a bid is not the same as accepting the mandate. Brokers and sponsor
# banks impose their own earlier cut-offs and mandates can lag by minutes.
SUBMIT_BUFFER_MIN = 60              # place the bid by 16:00
MANDATE_BUFFER_MIN = 20             # accept it by 16:40 at the latest


class Phase(str, Enum):
    PRE_OPEN = "pre_open"
    EARLY = "early"                 # day 1 / day 2
    FINAL_DAY_MORNING = "final_day_morning"
    DECISION_WINDOW = "decision_window"      # 14:00-16:30 on the closing day
    FINAL_CALL = "final_call"                # 16:30-17:00
    CLOSED = "closed"


class AlertLevel(str, Enum):
    INFO = "INFO"
    WATCH = "WATCH"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Action(str, Enum):
    APPLY = "APPLY"
    AVOID = "AVOID"
    WAIT = "WAIT"


@dataclass
class Alert:
    level: AlertLevel
    kind: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level.value, "kind": self.kind,
                "message": self.message}


@dataclass
class CutoffClock:
    now_ist: datetime
    close_date: date | None
    cutoff_at: datetime | None            # UPI mandate deadline, IST
    submit_by: datetime | None            # place the bid by this time
    mandate_by: datetime | None           # accept the mandate by this time
    phase: Phase
    is_final_day: bool
    seconds_to_cutoff: float | None
    source: str = "close_date + 17:00 IST"

    @property
    def minutes_to_cutoff(self) -> float | None:
        return None if self.seconds_to_cutoff is None else self.seconds_to_cutoff / 60.0

    def countdown(self) -> str:
        s = self.seconds_to_cutoff
        if s is None:
            return "—"
        if s <= 0:
            return "closed"
        h, rem = divmod(int(s), 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h {m:02d}m" if h else f"{m}m"

    def to_dict(self) -> dict[str, Any]:
        return {"now_ist": self.now_ist.isoformat(timespec="seconds"),
                "phase": self.phase.value, "is_final_day": self.is_final_day,
                "cutoff_at": self.cutoff_at.isoformat(timespec="minutes")
                if self.cutoff_at else None,
                "submit_by": self.submit_by.isoformat(timespec="minutes")
                if self.submit_by else None,
                "mandate_by": self.mandate_by.isoformat(timespec="minutes")
                if self.mandate_by else None,
                "countdown": self.countdown(), "source": self.source}


@dataclass
class SpikeSignal:
    category: str
    current_rate: float               # multiples per hour, final window
    baseline_mean: float
    baseline_sd: float
    z_score: float
    is_spike: bool
    samples: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@dataclass
class BiddingAdvice:
    action: Action
    category: str | None = None
    lots: int | None = None
    shares: int | None = None
    capital: float | None = None
    bid_price_instruction: str = ""
    p_allot: float | None = None
    expected_profit: float | None = None
    deadline_text: str = ""
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


@dataclass
class TDaySignal:
    symbol: str
    name: str
    clock: CutoffClock
    spikes: dict[str, SpikeSignal] = field(default_factory=dict)
    advice: BiddingAdvice | None = None
    alerts: list[Alert] = field(default_factory=list)

    @property
    def max_level(self) -> AlertLevel:
        order = [AlertLevel.INFO, AlertLevel.WATCH, AlertLevel.HIGH,
                 AlertLevel.CRITICAL]
        best = AlertLevel.INFO
        for a in self.alerts:
            if order.index(a.level) > order.index(best):
                best = a.level
        return best

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "name": self.name,
                "clock": self.clock.to_dict(),
                "spikes": {k: v.to_dict() for k, v in self.spikes.items()},
                "advice": self.advice.to_dict() if self.advice else None,
                "alerts": [a.to_dict() for a in self.alerts],
                "max_level": self.max_level.value}


class TDaySignalEngine:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store

    # --------------------------------------------------------------- clock
    @staticmethod
    def _parse_nse_cutoff(ipo: IPO) -> tuple[datetime | None, str]:
        """
        NSE states the mandate deadline per issue, e.g.
        "10-Sep-2026 (upto 5:00 PM)". Use it verbatim when present - it is
        authoritative and occasionally differs from the naive assumption.
        """
        raw = (ipo.meta or {}).get("raw_info") or {}
        text = ""
        for k, v in raw.items():
            if "upi mandate" in k.lower():
                text = str(v)
                break
        if not text:
            return None, ""
        m = re.search(r"(\d{1,2})-([A-Za-z]{3})-(\d{4}).{0,20}?"
                      r"(\d{1,2})[:.](\d{2})\s*([AP])\.?M", text, re.I)
        if not m:
            return None, ""
        months = {mm: i for i, mm in enumerate(
            ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
        mo = months.get(m.group(2).lower())
        if not mo:
            return None, ""
        hour = int(m.group(4)) % 12 + (12 if m.group(6).upper() == "P" else 0)
        try:
            return datetime(int(m.group(3)), mo, int(m.group(1)), hour,
                            int(m.group(5)), tzinfo=IST), "NSE issue circular"
        except ValueError:
            return None, ""

    def clock(self, ipo: IPO, now: datetime | None = None) -> CutoffClock:
        now_ist = datetime.now(IST) if now is None else to_ist(now)
        _, close = ipo.days()
        cutoff, source = self._parse_nse_cutoff(ipo)
        if cutoff is None and close:
            cutoff = datetime.combine(close, time(BID_CLOSE_HOUR, 0), tzinfo=IST)
            source = "close date + 17:00 IST"
        secs = (cutoff - now_ist).total_seconds() if cutoff else None
        is_final = bool(close and now_ist.date() == close)

        if close is None:
            phase = Phase.EARLY
        elif now_ist.date() > close or (secs is not None and secs <= 0):
            phase = Phase.CLOSED
        elif not is_final:
            o, _ = ipo.days()
            phase = Phase.PRE_OPEN if (o and now_ist.date() < o) else Phase.EARLY
        elif now_ist.time() >= FINAL_CALL_START:
            phase = Phase.FINAL_CALL
        elif now_ist.time() >= DECISION_WINDOW_START:
            phase = Phase.DECISION_WINDOW
        else:
            phase = Phase.FINAL_DAY_MORNING

        submit_by = mandate_by = None
        if cutoff:
            submit_by = cutoff - timedelta(minutes=SUBMIT_BUFFER_MIN)
            mandate_by = cutoff - timedelta(minutes=MANDATE_BUFFER_MIN)
        return CutoffClock(now_ist=now_ist, close_date=close, cutoff_at=cutoff,
                           submit_by=submit_by, mandate_by=mandate_by,
                           phase=phase, is_final_day=is_final,
                           seconds_to_cutoff=secs, source=source)

    # ---------------------------------------------------- spike detection
    def detect_spike(self, ipo: IPO, category: str = QIB,
                     now: datetime | None = None) -> SpikeSignal | None:
        """
        Flag institutional demand accelerating past 2 sigma of its own
        3-day baseline inside the final two hours.

        A QIB book that suddenly steepens at 15:30 is the single most
        actionable signal in the whole system: institutions have concluded
        their diligence and are committing, and there is still time to act.
        """
        if not self.store:
            return None
        series = self.store.subscription_series(ipo.symbol, category)
        if len(series) < 5:
            return None
        now_ist = datetime.now(IST) if now is None else to_ist(now)
        cutoff_ts = now_ist - timedelta(minutes=SPIKE_WINDOW_MIN)

        rates: list[tuple[datetime, float]] = []
        for (t0, x0), (t1, x1) in zip(series, series[1:]):
            try:
                a, b = to_ist(t0), to_ist(t1)
            except Exception:
                continue
            hours = (b - a).total_seconds() / 3600.0
            # ignore overnight gaps: nothing is bid between 17:00 and 10:00
            if hours <= 0.05 or hours > 8:
                continue
            rates.append((b, (x1 - x0) / hours))
        if len(rates) < 4:
            return None

        baseline = [r for ts, r in rates if ts < cutoff_ts]
        recent = [r for ts, r in rates if ts >= cutoff_ts]
        if len(baseline) < 3:
            return SpikeSignal(category, recent[-1] if recent else 0.0, 0.0, 0.0,
                               0.0, False, len(baseline),
                               "insufficient baseline for a 2-sigma test")
        mean = statistics.fmean(baseline)
        sd = statistics.pstdev(baseline) if len(baseline) > 1 else 0.0
        current = statistics.fmean(recent) if recent else (rates[-1][1])
        # a flat baseline gives sd ~= 0; fall back to a relative test
        if sd < 1e-6:
            z = 0.0 if mean <= 1e-9 else (current - mean) / max(abs(mean), 1e-9)
            spike = current > max(mean * 3.0, 0.5)
            note = "flat baseline; relative test applied"
        else:
            z = (current - mean) / sd
            spike = z >= SPIKE_SIGMA
            note = ""
        return SpikeSignal(category=category, current_rate=current,
                           baseline_mean=mean, baseline_sd=sd, z_score=z,
                           is_spike=bool(spike), samples=len(baseline), note=note)

    # ------------------------------------------------------------- advice
    # The composite grade is the single source of truth on whether an issue
    # is worth applying to. This engine decides HOW and WHEN, never whether -
    # a dashboard showing "AVOID" beside "APPLY" is worse than no dashboard.
    NEGATIVE_GRADES = ("AVOID", "STRONG AVOID")
    # A vetoed business can still be a valid listing-day trade, but the
    # instruction has to carry its own exit or the veto means nothing.
    FLIP_GRADE = "SPECULATIVE FLIP"
    # The execution floor lives in PortfolioConfig and must be read from
    # there, not restated. A local copy drifted once already: the optimiser
    # was raised to 61 to defend against the Winner's Curse while this stayed
    # at 50, so Prasol - refused capital by the optimiser at score 57 - still
    # fired a CRITICAL "the call is APPLY" alert two minutes before cut-off.
    # An alert is the most actionable surface in the system; it must never
    # contradict the plan.
    MIN_SCORE = PortfolioConfig().min_score

    def advise(self, ipo: IPO, clock: CutoffClock, *,
               allotment: dict[str, AllotmentOdds] | None,
               expected_gain_pct: float | None, score: float | None,
               grade: str | None = None, veto: dict[str, Any] | None = None,
               capital: float | None = None) -> BiddingAdvice:
        rationale: list[str] = []

        if clock.phase is Phase.CLOSED:
            return BiddingAdvice(Action.WAIT, deadline_text="Bidding has closed.",
                                 rationale=["window shut - nothing to do"])
        if clock.phase in (Phase.PRE_OPEN, Phase.EARLY, Phase.FINAL_DAY_MORNING):
            when = ("14:00 IST on the closing day" if clock.close_date
                    else "the closing afternoon")
            return BiddingAdvice(
                Action.WAIT,
                deadline_text=f"Decide from {when}; mandate cut-off "
                              f"{clock.cutoff_at:%d %b %H:%M} IST."
                              if clock.cutoff_at else "",
                rationale=["QIB demand lands late - the book is not yet "
                           "informative enough to commit against",
                           f"time to cut-off: {clock.countdown()}"])

        if expected_gain_pct is None or not allotment:
            return BiddingAdvice(Action.WAIT,
                                 rationale=["no listing estimate or allotment "
                                            "odds available yet"])

        # rank the concrete plans by return on the cash they lock up
        best_key = best = None
        best_roi = -1e9
        lot = ipo.lot_size or 0
        for key, odds in allotment.items():
            if odds.mechanism == "n/a" or odds.ticket_value <= 0:
                continue
            if capital is not None and odds.ticket_value > capital:
                continue
            profit = (odds.expected_lots * lot * (ipo.cap_price or 0.0)
                      * (expected_gain_pct / 100.0))
            roi = profit / odds.ticket_value * 100.0
            if roi > best_roi:
                best_key, best, best_roi = key, odds, roi

        if best is None:
            return BiddingAdvice(Action.AVOID,
                                 rationale=["no application plan fits the "
                                            "available capital"])

        grade_up = (grade or "").upper()
        is_flip = grade_up == self.FLIP_GRADE
        # A capped Value Trap grades NEUTRAL, not AVOID, so the grade alone
        # would still have let it through to an APPLY with a CRITICAL alert.
        # The veto has to travel with it.
        v = veto or {}
        vetoed = bool(v.get("value_trap")) and not bool(v.get("speculative_flip"))
        negative_grade = grade_up in self.NEGATIVE_GRADES or vetoed
        # a flip is capped below the NEUTRAL boundary by design, so the score
        # floor must not veto the very trade the flip exception permits
        weak_score = (score is not None and score < self.MIN_SCORE
                      and not is_flip)
        thin_gain = expected_gain_pct <= 1.0
        if negative_grade or weak_score or thin_gain:
            why = []
            if vetoed:
                why.append(v.get("reason")
                           or "Value Trap: earns below its cost of capital")
                why.append("the speculative-flip exception was not met, so "
                           "there is no case for blocking capital here")
            elif negative_grade:
                why.append(f"composite verdict is {grade}")
            elif weak_score:
                why.append(f"composite score {score:.0f} is below "
                           f"{self.MIN_SCORE:.0f}")
            if thin_gain:
                why.append(f"expected listing {expected_gain_pct:+.1f}% does not "
                           "justify blocking cash through allotment")
            return BiddingAdvice(
                Action.AVOID, rationale=why,
                deadline_text=f"Cut-off {clock.cutoff_at:%H:%M} IST"
                              if clock.cutoff_at else "")

        cat = best.category
        profit = (best.expected_lots * lot * (ipo.cap_price or 0.0)
                  * (expected_gain_pct / 100.0))

        # Cut-off pricing is a retail/employee privilege. NII must name a price.
        if cat in (RETAIL,):
            price = ("Tick CUT-OFF. It accepts whatever price is discovered, "
                     "so your bid stays valid if the issue prices at the cap.")
        elif cat in (SNII, BNII):
            price = (f"Bid at the CAP price of Rs {ipo.cap_price:,.0f} - cut-off "
                     "is not available to NII. A bid below the cap is excluded "
                     "from allotment if the issue prices at the cap.")
        else:
            price = f"Bid at the cap price of Rs {ipo.cap_price:,.0f}."

        if is_flip:
            rationale.append(
                "SPECULATIVE FLIP - this issuer does not earn its cost of "
                "capital. Sell on listing day; do not hold.")
        rationale.append(f"{best.mechanism}: {best.note}")
        rationale.append(f"~{best.p_single:.0%} chance of allotment, "
                         f"{best_roi:.2f}% return on the cash blocked")
        if cat == RETAIL and best.mechanism == "lottery":
            rationale.append("extra lots on this PAN would not raise the odds - "
                             "only extra PANs do")
        if clock.phase is Phase.FINAL_CALL:
            rationale.append("final call: place the bid and accept the mandate now")

        deadline = ""
        if clock.cutoff_at and clock.submit_by and clock.mandate_by:
            now = clock.now_ist
            place = ("PLACE THE BID NOW" if now >= clock.submit_by
                     else f"Place the bid by {clock.submit_by:%H:%M} IST")
            mandate = ("accept the UPI mandate IMMEDIATELY"
                       if now >= clock.mandate_by
                       else f"accept the UPI mandate by {clock.mandate_by:%H:%M} IST")
            deadline = (f"{place}, {mandate}. Hard cut-off "
                        f"{clock.cutoff_at:%H:%M} IST ({clock.countdown()} left). "
                        f"Accepting the mandate is a separate step in your banking "
                        f"app - a placed bid with an unaccepted mandate is not an "
                        f"application.")
        return BiddingAdvice(
            action=Action.APPLY, category=cat, lots=best.lots_applied,
            shares=best.lots_applied * lot, capital=best.ticket_value,
            bid_price_instruction=price, p_allot=best.p_single,
            expected_profit=profit, deadline_text=deadline,
            rationale=[r for r in rationale if r])

    # -------------------------------------------------------------- alerts
    def _alerts(self, ipo: IPO, clock: CutoffClock,
                spikes: dict[str, SpikeSignal],
                advice: BiddingAdvice | None) -> list[Alert]:
        out: list[Alert] = []
        mins = clock.minutes_to_cutoff
        if clock.phase is Phase.CLOSED:
            return out

        if clock.phase is Phase.FINAL_CALL and advice and advice.action is Action.APPLY:
            out.append(Alert(AlertLevel.CRITICAL, "cutoff",
                             f"{ipo.symbol}: {clock.countdown()} to the UPI mandate "
                             f"cut-off and the call is APPLY - act now"))
        elif clock.phase is Phase.DECISION_WINDOW:
            out.append(Alert(AlertLevel.HIGH, "window",
                             f"{ipo.symbol}: decision window open, "
                             f"{clock.countdown()} to cut-off"))
        elif clock.is_final_day and clock.phase is not Phase.CLOSED:
            out.append(Alert(AlertLevel.WATCH, "final_day",
                             f"{ipo.symbol}: closes today, "
                             f"{clock.countdown()} to cut-off"))

        for cat, sig in spikes.items():
            if sig.is_spike:
                out.append(Alert(
                    AlertLevel.HIGH, "institutional_spike",
                    f"{ipo.symbol}: {cat} demand accelerating at "
                    f"{sig.current_rate:.2f}x/hour, {sig.z_score:.1f} sigma above "
                    f"its {sig.baseline_mean:.2f}x/hour baseline"))
        if mins is not None and 0 < mins <= MANDATE_BUFFER_MIN and \
                advice and advice.action is Action.APPLY:
            out.append(Alert(AlertLevel.CRITICAL, "mandate",
                             f"{ipo.symbol}: under {int(mins)} minutes - accept the "
                             f"UPI mandate in your banking app immediately"))
        return out

    # --------------------------------------------------------------- main
    def evaluate(self, ipo: IPO, *, allotment: dict[str, AllotmentOdds] | None = None,
                 expected_gain_pct: float | None = None,
                 score: float | None = None, grade: str | None = None,
                 veto: dict[str, Any] | None = None,
                 capital: float | None = None,
                 now: datetime | None = None) -> TDaySignal:
        clock = self.clock(ipo, now)
        spikes: dict[str, SpikeSignal] = {}
        if clock.is_final_day or clock.phase is Phase.EARLY:
            for cat in (QIB, SNII, RETAIL):
                sig = self.detect_spike(ipo, cat, now)
                if sig is not None:
                    spikes[cat] = sig
        advice = self.advise(ipo, clock, allotment=allotment,
                             expected_gain_pct=expected_gain_pct, score=score,
                             grade=grade, veto=veto, capital=capital)
        signal = TDaySignal(symbol=ipo.symbol, name=ipo.name, clock=clock,
                            spikes=spikes, advice=advice)
        signal.alerts = self._alerts(ipo, clock, spikes, advice)
        return signal
