"""
Projecting FINAL subscription from a partially-filled book.

This matters because allotment is decided on the closing book, but you must
decide whether to apply before it closes. Indian IPO bidding is violently
back-loaded and the skew differs by category: QIBs and big NIIs overwhelmingly
bid in the last hours of the final day (funding cost, and not wanting to show
their hand), while retail fills in far more evenly.

Model: cumulative fraction of final demand f(t) = t**alpha, where t is the
fraction of the bidding window elapsed. alpha=1 is linear; higher alpha means
more back-loaded. Defaults below are seeded from typical mainboard behaviour
and then RE-FITTED from this database's own completed IPOs once enough have
been observed (`fit_exponents`), so the model sharpens on the market you are
actually trading.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

from ..models import (BNII, EMPLOYEE, IPO, NII, QIB, RETAIL, SHAREHOLDER,
                      SNII, SubscriptionSnapshot)
from ..store import Store

log = logging.getLogger("ipo_radar.projection")

# Chosen so that f(2/3) - the fraction of the final book in by the end of day
# 2, which is when most apply/skip decisions get made - matches typical
# mainboard behaviour: ~25% of final QIB, ~40% of NII, ~65% of retail.
DEFAULT_ALPHA: dict[str, float] = {
    QIB: 3.4,       # famously last-hour
    BNII: 2.8,      # funded bids arrive late
    NII: 2.4,
    SNII: 2.0,
    RETAIL: 1.15,   # close to linear, mild day-3 bump
    EMPLOYEE: 1.0,
    SHAREHOLDER: 1.1,
}
MIN_PROGRESS = 0.06        # below this the book says almost nothing
ALPHA_KEY = "projection_alphas"


def projection_confidence(t: float) -> float:
    """
    How much weight the projection deserves, given how much of the bid window
    has elapsed.

    A day-one QIB reading of 1.2x is compatible with a final book anywhere
    between 5x and 80x. Rather than hide that behind a confident point
    estimate, everything downstream scales its uncertainty by this number.
    """
    if t >= 0.999:
        return 1.0
    return max(0.10, min(1.0, 0.10 + 0.90 * (t ** 1.6)))


def max_growth(t: float) -> float:
    """Ceiling on how many times over the current book may still grow."""
    return 4.0 + 90.0 * (1.0 - min(max(t, 0.0), 1.0)) ** 3


def _alphas(store: Store | None) -> dict[str, float]:
    a = dict(DEFAULT_ALPHA)
    if store:
        learned = store.get_kv(ALPHA_KEY) or {}
        for k, v in learned.items():
            try:
                a[k] = float(v)
            except (TypeError, ValueError):
                pass
    return a


class SubscriptionProjector:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store
        self.alpha = _alphas(store)

    # ---------------------------------------------------------- project
    def project(self, ipo: IPO, sub: SubscriptionSnapshot | None,
                now: datetime | None = None) -> dict[str, float]:
        """Projected FINAL subscription multiple per category."""
        if not sub:
            return {}
        t = ipo.bid_progress(now)
        out: dict[str, float] = {}
        for cat, cb in sub.categories.items():
            x = cb.subscribed
            if t >= 1.0:
                out[cat] = x
                continue
            if t < MIN_PROGRESS:
                out[cat] = x            # too early to extrapolate honestly
                continue
            alpha = self.alpha.get(cat, 1.5)
            curve = x / max(t ** alpha, 1e-6)
            momentum = self._momentum(ipo, cat, x, t)
            # trust the curve early, the observed run-rate late
            w = min(1.0, max(0.0, (t - 0.3) / 0.6))
            proj = (1 - w) * curve + w * (momentum if momentum else curve)
            # never below the book, never a fantasy multiple of it
            out[cat] = max(x, min(proj, x * max_growth(t)))
        return out

    def confidence(self, ipo: IPO, now: datetime | None = None) -> float:
        return projection_confidence(ipo.bid_progress(now))

    def _momentum(self, ipo: IPO, category: str, current: float,
                  t: float) -> float | None:
        """Extrapolate the recent fill rate to the close."""
        if not self.store or t <= 0:
            return None
        series = self.store.subscription_series(ipo.symbol, category)
        if len(series) < 3:
            return None
        try:
            t0 = datetime.fromisoformat(series[0][0])
            recent = series[-min(6, len(series)):]
            ta = datetime.fromisoformat(recent[0][0])
            tb = datetime.fromisoformat(recent[-1][0])
            hours = (tb - ta).total_seconds() / 3600.0
            if hours <= 0.05:
                return None
            rate = (recent[-1][1] - recent[0][1]) / hours     # x per hour
        except Exception:
            return None
        remaining = self._hours_left(ipo)
        if remaining is None or rate <= 0:
            return None
        # late bids arrive faster than the trailing average; scale up mildly
        surge = 1.0 + 2.2 * max(0.0, t - 0.55)
        return current + rate * remaining * surge

    @staticmethod
    def _hours_left(ipo: IPO, now: datetime | None = None) -> float | None:
        now = now or datetime.now()
        _, close = ipo.days()
        if not close:
            return None
        end = datetime.combine(close, datetime.min.time()).replace(hour=17)
        if now >= end:
            return 0.0
        hours, cur = 0.0, now.date()
        while cur <= close:
            d_start = datetime.combine(cur, datetime.min.time()).replace(hour=10)
            d_end = datetime.combine(cur, datetime.min.time()).replace(hour=17)
            lo = max(now, d_start)
            if lo < d_end:
                hours += (d_end - lo).total_seconds() / 3600.0
            cur = cur.fromordinal(cur.toordinal() + 1)
        return hours

    # ------------------------------------------------------------ learn
    def fit_exponents(self, min_ipos: int = 4) -> dict[str, float]:
        """
        Re-fit alpha per category from completed IPOs in our own database.

        For each observation we know the cumulative multiple x(t) and the
        final multiple X, so alpha = log(x/X)/log(t). We take the median
        across all observations, which is robust to the odd stale snapshot.
        """
        if not self.store:
            return {}
        done = [i for i in self.store.get_ipos() if i.status in ("Closed", "Listed")]
        if len(done) < min_ipos:
            log.debug("only %d completed IPOs; keeping default alphas", len(done))
            return {}
        fitted: dict[str, float] = {}
        for cat in DEFAULT_ALPHA:
            samples: list[float] = []
            for ipo in done:
                series = self.store.subscription_series(ipo.symbol, cat)
                if len(series) < 4:
                    continue
                final = series[-1][1]
                if final <= 0.05:
                    continue
                for ts, x in series[:-1]:
                    try:
                        t = ipo.bid_progress(datetime.fromisoformat(ts))
                    except Exception:
                        continue
                    if not (MIN_PROGRESS < t < 0.97) or x <= 0:
                        continue
                    ratio = min(x / final, 0.999)
                    if ratio <= 0:
                        continue
                    a = math.log(ratio) / math.log(t)
                    if 0.3 <= a <= 12.0:
                        samples.append(a)
            if len(samples) >= 8:
                samples.sort()
                fitted[cat] = round(samples[len(samples) // 2], 3)
        if fitted:
            merged = {**self.alpha, **fitted}
            self.store.set_kv(ALPHA_KEY, merged)
            self.alpha = merged
            log.info("re-fitted projection alphas from %d IPOs: %s",
                     len(done), fitted)
        return fitted

    # ------------------------------------------------------------ shape
    @staticmethod
    def describe(projected: dict[str, float],
                 current: SubscriptionSnapshot | None) -> dict[str, Any]:
        """Human-facing summary of where demand is concentrated."""
        if not projected:
            return {}
        out: dict[str, Any] = {"projected": {k: round(v, 2)
                                             for k, v in projected.items()}}
        q, r = projected.get(QIB, 0.0), projected.get(RETAIL, 0.0)
        n = projected.get(NII, projected.get(SNII, 0.0))
        out["qib_led"] = q > max(r, n) * 1.3 and q > 2.0
        out["retail_led"] = r > max(q, n) * 1.3 and r > 2.0
        if current:
            out["current"] = {k: round(v.subscribed, 2)
                              for k, v in current.categories.items()}
        return out
