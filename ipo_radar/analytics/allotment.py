"""
Allotment probability under SEBI's actual allotment rules.

The mechanics that matter, and that most GMP sites get wrong:

RETAIL (RII)
  Bids are in whole lots; a bid must stay under Rs 2,00,000. When the retail
  book is oversubscribed, SEBI mandates that the MINIMUM BID LOT be allotted
  to as many applicants as possible, by LOTTERY. Consequence: past a certain
  point every winner receives exactly ONE lot, whether they bid 1 lot or 13.
  So bidding more than one lot does NOT raise your chance of getting an
  allotment - it only blocks more cash. Extra PANs raise your odds; extra
  lots on one PAN do not.

  With A = shares offered to retail, L = lot size, x = subscription multiple,
  and k = mean lots per application:
      applications  = (A*x/L) / k
      max allottees = A / L
      P(allot)      = max_allottees / applications = k / x

  k is unobservable from NSE data (they publish shares, not application
  counts), so it is estimated from demand heat and then CALIBRATED against
  real allotment outcomes over time - see brain/calibration.py.

sNII (Rs 2L-10L) and bNII (>Rs 10L)
  Since the April 2022 SEBI circular the NII book is split 1/3 sNII, 2/3
  bNII, and each is allotted proportionately BUT with a guaranteed minimum of
  one lot - which degenerates into a lottery once proportionate entitlement
  falls below one lot. Applicants overwhelmingly bid the minimum ticket, so
  k is simply the minimum lot count for that bracket, which is a far more
  reliable estimate than the retail case.

QIB
  Discretionary/proportionate and not open to individual applicants; reported
  for context only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..config import AllotmentParams
from ..models import (AllotmentOdds, BNII, EMPLOYEE, IPO, QIB, RETAIL,
                      SHAREHOLDER, SNII, SubscriptionSnapshot)

FULL = "full"
PROPORTIONATE = "proportionate"      # everyone gets >= 1 lot
LOTTERY = "lottery"                  # 1 lot to the lucky, nothing to the rest
NA = "n/a"


@dataclass
class CategoryPlan:
    """One concrete way to apply: category, lot count, cash blocked."""
    category: str
    lots: int
    shares: int
    ticket_value: float
    min_lots: int
    eligible: bool
    reason: str = ""


def mean_lots_retail(params: AllotmentParams, heat: float) -> float:
    """
    Mean lots per retail application, interpolated on `heat` (0=cold, 1=hot).

    Hot issues pull in applicants who bid the full Rs 2L; cold ones are
    dominated by single-lot punters. This is the most sensitive input in the
    whole engine, which is why it is a calibrated parameter and not a
    hard-coded constant.
    """
    heat = max(0.0, min(1.0, heat))
    lo, hi = params.avg_lots_retail_cold, params.avg_lots_retail_hot
    return lo + (hi - lo) * heat


def demand_heat(sub: SubscriptionSnapshot | None, gmp_pct: float | None) -> float:
    """Blend subscription and GMP into a 0-1 'how hot is this' scalar."""
    score = 0.0
    if sub:
        r = sub.times(RETAIL)
        # 0x -> 0, 5x -> ~0.5, 20x -> ~0.85
        score = max(score, 1.0 - math.exp(-r / 7.0))
    if gmp_pct is not None:
        score = max(score, 1.0 - math.exp(-max(0.0, gmp_pct) / 22.0))
    return max(0.0, min(1.0, score))


class AllotmentEngine:
    def __init__(self, params: AllotmentParams | None = None) -> None:
        self.p = params or AllotmentParams()

    # ------------------------------------------------------------- plans
    def plans(self, ipo: IPO, allow_snii: bool = True,
              allow_bnii: bool = False) -> list[CategoryPlan]:
        """Enumerate the sensible ways a retail-ish investor can apply."""
        out: list[CategoryPlan] = []
        lot, lv = ipo.lot_size, ipo.lot_value
        if not lot or not lv:
            return out

        max_retail = ipo.max_retail_lots(self.p.retail_max_ticket) or 1
        out.append(CategoryPlan(RETAIL, 1, lot, lv, 1, True,
                                "1 lot - the efficient retail bid"))
        if max_retail > 1:
            out.append(CategoryPlan(RETAIL, max_retail, max_retail * lot,
                                    max_retail * lv, 1, True,
                                    f"{max_retail} lots - max retail bid"))
        if allow_snii:
            ml = ipo.min_lots_for(self.p.snii_min_ticket) or 0
            if ml:
                out.append(CategoryPlan(SNII, ml, ml * lot, ml * lv, ml, True,
                                        "minimum sNII ticket"))
        if allow_bnii:
            ml = ipo.min_lots_for(self.p.bnii_min_ticket) or 0
            if ml:
                out.append(CategoryPlan(BNII, ml, ml * lot, ml * lv, ml, True,
                                        "minimum bNII ticket"))
        return out

    # -------------------------------------------------------------- core
    def odds(self, ipo: IPO, category: str, subscription_x: float,
             lots_applied: int, num_applications: int = 1,
             heat: float = 0.5) -> AllotmentOdds:
        lot = ipo.lot_size or 0
        lv = ipo.lot_value or 0.0
        ticket = lots_applied * lv
        x = max(0.0, float(subscription_x or 0.0))

        if category == QIB:
            return AllotmentOdds(category=QIB, subscription_x=x,
                                 lots_applied=0, ticket_value=0.0,
                                 p_single=0.0, expected_lots=0.0,
                                 p_any_multi=0.0, expected_lots_multi=0.0,
                                 mechanism=NA,
                                 note="QIB is not open to individual applicants")

        # mean lots per competing application in this category
        if category in (SNII, BNII):
            ticket_floor = (self.p.snii_min_ticket if category == SNII
                            else self.p.bnii_min_ticket)
            k = float(ipo.min_lots_for(ticket_floor) or lots_applied or 1)
        elif category in (RETAIL, EMPLOYEE, SHAREHOLDER):
            k = mean_lots_retail(self.p, heat)
        else:
            k = 1.0

        if x <= 0:
            return AllotmentOdds(category=category, subscription_x=x,
                                 lots_applied=lots_applied, ticket_value=ticket,
                                 p_single=1.0, expected_lots=float(lots_applied),
                                 p_any_multi=1.0,
                                 expected_lots_multi=float(lots_applied * num_applications),
                                 mechanism=FULL,
                                 note="category not yet subscribed")

        offered_shares = None      # filled by caller via subscription snapshot
        if x <= 1.0:
            mech, p, lots = FULL, 1.0, float(lots_applied)
            note = "undersubscribed - full allotment expected"
        elif x <= k:
            # enough lots for every applicant to get at least one
            mech, p = PROPORTIONATE, 1.0
            lots = max(1.0, math.floor(lots_applied / x))
            note = (f"proportionate with 1-lot floor "
                    f"(~{lots:.0f} of {lots_applied} lots)")
        else:
            mech, p = LOTTERY, min(1.0, k / x)
            lots = p * 1.0        # winners get exactly one lot
            note = (f"lottery for 1 lot; ~{k:.2f} mean lots/application "
                    f"against {x:.2f}x")

        n = max(1, int(num_applications))
        p_any = 1.0 - (1.0 - p) ** n
        exp_multi = lots * n

        return AllotmentOdds(
            category=category, subscription_x=x, lots_applied=lots_applied,
            ticket_value=ticket, p_single=p, expected_lots=lots,
            p_any_multi=p_any, expected_lots_multi=exp_multi, mechanism=mech,
            max_allottees=offered_shares, est_applications=None, note=note)

    # ------------------------------------------------------- convenience
    def analyse(self, ipo: IPO, sub: SubscriptionSnapshot | None,
                projected: dict[str, float] | None = None,
                num_applications: int = 1, gmp_pct: float | None = None,
                allow_snii: bool = True, allow_bnii: bool = False,
                ) -> dict[str, AllotmentOdds]:
        """
        Odds for every sensible application plan, computed against PROJECTED
        final subscription where available - allotment is decided on the final
        book, not on whatever the book looks like mid-window.
        """
        heat = demand_heat(sub, gmp_pct)
        out: dict[str, AllotmentOdds] = {}
        for plan in self.plans(ipo, allow_snii, allow_bnii):
            x = None
            if projected and plan.category in projected:
                x = projected[plan.category]
            elif sub:
                x = sub.times(plan.category)
            if x is None:
                continue
            key = f"{plan.category}:{plan.lots}"
            odds = self.odds(ipo, plan.category, x, plan.lots,
                             num_applications, heat)
            if sub:
                cb = sub.get(plan.category)
                if cb and cb.offered and ipo.lot_size:
                    odds.max_allottees = cb.offered / ipo.lot_size
                    kk = (float(ipo.min_lots_for(
                              self.p.snii_min_ticket if plan.category == SNII
                              else self.p.bnii_min_ticket) or 1)
                          if plan.category in (SNII, BNII)
                          else mean_lots_retail(self.p, heat))
                    if cb.bid:
                        odds.est_applications = (cb.bid / ipo.lot_size) / max(kk, 1e-9)
            out[key] = odds
        return out

    # ---------------------------------------------------------- economics
    @staticmethod
    def expected_value(odds: AllotmentOdds, ipo: IPO,
                       expected_gain_pct: float) -> dict[str, float]:
        """
        Rupee expectation of one application.

        `roi_on_blocked` is the number that actually matters when capital is
        scarce: profit per rupee of ASBA-blocked cash, for the ~7 days it is
        locked up.
        """
        lot = ipo.lot_size or 0
        price = ipo.cap_price or 0.0
        gain_per_share = price * (expected_gain_pct / 100.0)
        exp_shares = odds.expected_lots * lot
        exp_profit = exp_shares * gain_per_share
        blocked = odds.ticket_value
        return {
            "expected_shares": exp_shares,
            "expected_profit": exp_profit,
            "capital_blocked": blocked,
            "roi_on_blocked": (exp_profit / blocked * 100.0) if blocked else 0.0,
        }
