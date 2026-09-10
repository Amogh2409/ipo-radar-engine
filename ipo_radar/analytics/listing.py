"""
Listing-gain estimate.

Two independent views, blended:

  1. GMP view - the grey market is the single best predictor of listing pop,
     but it systematically OVERSTATES: premiums are quoted on thin, unofficial
     volume and decay into listing. We apply a realisation haircut.
  2. Demand view - final subscription, weighted towards QIB because
     institutional money is the informed money.

Both are then nudged by valuation/quality and issue size (small issues pop
harder). The output is a distribution, not a point: an IPO with a 20% central
estimate and a 25pp spread is a very different bet from one with a 20%
estimate and an 8pp spread.
"""
from __future__ import annotations

import math
from typing import Any

from datetime import datetime

from ..models import BNII, IPO, NII, QIB, RETAIL, SNII
from ..store import Store
from ..util import clamp

WEIGHTS_KEY = "listing_model_weights"

# ------------------------------------------------------- adverse selection
# The Winner's Curse, stated plainly: allotment odds are HIGH precisely when
# the market has declined the issue. Expected value alone therefore ranks a
# rejected book above a sought-after one, because 79% of a small gain beats
# 6% of a large one. Observed live - Prasol Chemicals sat at 0.05x QIB for
# most of its window with 79% retail odds and was funded ahead of a 215x book.
#
# An empty book late in the bidding is not a pricing inefficiency to harvest.
# It is the market's verdict, and it makes the grey market premium worthless:
# GMP is quoted on thin unofficial volume that does not clear against an
# order book nobody filled.
LATE_WINDOW = 0.75          # fraction of the bid window elapsed
UNDERSUBSCRIBED = 1.0       # below 1.0x the issue is not fully spoken for
ADVERSE_FLAG = ("Adverse Selection: Book undersubscribed near close. "
                "Gains forced to 0%.")


def adverse_selection(bid_progress: float, total_x: float | None,
                      qib_x: float | None) -> tuple[bool, list[str]]:
    """
    Late-window rejection test.

    A missing figure is never treated as a failure. `total_times` is absent
    from NSE's feed more often than not - it reads None on a fully subscribed
    book - so treating unknown as undersubscribed would condemn every issue.
    Only a number that is present AND below 1.0x counts against the issue.
    """
    if bid_progress <= LATE_WINDOW:
        return False, []
    reasons: list[str] = []
    if isinstance(total_x, (int, float)) and 0 < total_x < UNDERSUBSCRIBED:
        reasons.append(f"total book {total_x:.2f}x")
    if isinstance(qib_x, (int, float)) and qib_x < UNDERSUBSCRIBED:
        reasons.append(f"QIB {qib_x:.2f}x")
    return bool(reasons), reasons

DEFAULTS = {
    "gmp_realisation": 0.78,     # listing pop / GMP, historically <1
    "gmp_blend": 0.65,           # weight on the GMP view when GMP exists
    "sub_intercept": -6.0,
    "w_qib": 7.4,
    "w_nii": 3.1,
    "w_retail": 2.6,
    "w_size": 3.0,
    "w_valuation": 0.16,
    "base_sigma": 7.0,
    "sigma_slope": 0.42,
}


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class ListingModel:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store
        self.w = dict(DEFAULTS)
        if store:
            for k, v in (store.get_kv(WEIGHTS_KEY) or {}).items():
                if k in self.w:
                    try:
                        self.w[k] = float(v)
                    except (TypeError, ValueError):
                        pass

    # ------------------------------------------------------------ predict
    def predict(self, ipo: IPO, projected: dict[str, float],
                gmp_pct: float | None, valuation: dict[str, Any],
                projection_confidence: float = 1.0,
                total_x: float | None = None,
                now: datetime | None = None) -> dict[str, Any]:
        w = self.w
        qib = projected.get(QIB, 0.0)
        nii = projected.get(NII) or max(projected.get(SNII, 0.0),
                                        projected.get(BNII, 0.0))
        ret = projected.get(RETAIL, 0.0)

        # --- demand view
        size_cr = ipo.issue_size_cr or 500.0
        size_factor = clamp(math.log10(2000.0 / max(size_cr, 40.0)), -1.0, 1.6)
        val_score = (valuation or {}).get("valuation_score", 50.0)
        qual = (valuation or {}).get("quality_score", 50.0)

        demand_view = (w["sub_intercept"]
                       + w["w_qib"] * math.log1p(qib)
                       + w["w_nii"] * math.log1p(nii)
                       + w["w_retail"] * math.log1p(ret)
                       + w["w_size"] * size_factor
                       + w["w_valuation"] * ((val_score + qual) / 2.0 - 50.0))

        # --- GMP view
        if gmp_pct is not None:
            gmp_view = gmp_pct * w["gmp_realisation"]
            blend = w["gmp_blend"]
            # a GMP unsupported by the book is less trustworthy
            if gmp_pct > 12 and max(qib, nii, ret) < 1.2:
                blend *= 0.55
            central = blend * gmp_view + (1 - blend) * demand_view
            basis = "GMP + demand"
        else:
            central = demand_view
            basis = "demand only (no GMP)"

        central = clamp(central, -45.0, 160.0)
        sigma = w["base_sigma"] + w["sigma_slope"] * abs(central)
        # an early book is a weak book: widen the interval rather than pretend
        sigma *= 1.0 + 0.85 * (1.0 - clamp(projection_confidence, 0.0, 1.0))
        # an unsubscribed book at close is a genuine tail risk
        if projected and max(projected.values() or [0]) < 1.0:
            central = min(central, -2.0)
            sigma *= 1.25

        # --- adverse selection: the market's verdict outranks the model's
        progress = ipo.bid_progress(now)
        adverse, why = adverse_selection(progress, total_x, projected.get(QIB))
        if adverse:
            # min(0, central) - never a positive expectation on a rejected
            # book, but a model that already predicts a discount keeps it.
            forced = min(0.0, central)
            basis = f"{ADVERSE_FLAG} ({', '.join(why)} at {progress:.0%} elapsed)"
            return {
                "expected_gain_pct": round(forced, 1),
                "range": (round(forced - sigma, 1), round(min(0.0, forced + sigma), 1)),
                "sigma": round(sigma, 1),
                "p_positive": 0.0,
                "p_gain_over_10": 0.0,
                "basis": basis,
                "adverse_selection": True,
                "adverse_reasons": why,
                "pre_adverse_gain_pct": round(central, 1),
                "demand_view": round(demand_view, 1),
                "gmp_view": (round(gmp_pct * w["gmp_realisation"], 1)
                             if gmp_pct is not None else None),
                "size_factor": round(size_factor, 2),
                "projection_confidence": round(projection_confidence, 2),
            }

        return {
            "expected_gain_pct": round(central, 1),
            "range": (round(central - sigma, 1), round(central + sigma, 1)),
            "sigma": round(sigma, 1),
            "p_positive": round(_norm_cdf(central / max(sigma, 1e-6)), 3),
            "p_gain_over_10": round(1 - _norm_cdf((10.0 - central) / max(sigma, 1e-6)), 3),
            "basis": basis,
            "demand_view": round(demand_view, 1),
            "gmp_view": round(gmp_pct * w["gmp_realisation"], 1) if gmp_pct is not None else None,
            "size_factor": round(size_factor, 2),
            "projection_confidence": round(projection_confidence, 2),
            "adverse_selection": False,
        }

    # -------------------------------------------------------- calibrate
    def calibrate(self, min_samples: int = 12) -> dict[str, Any]:
        """
        Refit the GMP realisation factor against our own recorded outcomes.

        Deliberately conservative: only the single most important coefficient
        is refit, by robust median ratio, and only once there are enough
        completed IPOs. Fitting nine coefficients on twenty samples would
        just be curve-fitting noise.
        """
        if not self.store:
            return {}
        rows = self.store.training_rows()
        ratios = []
        for r in rows:
            f = r.get("features") or {}
            gmp = f.get("gmp_pct")
            actual = r.get("listing_gain_pct")
            if gmp and actual is not None and gmp > 3:
                ratios.append(actual / gmp)
        if len(ratios) < min_samples:
            return {"status": "insufficient", "samples": len(ratios)}
        ratios.sort()
        median = ratios[len(ratios) // 2]
        new = clamp(median, 0.3, 1.3)
        self.w["gmp_realisation"] = round(new, 3)
        self.store.set_kv(WEIGHTS_KEY, self.w)
        return {"status": "refit", "samples": len(ratios),
                "gmp_realisation": self.w["gmp_realisation"]}
