"""
Closing the loop: learn from what actually happened.

Three things get calibrated as outcomes accumulate:
  * projection alphas  - how back-loaded each category's bidding really is
  * GMP realisation    - how much of the quoted premium survives to listing
  * mean lots per retail application - the k in P(allot) = k / x, recovered
    from real allotment results

Until there is enough data these are no-ops that report "insufficient" rather
than fitting noise. That restraint is the point: a model that confidently
refits nine parameters on six observations is worse than the prior.
"""
from __future__ import annotations

import logging
from typing import Any

from ..config import CONFIG_PATH, Settings
from ..store import Store
from ..util import clamp
from ..analytics.listing import ListingModel
from ..analytics.subscription import SubscriptionProjector

log = logging.getLogger("ipo_radar.calibration")

RETAIL_K_KEY = "retail_k_observations"


def record_allotment_outcome(store: Store, symbol: str,
                             retail_subscription_x: float,
                             observed_allotment_rate: float) -> None:
    """
    Feed a real allotment result back in.

    `observed_allotment_rate` is applications allotted / applications made -
    e.g. you applied from 6 PANs and 2 got shares -> 0.333. Since
    P(allot) = k / x, each observation implies k = rate * x.
    """
    if retail_subscription_x <= 0 or not (0 <= observed_allotment_rate <= 1):
        return
    obs = store.get_kv(RETAIL_K_KEY) or []
    obs.append({"symbol": symbol, "x": retail_subscription_x,
                "rate": observed_allotment_rate,
                "k": observed_allotment_rate * retail_subscription_x})
    store.set_kv(RETAIL_K_KEY, obs[-200:])
    log.info("%s: recorded allotment rate %.3f at %.2fx -> implied k=%.2f",
             symbol, observed_allotment_rate, retail_subscription_x,
             observed_allotment_rate * retail_subscription_x)


def calibrate_all(store: Store, settings: Settings) -> dict[str, Any]:
    report: dict[str, Any] = {}

    report["projection"] = SubscriptionProjector(store).fit_exponents() or \
        {"status": "insufficient"}
    report["listing"] = ListingModel(store).calibrate()

    # --- retail k, split into a cold and a hot regime
    obs = store.get_kv(RETAIL_K_KEY) or []
    usable = [o for o in obs if isinstance(o.get("k"), (int, float))
              and 0.5 <= o["k"] <= 6.0]
    if len(usable) >= 6:
        cold = sorted(o["k"] for o in usable if o["x"] < 8)
        hot = sorted(o["k"] for o in usable if o["x"] >= 8)
        changed = False
        if len(cold) >= 3:
            settings.allotment.avg_lots_retail_cold = round(
                clamp(cold[len(cold) // 2], 1.0, 3.0), 2)
            changed = True
        if len(hot) >= 3:
            settings.allotment.avg_lots_retail_hot = round(
                clamp(hot[len(hot) // 2], 1.0, 5.0), 2)
            changed = True
        if changed:
            settings.save()
            report["retail_k"] = {
                "status": "refit", "samples": len(usable),
                "cold": settings.allotment.avg_lots_retail_cold,
                "hot": settings.allotment.avg_lots_retail_hot,
                "written_to": str(CONFIG_PATH)}
        else:
            report["retail_k"] = {"status": "insufficient per-regime",
                                  "samples": len(usable)}
    else:
        report["retail_k"] = {"status": "insufficient", "samples": len(usable)}

    # --- how well have our own listing forecasts done?
    rows = store.training_rows()
    errs = [abs(r["predicted"] - r["listing_gain_pct"])
            for r in rows
            if r.get("metric") == "listing_gain_pct"
            and r.get("listing_gain_pct") is not None]
    if errs:
        errs.sort()
        report["self_score"] = {
            "n": len(errs),
            "median_abs_error_pp": round(errs[len(errs) // 2], 1),
            "mean_abs_error_pp": round(sum(errs) / len(errs), 1)}
    return report
