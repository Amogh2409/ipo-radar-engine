"""
Macro regime scoring, and why it must not be applied uniformly.

An IPO is a long-duration equity claim. When the discount rate rises, the
present value of cash flows that arrive in year eight falls far more than the
value of cash flows arriving next year. So a rate shock is not one number
applied to every issue alike: a 40x-earnings growth story and a 9x-earnings
cash-generating manufacturer live in genuinely different worlds when the
10-year yield jumps 60bp.

Two ideas do the work here:

  * Penalties are NON-LINEAR in the size of the move. A 10bp drift is noise
    and should cost nothing; a 60bp repricing should hurt. A linear function
    cannot express both, so stress rises with an exponent above one.
  * Penalties are scaled by the ISSUE'S OWN duration. `issue_sensitivity()`
    reads how much of the valuation rests on distant growth, and the regime
    penalty is multiplied by it.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from ..util import clamp

log = logging.getLogger("ipo_radar.regime")

# Roughly neutral for India through this cycle. Above the upper bound policy
# is restrictive and risk appetite for new paper thins out.
NEUTRAL_10Y = 6.60
RESTRICTIVE_10Y = 7.40
# A daily move only starts to matter beyond the noise band.
YIELD_NOISE_BP = 12.0
YIELD_SHOCK_BP = 45.0
SPREAD_NOISE_BP = 10.0
SPREAD_SHOCK_BP = 35.0
STRESS_EXPONENT = 1.7        # >1 so small moves are cheap, large ones bite

# Stress budget. Sums to 1.0 so a fully-stressed regime lands at SCORE_FLOOR
# rather than saturating - a regime that always reads zero cannot rank issues.
W_YIELD, W_SPREAD, W_LEVEL, W_EQUITY = 0.45, 0.25, 0.18, 0.12
SCORE_BASE, SCORE_RANGE = 88.0, 80.0     # stress 0 -> 88, stress 1 -> 8


@dataclass
class MacroRegime:
    score: float = 55.0                  # 0-100, higher is friendlier
    stress: float = 0.0                  # 0-1, drives Bayesian noise inflation
    india_10y: float | None = None
    credit_spread: float | None = None
    credit_spread_source: str = ""
    yield_change_bp: float | None = None
    spread_change_bp: float | None = None
    index_change_pct: float | None = None
    baseline_days: int = 0
    drivers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 1), "stress": round(self.stress, 3),
                "india_10y": self.india_10y,
                "credit_spread": self.credit_spread,
                "credit_spread_source": self.credit_spread_source,
                "yield_change_bp": self.yield_change_bp,
                "spread_change_bp": self.spread_change_bp,
                "index_change_pct": self.index_change_pct,
                "baseline_days": self.baseline_days, "drivers": self.drivers}


def _nonlinear(move_bp: float, noise: float, shock: float) -> float:
    """0 below the noise band, 1 at a full shock, convex in between."""
    if move_bp <= noise:
        return 0.0
    span = max(shock - noise, 1e-6)
    # Capped at 1.0: beyond a full shock the marginal information is nil, and
    # letting components exceed 1 saturates the total so every bad regime
    # scores identically zero, destroying the discrimination we need.
    return clamp(((move_bp - noise) / span) ** STRESS_EXPONENT, 0.0, 1.0)


def assess(snapshot: Any, history: list[dict[str, Any]] | None = None
           ) -> MacroRegime:
    """
    Turn a macro snapshot plus its own history into a regime score.

    Levels alone are nearly useless - a 6.7% 10-year is only informative
    relative to where it has been. With no history yet the function returns a
    near-neutral score and says so, rather than inventing a trend.
    """
    d = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot or {})
    reg = MacroRegime(india_10y=d.get("india_10y"),
                      credit_spread=d.get("credit_spread"),
                      credit_spread_source=d.get("credit_spread_source", ""),
                      index_change_pct=d.get("index_change_pct"))
    drivers: list[str] = []
    stress = 0.0

    hist = [h for h in (history or []) if h.get("india_10y")]
    reg.baseline_days = len(hist)

    # --- change versus a trailing baseline
    if len(hist) >= 3:
        base_y = statistics.median([h["india_10y"] for h in hist[:-1]] or
                                   [hist[0]["india_10y"]])
        if reg.india_10y is not None:
            reg.yield_change_bp = round((reg.india_10y - base_y) * 100, 1)
            s = _nonlinear(reg.yield_change_bp, YIELD_NOISE_BP, YIELD_SHOCK_BP)
            stress += W_YIELD * s
            if s > 0.25:
                drivers.append(
                    f"10y yield {reg.yield_change_bp:+.0f}bp vs its recent "
                    f"baseline - liquidity tightening")
        spreads = [h["credit_spread"] for h in hist[:-1] if h.get("credit_spread")]
        if spreads and reg.credit_spread is not None:
            base_s = statistics.median(spreads)
            reg.spread_change_bp = round((reg.credit_spread - base_s) * 100, 1)
            s = _nonlinear(reg.spread_change_bp, SPREAD_NOISE_BP, SPREAD_SHOCK_BP)
            stress += W_SPREAD * s
            if s > 0.25:
                drivers.append(
                    f"credit spread {reg.spread_change_bp:+.0f}bp wider "
                    f"({reg.credit_spread_source})")
    else:
        drivers.append("no macro baseline yet - regime treated as neutral")

    # --- absolute level: restrictive rates thin out appetite for new paper
    if reg.india_10y is not None:
        if reg.india_10y > NEUTRAL_10Y:
            over = (reg.india_10y - NEUTRAL_10Y) / max(
                RESTRICTIVE_10Y - NEUTRAL_10Y, 1e-6)
            lvl = clamp(over ** STRESS_EXPONENT, 0.0, 1.0)
            stress += W_LEVEL * lvl
            if lvl > 0.4:
                drivers.append(f"10y at {reg.india_10y:.2f}% is above the "
                               f"{NEUTRAL_10Y:.2f}% neutral anchor")
        else:
            stress -= 0.10
            drivers.append(f"10y at {reg.india_10y:.2f}% is accommodative")

    # --- same-day equity tape
    idx = reg.index_change_pct
    if idx is not None:
        if idx < -0.75:
            stress += W_EQUITY * clamp(abs(idx) / 2.5, 0.0, 1.0)
            drivers.append(f"NIFTY 50 {idx:+.2f}% - risk-off tape")
        elif idx > 0.75:
            stress -= 0.08

    reg.stress = clamp(stress, 0.0, 1.0)
    reg.score = clamp(SCORE_BASE - SCORE_RANGE * reg.stress, 0.0, 100.0)
    reg.drivers = drivers[:6]
    return reg


def issue_sensitivity(valuation: dict[str, Any] | None) -> float:
    """
    How exposed THIS issue is to a discount-rate shock, as a multiplier on the
    regime penalty. ~0.55 for a cheap, cash-generating issuer; ~1.60 for an
    expensive growth story whose value sits in the distant future.
    """
    v = valuation or {}
    sens = 1.0

    rel_pe = v.get("pe_vs_benchmark")
    if isinstance(rel_pe, (int, float)):
        sens += clamp((rel_pe - 1.0) * 0.55, -0.35, 0.55)
    rel_ev = v.get("ev_vs_benchmark")
    if isinstance(rel_ev, (int, float)):
        sens += clamp((rel_ev - 1.0) * 0.35, -0.25, 0.35)

    pe = v.get("pe")
    if isinstance(pe, (int, float)) and pe > 0:
        # absolute multiple is a duration proxy in its own right
        sens += clamp((pe - 25.0) / 55.0, -0.25, 0.40)
    pb = v.get("pb")
    if isinstance(pb, (int, float)) and pb > 8:
        sens += 0.15

    if v.get("loss_making"):
        sens += 0.35                      # all of the value is in the future
    cagr = v.get("eps_cagr_pct")
    if isinstance(cagr, (int, float)) and cagr > 45:
        sens += 0.12                      # priced on growth persisting
    ronw = v.get("ronw_pct")
    if isinstance(ronw, (int, float)) and ronw >= 25 and not v.get("loss_making"):
        sens -= 0.18                      # cash today, not a promise
    return clamp(sens, 0.50, 1.65)


def effective_regime_score(base_score: float, sensitivity: float) -> float:
    """
    Apply the regime to an issue in proportion to its rate sensitivity -
    ASYMMETRICALLY.

    A hostile regime costs a high-duration growth story far more than a
    cash-generating value issue, so the penalty is amplified by sensitivity.
    A benign regime is NOT amplified in the same way. The symmetric version of
    this function scored an expensive loss-making issue 83 against a base of
    70 purely because easy money flatters long-duration assets, which is
    exactly the reasoning that loses money at the top of a cycle. Cheap credit
    is a reason to be less afraid, never a reason to pay more.

    So: sensitivity amplifies downside only. Upside passes through unchanged.
    """
    if base_score >= 50.0:
        return clamp(base_score, 0.0, 100.0)
    # floored rather than zeroed so extreme regimes still rank issues
    return clamp(50.0 - (50.0 - base_score) * sensitivity, 3.0, 100.0)
