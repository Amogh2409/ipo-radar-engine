"""
Composite conviction score.

Seven weighted factors, each normalised to 0-100, combined per config
ScoreWeights. Every factor keeps its own sub-score so a report can always
answer "why did this get 72?" - a black-box number would be useless for a
decision you have to make with your own money.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

from ..config import ScoreWeights
from ..models import (BNII, EMPLOYEE, IPO, NII, QIB, RETAIL, SNII, Verdict)
from ..util import clamp, now_iso
from .anchors import profile_anchors

# ---------------------------------------------------------- terminal veto
# A business earning less on equity than equity costs is destroying capital.
# No amount of grey-market enthusiasm changes that, so a Value Trap is capped
# below the APPLY band no matter how hot the book gets.
VETO_CEILING = 59.0                 # top of NEUTRAL

# The one exception. A pop is a liquidity event, not an endorsement: if the
# expected listing gain is large AND institutions are genuinely crowding the
# book, the trade can be taken - but it is renamed so nobody mistakes it for
# an investment.
FLIP_MIN_GAIN_PCT = 20.0
FLIP_MIN_QIB_X = 5.0

GRADE_SPECULATIVE_FLIP = "SPECULATIVE FLIP"
# The flip exception bypasses the VETO CEILING, not the ordinary bands. An
# issue whose composite is weak on its own merits stays weak - otherwise a hot
# grey market could promote a genuinely bad business out of AVOID, which is
# the opposite of what a veto is for.
FLIP_MIN_COMPOSITE = 61.0           # the APPLY threshold
LONGTERM_VALUE_TRAP = ("AVOID - Value Trap: Fails to earn its Cost of Capital. "
                       "Do not hold post-listing.")


def is_value_trap(valuation: dict[str, Any] | None) -> bool:
    """True when the EVA gate found RoNW below the cost of equity."""
    if not valuation:
        return False
    if valuation.get("eva_verdict") == "value destroyer":
        return True
    spread = valuation.get("eva_spread_pct")
    return isinstance(spread, (int, float)) and spread < 0


def speculative_flip_allowed(listing: dict[str, Any] | None,
                             projected: dict[str, float] | None
                             ) -> tuple[bool, dict[str, Any]]:
    """
    Does a Value Trap still clear the bar for a pure listing-day flip?

    Both conditions must hold: a large expected pop, and institutional money
    actually crowding the book. Retail enthusiasm alone is not enough - it is
    the least informed capital in the offer and the most likely to be exiting
    into itself.
    """
    gain = (listing or {}).get("expected_gain_pct")
    qib = (projected or {}).get(QIB)
    facts = {"expected_gain_pct": gain, "projected_qib_x": qib,
             "min_gain_pct": FLIP_MIN_GAIN_PCT, "min_qib_x": FLIP_MIN_QIB_X}
    ok = (isinstance(gain, (int, float)) and gain > FLIP_MIN_GAIN_PCT
          and isinstance(qib, (int, float)) and qib > FLIP_MIN_QIB_X)
    return bool(ok), facts


# A book still unsubscribed this late is a structural rejection, not a slow
# start, and the composite must fall far enough to clear the APPLY band -
# otherwise the high allotment odds that come with an empty book make the
# issue look attractive on expected value alone.
COLLAPSE_PROGRESS = 0.75
COLLAPSE_CEILING = 10.0


def _demand_score(projected: dict[str, float], bid_progress: float = 0.0,
                  total_x: float | None = None) -> tuple[float, list[str]]:
    notes: list[str] = []
    if not projected:
        return 45.0, ["no demand data yet"]
    qib = projected.get(QIB, 0.0)
    nii = projected.get(NII) or max(projected.get(SNII, 0.0),
                                    projected.get(BNII, 0.0))
    ret = projected.get(RETAIL, 0.0)

    def band(x: float, strong: float, ok: float) -> float:
        if x >= strong:
            return 95.0
        if x >= ok:
            return 70.0 + 25.0 * (x - ok) / max(strong - ok, 1e-6)
        if x >= 1.0:
            return 45.0 + 25.0 * (x - 1.0) / max(ok - 1.0, 1e-6)
        return clamp(45.0 * x, 0, 45.0)

    # QIB carries the most information: institutions do the deepest diligence
    s = 0.50 * band(qib, 25, 3) + 0.28 * band(nii, 30, 3) + 0.22 * band(ret, 12, 2)
    if qib >= 10:
        notes.append(f"institutional book strong at {qib:.1f}x projected QIB")
    elif qib < 1.0:
        notes.append(f"QIB book weak ({qib:.2f}x) - institutions are passing")
    if ret >= 8:
        notes.append(f"heavy retail demand ({ret:.1f}x) - allotment will be a lottery")
    if max(qib, nii, ret) < 1.0:
        notes.append("issue may not reach full subscription")

    # --- late-window collapse. A present figure below 1.0x is decisive; a
    # missing one is not, because NSE omits the Total row more often than not.
    if bid_progress > COLLAPSE_PROGRESS:
        failed: list[str] = []
        if isinstance(total_x, (int, float)) and 0 < total_x < 1.0:
            failed.append(f"total book {total_x:.2f}x")
        if qib < 1.0:
            failed.append(f"QIB {qib:.2f}x")
        if failed:
            s = min(s, COLLAPSE_CEILING)
            notes.append(
                f"DEMAND COLLAPSE - {' and '.join(failed)} with "
                f"{bid_progress:.0%} of the window gone; the market has "
                f"declined this issue")
    return clamp(s, 0, 100), notes


def _gmp_score(gmp_pct: float | None, gmp_trend: float | None
               ) -> tuple[float, list[str]]:
    if gmp_pct is None:
        return 50.0, ["no GMP observed"]
    notes = []
    s = clamp(50.0 + 45.0 * math.tanh(gmp_pct / 22.0), 0, 100)
    if gmp_pct <= 0:
        notes.append("grey market flat or negative")
    elif gmp_pct >= 25:
        notes.append(f"strong grey market premium (+{gmp_pct:.1f}%)")
    if gmp_trend is not None:
        if gmp_trend < -1.5:
            s -= 12
            notes.append("GMP fading - late enthusiasm is cooling")
        elif gmp_trend > 1.5:
            s += 8
            notes.append("GMP firming into close")
    return clamp(s, 0, 100), notes


def _structure_score(ipo: IPO, metrics: dict[str, Any]
                     ) -> tuple[float, list[str], list[str], dict[str, Any]]:
    """Returns (score, positives, risks, anchor_profile)."""
    s, notes, risks = 55.0, [], []
    fresh, ofs = ipo.fresh_issue_cr, ipo.ofs_cr
    if fresh is not None and ofs is not None and (fresh + ofs) > 0:
        ofs_share = ofs / (fresh + ofs)
        if ofs_share > 0.85:
            s -= 16
            notes.append(f"{ofs_share:.0%} of the issue is an exit for existing "
                         "holders - no money reaches the company")
        elif ofs_share < 0.35:
            s += 10
            notes.append(f"mostly fresh capital ({1 - ofs_share:.0%}) into the business")
    # --- anchor book: not just who is in it, but how long they will stay.
    # SEBI unlocks 50% of anchor shares at 30 days and the rest at 90, so a
    # book of tactical money is a scheduled supply overhang.
    # Share counts drive the day-30 float expansion. `issue_size_shares` is
    # NSE's authoritative offer size; promoter lock-in is rarely parseable so
    # the profiler falls back to the listing float and says which basis it used.
    promoter_locked = metrics.get("promoter_locked_shares")
    profile = profile_anchors(
        metrics.get("anchor_investors"), ipo.name,
        anchor_shares=metrics.get("anchor_shares"),
        total_shares=metrics.get("shares_outstanding"),
        promoter_locked_shares=promoter_locked,
        offer_shares=ipo.issue_size_shares)
    anchor_info = profile.to_dict()
    if profile.overhang_risk != "UNKNOWN":
        # centre the 0-100 anchor quality on the structure score
        s += clamp((profile.quality_score - 50.0) * 0.34, -20.0, 20.0)
        if profile.overhang_risk == "HIGH":
            risks.extend(profile.flags)
        elif profile.overhang_risk == "MEDIUM":
            risks.extend(profile.flags)
        else:
            notes.extend(profile.flags)
    elif profile.flags:
        notes.extend(profile.flags)
    anchor_price, cap = metrics.get("anchor_price"), ipo.cap_price
    if anchor_price and cap:
        if anchor_price >= cap - 0.5:
            s += 6
            notes.append("anchors came in at the full cap price")
        elif anchor_price < cap * 0.97:
            s -= 8
            notes.append("anchors priced below the cap")
    iem = metrics.get("insider_exit_multiple")
    if iem:
        if iem >= 6:
            s -= 14
            notes.append(f"selling holders exiting at {iem:.1f}x their cost")
        elif iem <= 1.6:
            s += 8
            notes.append(f"offer priced close to recent insider cost ({iem:.2f}x)")
    if ipo.board == "SME":
        s -= 6
        risks.append("SME board - thinner liquidity, larger lot, higher risk")
    return clamp(s, 0, 100), notes, risks, anchor_info


def _regime_score(regime: dict[str, Any] | None,
                  valuation: dict[str, Any] | None = None
                  ) -> tuple[float, list[str]]:
    """
    Macro regime, scaled by how rate-sensitive THIS issue actually is.

    The same 60bp move is a different event for a 45x growth story than for a
    9x cash-generating manufacturer, so the penalty is pivoted about neutral
    by `issue_sensitivity` rather than applied flat.
    """
    if not regime:
        return 55.0, []
    from .regime import effective_regime_score, issue_sensitivity

    notes: list[str] = []
    macro = regime.get("macro") or {}
    base = macro.get("score")

    if base is None:                       # no macro layer: legacy behaviour
        base = 55.0
        idx = regime.get("index_change_pct")
        if idx is not None:
            base += clamp(idx * 4.0, -18, 18)
            if idx < -1.0:
                notes.append(f"market weak today ({idx:+.2f}% on the index)")
    else:
        notes.extend(macro.get("drivers", [])[:3])

    sens = issue_sensitivity(valuation)
    score = effective_regime_score(float(base), sens)
    if abs(score - base) >= 4:
        direction = "amplified" if score < base else "cushioned"
        notes.append(
            f"macro impact {direction} by this issue's rate sensitivity "
            f"({sens:.2f}x): regime {base:.0f} -> {score:.0f}")

    recent = regime.get("recent_listing_avg_pct")
    if recent is not None:
        score = clamp(score + clamp(recent * 0.5, -18, 18), 0, 100)
        notes.append(f"recent IPOs listed {recent:+.1f}% on average")
    return clamp(score, 0, 100), notes


class Scorer:
    def __init__(self, weights: ScoreWeights | None = None) -> None:
        self.w = weights or ScoreWeights()

    def score(self, ipo: IPO, *, valuation: dict[str, Any],
              projected: dict[str, float], gmp_pct: float | None,
              gmp_trend: float | None, metrics: dict[str, Any],
              listing: dict[str, Any], regime: dict[str, Any] | None = None,
              llm: dict[str, Any] | None = None,
              allotment: dict[str, Any] | None = None,
              projection_confidence: float = 1.0,
              weights: ScoreWeights | None = None,
              total_x: float | None = None,
              now: Any = None) -> Verdict:
        reasons: list[str] = []
        risks: list[str] = []

        val = float(valuation.get("valuation_score", 50.0))
        qual = float(valuation.get("quality_score", 50.0))
        dem, n1 = _demand_score(projected, ipo.bid_progress(now), total_x)
        collapsed = [n for n in n1 if n.startswith("DEMAND COLLAPSE")]
        # Early in the window the projected book is mostly assumption, so pull
        # the demand factor towards neutral instead of letting it dominate.
        pc = clamp(projection_confidence, 0.0, 1.0)
        dem = 50.0 + (dem - 50.0) * (0.45 + 0.55 * pc)
        # ...but a collapse is an OBSERVED book, not a projection, so the
        # damping must not walk it back towards neutral. Re-apply the cap.
        if collapsed:
            dem = min(dem, COLLAPSE_CEILING)
        if pc < 0.5:
            n1.append(f"demand still early ({pc:.0%} of the bid window elapsed) "
                      "- projections are indicative only")
        gmp_s, n2 = _gmp_score(gmp_pct, gmp_trend)
        struct, n3, struct_risks, anchor_info = _structure_score(ipo, metrics)
        reg, n4 = _regime_score(regime, valuation)

        llm = llm or {}
        qual_llm = llm.get("qualitative_score")
        llm_score = float(qual_llm) if isinstance(qual_llm, (int, float)) else 50.0

        factors = {"valuation": val, "financials": qual, "demand": dem,
                   "gmp": gmp_s, "structure": struct, "regime": reg,
                   "qualitative": llm_score}
        # weights ramp with the bid window: demand earns its influence as the
        # book fills, and gives it back to valuation/fundamentals when it cannot
        w = weights or self.w
        total_w = w.total() or 100.0
        composite = (val * w.valuation + qual * w.financials + dem * w.demand
                     + gmp_s * w.gmp + struct * w.structure + reg * w.regime
                     + llm_score * w.qualitative) / total_w

        # --- narrative
        pe, bench = valuation.get("pe"), valuation.get("benchmark_pe")
        if pe and bench:
            rel = pe / bench
            if rel < 0.8:
                reasons.append(f"priced at {pe:.1f}x vs {bench:.0f}x benchmark - a discount")
            elif rel > 1.5:
                risks.append(f"priced at {pe:.1f}x vs {bench:.0f}x benchmark - demanding")
        if valuation.get("ronw_pct") is not None:
            r = valuation["ronw_pct"]
            (reasons if r >= 18 else risks).append(
                f"RoNW {r:.1f}%" + (" - strong returns on capital" if r >= 18
                                    else " - modest returns on capital"))
        if valuation.get("eps_cagr_pct") is not None:
            g = valuation["eps_cagr_pct"]
            (reasons if g >= 20 else risks).append(f"EPS CAGR {g:+.0f}%")
        if valuation.get("loss_making"):
            risks.append("loss-making at the latest reported year")

        # EVA gate: earning less on equity than equity costs is a first-order
        # problem, so its flags go to risks, never to the positives column.
        for flag in (valuation.get("eva_risk_flags") or []):
            risks.append(flag)
        if valuation.get("eva_verdict") == "high-spread compounder":
            reasons.append(valuation.get("eva_flag", ""))

        risks.extend(struct_risks)
        risks.extend(collapsed)
        reasons.extend([n for n in n1 if n not in collapsed] + n2 + n3 + n4)
        for lst, key in ((reasons, "positives"), (risks, "risks")):
            extra = llm.get(key)
            if isinstance(extra, list):
                lst.extend(str(x) for x in extra[:5])

        # --- terminal veto: economics outrank sentiment
        value_trap = is_value_trap(valuation)
        flip_ok, flip_facts = False, {}
        veto: dict[str, Any] = {}
        if value_trap:
            flip_ok, flip_facts = speculative_flip_allowed(listing, projected)
            # the exception lifts the ceiling only for an issue that would
            # otherwise have earned an APPLY on its own composite
            flip_ok = flip_ok and composite >= FLIP_MIN_COMPOSITE
            flip_facts["min_composite"] = FLIP_MIN_COMPOSITE
            flip_facts["composite_before_veto"] = round(composite, 1)
            if not flip_ok:
                composite = min(composite, VETO_CEILING)
            ke = valuation.get("cost_of_equity_pct")
            ronw_v = valuation.get("ronw_pct")
            veto = {"value_trap": True, "speculative_flip": flip_ok,
                    "ceiling": VETO_CEILING, "cost_of_equity_pct": ke,
                    "ronw_pct": ronw_v,
                    "eva_spread_pct": valuation.get("eva_spread_pct"),
                    "projection_confidence": round(projection_confidence, 3),
                    **flip_facts}
            if flip_ok:
                veto["reason"] = (
                    f"Value Trap, but the listing case clears the flip bar: "
                    f"{flip_facts.get('expected_gain_pct'):.1f}% expected gain "
                    f"and {flip_facts.get('projected_qib_x'):.1f}x projected QIB. "
                    f"Traded as a liquidity event, not an investment.")
                risks.insert(0, (
                    "SPECULATIVE FLIP ONLY - this business does not earn its "
                    "cost of capital. Exit on listing; do not hold."))
            else:
                veto["reason"] = (
                    f"Value Trap: RoNW "
                    f"{ronw_v:.1f}% is below cost of equity {ke:.1f}%"
                    if isinstance(ronw_v, (int, float))
                    and isinstance(ke, (int, float))
                    else "Value Trap: RoNW is below cost of equity")
                veto["reason"] += (f"; composite capped at {VETO_CEILING:.0f}")
                risks.insert(0, (
                    f"Score capped at {VETO_CEILING:.0f} - a business earning "
                    f"below its cost of capital cannot be an APPLY for "
                    f"long-term capital"))

        # --- stances
        gain = listing.get("expected_gain_pct")
        p_pos = listing.get("p_positive")
        stance_listing = self._listing_stance(composite, gain, p_pos)
        if value_trap and flip_ok:
            stance_listing = ("speculative flip only - take the pop, exit on "
                              "listing day")
        elif value_trap:
            # A denied flip previously still read "apply for listing gains",
            # directly contradicting the veto sitting beside it.
            stance_listing = ("not worth the blocked capital - fails its cost "
                              "of capital and the flip bar was not met")
        stance_long = self._longterm_stance(composite, val, qual, value_trap)
        grade = self._grade(composite, value_trap=value_trap,
                            speculative_flip=flip_ok)

        # confidence reflects how much of the picture we actually have
        have = sum(x is not None for x in
                   (pe, gmp_pct, projected.get(QIB), metrics.get("anchor_shares")))
        confidence = clamp((0.30 + 0.13 * have + (0.10 if llm else 0.0)) * (0.55 + 0.45 * pc),
                           0.15, 0.95)

        verdict_out = Verdict(
            symbol=ipo.symbol, ts=now_iso(), score=round(composite, 1),
            grade=grade, stance_listing=stance_listing,
            stance_longterm=stance_long, confidence=round(confidence, 2),
            factors={k: round(v, 1) for k, v in factors.items()},
            reasons=reasons[:12], risks=risks[:12],
            expected_listing_gain_pct=gain,
            listing_gain_range=listing.get("range"),
            p_positive_listing=p_pos,
            projected_subscription={k: round(v, 2) for k, v in projected.items()},
            llm=llm,
        )
        verdict_out.anchors = anchor_info
        verdict_out.veto = veto
        return verdict_out

    @staticmethod
    def _grade(s: float, *, value_trap: bool = False,
               speculative_flip: bool = False) -> str:
        """
        Band the composite, honouring the terminal veto.

        The veto lives HERE rather than only at the call site so that any
        later re-grade - the LLM critique adjustment in engine.analyse, for
        instance - cannot quietly restore an APPLY on a wealth-destroying
        business by recomputing the band from a raw number.
        """
        if speculative_flip and s >= FLIP_MIN_COMPOSITE:
            return GRADE_SPECULATIVE_FLIP
        if value_trap:
            s = min(s, VETO_CEILING)
        if s >= 72:
            return "STRONG APPLY"
        if s >= 61:
            return "APPLY"
        if s >= 50:
            return "NEUTRAL"
        if s >= 38:
            return "AVOID"
        return "STRONG AVOID"

    @staticmethod
    def _listing_stance(score: float, gain: float | None,
                        p_pos: float | None) -> str:
        if gain is None:
            return "insufficient data"
        if gain >= 18 and (p_pos or 0) >= 0.72:
            return "apply for listing gains - attractive risk/reward"
        if gain >= 8 and (p_pos or 0) >= 0.62:
            return "apply for listing gains - moderate upside"
        if gain <= -3 or (p_pos or 1) < 0.42:
            return "avoid - listing likely at or below issue price"
        return "marginal - only worth it if capital is otherwise idle"

    @staticmethod
    def _longterm_stance(score: float, val: float, qual: float,
                         value_trap: bool = False) -> str:
        # A Value Trap has exactly one long-term answer, and it is not a
        # judgement call that hot demand or a cheap multiple can override.
        if value_trap:
            return LONGTERM_VALUE_TRAP
        if qual >= 68 and val >= 55:
            return "worth holding - good business at a defensible price"
        if qual >= 68 and val < 45:
            return "good business, rich price - consider buying post-listing dips"
        if qual < 45:
            return "weak fundamentals - not a hold"
        return "average - no strong long-term case"
