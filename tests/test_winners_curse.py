#!/usr/bin/env python3
"""
The "Prasol" regression - a hard defence against adverse selection.

THE FAILURE, OBSERVED LIVE
--------------------------
Prasol Chemicals opened its final day with QIB at 0.05x. An institutional
book that empty means almost every retail applicant gets filled, so its
allotment odds read 78.8% against ~6% for a 215x-subscribed peer. The
optimiser ranks by expected profit - odds x gain - so 78.8% of a small gain
beat 6% of a large one and capital was routed to the one issue the market
had actively declined.

High allotment odds are not an opportunity. They are the market's verdict
arriving in a form that flatters expected-value arithmetic.

WHY THE PRE-EXISTING GUARD MISSED IT
------------------------------------
listing.py already collapsed the estimate when EVERY category was under 1.0x.
The real book never satisfied that: at 05:08 it read

    QIB 0.05x   NII 0.66x   RETAIL 1.14x   (derived total 0.72x)

Retail alone was above 1.0x - and retail is exactly the category that is
loudest and least informed - so `max(...) < 1.0` was False and the guard
stayed silent. Informed money had refused the issue while the guard watched
the uninformed money and saw nothing wrong.

THREE LAYERED DEFENCES, asserted independently so no single threshold is
load-bearing:

  1. ListingModel forces expected gain to <= 0 on a late, unsubscribed book
  2. Scorer collapses the demand factor, dragging the composite under APPLY
  3. PortfolioOptimizer refuses anything below the execution floor

Run directly (`python3 tests/test_winners_curse.py`) or under pytest.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipo_radar.analytics import listing as listing_mod                # noqa: E402
from ipo_radar.analytics.listing import ListingModel                  # noqa: E402
from ipo_radar.analytics.portfolio import PortfolioOptimizer          # noqa: E402
from ipo_radar.analytics.scoring import Scorer                        # noqa: E402
from ipo_radar.config import PortfolioConfig                          # noqa: E402
from ipo_radar.models import (AllotmentOdds, IPO, NII, QIB,           # noqa: E402
                              RETAIL)

# --------------------------------------------------------------- fixtures
# Prasol's real window. 14:54 on the closing day is 90% elapsed against the
# 10:00-17:00 bidding hours the engine models.
OPEN_DATE, CLOSE_DATE = "2026-09-08", "2026-09-10"
NOW = datetime(2026, 9, 10, 14, 54)
GMP_PCT = 15.0                     # grey market still quoting a fat premium

# As specified: a book undersubscribed across the board.
SPEC_BOOK = {QIB: 0.05, NII: 0.6, RETAIL: 0.9}
SPEC_TOTAL = 0.6

# As actually observed at 05:08 on the closing day. Retail above 1.0x is what
# defeated the old guard, so this is the fixture that proves the new rule
# closes a gap rather than merely restating one that was already covered.
LIVE_BOOK = {QIB: 0.05, NII: 0.66, RETAIL: 1.14}
LIVE_TOTAL = 0.72
LIVE_SCORE = 55.5                  # the composite the engine actually stored
LIVE_ODDS = 0.788                  # the seductive number


def toxic_ipo(symbol: str = "PRASOLTEST") -> IPO:
    return IPO(symbol=symbol, name="Prasol Test Chemicals Limited",
               status="Active", open_date=OPEN_DATE, close_date=CLOSE_DATE,
               price_low=643, price_high=676, lot_size=22,
               issue_size_shares=5_443_229, issue_size_cr=368.0,
               fresh_issue_cr=80.0, ofs_cr=288.0, sector="chemical")


def weak_valuation() -> dict:
    """Prasol's real profile: 90x earnings, RoNW only just over Ke."""
    return {"pe": 90.0, "benchmark_pe": 24.0, "pe_vs_benchmark": 3.75,
            "valuation_score": 12.0, "quality_score": 70.0,
            "ronw_pct": 18.5, "cost_of_equity_pct": 12.9,
            "eva_verdict": "marginal", "eva_spread_pct": 5.6,
            "eps_cagr_pct": 40.0, "loss_making": False,
            "capital_intensity": 0.70, "sector_key": "chemical"}


def odds(ipo: IPO, sub_x: float, p: float) -> dict[str, AllotmentOdds]:
    lot_value = (ipo.cap_price or 0) * (ipo.lot_size or 0)
    return {"RETAIL:1": AllotmentOdds(
        category=RETAIL, subscription_x=sub_x, lots_applied=1,
        ticket_value=lot_value, p_single=p, expected_lots=p,
        p_any_multi=p, expected_lots_multi=p, mechanism="lottery",
        note="thin book - near-certain allotment")}


def predict(book: dict, total_x: float | None, *, rule: bool = True) -> dict:
    """Run ListingModel, optionally with the adverse-selection rule disabled."""
    original = listing_mod.LATE_WINDOW
    if not rule:                       # no window can exceed 1.0 -> never fires
        listing_mod.LATE_WINDOW = 1.01
    try:
        return ListingModel().predict(
            toxic_ipo(), book, GMP_PCT, weak_valuation(),
            projection_confidence=0.93, total_x=total_x, now=NOW)
    finally:
        listing_mod.LATE_WINDOW = original


# ------------------------------------------------------------------ tests
def test_fixture_is_in_the_late_window() -> float:
    p = toxic_ipo().bid_progress(NOW)
    assert p > 0.75, f"fixture is not in the late window: {p:.2f}"
    return p


def test_the_old_guard_could_not_see_the_live_book() -> None:
    """
    Establishes that this regression covers new ground.

    The pre-existing collapse fired only when every category was under 1.0x.
    The live book had retail at 1.14x, so it did not qualify.
    """
    assert max(LIVE_BOOK.values()) > 1.0, (
        "the live fixture no longer defeats the old guard - it has lost its "
        "point as a regression")
    assert LIVE_BOOK[QIB] < 1.0, "informed money must be absent"


def test_listing_model_forces_zero_gain_spec() -> dict:
    """Defence 1, on the specified book: gain forced to <= 0."""
    out = predict(SPEC_BOOK, SPEC_TOTAL)
    assert out["expected_gain_pct"] <= 0.0, (
        f"expected gain {out['expected_gain_pct']} must be <= 0 on an "
        f"undersubscribed late book")
    assert out.get("adverse_selection") is True
    assert "Adverse Selection" in out["basis"]
    assert out["p_positive"] == 0.0
    return out


def test_listing_model_forces_zero_gain_live() -> dict:
    """
    Defence 1, on the live book, with a control.

    Without the rule the model returns a positive expectation and a majority
    chance of a positive listing. That is the bug, reproduced.
    """
    before = predict(LIVE_BOOK, LIVE_TOTAL, rule=False)
    after = predict(LIVE_BOOK, LIVE_TOTAL)

    assert before["expected_gain_pct"] > 0.0, (
        "control did not reproduce the bug - the live fixture should yield a "
        "positive gain when the rule is disabled")
    assert before["p_positive"] > 0.5, "control should look attractive"

    assert after["expected_gain_pct"] <= 0.0, (
        f"expected gain {after['expected_gain_pct']} must be forced to <= 0")
    assert after.get("adverse_selection") is True
    assert after["p_positive"] == 0.0
    assert "QIB 0.05x" in after["basis"], "the reason must name the empty book"
    return after


def test_listing_model_spares_a_healthy_book() -> None:
    """Guard against over-correction: a subscribed issue is untouched."""
    healthy = predict({QIB: 20.0, NII: 15.0, RETAIL: 8.0}, 12.0)
    assert not healthy.get("adverse_selection")
    assert healthy["expected_gain_pct"] > 0.0


def test_missing_total_never_condemns() -> None:
    """
    NSE omits the Total row on most issues, including fully subscribed ones.
    Unknown must never be read as undersubscribed.
    """
    unknown = predict({QIB: 20.0, NII: 15.0, RETAIL: 8.0}, None)
    assert not unknown.get("adverse_selection")
    assert unknown["expected_gain_pct"] > 0.0


def score_book(book: dict, total_x: float, listing: dict) -> float:
    ipo = toxic_ipo()
    verdict = Scorer().score(
        ipo, valuation=weak_valuation(), projected=book, gmp_pct=GMP_PCT,
        gmp_trend=1.0, metrics={}, listing=listing, regime=None, llm=None,
        allotment=odds(ipo, book[RETAIL], LIVE_ODDS),
        projection_confidence=0.93, total_x=total_x, now=NOW)
    floor = PortfolioConfig().min_score
    assert verdict.score < floor, (
        f"score {verdict.score} must fall below the {floor} execution floor")
    assert verdict.factors["demand"] <= 10.0, (
        f"demand factor {verdict.factors['demand']} should be collapsed")
    assert any("DEMAND COLLAPSE" in r for r in verdict.risks), (
        "the collapse must surface as a risk, not be buried in reasons")
    return verdict.score


def test_scorer_drops_below_floor_spec(listing: dict) -> float:
    return score_book(SPEC_BOOK, SPEC_TOTAL, listing)


def test_scorer_drops_below_floor_live(listing: dict) -> float:
    return score_book(LIVE_BOOK, LIVE_TOTAL, listing)


def allocate(gain: float, score: float, p: float = LIVE_ODDS):
    ipo = toxic_ipo()
    cand = {"ipo": ipo, "verdict": None, "allotment": odds(ipo, 1.14, p),
            "expected_gain_pct": gain, "score": score, "veto": {}}
    return ipo, PortfolioOptimizer(500_000.0, num_pans=1,
                                   today=NOW.date()).optimise([cand])


def test_optimizer_allocates_nothing(listing: dict, score: float) -> None:
    """Defence 3: capital is refused, and the refusal is recorded."""
    ipo, alloc = allocate(listing["expected_gain_pct"], score)
    assert not alloc.applications, (
        f"optimiser allocated {len(alloc.applications)} application(s) to a "
        f"market-rejected issue")
    assert sum(a.lots for a in alloc.applications) == 0
    assert alloc.expected_profit == 0.0
    assert any(s["symbol"] == ipo.symbol for s in alloc.skipped), (
        "the refusal must be recorded with a reason, not silently dropped")


def test_score_floor_alone_rejects() -> None:
    """
    Defence 3 in isolation.

    Even handed the pre-fix numbers - a positive expected gain and the real
    78.8% odds - the raised floor alone refuses the issue. Defence 1 and 2
    are not load-bearing here.
    """
    _, alloc = allocate(gain=3.30, score=LIVE_SCORE)
    assert not alloc.applications, (
        f"the {PortfolioConfig().min_score} floor must reject a "
        f"{LIVE_SCORE}-score issue on its own")


def test_alert_floor_matches_execution_floor() -> None:
    """
    The alerting engine must not contradict the plan.

    T-Day alerts kept their own copy of the floor at 50.0. When the optimiser
    was raised to 61.0, Prasol - refused capital at score 57 - still fired a
    CRITICAL "the call is APPLY" push two minutes before cut-off. Whichever
    surface a user acts on, it has to agree with the others.
    """
    from ipo_radar.analytics.tday_signals import TDaySignalEngine
    assert TDaySignalEngine.MIN_SCORE == PortfolioConfig().min_score, (
        f"T-Day alert floor {TDaySignalEngine.MIN_SCORE} has drifted from the "
        f"execution floor {PortfolioConfig().min_score}")
    assert LIVE_SCORE < TDaySignalEngine.MIN_SCORE, (
        "Prasol's real score must not clear the alert floor")


def test_optimizer_still_funds_a_strong_issue() -> None:
    """
    Guard against over-correction.

    A floor that starves everything is not a fix. A genuinely strong issue
    with modest odds must still be funded.
    """
    ipo = IPO(symbol="GOODCO", name="Good Co Limited", status="Active",
              open_date=OPEN_DATE, close_date=CLOSE_DATE,
              price_low=600, price_high=632, lot_size=23, issue_size_cr=739.0)
    cand = {"ipo": ipo, "verdict": None, "allotment": odds(ipo, 17.0, 0.10),
            "expected_gain_pct": 39.0, "score": 76.0, "veto": {}}
    alloc = PortfolioOptimizer(500_000.0, num_pans=1,
                               today=NOW.date()).optimise([cand])
    assert alloc.applications, "a 76-score issue must still be funded"


def main() -> int:
    print("\n  The 'Prasol' Winner's Curse regression\n  " + "-" * 60)
    results: list[tuple[str, bool, str]] = []

    def run(label, fn, *a):
        try:
            out = fn(*a)
            results.append((label, True, ""))
            return out
        except AssertionError as exc:
            results.append((label, False, str(exc)))
        except Exception as exc:
            results.append((label, False, f"{type(exc).__name__}: {exc}"))
        return None

    progress = run("fixture sits in the late window",
                   test_fixture_is_in_the_late_window)
    run("live book defeats the OLD guard",
        test_the_old_guard_could_not_see_the_live_book)

    spec_l = run("ListingModel forces gain <= 0  (spec book)",
                 test_listing_model_forces_zero_gain_spec)
    live_l = run("ListingModel forces gain <= 0  (live book)",
                 test_listing_model_forces_zero_gain_live)
    run("healthy book untouched", test_listing_model_spares_a_healthy_book)
    run("missing total never condemns", test_missing_total_never_condemns)

    spec_s = run("Scorer drops below floor      (spec book)",
                 test_scorer_drops_below_floor_spec, spec_l) if spec_l else None
    live_s = run("Scorer drops below floor      (live book)",
                 test_scorer_drops_below_floor_live, live_l) if live_l else None

    if spec_l and spec_s is not None:
        run("Optimizer allocates nothing", test_optimizer_allocates_nothing,
            spec_l, spec_s)
    run("score floor alone rejects", test_score_floor_alone_rejects)
    run("alert floor matches plan floor",
        test_alert_floor_matches_execution_floor)
    run("strong issue still funded", test_optimizer_still_funds_a_strong_issue)

    for label, ok, err in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if err:
            print(f"         {err}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {passed}/{len(results)} pass")

    if live_l:
        control = predict(LIVE_BOOK, LIVE_TOTAL, rule=False)
        print(f"\n  live book   QIB 0.05x  NII 0.66x  RETAIL 1.14x  "
              f"total 0.72x  GMP +15%")
        print(f"  progress    {progress:.0%} of the bid window elapsed")
        print(f"  pre-fix     {control['expected_gain_pct']:+.2f}%  "
              f"p(positive) {control['p_positive']:.3f}   <- funded ahead of a "
              f"215x book")
        print(f"  post-fix    {live_l['expected_gain_pct']:+.2f}%  "
              f"p(positive) {live_l['p_positive']:.3f}")
        print(f"  composite   {live_s}  against a "
              f"{PortfolioConfig().min_score} floor")
    print()
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
