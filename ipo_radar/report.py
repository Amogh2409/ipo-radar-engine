"""Terminal dashboard and on-disk reports."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import shutil

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analytics.portfolio import Allocation
from .config import REPORT_DIR
from .models import BNII, IPO, QIB, RETAIL, SNII, Verdict
from .store import Store

# When output is piped, rich falls back to 80 columns and shreds the table.
_TERM_W = shutil.get_terminal_size((132, 40)).columns
console = Console(width=max(_TERM_W, 118))

GRADE_STYLE = {
    "STRONG APPLY": "bold green", "APPLY": "green", "NEUTRAL": "yellow",
    "AVOID": "red", "STRONG AVOID": "bold red",
    # Deliberately not green: a flip is a trade to exit, not a holding.
    "SPECULATIVE FLIP": "bold magenta",
}


def _pct(v: float | None, dp: int = 1) -> str:
    return "—" if v is None else f"{v:+.{dp}f}%"


def _x(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}x"


def _days_left(ipo: IPO) -> str:
    _, close = ipo.days()
    if not close:
        return "—"
    d = (close - date.today()).days
    if d < 0:
        return "closed"
    return "closes today" if d == 0 else f"{d}d left"


# --------------------------------------------------------------- dashboard
ACTION_STYLE = {"APPLY": "bold green", "AVOID": "red", "WAIT": "dim"}


def dashboard(store: Store, verdicts: list[Verdict],
              allocation: Allocation | None = None,
              regime: dict[str, Any] | None = None,
              signals: dict[str, Any] | None = None) -> Panel:
    t = Table(expand=True, header_style="bold cyan", box=None, pad_edge=False)
    for col, just in (("IPO", "left"), ("Closes", "left"), ("QIB", "right"),
                      ("NII", "right"), ("RET", "right"), ("GMP", "right"),
                      ("Est.list", "right"), ("P(allot)", "right"),
                      ("Conf", "right"), ("Score", "right"), ("Verdict", "left"),
                      ("Cut-off", "right"), ("Do", "left")):
        t.add_column(col, justify=just, no_wrap=True)

    for v in verdicts:
        ipo = store.get_ipo(v.symbol)
        if not ipo:
            continue
        gmp = store.latest_gmp(v.symbol)
        proj = v.projected_subscription or {}
        retail_key = next((k for k in v.allotment if k.startswith(f"{RETAIL}:")), None)
        odds = v.allotment.get(retail_key) if retail_key else None
        p_txt = "—"
        if odds:
            p = odds.p_single if isinstance(odds.p_single, float) else None
            if p is not None:
                p_txt = f"{p:.0%}"
        t.add_row(
            Text(v.symbol[:11], style="bold"),
            _days_left(ipo),
            _x(proj.get(QIB)),
            _x(proj.get("NII") or proj.get(SNII)),
            _x(proj.get(RETAIL)),
            _pct(gmp.est_gain_pct if gmp else None, 0),
            _pct(v.expected_listing_gain_pct, 0),
            p_txt,
            f"{v.confidence:.0%}",
            Text(f"{v.score:.0f}", style=GRADE_STYLE.get(v.grade, "white")),
            Text(v.grade, style=GRADE_STYLE.get(v.grade, "white")),
            *_tday_cells(v, signals),
        )

    parts: list[Any] = [t]

    live_alerts = _live_alerts(verdicts, signals)
    if live_alerts:
        parts.append(Text("\n  Execution alerts", style="bold red"))
        for level, msg in live_alerts:
            style = "bold red" if level == "CRITICAL" else "bold yellow"
            parts.append(Text(f"   [{level}] {msg}", style=style))

    if allocation and allocation.applications:
        mode = getattr(allocation, "objective", "maximize_roi")
        label = ("maximising absolute profit"
                 if mode == "maximize_absolute_profit"
                 else "maximising return on blocked capital")
        parts.append(Text(f"\n  Suggested application plan  ({label})",
                          style="bold cyan"))
        a = Table(expand=True, header_style="bold", box=None)
        for c in ("PAN", "IPO", "Category", "Lots", "Capital", "Blocked",
                  "Deployable", "P(allot)", "Exp.profit", "ROI"):
            a.add_column(c, justify="right" if c not in ("IPO", "Category",
                                                         "Deployable") else "left")
        for app in allocation.applications:
            sym = Text(app.symbol, style="bold cyan" if app.reserved else "")
            a.add_row(f"#{app.pan_index}", sym, app.category,
                      str(app.lots), f"₹{app.capital:,.0f}",
                      f"{app.days_blocked}d",
                      # available_from, not unlock_on: the mandate releases in
                      # an evening batch, so the release date is precisely the
                      # day this capital CANNOT back a new bid.
                      _short_date(app.available_from or app.unlock_on),
                      f"{app.p_allot:.0%}", f"₹{app.expected_profit:,.0f}",
                      f"{app.roi_on_blocked:.2f}%")
        parts.append(a)
        summary = (f"  peak capital blocked ₹{allocation.capital_used:,.0f} of "
                   f"₹{allocation.capital_available:,.0f}  ·  expected profit "
                   f"₹{allocation.expected_profit:,.0f} "
                   f"({allocation.expected_roi_pct:.2f}% on peak blocked)")
        if allocation.reserved_symbols:
            summary += (f"  ·  reserved: "
                        f"{', '.join(allocation.reserved_symbols)}")
        parts.append(Text(summary, style="dim"))
        parts.extend(_cash_flow_rows(allocation))

    sub = f"{datetime.now():%d %b %Y  %H:%M:%S}"
    if regime and regime.get("index_change_pct") is not None:
        sub += f"   ·   NIFTY 50 {regime['index_change_pct']:+.2f}%"
    return Panel(Group(*parts), title="[bold]IPO RADAR[/bold]",
                 subtitle=sub, border_style="cyan")


def _short_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return date.fromisoformat(iso).strftime("%d %b")
    except (TypeError, ValueError):
        return str(iso)


def _cash_flow_rows(allocation: Allocation) -> list[Any]:
    """
    When the money comes back.

    ASBA does not spend capital, it sterilises it for a few days, so the
    number that matters is not "how much did I commit" but "how much is free
    on any given day". This is what tells you whether an issue closing next
    week is still fundable.
    """
    timeline = getattr(allocation, "timeline", None) or []
    if not timeline:
        return []
    out: list[Any] = [Text("\n  Cash flow projection (ASBA T+3 unlock)",
                           style="bold cyan")]
    t = Table(expand=True, header_style="bold", box=None)
    for col, just in (("Date", "left"), ("Blocked", "right"),
                      ("Released", "right"), ("Outstanding", "right"),
                      ("Free to deploy", "right"), ("Movements", "left")):
        t.add_column(col, justify=just, no_wrap=True)
    for ev in timeline:
        moves = " ".join([f"+{s}" for s in ev.symbols_in]
                         + [f"-{s}" for s in ev.symbols_out])
        low = ev.headroom < allocation.capital_available * 0.15
        t.add_row(
            ev.day.strftime("%a %d %b"),
            f"₹{ev.committed:,.0f}" if ev.committed else "—",
            f"₹{ev.released:,.0f}" if ev.released else "—",
            f"₹{ev.outstanding:,.0f}",
            Text(f"₹{ev.headroom:,.0f}", style="red" if low else "green"),
            moves[:60] or "—")
    out.append(t)
    return out


def _signal_dict(v: Verdict, signals: dict[str, Any] | None) -> dict[str, Any]:
    """Signals may arrive as live objects (daemon) or as stored dicts."""
    sig = (signals or {}).get(v.symbol)
    if sig is None:
        return v.tday or {}
    return sig.to_dict() if hasattr(sig, "to_dict") else sig


def _tday_cells(v: Verdict, signals: dict[str, Any] | None) -> tuple[Any, Any]:
    d = _signal_dict(v, signals)
    if not d:
        return "—", "—"
    clock = d.get("clock") or {}
    advice = d.get("advice") or {}
    action = str(advice.get("action") or "—")
    return (clock.get("countdown") or "—",
            Text(action, style=ACTION_STYLE.get(action, "white")))


def _live_alerts(verdicts: list[Verdict],
                 signals: dict[str, Any] | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for v in verdicts:
        for a in (_signal_dict(v, signals).get("alerts") or []):
            if a.get("level") in ("HIGH", "CRITICAL"):
                out.append((a["level"], a["message"]))
    out.sort(key=lambda x: 0 if x[0] == "CRITICAL" else 1)
    return out[:8]


def print_dashboard(store: Store, verdicts: list[Verdict],
                    allocation: Allocation | None = None,
                    regime: dict[str, Any] | None = None,
                    signals: dict[str, Any] | None = None) -> None:
    console.print(dashboard(store, verdicts, allocation, regime, signals))


# ------------------------------------------------------------ deep report
def detail(store: Store, verdict: Verdict) -> str:
    """Full markdown write-up for one IPO."""
    ipo = store.get_ipo(verdict.symbol)
    if not ipo:
        return ""
    gmp = store.latest_gmp(ipo.symbol)
    sub = store.latest_subscription(ipo.symbol)
    fin = store.get_financials(ipo.symbol).get("rhp", {}) or {}
    llm = verdict.llm or {}
    syn = llm.get("synthesis") or {}
    L: list[str] = []
    A = L.append

    A(f"# {ipo.name} ({ipo.symbol})")
    A("")
    A(f"**{verdict.grade}** · score **{verdict.score:.0f}/100** · "
      f"confidence {verdict.confidence:.0%}")
    if syn.get("headline"):
        A("")
        A(f"> {syn['headline']}")
    A("")
    A("## Issue")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| Price band | ₹{ipo.price_low:.0f} – ₹{ipo.price_high:.0f} |")
    A(f"| Lot size | {ipo.lot_size} shares (₹{ipo.lot_value:,.0f}) |"
      if ipo.lot_size else "| Lot size | — |")
    A(f"| Issue size | ₹{ipo.issue_size_cr:,.0f} Cr |"
      if ipo.issue_size_cr else "| Issue size | — |")
    A(f"| Dates | {ipo.open_date} → {ipo.close_date} |")
    A(f"| Registrar | {ipo.registrar or '—'} |")
    if ipo.meta.get("brlms"):
        A(f"| Lead managers | {', '.join(ipo.meta['brlms'][:4])} |")
    A("")

    A("## Valuation")
    A("")
    val = verdict.valuation or {}
    rows = [("P/E at cap price", fin.get("pe_cap_diluted") or fin.get("pe_cap_basic"), "x"),
            ("P/B at cap price", fin.get("pb_cap_pre_issue"), "x"),
            ("EV/EBITDA at cap price", val.get("ev_ebitda"), "x"),
            ("Enterprise value", val.get("enterprise_value_cr"), "Cr"),
            ("Market cap (implied)", val.get("market_cap_cr"), "Cr"),
            ("Net debt", val.get("net_debt_cr"), "Cr"),
            ("EBITDA (latest FY)", val.get("ebitda_cr"), "Cr"),
            ("Cash flow from operations", val.get("cfo_cr"), "Cr"),
            ("Cash conversion (CFO/EBITDA)", val.get("cash_conversion"), "x"),
            ("RoNW (latest FY)", fin.get("ronw_pct"), "%"),
            ("NAV per share", fin.get("nav_pre_issue"), "₹"),
            ("Insider exit multiple", fin.get("insider_exit_multiple"), "x")]
    A("| Metric | Value |")
    A("|---|---|")
    for label, value, unit in rows:
        if value is None:
            continue
        if unit == "Cr":
            A(f"| {label} | ₹{value:,.0f} Cr |")
        elif unit == "₹":
            A(f"| {label} | ₹{value:,.2f} |")
        else:
            A(f"| {label} | {value:,.2f}{unit} |")
    eps = fin.get("eps_by_fy") or {}
    if eps:
        A("")
        A("EPS trajectory: " + " → ".join(
            f"FY{fy[-2:]} ₹{eps[fy]['basic']:.2f}" for fy in sorted(eps)))
    if val.get("valuation_basis"):
        A("")
        A(f"_Valuation basis: {val['valuation_basis']}._")
        if val.get("ev_vs_benchmark"):
            A(f"_EV/EBITDA is {val['ev_vs_benchmark']:.2f}x the "
              f"{val.get('ev_ebitda_benchmark')}x sector anchor._")
    elif val.get("ev_status") and val.get("ev_status") != "ok":
        A("")
        A(f"_EV/EBITDA unavailable: {val['ev_status']}._")
    if fin.get("peer_note"):
        A("")
        A(f"*{fin['peer_note']}*")
    A("")

    A("## Demand")
    A("")
    if sub:
        A("| Category | Offered | Bid | Current | Projected final |")
        A("|---|---|---|---|---|")
        for cat, cb in sub.categories.items():
            proj = verdict.projected_subscription.get(cat)
            A(f"| {cat} | {cb.offered:,.0f} | {cb.bid:,.0f} | "
              f"{cb.subscribed:.2f}x | {proj:.2f}x |" if proj is not None else
              f"| {cat} | {cb.offered:,.0f} | {cb.bid:,.0f} | {cb.subscribed:.2f}x | — |")
    else:
        A("_No bidding data yet._")
    if gmp and gmp.gmp is not None:
        A("")
        A(f"Grey market premium: **₹{gmp.gmp:.0f}** "
          f"({_pct(gmp.est_gain_pct)}) via {gmp.source}")
    A("")

    if verdict.posteriors:
        A("## Demand — posterior view")
        A("")
        A("_Final subscription as a distribution. The 80% interval is the "
          "honest width of what is still unknown._")
        A("")
        A("| Category | Now | Posterior median | 80% interval | Confidence |")
        A("|---|---|---|---|---|")
        for cat, pd in verdict.posteriors.items():
            if not isinstance(pd, dict):
                continue
            lo, hi = (pd.get("ci80") or [None, None])[:2]
            A(f"| {cat} | {pd.get('observed', 0):.2f}x | "
              f"**{pd.get('median', 0):.2f}x** | "
              f"{lo:.2f}x – {hi:.2f}x | {pd.get('confidence', 0):.0%} |"
              if lo is not None else
              f"| {cat} | {pd.get('observed', 0):.2f}x | "
              f"{pd.get('median', 0):.2f}x | — | — |")
        A("")
        if verdict.weights_used:
            A(f"_Score weighting right now: demand "
              f"{verdict.weights_used.get('demand')}%, valuation "
              f"{verdict.weights_used.get('valuation')}%, financials "
              f"{verdict.weights_used.get('financials')}% — demand earns "
              f"influence as the book fills._")
            A("")

    A("## Listing expectation")
    A("")
    lo, hi = verdict.listing_gain_range or (None, None)
    A(f"- Central estimate: **{_pct(verdict.expected_listing_gain_pct)}**")
    if lo is not None:
        A(f"- Likely range: {lo:+.1f}% to {hi:+.1f}%")
    if verdict.p_positive_listing is not None:
        A(f"- Probability of a positive listing: **{verdict.p_positive_listing:.0%}**")
    A("")

    A("## Allotment odds")
    A("")
    A("| Apply as | Lots | Capital blocked | Mechanism | P(allotment) | E[shares] |")
    A("|---|---|---|---|---|---|")
    for key, o in verdict.allotment.items():
        if o.mechanism == "n/a":
            continue
        A(f"| {o.category} | {o.lots_applied} | ₹{o.ticket_value:,.0f} | "
          f"{o.mechanism} | {o.p_single:.1%} | "
          f"{o.expected_lots * (ipo.lot_size or 0):,.0f} |")
    if any(o.mechanism == "lottery" for o in verdict.allotment.values()):
        A("")
        A("> Retail is a lottery here: every winner gets exactly one lot, so a "
          "larger bid on one PAN does not improve your odds. Additional PANs do.")
    A("")

    anchors = verdict.anchors or {}
    d30 = anchors.get("day30") or {}
    if d30.get("day30_impact_pct") is not None:
        A("## Anchor lock-in supply")
        A("")
        A(f"- 30-day tranche releases **{d30['day30_released_shares']:,}** "
          f"shares (50% of the anchor allocation)")
        A(f"- That expands the tradable float by "
          f"**{d30['day30_impact_pct']:.1f}%** "
          f"(basis: {d30.get('day30_basis', '—')})")
        A(f"- Anchor book: {anchors.get('compounder_ratio', 0):.0%} patient "
          f"capital, {anchors.get('flipper_ratio', 0):.0%} tactical, "
          f"overhang risk **{anchors.get('overhang_risk', '—')}**")
        if d30.get("day30_warning"):
            A("")
            A("> Both conditions are met: a large float expansion AND a book "
              "weighted to holders who trade. Expect supply pressure around "
              "the 30-day unlock.")
        A("")

    sb = (verdict.regime or {}).get("structural_break") or {}
    if sb.get("active"):
        A("## Structural break")
        A("")
        A(f"- {sb.get('reason', '')}")
        A("- Subscription posteriors widened and pulled back toward the "
          "issue-size prior; credible intervals reflect execution risk, not "
          "just sampling noise.")
        A("")

    macro = (verdict.regime or {}).get("macro") or {}
    if macro:
        A("## Macro regime")
        A("")
        bits = []
        if macro.get("india_10y") is not None:
            bits.append(f"India 10y **{macro['india_10y']:.2f}%**")
        if macro.get("yield_change_bp") is not None:
            bits.append(f"{macro['yield_change_bp']:+.0f}bp vs baseline")
        if macro.get("credit_spread") is not None:
            bits.append(f"spread {macro['credit_spread']:.2f}pp "
                        f"({macro.get('credit_spread_source', '')})")
        if bits:
            A("- " + " · ".join(bits))
        A(f"- Regime score **{macro.get('score', '—')}/100**, "
          f"stress {macro.get('stress', 0):.2f}")
        if macro.get("bayes_noise_gain"):
            A(f"- Regime below the stress threshold — subscription posteriors "
              f"widened by {macro['bayes_noise_gain']:.2f}")
        for d in (macro.get("drivers") or [])[:4]:
            A(f"  - {d}")
        A("")

    td = verdict.tday or {}
    if td:
        clock = td.get("clock") or {}
        advice = td.get("advice") or {}
        A("## Execution")
        A("")
        A(f"- Phase: **{clock.get('phase', '—')}**, "
          f"{clock.get('countdown', '—')} to the UPI mandate cut-off")
        if clock.get("cutoff_at"):
            A(f"- Mandate cut-off: **{clock['cutoff_at']} IST** "
              f"({clock.get('source', '')})")
        action = advice.get("action")
        if action:
            A(f"- Call: **{action}**"
              + (f" — {advice.get('category')} × {advice.get('lots')} lot(s), "
                 f"₹{advice.get('capital', 0):,.0f} blocked"
                 if action == "APPLY" else ""))
        if advice.get("bid_price_instruction"):
            A(f"- Price: {advice['bid_price_instruction']}")
        if advice.get("deadline_text"):
            A(f"- {advice['deadline_text']}")
        for r in (advice.get("rationale") or [])[:4]:
            A(f"  - {r}")
        spikes = td.get("spikes") or {}
        hot = {k: v for k, v in spikes.items() if v.get("is_spike")}
        if hot:
            A("")
            for cat, sp in hot.items():
                A(f"- **{cat} demand spike**: {sp['current_rate']:.2f}x/hour vs "
                  f"{sp['baseline_mean']:.2f}x/hour baseline "
                  f"({sp['z_score']:.1f} sigma)")
        A("")

    if verdict.guardrails and verdict.guardrails.get("violations"):
        g = verdict.guardrails
        A("## Model guardrails")
        A("")
        A(f"{g['violations']} model claim(s) were rejected before scoring"
          + (f" — analysis fell back to pure-Python rules ({g.get('fallback_reason')})"
             if g.get("used_fallback") else "") + ".")
        A("")
        for d in (g.get("details") or [])[:6]:
            A(f"- `{d['kind']}` in {d['field']}: {d['detail']}"
              + (f"\n  - removed: \"{d['removed'][:120]}\"" if d.get("removed") else ""))
        A("")

    if verdict.reasons:
        A("## Why it scores where it does")
        A("")
        for r in verdict.reasons:
            A(f"- {r}")
        A("")
    if verdict.risks:
        A("## Risks")
        A("")
        for r in verdict.risks:
            A(f"- {r}")
        A("")

    if llm:
        A("## Analyst view")
        A("")
        if llm.get("business_summary"):
            A(llm["business_summary"])
            A("")
        for label, key in (("Moat", "competitive_moat"),
                           ("Use of proceeds", "use_of_proceeds")):
            if llm.get(key):
                A(f"**{label}:** {llm[key]}")
                A("")
        if llm.get("red_flags"):
            A("**Red flags**")
            A("")
            for f in llm["red_flags"]:
                A(f"- {f}")
            A("")
        crit = llm.get("critique") or {}
        if crit.get("strongest_bear_case"):
            A("**The bear case**")
            A("")
            A(crit["strongest_bear_case"])
            A("")
        if syn:
            A("**Bottom line**")
            A("")
            A(f"- Recommendation: **{syn.get('recommendation', '—')}**")
            A(f"- Where to apply: **{syn.get('apply_in_category', '—')}**")
            if syn.get("one_line_reason"):
                A(f"- {syn['one_line_reason']}")
            A("")

    A("---")
    A(f"*Generated {datetime.now():%d %b %Y %H:%M} by IPO Radar. "
      "Analysis only — not investment advice.*")
    return "\n".join(L)


def write_reports(store: Store, verdicts: list[Verdict],
                  allocation: Allocation | None = None,
                  signals: dict[str, Any] | None = None) -> list[Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for v in verdicts:
        p = REPORT_DIR / f"{v.symbol}.md"
        p.write_text(detail(store, v), encoding="utf-8")
        written.append(p)

    idx: list[str] = ["# IPO Radar — live board", "",
                      f"_{datetime.now():%d %b %Y %H:%M}_", "",
                      "| IPO | Closes | Score | Verdict | Est. listing | "
                      "Retail P(allot) |", "|---|---|---|---|---|---|"]
    for v in verdicts:
        ipo = store.get_ipo(v.symbol)
        rk = next((k for k in v.allotment if k.startswith(f"{RETAIL}:")), None)
        o = v.allotment.get(rk) if rk else None
        idx.append(
            f"| [{v.symbol}]({v.symbol}.md) | {ipo.close_date if ipo else '—'} | "
            f"{v.score:.0f} | {v.grade} | {_pct(v.expected_listing_gain_pct)} | "
            f"{o.p_single:.0%} |" if o else
            f"| [{v.symbol}]({v.symbol}.md) | {ipo.close_date if ipo else '—'} | "
            f"{v.score:.0f} | {v.grade} | {_pct(v.expected_listing_gain_pct)} | — |")

    if allocation and allocation.applications:
        idx += ["", "## Suggested applications", "",
                "| PAN | IPO | Category | Lots | Capital | P(allot) | E[profit] |",
                "|---|---|---|---|---|---|---|"]
        for a in allocation.applications:
            idx.append(f"| #{a.pan_index} | {a.symbol} | {a.category} | {a.lots} | "
                       f"₹{a.capital:,.0f} | {a.p_allot:.0%} | "
                       f"₹{a.expected_profit:,.0f} |")
        idx += ["", f"Capital deployed **₹{allocation.capital_used:,.0f}** of "
                    f"₹{allocation.capital_available:,.0f}, expected profit "
                    f"**₹{allocation.expected_profit:,.0f}** "
                    f"({allocation.expected_roi_pct:.2f}% on blocked funds)."]
        if getattr(allocation, "timeline", None):
            idx += ["", "### Cash flow projection", "",
                    "_ASBA blocks capital from the close date until roughly "
                    "T+3 business days, so what matters is how much is free on "
                    "any given day._", "",
                    "| Date | Blocked | Released | Outstanding | Free to deploy | Movements |",
                    "|---|---|---|---|---|---|"]
            for ev in allocation.timeline:
                moves = " ".join([f"+{x}" for x in ev.symbols_in]
                                 + [f"-{x}" for x in ev.symbols_out]) or "—"
                idx.append(
                    f"| {ev.day.strftime('%a %d %b')} | "
                    f"{('₹%s' % format(ev.committed, ',.0f')) if ev.committed else '—'} | "
                    f"{('₹%s' % format(ev.released, ',.0f')) if ev.released else '—'} | "
                    f"₹{ev.outstanding:,.0f} | ₹{ev.headroom:,.0f} | {moves} |")
        if allocation.skipped:
            idx += ["", "### Skipped", ""]
            for s in allocation.skipped:
                idx.append(f"- **{s['symbol']}** — {s['why']}")

    p = REPORT_DIR / "index.md"
    p.write_text("\n".join(idx), encoding="utf-8")
    written.append(p)

    snap = REPORT_DIR / "latest.json"
    snap.write_text(json.dumps([v.to_dict() for v in verdicts], indent=2,
                               default=str), encoding="utf-8")
    written.append(snap)
    return written
