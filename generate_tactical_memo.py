#!/usr/bin/env python3
"""
Tactical action memo - a dense 1-2 page execution sheet.

The Friday briefing is a document you read. This is a sheet you act from, and
the difference drives every design choice here: no history, no narrative, no
analyst prose. Only issues you can still bid on, only the numbers that change
what you do in the next few hours, and the vetoes that stop you putting
capital somewhere it should not go.

    python3 generate_tactical_memo.py
    python3 run.py memo                       # same thing, wired into the CLI
    python3 generate_tactical_memo.py --hours 48 --capital 750000

Rendering reuses the cascading chain from generate_briefing.py
(WeasyPrint -> headless Chrome -> wkhtmltopdf -> ReportLab), so it needs no
new dependency and cannot hit a PEP 668 wall.

The two-page limit is enforced, not hoped for: the memo is rendered, its page
count measured, and if it overflows it is re-rendered at a tighter scale until
it fits or the scale floor is reached.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_briefing import (Block, blocks_to_html, render_pdf,  # noqa: E402
                               collect_health)

log = logging.getLogger("memo")

DB_PATH = ROOT / "data" / "ipo_radar.db"
DEFAULT_HORIZON_HOURS = 96
MAX_PAGES = 2
SCALE_STEPS = (1.00, 0.94, 0.88, 0.82, 0.76)   # shrink until it fits


# =====================================================================
# Data
# =====================================================================
@dataclass
class Candidate:
    symbol: str
    name: str
    status: str
    close_date: date | None
    cap_price: float | None
    lot_size: int | None
    issue_size_cr: float | None
    fresh_cr: float | None
    ofs_cr: float | None
    score: float = 0.0
    grade: str = ""
    gain_pct: float | None = None
    verdict: dict[str, Any] = field(default_factory=dict)

    @property
    def valuation(self) -> dict[str, Any]:
        return self.verdict.get("valuation") or {}

    @property
    def veto(self) -> dict[str, Any]:
        return self.verdict.get("veto") or {}

    @property
    def anchors(self) -> dict[str, Any]:
        return self.verdict.get("anchors") or {}

    @property
    def tday(self) -> dict[str, Any]:
        return self.verdict.get("tday") or {}

    @property
    def days_to_close(self) -> int | None:
        if not self.close_date:
            return None
        from ipo_radar.util import today_ist
        return (self.close_date - today_ist()).days

    @property
    def ofs_pct(self) -> float | None:
        f, o = self.fresh_cr, self.ofs_cr
        if isinstance(f, (int, float)) and isinstance(o, (int, float)) and (f + o) > 0:
            return o / (f + o) * 100.0
        return None


def load_candidates(db_path: Path, horizon_hours: int) -> tuple[list[Candidate], str]:
    """Live issues whose bidding is still open inside the horizon."""
    from ipo_radar.util import today_ist
    today = today_ist()
    # Close dates are day-granular, so the hour horizon becomes whole days by
    # FLOOR division: 96h -> 4 days (today plus the next four close dates),
    # 24h -> tomorrow, and anything under a day -> today only. Rounding up
    # instead made "--hours 1" silently include tomorrow's issues.
    horizon = today + timedelta(days=horizon_hours // 24)
    out: list[Candidate] = []
    note = ""
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM ipos WHERE status IN ('Active','Forthcoming')"
        ).fetchall()
        for r in rows:
            try:
                close = date.fromisoformat(r["close_date"]) if r["close_date"] else None
            except (TypeError, ValueError):
                close = None
            # Closed issues are gone; anything past the horizon is not this
            # memo's problem - that is what the weekly briefing is for.
            if close is None or close < today or close > horizon:
                continue
            vr = conn.execute(
                "SELECT payload FROM verdicts WHERE symbol=? ORDER BY ts DESC LIMIT 1",
                (r["symbol"],)).fetchone()
            payload: dict[str, Any] = {}
            if vr:
                try:
                    payload = json.loads(vr["payload"])
                except Exception:
                    payload = {}
            out.append(Candidate(
                symbol=r["symbol"], name=r["name"] or r["symbol"],
                status=r["status"] or "", close_date=close,
                cap_price=r["price_high"] or r["price_low"],
                lot_size=r["lot_size"], issue_size_cr=r["issue_size_cr"],
                fresh_cr=r["fresh_issue_cr"], ofs_cr=r["ofs_cr"],
                score=float(payload.get("score") or 0.0),
                grade=str(payload.get("grade") or ""),
                gain_pct=payload.get("expected_listing_gain_pct"),
                verdict=payload))
    except Exception as exc:
        note = f"database read failed: {type(exc).__name__}: {exc}"
        log.warning(note)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    out.sort(key=lambda c: (c.close_date or date.max, -c.score))
    return out, note


def build_allocation(cands: Sequence[Candidate], capital: float, pans: int,
                     objective: str) -> tuple[Any, str]:
    """
    Re-run the live optimiser rather than reading a stored plan.

    The allocation is a function of capital, PAN count and objective - all of
    which the operator can override on this invocation - so a cached plan
    would silently answer a different question than the one being asked.
    """
    try:
        from ipo_radar.analytics.portfolio import PortfolioOptimizer
        from ipo_radar.config import Settings
        from ipo_radar.models import AllotmentOdds, IPO
    except Exception as exc:
        return None, f"optimiser unavailable: {exc}"

    s = Settings.load()
    payload = []
    for c in cands:
        odds = {}
        for key, raw in (c.verdict.get("allotment") or {}).items():
            try:
                odds[key] = AllotmentOdds(**raw)
            except Exception:
                continue
        if not odds:
            continue
        payload.append({
            "ipo": IPO(symbol=c.symbol, name=c.name, status=c.status,
                       close_date=c.close_date.isoformat() if c.close_date else None,
                       price_low=c.cap_price, price_high=c.cap_price,
                       lot_size=c.lot_size, issue_size_cr=c.issue_size_cr,
                       fresh_issue_cr=c.fresh_cr, ofs_cr=c.ofs_cr),
            "verdict": None, "allotment": odds,
            "expected_gain_pct": c.gain_pct, "score": c.score,
            "veto": c.veto})
    if not payload:
        return None, "no issue carried usable allotment odds"
    try:
        opt = PortfolioOptimizer(
            capital, pans, objective=objective,
            min_score=s.portfolio.min_score,
            lookahead_days=s.portfolio.lookahead_days,
            settlement_days=s.portfolio.asba_settlement_days,
            reserve_min_score=s.portfolio.reserve_min_score)
        return opt.optimise(payload), ""
    except Exception as exc:
        return None, f"allocation failed: {type(exc).__name__}: {exc}"


# =====================================================================
# Formatting helpers
# =====================================================================
def rs(v: Any, dp: int = 0) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    return f"₹{v:,.{dp}f}"


def pct(v: Any, dp: int = 1, sign: bool = False) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:+.{dp}f}%" if sign else f"{v:.{dp}f}%"


def num(v: Any, dp: int = 2, suffix: str = "") -> str:
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:,.{dp}f}{suffix}"


def short_day(d: Any) -> str:
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d[:10])
        except ValueError:
            return d
    return d.strftime("%a %d %b") if isinstance(d, date) else "—"


def price_rule(category: str, cap_price: float | None) -> str:
    """
    Cut-off is a retail/employee privilege. An NII bid must name a price, and
    naming anything below the cap risks exclusion if the issue prices there -
    which oversubscribed issues almost always do.
    """
    if category.upper().startswith("RETAIL"):
        return "CUT-OFF"
    return f"CAP {rs(cap_price)}" if cap_price else "CAP price"


# =====================================================================
# Page 1 - the execution desk
# =====================================================================
def capital_header(alloc: Any, capital: float, cands: Sequence[Candidate],
                   objective: str) -> list[Block]:
    from ipo_radar.util import today_ist
    today = today_ist()
    committed_today = outstanding_today = 0.0
    if alloc is not None:
        committed_today = sum(
            a.capital for a in alloc.applications
            if (a.block_from or "") == today.isoformat())
        for ev in (getattr(alloc, "timeline", None) or []):
            if ev.day == today:
                outstanding_today = ev.outstanding
                break
        else:
            outstanding_today = committed_today
    reserve = max(0.0, capital - outstanding_today)
    peak = getattr(alloc, "peak_blocked", 0.0) if alloc else 0.0
    profit = getattr(alloc, "expected_profit", 0.0) if alloc else 0.0
    n_today = sum(1 for c in cands if c.close_date == today)

    cells = [
        ("Total budget", rs(capital), ""),
        ("Committed today", rs(committed_today), "warn" if committed_today else ""),
        ("Liquid reserve", rs(reserve), "ok" if reserve > capital * 0.2 else "warn"),
        ("Peak block", rs(peak), ""),
        ("Expected profit", rs(profit), "ok" if profit > 0 else ""),
    ]
    tiles = "".join(
        f'<div class="tile {cls}"><div class="tl">{label}</div>'
        f'<div class="tv">{value}</div></div>' for label, value, cls in cells)
    return [Block("raw", (
        f'<div class="hdr">'
        f'<div class="hl"><span class="ht">TACTICAL ACTION MEMO</span>'
        f'<span class="hs">{datetime.now():%d %b %Y · %H:%M} IST · '
        f'{len(cands)} live issue{"s" if len(cands) != 1 else ""} · '
        f'{n_today} closing today · {objective.replace("_", " ")}</span></div>'
        f'</div><div class="tiles">{tiles}</div>'))]


def act_now_table(alloc: Any, cands: Sequence[Candidate]) -> list[Block]:
    """Only what can still be acted on today or tomorrow."""
    from ipo_radar.util import today_ist
    today = today_ist()
    tomorrow = today + timedelta(days=1)
    by_symbol = {c.symbol: c for c in cands}
    rows: list[list[str]] = []

    apps = sorted(getattr(alloc, "applications", []) or [],
                  key=lambda a: (a.close_date or "", -a.expected_profit))
    for a in apps:
        c = by_symbol.get(a.symbol)
        if not c or c.close_date not in (today, tomorrow):
            continue
        clock = (c.tday.get("clock") or {})
        cut = clock.get("cutoff_at") or ""
        cut_txt = (f"{cut[11:16]} {short_day(cut[:10]).split()[0]}"
                   if len(cut) >= 16 else "17:00")
        if c.close_date == today:
            cut_txt = f"{cut[11:16] if len(cut) >= 16 else '17:00'} TODAY"
        rows.append([
            f"**{a.symbol}**", cut_txt, a.category, str(a.lots),
            price_rule(a.category, c.cap_price),
            pct(a.p_allot * 100.0, 0),
            pct(c.gain_pct, 1, sign=True),
            rs(a.capital), rs(a.expected_profit)])

    out = [Block("h2", "ACT NOW — bids that must be placed today or tomorrow")]
    if not rows:
        out.append(Block("quote", "**No action required in this window.** "
                                  "Nothing open clears the score, gain and "
                                  "capital-timing filters."))
        return out
    out.append(Block("table",
                     header=["Symbol", "Cut-off", "Category", "Lots", "Price rule",
                             "Odds", "Exp. gain", "Capital", "Net profit"],
                     rows=rows))
    total_cap = sum(a.capital for a in apps
                    if by_symbol.get(a.symbol) and
                    by_symbol[a.symbol].close_date in (today, tomorrow))
    total_pf = sum(a.expected_profit for a in apps
                   if by_symbol.get(a.symbol) and
                   by_symbol[a.symbol].close_date in (today, tomorrow))
    out.append(Block("p",
        f"**{len(rows)} application(s), {rs(total_cap)} blocked, "
        f"{rs(total_pf)} expected.** Retail bids go at CUT-OFF so they stay "
        f"valid if the issue prices at the cap; NII bids must name the cap "
        f"price or risk exclusion. A placed bid with an unaccepted UPI mandate "
        f"is not an application."))
    return out


def roadmap_blocks(alloc: Any) -> list[Block]:
    """When blocked cash comes back, and what it can then fund."""
    events = getattr(alloc, "timeline", None) or []
    out = [Block("h2", "CAPITAL DEPLOYMENT ROADMAP")]
    if not events:
        out.append(Block("p", "_No capital scheduled._"))
        return out
    rows = []
    for ev in events:
        moves = " ".join([f"+{s}" for s in ev.symbols_in]
                         + [f"−{s}" for s in ev.symbols_out]) or "—"
        rows.append([short_day(ev.day),
                     rs(ev.committed) if ev.committed else "—",
                     rs(ev.released) if ev.released else "—",
                     rs(ev.outstanding), rs(ev.headroom), moves])
    out.append(Block("table",
                     header=["Date", "Blocked", "Released", "Outstanding",
                             "Free to deploy", "Movements"], rows=rows))
    out.append(Block("p", "_Funds released on day T are only deployable from "
                          "T+1: sponsor banks push ASBA revocations in an "
                          "evening batch, hours after the 17:00 mandate "
                          "cut-off._"))
    return out


# =====================================================================
# Page 2 - forensic due diligence
# =====================================================================
def veto_blocks(cands: Sequence[Candidate]) -> list[Block]:
    out = [Block("pagebreak"),
           Block("h2", "HARD VETO — capital must not be allocated here")]
    entries: list[str] = []
    for c in cands:
        v, val = c.veto, c.valuation
        reasons: list[str] = []
        if v.get("value_trap"):
            ke, ronw = v.get("cost_of_equity_pct"), v.get("ronw_pct")
            reasons.append(
                f"**Value Trap** — RoNW {num(ronw, 1, '%')} is below its "
                f"{num(ke, 1, '%')} cost of equity "
                f"(spread {num(v.get('eva_spread_pct'), 1, 'pp')}). "
                + ("Flip exception met, so it is tradeable for the listing pop "
                   "ONLY — sell on listing day."
                   if v.get("speculative_flip") else
                   f"Score capped at {num(v.get('ceiling'), 0)}; not an APPLY "
                   f"for long-term capital."))
        if val.get("accruals_triggered"):
            reasons.append(
                f"**Aggressive Accruals** — {val.get('cash_conversion_status', '')}. "
                f"Reported profit is not converting into operating cash.")
        if reasons:
            entries.append(f"**{c.symbol}** ({c.grade}, score {c.score:.0f}) — "
                           + " ".join(reasons))
    if entries:
        out.append(Block("raw", '<div class="veto">'
                         + "".join(f"<p>{__import__('generate_briefing').inline_html(e)}</p>"
                                   for e in entries) + "</div>"))
    else:
        out.append(Block("p", "**No hard vetoes among the live issues.** No "
                              "Value Trap and no accruals warning in scope."))
    return out


def scorecard_blocks(cands: Sequence[Candidate]) -> list[Block]:
    rows = []
    for c in cands:
        v = c.valuation
        pe, bench = v.get("pe"), v.get("benchmark_pe")
        rel = f"{pe / bench:.2f}x" if (pe and bench) else "—"
        ev, evb = v.get("ev_ebitda"), v.get("ev_ebitda_benchmark")
        ev_txt = (f"{num(ev, 1, 'x')} / {num(evb, 0, 'x')}"
                  if ev else ("n/a — lender" if v.get("capital_intensity") == 0
                              else "—"))
        rows.append([
            f"**{c.symbol}**", f"{c.score:.0f}", c.grade[:13],
            num(pe, 1, "x"), num(bench, 0, "x"), rel, ev_txt,
            num(v.get("ronw_pct"), 1, "%"),
            num(v.get("cost_of_equity_pct"), 1, "%"),
            pct(c.ofs_pct, 0) if c.ofs_pct is not None else "—"])
    out = [Block("h2", "FUNDAMENTAL SCORECARD")]
    if not rows:
        out.append(Block("p", "_No valuation data in scope._"))
        return out
    out.append(Block("table",
        header=["Symbol", "Score", "Grade", "P/E", "Bench", "vs",
                "EV/EBITDA", "RoNW", "Ke", "% OFS"], rows=rows))
    out.append(Block("p", "_EV/EBITDA carries more of the valuation score as "
                          "capital intensity rises; it is meaningless for "
                          "lenders, whose borrowings are raw material. A high "
                          "% OFS means the money goes to selling holders, not "
                          "into the business._"))
    return out


def anchor_blocks(cands: Sequence[Candidate]) -> list[Block]:
    rows = []
    for c in cands:
        a = c.anchors
        d = a.get("day30") or {}
        if not a.get("total"):
            continue
        impact = d.get("day30_impact_pct")
        thr = d.get("day30_threshold_pct")
        flag = "⚠ DUMP RISK" if d.get("day30_warning") else a.get("overhang_risk", "—")
        rows.append([
            f"**{c.symbol}**", str(a.get("total", 0)),
            pct((a.get("compounder_ratio") or 0) * 100, 0),
            pct((a.get("flipper_ratio") or 0) * 100, 0),
            pct(impact, 1) if impact is not None else "—",
            pct(thr, 0) if thr is not None else "—",
            flag])
    out = [Block("h2", "ANCHOR LOCK-IN — day-30 supply risk")]
    if not rows:
        out.append(Block("p", "_No anchor allocation parsed for the live "
                              "issues; the anchor letter is published on the "
                              "eve of listing._"))
        return out
    out.append(Block("table",
        header=["Symbol", "Names", "Patient", "Flippers", "Day-30 float +",
                "Trigger", "Verdict"], rows=rows))
    out.append(Block("p", "_SEBI releases 50% of the anchor allocation at 30 "
                          "days and the rest at 90. Supply only matters if the "
                          "holders will sell, so the warning needs both a large "
                          "float expansion and a flipper-heavy book._"))
    return out


def footer_blocks(alloc: Any, notes: Sequence[str]) -> list[Block]:
    skipped = getattr(alloc, "skipped", None) or []
    out: list[Block] = []
    if skipped:
        out.append(Block("h3", "Excluded from the plan"))
        out.append(Block("ul", items=[f"**{s['symbol']}** — {s['why']}"
                                      for s in skipped[:8]]))
    if notes:
        out.append(Block("ul", items=[f"_{n}_" for n in notes]))
    out.append(Block("raw",
        '<div class="fine">Analysis only, not investment advice. Grey-market '
        'premium is unofficial and easily manipulated; projections before the '
        'final day are wide by construction. Verify every cut-off against your '
        'broker before bidding.</div>'))
    return out


# =====================================================================
# Styling - dense by design
# =====================================================================
def memo_css(scale: float = 1.0) -> str:
    f = lambda pt: f"{pt * scale:.2f}pt"                      # noqa: E731
    return f"""
@page {{ size: A4; margin: 9mm 9mm 8mm 9mm;
  @bottom-right {{ content: counter(page) " of " counter(pages);
    font-family: Helvetica, Arial, sans-serif; font-size: {f(6.5)}; color: #98a0ac; }} }}
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: {f(7.6)}; line-height: 1.28; color: #16202e; margin: 0; }}
.hdr {{ border-bottom: 2px solid #16202e; padding-bottom: 3pt; margin-bottom: 5pt; }}
.ht {{ font-size: {f(15)}; font-weight: 700; letter-spacing: -0.4pt; }}
.hs {{ float: right; font-size: {f(7)}; color: #6b7480; padding-top: {f(6)}; }}
.tiles {{ display: flex; gap: 4pt; margin-bottom: 6pt; }}
.tile {{ flex: 1; border: 0.6pt solid #d6dce5; border-radius: 2pt;
  padding: 3pt 5pt; background: #fafbfd; }}
.tile.ok {{ border-left: 2.5pt solid #1d8a44; }}
.tile.warn {{ border-left: 2.5pt solid #c8871a; }}
.tl {{ font-size: {f(6.2)}; text-transform: uppercase; letter-spacing: 0.3pt;
  color: #78818f; }}
.tv {{ font-size: {f(11)}; font-weight: 700; color: #0f1826; }}
h2 {{ font-size: {f(8.6)}; margin: 7pt 0 3pt; text-transform: uppercase;
  letter-spacing: 0.4pt; color: #0f1826;
  border-bottom: 0.8pt solid #c9d2de; padding-bottom: 1.5pt; }}
h3 {{ font-size: {f(7.6)}; margin: 5pt 0 2pt; color: #3a465c; }}
p {{ margin: 0 0 3pt; }}
ul {{ margin: 0 0 4pt; padding-left: 11pt; }}
li {{ margin: 0 0 1pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 2pt 0 4pt;
  font-size: {f(7.0)}; }}
thead {{ display: table-header-group; }}
tr {{ page-break-inside: avoid; }}
th {{ background: #eef2f7; text-align: left; font-weight: 600; color: #1e2a3d;
  border-bottom: 0.9pt solid #b9c4d2; padding: 2pt 3.5pt; white-space: nowrap; }}
td {{ border-bottom: 0.4pt solid #e6eaf0; padding: 1.8pt 3.5pt;
  vertical-align: top; }}
tbody tr:nth-child(even) td {{ background: #fafbfd; }}
blockquote {{ margin: 3pt 0; padding: 4pt 7pt; background: #f2f6fb;
  border-left: 2.5pt solid #3f6fae; }}
.veto {{ border: 1pt solid #c0392b; border-left: 3.5pt solid #c0392b;
  background: #fdf4f3; padding: 5pt 8pt; margin: 3pt 0 5pt; }}
.veto p {{ margin: 0 0 3pt; font-size: {f(7.4)}; }}
.fine {{ font-size: {f(6.2)}; color: #98a0ac; border-top: 0.5pt solid #e6eaf0;
  padding-top: 3pt; margin-top: 5pt; }}
.pagebreak {{ page-break-before: always; }}
strong {{ color: #0f1826; }}
em {{ color: #5a6472; }}
"""


def build_html(blocks: Sequence[Block], title: str, scale: float) -> str:
    import html as _html
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{_html.escape(title)}</title>"
            f"<style>{memo_css(scale)}</style></head>"
            f"<body>{blocks_to_html(blocks)}</body></html>")


def page_count(pdf: Path) -> int | None:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf)).pages)
    except Exception:
        return None


# =====================================================================
# Assembly
# =====================================================================
def build_memo(db_path: Path, out_dir: Path, hours: int, capital: float | None,
               pans: int | None, objective: str | None, engine: str,
               keep_html: bool, max_pages: int) -> int:
    try:
        from ipo_radar.config import Settings
        s = Settings.load()
        capital = capital if capital is not None else s.portfolio.total_capital
        pans = pans if pans is not None else s.portfolio.num_pans
        objective = objective or s.portfolio.objective
    except Exception as exc:
        log.warning("settings unavailable (%s) - using defaults", exc)
        capital = capital if capital is not None else 500_000.0
        pans = pans or 1
        objective = objective or "maximize_roi"

    notes: list[str] = []
    cands, note = load_candidates(db_path, hours)
    if note:
        notes.append(note)
    if not cands:
        print(f"\n  No issues open or closing within {hours}h. "
              f"Nothing to act on.\n", file=sys.stderr)
        return 2

    alloc, alloc_note = build_allocation(cands, capital, pans, objective)
    if alloc_note:
        notes.append(alloc_note)

    blocks: list[Block] = []
    for part in (capital_header(alloc, capital, cands, objective),
                 act_now_table(alloc, cands),
                 roadmap_blocks(alloc),
                 veto_blocks(cands),
                 scorecard_blocks(cands),
                 anchor_blocks(cands),
                 footer_blocks(alloc, notes)):
        try:
            blocks.extend(part)
        except Exception as exc:            # one bad section must not sink it
            log.warning("section failed: %s", exc)
            notes.append(f"a section could not be rendered: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    stem = f"Tactical_Action_Memo_{today:%Y-%m-%d}"
    pdf_path = out_dir / f"{stem}.pdf"
    html_path = out_dir / f"{stem}.html"
    title = f"Tactical Action Memo {today:%d %b %Y}"

    used = None
    pages = None
    html_text = ""
    # Enforce the page budget rather than hoping the content fits.
    for scale in SCALE_STEPS:
        html_text = build_html(blocks, title, scale)
        used = render_pdf(blocks, html_text, pdf_path, title, engine)
        if not used:
            break
        pages = page_count(pdf_path)
        if pages is None or pages <= max_pages:
            if scale < 1.0:
                notes.append(f"typeset at {scale:.0%} to hold {max_pages} pages")
            break
        log.info("%d pages at scale %.2f - tightening", pages, scale)

    try:
        html_path.write_text(html_text, encoding="utf-8")
    except Exception as exc:
        log.warning("could not write HTML: %s", exc)

    print()
    if used:
        size = pdf_path.stat().st_size / 1024
        fit = (f"{pages} page{'s' if pages != 1 else ''}"
               if pages else "page count unknown")
        print(f"  Memo     : {pdf_path}")
        print(f"  Engine   : {used}   ({size:,.0f} KB, {fit})")
    else:
        print(f"  No PDF engine available - wrote HTML instead:\n  {html_path}")
    n_act = sum(1 for b in blocks if b.kind == "table"
                and b.header and b.header[0] == "Symbol"
                and "Cut-off" in b.header)
    vetoed = sum(1 for c in cands
                 if c.veto.get("value_trap")
                 or (c.valuation or {}).get("accruals_triggered"))
    print(f"  Scope    : {len(cands)} live issue(s) within {hours}h, "
          f"{vetoed} under hard veto")
    if alloc is not None:
        print(f"  Capital  : {rs(getattr(alloc, 'peak_blocked', 0))} peak of "
              f"{rs(capital)}, {rs(getattr(alloc, 'expected_profit', 0))} expected")
    for n in notes:
        print(f"  Note     : {n}")
    if not keep_html and used and html_path.exists():
        try:
            html_path.unlink()
        except Exception:
            pass
    print()
    return 0 if used else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="generate_tactical_memo",
        description="Dense 1-2 page execution memo for IPOs closing inside the "
                    "next few days.")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "briefings")
    ap.add_argument("--hours", type=int, default=DEFAULT_HORIZON_HOURS,
                    help="close-date horizon (default 96)")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--pans", type=int, default=None)
    ap.add_argument("--objective", default=None,
                    choices=("maximize_roi", "maximize_absolute_profit"))
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES)
    ap.add_argument("--keep-html", action="store_true")
    ap.add_argument("--log", default="WARNING")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.WARNING),
                        format="%(levelname)-7s %(name)s  %(message)s")
    try:
        return build_memo(args.db, args.output_dir, max(1, args.hours),
                          args.capital, args.pans, args.objective, args.engine,
                          args.keep_html, max(1, args.max_pages))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger("memo").exception("memo failed: %s", exc)
        print(f"\n  Memo failed: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
