"""
Valuation and financial-quality scoring from RHP-derived numbers.

The RHP leaves every multiple as "[<bullet>]" because it predates the price band,
so documents.py computes P/E and P/B from EPS/NAV against the actual band.
This module turns those into comparable 0-100 scores.
"""
from __future__ import annotations

import math
from typing import Any

import re

from ..models import IPO
from ..util import clamp

# Rough fair-P/E anchors by broad sector. Deliberately coarse - this is a
# sanity rail, not a DCF. Overridden whenever the RHP lists real peers.
SECTOR_PE = {
    "bank": 14.0, "nbfc": 18.0, "finance": 18.0, "insurance": 30.0,
    "it": 26.0, "software": 30.0, "tech": 34.0, "internet": 45.0,
    "pharma": 30.0, "healthcare": 34.0, "hospital": 40.0,
    "chemical": 24.0, "specialty chemical": 28.0,
    "auto": 22.0, "ancillary": 20.0, "capital goods": 32.0,
    "engineering": 28.0, "infra": 20.0, "construction": 18.0,
    "realty": 24.0, "cement": 22.0, "metal": 12.0, "steel": 12.0,
    "fmcg": 45.0, "retail": 45.0, "consumer": 42.0,
    "textile": 16.0, "logistics": 28.0, "power": 18.0, "energy": 16.0,
    "defence": 45.0, "renewable": 30.0, "hotel": 35.0, "media": 22.0,
}
DEFAULT_FAIR_PE = 24.0

# Fair EV/EBITDA anchors. EV/EBITDA is capital-structure neutral and ignores
# the D&A that swamps reported earnings in asset-heavy businesses, which is
# precisely why it should outrank P/E for them.
SECTOR_EV_EBITDA = {
    "power": 10.0, "utility": 10.0, "energy": 8.0, "oil": 7.5,
    "telecom": 8.5, "infra": 9.5, "construction": 9.0, "road": 9.0,
    "port": 12.0, "shipping": 7.0, "logistics": 14.0, "airline": 7.0,
    "cement": 13.0, "metal": 7.0, "steel": 6.5, "mining": 6.0,
    "realty": 12.0, "hotel": 18.0, "hospital": 21.0, "healthcare": 19.0,
    "renewable": 11.0, "defence": 24.0, "capital goods": 18.0,
    "engineering": 15.0, "auto": 13.0, "ancillary": 11.0,
    "chemical": 14.0, "specialty chemical": 16.0, "pharma": 18.0,
    "textile": 9.0, "fmcg": 26.0, "consumer": 24.0, "retail": 22.0,
    "it": 18.0, "software": 20.0, "tech": 22.0, "internet": 25.0,
    "bank": 0.0, "nbfc": 0.0, "finance": 0.0, "insurance": 0.0,
}
DEFAULT_FAIR_EV_EBITDA = 13.0

# Issuer names and RHP prose rarely use the tidy sector word. These map the
# language companies actually use onto the benchmark tables above.
SECTOR_SYNONYMS = {
    "electrical": "capital goods", "electricals": "capital goods",
    "cable": "capital goods", "cables": "capital goods",
    "wire": "capital goods", "wires": "capital goods",
    "transformer": "capital goods", "switchgear": "capital goods",
    "boiler": "capital goods", "steam": "capital goods",
    "solar": "renewable", "wind": "renewable",
    "transmission": "power", "grid": "power", "electricity": "power",
    "developer": "realty", "developers": "realty", "housing": "realty",
    "facade": "construction", "glazing": "construction",
    "fenestration": "construction", "epc": "construction",
    "infrastructure": "infra", "projects": "construction",
    "plastech": "chemical", "plastic": "chemical", "polymer": "chemical",
    "chemicals": "chemical", "speciality chemical": "specialty chemical",
    "payment": "tech", "payments": "tech", "fintech": "tech",
    "identity": "tech", "saas": "software", "platform": "internet",
    "rental": "internet", "rentals": "internet", "subscription": "internet",
    "asset reconstruction": "nbfc", "securitisation": "nbfc",
    "pharmaceutical": "pharma", "pharmaceuticals": "pharma",
    "hospitals": "hospital", "diagnostics": "healthcare",
    "engineering": "engineering", "forging": "engineering",
    "castings": "engineering", "fabrication": "engineering",
    "glass": "construction", "glasswall": "construction",
    "aluminium": "metal", "steels": "steel", "ispat": "steel",
    "furniture": "retail", "appliances": "retail",
    "electronics": "capital goods", "motors": "capital goods",
    "pipes": "capital goods", "tubes": "capital goods",
}


def _canonical_sector(hay: str) -> str:
    """Longest synonym wins, so 'specialty chemical' beats 'chemical'."""
    best, blen = "", 0
    for word, canon in SECTOR_SYNONYMS.items():
        if re.search(rf"\b{re.escape(word)}\b", hay) and len(word) > blen:
            best, blen = canon, len(word)
    return best

# 0 = asset-light (earnings are meaningful), 1 = capital-intensive (D&A and
# leverage dominate, so enterprise-level multiples carry the signal).
CAPITAL_INTENSITY = {
    "power": 0.95, "utility": 0.95, "telecom": 0.95, "infra": 0.90,
    "construction": 0.80, "road": 0.90, "port": 0.90, "shipping": 0.90,
    "airline": 0.90, "cement": 0.90, "metal": 0.90, "steel": 0.90,
    "mining": 0.85, "realty": 0.80, "renewable": 0.95, "hotel": 0.80,
    "hospital": 0.75, "oil": 0.90, "energy": 0.85, "logistics": 0.65,
    "auto": 0.65, "ancillary": 0.60, "chemical": 0.70,
    "specialty chemical": 0.65, "engineering": 0.55, "capital goods": 0.55,
    "textile": 0.65, "defence": 0.50, "pharma": 0.50, "healthcare": 0.55,
    "fmcg": 0.30, "consumer": 0.30, "retail": 0.35, "it": 0.15,
    "software": 0.10, "tech": 0.15, "internet": 0.15,
    "bank": 0.0, "nbfc": 0.0, "finance": 0.0, "insurance": 0.0,
}
# ------------------------------------------------------- accruals gate
# Reported profit is an opinion; cash is a fact. A company can book revenue it
# has not collected, and an issuer polishing its numbers before an IPO does it
# by letting receivables and inventory balloon - which shows up as profit
# rising while operating cash flow does not follow.
MIN_CASH_CONVERSION = 0.35
# Above this the extraction is wrong, not the company. CFO exceeding EBITDA
# several times over means a mis-parsed EBITDA, and a confident warning built
# on a broken denominator is worse than no warning at all.
MAX_PLAUSIBLE_CONVERSION = 10.0
ACCRUALS_FLAG = ("Aggressive Accruals Warning: Negative/weak operating cash "
                 "flow (CFO) despite reported accounting profit.")


def accruals_gate(cfo_cr: float | None, ebitda_cr: float | None,
                  pat_positive: bool, pat_growing: bool) -> dict[str, Any]:
    """
    Cash conversion test: CFO / EBITDA.

    Only fires when profit is positive AND growing, because that is the
    combination the warning is about - an issuer showing an improving P&L
    that the cash flow statement does not corroborate. A loss-maker with
    negative CFO is not an accruals problem, it is simply a loss-maker, and
    the EVA gate already handles it.
    """
    out: dict[str, Any] = {}
    if not isinstance(cfo_cr, (int, float)):
        out["cash_conversion_status"] = "CFO not disclosed in parsed sections"
        return out
    out["cfo_cr"] = round(float(cfo_cr), 1)

    if not (isinstance(ebitda_cr, (int, float)) and ebitda_cr > 0):
        # CFO alone still decides the negative-cash case
        if cfo_cr < 0 and pat_positive and pat_growing:
            out["cash_conversion_status"] = "negative CFO (EBITDA unavailable)"
            out["accruals_flag"] = ACCRUALS_FLAG
            out["accruals_triggered"] = True
        else:
            out["cash_conversion_status"] = "EBITDA unavailable"
        return out

    conv = float(cfo_cr) / float(ebitda_cr)
    out["ebitda_cr"] = round(float(ebitda_cr), 1)
    out["cash_conversion"] = round(conv, 2)

    # Negative CFO is decisive on its own and is settled BEFORE the
    # plausibility guard. Testing abs(conv) first exempted the worst cash
    # burners: a company burning cash against a small EBITDA produces a large
    # NEGATIVE ratio, whose absolute value trips the "extraction suspect"
    # branch and returns with no flag at all. A present EBITDA would then
    # suppress a warning that a missing one raises - precisely backwards.
    if cfo_cr < 0 and pat_positive and pat_growing:
        out["accruals_flag"] = ACCRUALS_FLAG
        out["accruals_triggered"] = True
        out["cash_conversion_status"] = (
            f"operating cash flow is negative ({conv:.2f}x EBITDA)")
        return out

    # One-sided now: only an implausibly HIGH ratio indicates a mis-parsed
    # denominator. A low or negative one is a finding, not a parsing failure.
    if conv > MAX_PLAUSIBLE_CONVERSION:
        out["cash_conversion_status"] = (
            f"implausible ratio ({conv:.1f}x) - extraction suspect, not assessed")
        return out

    weak = cfo_cr < 0 or conv < MIN_CASH_CONVERSION
    if weak and pat_positive and pat_growing:
        out["accruals_flag"] = ACCRUALS_FLAG
        out["accruals_triggered"] = True
        out["cash_conversion_status"] = (
            f"CFO converts only {conv:.0%} of EBITDA against a "
            f"{MIN_CASH_CONVERSION:.0%} floor"
            if cfo_cr >= 0 else "operating cash flow is negative")
    elif weak:
        out["cash_conversion_status"] = (
            f"weak conversion ({conv:.0%}) but profit is not both positive "
            f"and growing - no accruals inference drawn")
    else:
        out["cash_conversion_status"] = f"healthy ({conv:.0%} of EBITDA)"
    return out


# ---------------------------------------------------------------- EVA gate
# A company that earns less on equity than equity costs is destroying value,
# however fast it is growing. With a live risk-free rate from the G-Sec curve
# we can test that directly instead of guessing at "good" returns.
EQUITY_RISK_PREMIUM = 6.5          # percentage points, India long-run
FALLBACK_RISK_FREE = 6.75          # used only when no macro snapshot exists
HIGH_SPREAD_MARGIN = 6.0           # RoNW above Ke by this much = compounder

# Beta by capital intensity. Regulated, asset-heavy businesses have stable
# cash flows and trade at low beta; asset-light growth is far more volatile.
# So beta FALLS as capital intensity rises - the opposite of the intuition
# that heavy balance sheets are risky.
BETA_BASE, BETA_SLOPE = 1.30, 0.50     # beta = BASE - SLOPE * capital_intensity

# Explicit overrides where the capital-intensity proxy is a poor guide.
SECTOR_BETA = {
    "bank": 1.25, "nbfc": 1.40, "finance": 1.35, "insurance": 1.05,
    "power": 0.80, "utility": 0.75, "telecom": 0.95, "infra": 0.95,
    "realty": 1.35, "metal": 1.45, "steel": 1.50, "mining": 1.40,
    "internet": 1.40, "tech": 1.30, "software": 1.20, "it": 1.15,
    "fmcg": 0.70, "consumer": 0.85, "pharma": 0.80, "hospital": 0.85,
    "airline": 1.55, "shipping": 1.35, "hotel": 1.20, "defence": 1.10,
    # Cyclical heavies: high capital intensity but HIGH beta, the one place
    # the intensity proxy inverts. Listed explicitly so it is never used.
    "cement": 1.20, "construction": 1.25, "engineering": 1.10,
    "capital goods": 1.05, "auto": 1.15, "ancillary": 1.10,
    "chemical": 1.05, "specialty chemical": 1.00, "textile": 1.15,
    "logistics": 1.05, "port": 1.00, "mining": 1.40, "renewable": 1.05,
    "oil": 1.10, "energy": 1.05, "retail": 1.05, "healthcare": 0.85,
}


FINANCIAL_BETA = 1.30          # lenders: leveraged, rate-sensitive, cyclical


def proxy_beta(capital_intensity: float, sector_key: str = "",
               is_financial: bool = False) -> float:
    """
    Sector override where we have one, else the capital-intensity proxy.

    `is_financial` must be passed explicitly. classify_sector sets
    capital_intensity to 0.0 for lenders as a SENTINEL meaning "EBITDA and
    enterprise value do not apply here" - not as a measurement of asset
    lightness. Feeding that 0.0 into the proxy silently produced the
    asset-light beta of 1.30, which was right by coincidence and wrong by
    construction.
    """
    if sector_key and sector_key in SECTOR_BETA:
        return SECTOR_BETA[sector_key]
    if is_financial:
        return FINANCIAL_BETA
    return clamp(BETA_BASE - BETA_SLOPE * clamp(capital_intensity, 0.0, 1.0),
                 0.80, 1.30)


def cost_of_equity(risk_free_pct: float, beta: float,
                   erp: float = EQUITY_RISK_PREMIUM) -> float:
    """CAPM. Ke = Rf + beta x ERP, all in percentage points."""
    return risk_free_pct + beta * erp


def eva_gate(ronw_pct: float | None, ke_pct: float,
             post_issue_ronw_pct: float | None = None) -> dict[str, Any]:
    """
    Does this business earn more on equity than equity costs?

    The test uses REPORTED RoNW as the headline because that is what the RHP
    certifies, but a post-issue estimate is reported alongside it where the
    fresh-issue size is known: raising equity mechanically dilutes return on
    equity, and an issuer sitting just above its cost of capital before the
    offer can fall below it the moment the money lands.
    """
    out: dict[str, Any] = {"cost_of_equity_pct": round(ke_pct, 2)}
    if not isinstance(ronw_pct, (int, float)):
        out["eva_verdict"] = "unknown"
        out["eva_note"] = "RoNW not available - EVA gate not applied"
        return out

    spread = ronw_pct - ke_pct
    out["eva_spread_pct"] = round(spread, 2)
    if post_issue_ronw_pct is not None:
        out["post_issue_ronw_pct"] = round(post_issue_ronw_pct, 2)
        out["post_issue_eva_spread_pct"] = round(post_issue_ronw_pct - ke_pct, 2)

    if spread < 0:
        out["eva_verdict"] = "value destroyer"
        out["eva_flag"] = (f"Value Trap: RoNW ({ronw_pct:.1f}%) is below "
                           f"estimated Cost of Equity ({ke_pct:.1f}%)")
    elif spread >= HIGH_SPREAD_MARGIN:
        out["eva_verdict"] = "high-spread compounder"
        out["eva_flag"] = (f"High-Spread Compounder: RoNW ({ronw_pct:.1f}%) "
                           f"exceeds Cost of Equity ({ke_pct:.1f}%) by "
                           f"{spread:.1f}pp")
    else:
        out["eva_verdict"] = "marginal"
        out["eva_flag"] = (f"RoNW ({ronw_pct:.1f}%) only just clears Cost of "
                           f"Equity ({ke_pct:.1f}%) - thin economic spread")

    # A post-issue fall below Ke is its own warning even when reported RoNW passes
    pi = out.get("post_issue_eva_spread_pct")
    if spread >= 0 and isinstance(pi, (int, float)) and pi < 0:
        out["eva_dilution_warning"] = (
            f"Post-issue RoNW (~{post_issue_ronw_pct:.1f}%) falls below Cost of "
            f"Equity ({ke_pct:.1f}%) once the fresh issue dilutes equity")
    return out


# EBITDA and enterprise value are meaningless for lenders: borrowings are raw
# material, not financing. These are valued on P/B and RoE alone.
FINANCIALS = ("bank", "nbfc", "finance", "insurance", "asset reconstruction",
              "housing finance", "microfinance")


def _soft_cap(x: float, knee: float = 82.0, squeeze: float = 0.38) -> float:
    """
    Compress the top of a 0-100 score instead of clipping it.

    Additive factor models overshoot: a genuinely good issuer collects the
    RoNW bonus, the growth bonus, the consistency bonus and the EVA bonus and
    lands past 100. A hard clamp then ties every strong issuer at exactly
    100.0 - five of eleven live issues did precisely that - which throws away
    the ranking right where it matters most. Squeezing above the knee keeps
    the ordering intact and leaves headroom.
    """
    if x <= knee:
        return x
    return knee + (x - knee) * squeeze


def _cagr(first: float, last: float, years: int) -> float | None:
    if years <= 0 or first is None or last is None:
        return None
    if first <= 0:
        return None if last <= 0 else 100.0     # loss -> profit: cap the story
    return ((last / first) ** (1.0 / years) - 1.0) * 100.0


def _match_sector(hay: str, table: dict[str, float],
                  default: float) -> tuple[float, str]:
    """
    Word-boundary matching. Naive substring matching classified every issuer
    as "it", because the word "Limited" contains it - and "Kanohar Electricals
    Limited" is not a software company.
    """
    best, blen, key_hit = default, 0, ""
    for key, val in table.items():
        if re.search(rf"\b{re.escape(key)}\b", hay) and len(key) > blen:
            best, blen, key_hit = val, len(key), key
    return best, key_hit


def sector_fair_pe(sector: str | None, name: str = "", context: str = "") -> float:
    hay = f"{sector or ''} {name or ''} {context or ''}".lower()
    return _match_sector(hay, SECTOR_PE, DEFAULT_FAIR_PE)[0]


def classify_sector(ipo: IPO, context: str = "") -> dict[str, Any]:
    """
    Best-effort sector read from the declared sector, the company name, and a
    slice of the RHP's own business description. Coarse by design: it selects
    a benchmark and a capital-intensity weight, nothing finer.
    """
    name_hay = f"{ipo.sector or ''} {ipo.name or ''}".lower()
    full_hay = f"{name_hay} {context or ''}".lower()
    # The issuer's own NAME is the most reliable signal - Indian issuers say
    # what they do ("LCC Projects", "Kanohar Electricals", "Manika Plastech").
    # RHP prose is a fallback, because a passing mention of "rental" in the
    # objects section once reclassified a construction company as internet.
    canon = _canonical_sector(name_hay) or _canonical_sector(full_hay)
    hay = f"{canon} {full_hay}" if canon else full_hay
    fair_pe, pe_key = _match_sector(hay, SECTOR_PE, DEFAULT_FAIR_PE)
    fair_ev, ev_key = _match_sector(hay, SECTOR_EV_EBITDA, DEFAULT_FAIR_EV_EBITDA)
    intensity, ci_key = _match_sector(hay, CAPITAL_INTENSITY, 0.45)
    is_financial = any(f in hay for f in FINANCIALS)
    return {"sector_key": ev_key or pe_key or ci_key or "unclassified",
            "fair_pe": fair_pe, "fair_ev_ebitda": fair_ev,
            "capital_intensity": 0.0 if is_financial else intensity,
            "is_financial": is_financial}


def enterprise_value(ipo: IPO, m: dict[str, Any]) -> dict[str, Any]:
    """
    EV = market cap + net debt, built from RHP figures with every step gated.

    Share count is derived as PAT / EPS rather than parsed, because the
    capital-structure tables are far messier than the KPI table. Two
    independent gates catch a bad extraction before it can produce a
    confident-looking wrong multiple:

      1. RoNW reconciliation - EPS/NAV must equal the separately-extracted
         RoNW, since both reduce to PAT/net-worth.
      2. Scale sanity - the offer must land at 2-65% of implied market cap.
         An IPO is never 300% or 0.1% of the company.
    """
    out: dict[str, Any] = {}
    eps = m.get("eps_basic")
    pat = m.get("kpi_pat")
    ebitda = m.get("kpi_ebitda")
    nav = m.get("nav_pre_issue")
    cap = ipo.cap_price
    if not (cap and eps and pat):
        out["ev_status"] = "missing EPS or PAT"
        return out
    # A loss-maker has negative EPS and negative PAT, whose ratio is still a
    # positive share count. Requiring eps > 0 excluded exactly the issuers
    # EV/EBITDA exists to value - a company can generate real EBITDA while
    # reporting a loss after depreciation and interest - and left the
    # loss-making branch of the blend below permanently unreachable.
    if (pat > 0) != (eps > 0):
        out["ev_status"] = "EPS and PAT disagree in sign - extraction suspect"
        return out

    # KPI tables are quoted in Rs million unless they say otherwise
    scale = m.get("kpi_pat_scale")
    if scale is None:
        scale = 0.1
        out["ev_scale_assumed"] = "Rs million"
    pat_cr = pat * scale
    shares = (pat_cr * 1e7) / eps                    # PAT in rupees / EPS
    if shares <= 0:
        out["ev_status"] = "non-positive share count"
        return out

    # gate 1: RoNW reconciliation (only meaningful for a profitable issuer;
    # for a loss-maker both sides are negative and the ratio is unstable)
    ronw = m.get("ronw_pct")
    if nav and nav > 0 and isinstance(ronw, (int, float)) and ronw > 0 and eps > 0:
        implied = eps / nav * 100.0
        out["ronw_implied_pct"] = round(implied, 2)
        if abs(implied - ronw) / ronw > 0.25:
            out["ev_status"] = (f"RoNW reconciliation failed "
                                f"(implied {implied:.1f}% vs reported {ronw:.1f}%)")
            return out

    mcap_cr = cap * shares / 1e7
    # gate 2: an offer is a slice of the company, not a multiple of it
    if ipo.issue_size_cr and mcap_cr > 0:
        ratio = ipo.issue_size_cr / mcap_cr
        out["offer_pct_of_mcap"] = round(ratio * 100.0, 1)
        if not (0.02 <= ratio <= 0.65):
            out["ev_status"] = (f"scale check failed - offer is "
                                f"{ratio * 100:.0f}% of implied market cap")
            return out

    # Both gates have passed, so these are trustworthy. Recorded before the
    # remaining early returns so the EVA dilution check can still use net
    # worth when EV itself cannot be completed for want of debt data - but
    # NOT before the gates, or a rejected extraction leaks downstream. (An
    # earlier revision wrote net worth up front and published 62 Cr for an
    # issuer whose real figure is nearer 250 Cr, from a PAT the scale gate
    # went on to reject.)
    out["shares_outstanding"] = round(shares)
    if nav and nav > 0:
        out["net_worth_cr"] = round(shares * nav / 1e7, 1)

    # net debt: prefer the ratio (one number, one scale) over debt minus cash
    net_debt_cr = None
    nde = m.get("kpi_net_debt_eq")
    net_worth_cr = (shares * nav / 1e7) if (nav and nav > 0) else None
    if isinstance(nde, (int, float)) and net_worth_cr and abs(nde) < 15:
        net_debt_cr = nde * net_worth_cr
        out["net_debt_source"] = "net-debt-to-equity x net worth"
    else:
        borrow, cash = m.get("kpi_borrowings"), m.get("kpi_cash")
        if isinstance(borrow, (int, float)):
            b_scale = m.get("kpi_borrowings_scale") or 0.1
            c_scale = m.get("kpi_cash_scale") or 0.1
            net_debt_cr = borrow * b_scale - (cash * c_scale
                                              if isinstance(cash, (int, float)) else 0.0)
            out["net_debt_source"] = "total borrowings less cash"
    if net_debt_cr is None:
        out["ev_status"] = "no debt data"
        out["market_cap_cr"] = round(mcap_cr, 1)
        out["shares_outstanding"] = round(shares)
        return out

    ev_cr = mcap_cr + net_debt_cr
    out.update({"shares_outstanding": round(shares),
                "market_cap_cr": round(mcap_cr, 1),
                "net_debt_cr": round(net_debt_cr, 1),
                "net_worth_cr": round(net_worth_cr, 1) if net_worth_cr else None,
                "enterprise_value_cr": round(ev_cr, 1),
                "ev_status": "ok"})
    if isinstance(ebitda, (int, float)):
        e_scale = m.get("kpi_ebitda_scale") or 0.1
        ebitda_cr = ebitda * e_scale
        out["ebitda_cr"] = round(ebitda_cr, 1)
        if ebitda_cr > 0:
            out["ev_ebitda"] = round(ev_cr / ebitda_cr, 2)
        else:
            out["ev_status"] = "non-positive EBITDA"
    else:
        out["ev_status"] = "no EBITDA"
    return out


class ValuationEngine:
    def analyse(self, ipo: IPO, metrics: dict[str, Any],
                context: str = "",
                risk_free_pct: float | None = None) -> dict[str, Any]:
        m = metrics or {}
        out: dict[str, Any] = {}
        profile = classify_sector(ipo, context)
        out["sector_key"] = profile["sector_key"]
        out["capital_intensity"] = round(profile["capital_intensity"], 2)
        ev = enterprise_value(ipo, m)
        out.update({k: v for k, v in ev.items()})

        pe = m.get("pe_cap_diluted") or m.get("pe_cap_basic")
        pb = m.get("pb_cap_pre_issue")
        ronw = m.get("ronw_pct")
        out["pe"] = pe
        out["pb"] = pb
        out["ronw_pct"] = ronw
        out["loss_making"] = bool(m.get("loss_making"))

        # --- growth from the 3-year EPS ladder in the RHP
        eps_by_fy: dict[str, dict[str, float]] = m.get("eps_by_fy") or {}
        eps_cagr = None
        if len(eps_by_fy) >= 2:
            fys = sorted(eps_by_fy)
            first = eps_by_fy[fys[0]].get("basic")
            last = eps_by_fy[fys[-1]].get("basic")
            eps_cagr = _cagr(first, last, len(fys) - 1)
        out["eps_cagr_pct"] = eps_cagr

        # --- peer / sector benchmark
        peer_pe = m.get("peer_pe_avg")
        if peer_pe is None:
            peers = m.get("peers") or []
            vals = [p.get("value2") or p.get("value1") for p in peers
                    if isinstance(p, dict)]
            vals = [v for v in vals if v and 0 < v < 300]
            if vals:
                peer_pe = sum(vals) / len(vals)
        benchmark = peer_pe or profile["fair_pe"]
        out["benchmark_pe"] = round(benchmark, 1)
        out["benchmark_source"] = ("RHP peer group" if peer_pe
                                   else "sector heuristic")
        if m.get("peer_note"):
            out["peer_note"] = m["peer_note"]

        # --- valuation score: cheaper than benchmark = better
        if pe and pe > 0:
            rel = pe / benchmark
            out["pe_vs_benchmark"] = round(rel, 2)
            # rel 0.6 -> ~88, 1.0 -> 60, 1.6 -> ~32, 2.5+ -> ~10
            val_score = clamp(60.0 - 55.0 * math.log(rel) / math.log(2.2), 0, 100)
        elif out["loss_making"]:
            val_score = 22.0
            out["pe_vs_benchmark"] = None
        else:
            val_score = 45.0            # unknown: sit near neutral
            out["pe_vs_benchmark"] = None

        # PEG-style adjustment: a rich multiple is forgivable if growth is real
        if pe and eps_cagr and eps_cagr > 0:
            peg = pe / eps_cagr
            out["peg"] = round(peg, 2)
            if peg < 0.8:
                val_score += 12
            elif peg < 1.3:
                val_score += 6
            elif peg > 3.0:
                val_score -= 12

        # P/B only condemns when it is not backed by returns
        if pb and ronw:
            implied = pb / max(ronw / 100.0, 0.01)     # ~ P/E equivalent
            out["pb_implied_pe"] = round(implied, 1)
            if pb > 12 and ronw < 20:
                val_score -= 14
        elif pb and pb > 12:
            val_score -= 8

        # insiders exiting at a huge markup to what they paid is a real flag
        iem = m.get("insider_exit_multiple")
        if iem:
            out["insider_exit_multiple"] = iem
            if iem >= 6:
                val_score -= 16
            elif iem >= 3.5:
                val_score -= 9
            elif iem <= 1.6:
                val_score += 6
        # ------------------------------------------------ EV/EBITDA overlay
        # For capital-intensive issuers, reported earnings are largely an
        # artefact of depreciation policy and leverage. EV/EBITDA sees through
        # both, so it takes the larger share of the valuation score exactly
        # where P/E is least trustworthy.
        ev_multiple = out.get("ev_ebitda")
        fair_ev = profile["fair_ev_ebitda"]
        intensity = profile["capital_intensity"]
        ev_score = None
        if ev_multiple and fair_ev and fair_ev > 0 and ev_multiple > 0:
            rel_ev = ev_multiple / fair_ev
            ev_score = clamp(60.0 - 55.0 * math.log(rel_ev) / math.log(2.2), 0, 100)
            out["ev_ebitda_benchmark"] = fair_ev
            out["ev_vs_benchmark"] = round(rel_ev, 2)
            out["ev_ebitda_score"] = round(ev_score, 1)

        pe_score = clamp(val_score, 0, 100)
        out["pe_score"] = round(pe_score, 1)
        if ev_score is not None and not profile["is_financial"]:
            # 0.25 weight on EV when asset-light, up to 0.75 when heavy
            w_ev = clamp(0.25 + 0.50 * intensity, 0.0, 0.80)
            if m.get("loss_making"):
                # negative earnings make P/E uninformative; lean on EV
                w_ev = max(w_ev, 0.70)
            blended = (1.0 - w_ev) * pe_score + w_ev * ev_score
            out["ev_weight"] = round(w_ev, 2)
            out["valuation_basis"] = (
                f"{(1 - w_ev) * 100:.0f}% P/E + {w_ev * 100:.0f}% EV/EBITDA "
                f"({profile['sector_key']}, capital intensity "
                f"{intensity:.2f})")
            val_score = blended
        else:
            out["ev_weight"] = 0.0
            out["valuation_basis"] = (
                "P/E only - lenders are valued on book and returns, not EBITDA"
                if profile["is_financial"] else
                f"P/E only - EV/EBITDA unavailable ({out.get('ev_status', 'n/a')})")

        out["valuation_score"] = round(clamp(_soft_cap(val_score), 0, 100), 1)
        out["valuation_score_raw"] = round(val_score, 1)

        # --- financial quality
        q = 45.0
        low_ronw_charged = 0.0
        if ronw is not None:
            if ronw >= 30:
                q += 22
            elif ronw >= 20:
                q += 16
            elif ronw >= 15:
                q += 9
            elif ronw >= 10:
                q += 3
            else:
                q -= 12
                low_ronw_charged = 12.0
        if eps_cagr is not None:
            if eps_cagr >= 45:
                q += 20
            elif eps_cagr >= 25:
                q += 14
            elif eps_cagr >= 12:
                q += 7
            elif eps_cagr < 0:
                q -= 18
        if eps_by_fy:
            vals = [eps_by_fy[f].get("basic") for f in sorted(eps_by_fy)]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 3:
                if all(b > a for a, b in zip(vals, vals[1:])):
                    q += 10                      # monotonically improving
                    out["eps_trend"] = "rising every year"
                elif any(v <= 0 for v in vals):
                    q -= 15
                    out["eps_trend"] = "loss in at least one year"
                else:
                    out["eps_trend"] = "uneven"
        if out["loss_making"]:
            q -= 25

        # ------------------------------------------------- EVA / Ke gate
        rf = risk_free_pct if isinstance(risk_free_pct, (int, float)) \
            else FALLBACK_RISK_FREE
        beta = proxy_beta(profile["capital_intensity"], profile["sector_key"],
                          is_financial=profile["is_financial"])
        ke = cost_of_equity(rf, beta)
        out["risk_free_pct"] = round(rf, 2)
        out["beta"] = round(beta, 2)
        out["risk_free_source"] = ("live G-Sec 10y"
                                   if isinstance(risk_free_pct, (int, float))
                                   else "fallback assumption")

        # Post-issue RoNW: fresh equity dilutes return on equity mechanically.
        post_ronw = None
        nw_cr = out.get("net_worth_cr")
        fresh_cr = ipo.fresh_issue_cr
        if (isinstance(nw_cr, (int, float)) and nw_cr > 0
                and isinstance(fresh_cr, (int, float)) and fresh_cr > 0
                and isinstance(ronw, (int, float))):
            pat_cr = nw_cr * ronw / 100.0
            post_nw = nw_cr + fresh_cr
            if post_nw > 0:
                post_ronw = pat_cr / post_nw * 100.0

        # --- accruals / cash-conversion gate
        cfo_cr = None
        if isinstance(m.get("kpi_cfo"), (int, float)):
            cfo_cr = m["kpi_cfo"] * (m.get("kpi_cfo_scale") or 0.1)
        ebitda_cr = out.get("ebitda_cr")
        if ebitda_cr is None and isinstance(m.get("kpi_ebitda"), (int, float)):
            ebitda_cr = m["kpi_ebitda"] * (m.get("kpi_ebitda_scale") or 0.1)
        pat_val = m.get("kpi_pat")
        pat_positive = (bool(pat_val > 0) if isinstance(pat_val, (int, float))
                        else (not out["loss_making"]))
        pat_growing = bool(isinstance(eps_cagr, (int, float)) and eps_cagr > 0)
        accruals = accruals_gate(cfo_cr, ebitda_cr, pat_positive, pat_growing)
        out.update(accruals)

        eva = eva_gate(ronw, ke, post_ronw)
        out.update(eva)

        flags: list[str] = []
        verdict_eva = eva.get("eva_verdict")
        if verdict_eva == "value destroyer":
            # Growth funded below the cost of capital compounds the loss.
            shortfall = abs(eva.get("eva_spread_pct") or 0.0)
            # Net of anything the RoNW band already charged: a sub-10% RoNW
            # that is also below Ke is one fact, not two, and double-charging
            # it pushed otherwise-ordinary issuers to the floor.
            penalty = clamp(18.0 + 1.8 * shortfall, 18.0, 34.0)
            q -= max(0.0, penalty - low_ronw_charged)
            flags.append(eva["eva_flag"])
        elif verdict_eva == "high-spread compounder":
            # A compounder that is not converting profit into cash has not yet
            # earned the full benefit of the doubt, so the spread bonus is
            # halved rather than granted outright.
            bonus = 10.0
            if accruals.get("accruals_triggered"):
                bonus /= 2.0
                out["eva_bonus_halved_by_accruals"] = True
            q += bonus
        elif verdict_eva == "marginal":
            q -= 4
            flags.append(eva["eva_flag"])
        if eva.get("eva_dilution_warning"):
            q -= 6
            flags.append(eva["eva_dilution_warning"])
        if accruals.get("accruals_triggered"):
            q -= 12
            flags.insert(0, accruals["accruals_flag"])
        if flags:
            out["eva_risk_flags"] = flags

        out["quality_score"] = round(clamp(_soft_cap(q), 0, 100), 1)
        out["quality_score_raw"] = round(q, 1)

        # --- headline market cap at the cap price
        if ipo.cap_price and m.get("eps_basic") and m["eps_basic"] > 0:
            # PAT is unknown here; approximate shares from EPS when available
            out["implied_pe_note"] = (
                f"{pe:.1f}x FY{m.get('eps_fy', '')} diluted EPS"
                if pe else None)
        return out
