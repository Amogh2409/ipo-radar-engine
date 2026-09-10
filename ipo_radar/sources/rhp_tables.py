"""
Parsing the "Basis for the Offer Price" tables out of an RHP.

RHP typography is not standardised. Across eleven live issues this code was
built against, the same three numbers appear in at least four layouts:

  * fiscal-per-row     "March 31, 2026   10.42  10.10  3"
  * fiscal relabelled  "Fiscal 2026      17.43  17.43  3"
  * collapsed column   "Fiscal 2026       2.36         3"   (basic == diluted)
  * peer-comparison    "LCC Projects Limited 5.00 36,002.52 10.42 10.42 - 32.24% 32.66"

plus placeholder cells ("-", "NA", "[<bullet>]") wherever a figure is pending, and a
glossary page that defines "EPS" and would otherwise win any naive search.

So: locate each sub-table independently by SCORING candidate pages (label plus
fiscal rows plus a weights column), parse the layouts in order of specificity,
and bound the results so a stray year can never be mistaken for a NAV.
Whatever is still missing is handed to the local model in brain/analyst.py.
"""
from __future__ import annotations

import re

NUM = r"\(?-?\d[\d,]*(?:\.\d+)?\)?"
# Cells that mean "no figure": a dash, NA, Nil, or the "[<bullet>]" the RHP uses for
# anything still pending at the prospectus stage.
PH  = r"(?:[-–—]|N\.?A\.?|Nil|\[[^\]]{0,4}\])"
VAL = rf"(?:{NUM}|{PH})"

# Fiscal-period labels seen in the wild: "March 31, 2026", "As at March 31,
# 2026", "Fiscal 2026", "Fiscal Year 2026", "Financial Year 2026",
# "Financial Year ended March 31, 2025", "FY2026", "Period ended Fiscal 2026".
FY  = (r"(?:(?:As\s+(?:on|at)\s+)?(?:Financial\s+Year|Fiscal(?:\s+Year)?|Period)?\s*(?:ended\s+)?March\s*31,?\s*|(?:Financial\s+Year|Fiscal(?:\s+Year)?)\s+(?:ended\s+)?|FY\s*)(20\d\d)")
TR  = r"[%*^$#\s]*(?:\(\d{1,2}\))?[%*^$#\s]*"           # %, footnote refs, marks

def _n(t):
    if t is None: return None
    t=str(t).strip().replace(",","").rstrip("%*^$#")
    if re.fullmatch(PH, t): return None
    neg=t.startswith("(") and t.endswith(")"); t=t.strip("()")
    try: v=float(t)
    except ValueError: return None
    return -v if neg else v

def _window(pages, rx, span=5, prefer="first"):
    hits=[i for i,t in enumerate(pages) if re.search(rx,t)]
    if not hits: return ""
    return " ".join(pages[(hits[-1] if prefer=="last" else hits[0]):][:span])


# A label alone is not enough: every RHP defines "EPS" in its glossary and
# cross-references the section from the risk factors. The real table is the
# page where the label sits next to fiscal-year rows carrying numbers.
_FY_ROW = re.compile(r"(?:March\s*31,?\s*|Fiscal\s*|FY\s*)20\d\d\s+\(?\d")

def _table_window(pages, label_rx, span=5, corroborate=()):
    best, best_score = None, 0
    for i, t in enumerate(pages):
        if not re.search(label_rx, t):
            continue
        score = 1 + 3 * len(_FY_ROW.findall(t))
        if re.search(r"\bWeight\b", t):
            score += 3
        for rx in corroborate:
            if re.search(rx, t):
                score += 2
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score <= 1:
        return ""
    return " ".join(pages[best:best + span])

def _slice(text, start_rx, end_rxs, limit=1800):
    m=re.search(start_rx, text)
    if not m: return ""
    tail=text[m.start(): m.start()+limit]
    best=len(tail)
    for e in end_rxs:
        em=re.search(e, tail[len(m.group(0)):])
        if em: best=min(best, len(m.group(0))+em.start())
    return tail[:best]

# The "Comparison with listed industry peers" table is column-oriented with one
# row per company: <name> <face value> <revenue> <basic EPS> <diluted EPS>
# <P/E> <RoNW>% <NAV>. Column ORDER varies between RHPs but the RoNW cell is
# the one carrying a per-cent sign, which makes it a reliable anchor to index
# the neighbouring cells from.
_PEER_ROW = re.compile(
    r"([A-Z][A-Za-z&.,'\-() ]{3,60}?(?:Limited|Ltd\.?))\s+"
    r"((?:(?:\(?-?[\d,]+(?:\.\d+)?\)?%?|[-–—]|NA|N\.A\.)\s+){4,8})")


def parse_peer_table(pages, company_name=""):
    win = _table_window(pages, r"Comparison with (?:our )?listed industry peer|"
                               r"Industry Peer Group|Listed [Pp]eers",
                        span=2, corroborate=(r"P/E", r"NAV", r"RoNW|Return on"))
    if not win:
        return {}
    out, peer_pes = {}, []
    key = re.sub(r"[^a-z]", "", (company_name or "").lower())[:12]
    for m in _PEER_ROW.finditer(win):
        name = m.group(1).strip()
        cells = m.group(2).split()
        pct = next((i for i, c in enumerate(cells) if c.endswith("%")), None)
        if pct is None or pct < 3:
            continue
        row = {"name": name,
               "eps_basic": _n(cells[pct - 3]), "eps_diluted": _n(cells[pct - 2]),
               "pe": _n(cells[pct - 1]), "ronw": _n(cells[pct]),
               "nav": _n(cells[pct + 1]) if pct + 1 < len(cells) else None}
        is_issuer = bool(key) and key[:8] in re.sub(r"[^a-z]", "", name.lower())
        if is_issuer:
            out["peer_row_self"] = row
        elif row["pe"] and 0 < row["pe"] < 300:
            peer_pes.append(row["pe"])
            out.setdefault("peers", []).append(row)
    if peer_pes:
        peer_pes.sort()
        out["peer_pe_avg"] = round(peer_pes[len(peer_pes) // 2], 2)
        out["peer_pe_low"], out["peer_pe_high"] = peer_pes[0], peer_pes[-1]
    return out


def extract(pages, company_name=""):
    """
    Each sub-table gets its OWN scored window. They are frequently on
    different pages (EPS on one, RoNW and NAV on the next), so a single
    shared window always loses one of them to whichever page scores highest.
    """
    out = {}

    # ---------------------------------------------------------------- EPS
    eps_win = _table_window(
        pages, r"Basic EPS|Basic (?:and|&) [Dd]iluted [Ee]arnings [Pp]er|"
               r"Basic (?:and|&) [Dd]iluted EPS",
        span=2, corroborate=(r"\bWeight\b",))
    if eps_win:
        eps_seg = _slice(eps_win, r"Basic EPS|Basic (?:and|&) [Dd]iluted",
                         [r"Price/Earning", r"P/E ratio",
                          r"Return on [Nn]et [Ww]orth", r"RoNW"]) or eps_win
        eps = {}
        for fy, b, d, _w in re.findall(rf"{FY}\s+({VAL}){TR}\s+({VAL}){TR}\s+(\d)\b", eps_seg):
            bv, dv = _n(b), _n(d)
            if bv is not None:
                eps[fy] = {"basic": bv, "diluted": dv if dv is not None else bv}
        if not eps:   # "Basic & Diluted EPS" collapsed into a single column
            for fy, v, _w in re.findall(rf"{FY}\s+({VAL}){TR}\s+(\d)\b", eps_seg):
                val = _n(v)
                if val is not None:
                    eps[fy] = {"basic": val, "diluted": val}
        if eps:
            out["eps_by_fy"] = eps
        m = re.search(rf"Weighted [Aa]verage(?: EPS)?[^\d\-(]{{0,60}}({NUM}){TR}\s*({NUM})?", eps_seg)
        if m:
            out["eps_basic_weighted"] = _n(m.group(1))
            if m.group(2):
                out["eps_diluted_weighted"] = _n(m.group(2))

    # --------------------------------------------------------------- RoNW
    ronw_win = _table_window(pages, r"Return on [Nn]et [Ww]orth|\bRoNW\b",
                             span=2, corroborate=(r"\bWeight\b",))
    if ronw_win:
        seg = _slice(ronw_win, r"Return on [Nn]et [Ww]orth|\bRoNW\b",
                     [r"Net [Aa]sset [Vv]alue", r"NAV per"]) or ronw_win
        ronw = {fy: _n(v) for fy, v in re.findall(rf"{FY}\s+({VAL}){TR}\s+\d\b", seg)
                if _n(v) is not None}
        if ronw:
            out["ronw_by_fy"] = ronw
        wm = re.search(rf"Weighted [Aa]verage[^\d\-(]{{0,50}}({NUM})", seg)
        if wm:
            out["ronw_weighted_pct"] = _n(wm.group(1))

    # ---------------------------------------------------------------- NAV
    nav_win = _table_window(pages, r"Net [Aa]sset [Vv]alue",
                            span=2, corroborate=(r"As\s+(?:on|at)", r"Cap Price"))
    if nav_win:
        seg = _slice(nav_win, r"Net [Aa]sset [Vv]alue",
                     [r"Comparison", r"Industry Peer", r"Key [Pp]erformance"], 1100) or nav_win
        navs = {fy: _n(v) for fy, v in
                re.findall(rf"As\s+(?:on|at)\s+March\s*31,?\s*(20\d\d){TR}\s*({NUM})", seg)
                if _n(v) is not None}
        if not navs:
            navs = {fy: _n(v) for fy, v in re.findall(rf"{FY}{TR}\s+({NUM})", seg)
                    if _n(v) is not None}
        if navs:
            out["nav_by_fy"] = navs
            out["nav_pre_issue"] = navs[max(navs)]

    # -------------------------------------------------------------- peers
    peer_win = _table_window(pages, r"Industry Peer Group|Comparison with .{0,40}[Pp]eer",
                             span=2)
    peer_seg = peer_win or ""
    if re.search(r"there are no listed compan", peer_seg, re.I):
        out["peers"] = []
        out["peer_note"] = "RHP states no comparable listed peers"
    for lbl, key in ((r"Highest", "peer_pe_high"), (r"Lowest", "peer_pe_low"),
                     (r"Average|Median", "peer_pe_avg")):
        m = re.search(lbl + rf"\s+({NUM})", peer_seg)
        if m:
            out[key] = _n(m.group(1))

    # --------------------------------------------------------------- WACA
    waca = _window(pages, r"Weighted [Aa]verage [Cc]ost of [Aa]cquisition[^.]{0,80}"
                          r"(?:per Equity Share|per share)", span=2)
    if not re.search(r"Last\s+(?:one|1|18|three|3)\b", waca):
        waca = _window(pages, r"Last\s+18\s+months preceding|Last\s+one\s+year preceding",
                       span=2)
    for lbl, key in ((r"Last\s+(?:one|1)\s+year", "waca_1y"),
                     (r"Last\s+18\s+months", "waca_18m"),
                     (r"Last\s+(?:three|3)\s+years", "waca_3y")):
        m = re.search(lbl + r"[^\n]{0,70}?preceding[^\d\-(N]{0,60}?" + rf"({VAL})", waca)
        if not m:
            m = re.search(lbl + rf"[^\d\-(N]{{0,40}}({VAL})", waca)
        if m and _n(m.group(1)) is not None:
            out[key] = _n(m.group(1))

    # The peer-comparison table repeats the issuer's own EPS/P/E/RoNW/NAV and
    # is sometimes the ONLY place they appear in parsed form.
    peer = parse_peer_table(pages, company_name)
    for k in ("peer_pe_avg", "peer_pe_low", "peer_pe_high"):
        if peer.get(k) is not None and out.get(k) is None:
            out[k] = peer[k]
    if peer.get("peers"):
        out.setdefault("peers", peer["peers"])
    self_row = peer.get("peer_row_self") or {}
    if not out.get("eps_by_fy") and self_row.get("eps_basic") is not None:
        out["eps_by_fy"] = {"latest": {"basic": self_row["eps_basic"],
                                       "diluted": self_row.get("eps_diluted")
                                       or self_row["eps_basic"]}}
    if out.get("ronw_pct") is None and self_row.get("ronw") is not None:
        out["ronw_pct"] = self_row["ronw"]
    if out.get("nav_pre_issue") is None and self_row.get("nav") is not None:
        out["nav_pre_issue"] = self_row["nav"]

    # sanity rails: a bare fiscal year must never survive as a NAV or an EPS
    def _sane(v, lo, hi):
        return isinstance(v, (int, float)) and lo <= abs(v) <= hi and not (
            1990 <= v <= 2100 and float(v).is_integer())
    if out.get("nav_pre_issue") is not None and not _sane(out["nav_pre_issue"], 0.1, 100000):
        out.pop("nav_pre_issue", None); out.pop("nav_by_fy", None)
    if out.get("ronw_pct") is not None and not (-200 <= out["ronw_pct"] <= 300):
        out.pop("ronw_pct", None)

    out.update(extract_cfo(pages))

    kpis = extract_kpis(pages)
    for key in ("ebitda", "pat", "revenue", "net_worth", "borrowings", "cash",
                "net_debt_eq", "debt_eq", "roce", "ebitda_margin"):
        v = kpis.get(key)
        if not v or not v["values"]:
            continue
        # KPI columns run newest-first in every RHP examined
        out[f"kpi_{key}"] = v["values"][0]
        out[f"kpi_{key}_series"] = v["values"]
        if v.get("scale") is not None:
            out[f"kpi_{key}_scale"] = v["scale"]
    if kpis.get("_kpi_page") is not None:
        out["kpi_page"] = kpis["_kpi_page"]

    if out.get("eps_by_fy"):
        latest=max(out["eps_by_fy"]); out["eps_fy"]=latest
        out["eps_basic"]=out["eps_by_fy"][latest]["basic"]
        out["eps_diluted"]=out["eps_by_fy"][latest]["diluted"]
    if out.get("ronw_by_fy"): out["ronw_pct"]=out["ronw_by_fy"][max(out["ronw_by_fy"])]
    return out


def ratios_context(pages, budget: int = 5000) -> str:
    """
    The slice of the RHP most likely to contain EPS / RoNW / NAV, for the LLM
    fallback to read when the table parsers come up empty. Deliberately small:
    a 4B model given 400 pages will hallucinate, given three will not.
    """
    chunks, seen = [], set()
    for rx, span in ((r"Basic EPS|Basic (?:and|&) [Dd]iluted [Ee]arnings [Pp]er", 2),
                     (r"Return on [Nn]et [Ww]orth|\bRoNW\b", 2),
                     (r"Net [Aa]sset [Vv]alue", 2),
                     (r"Comparison with (?:our )?listed industry peer", 2)):
        for i, t in enumerate(pages):
            if not re.search(rx, t) or i in seen:
                continue
            # The rows often sit on the FOLLOWING page (a table split by a page
            # break), so accept a hit whose fiscal rows land next door. This
            # still excludes the glossary, which has neither.
            nxt = pages[i + 1] if i + 1 < len(pages) else ""
            if not (_FY_ROW.search(t) or _FY_ROW.search(nxt)):
                continue
            for j in range(i, min(i + span, len(pages))):
                if j not in seen:
                    seen.add(j)
                    chunks.append(pages[j])
            break
    return " ".join(chunks)[:budget]


# =====================================================================
# Key Performance Indicators — the absolute figures Enterprise Value needs
# =====================================================================
# The KPI table is metric-per-row with a units column:
#     EBITDA (1)  Rs million  1,051.98  730.09  547.03
# "(1)" is a footnote reference, never a value. A genuine bracketed negative
# in these tables always carries a decimal point ("(0.30)"), so stripping
# integers-in-brackets is safe.
FOOTNOTE = re.compile(r"\((\d{1,2})\)")

KPI_LABELS = {
    "ebitda":        r"\bEBITDA\b(?!\s*Margin)",
    "ebitda_margin": r"EBITDA\s*Margin",
    "pat":           r"Profit for the (?:Year|year|period)\s*\(?.{0,8}PAT|\bPAT\b(?!\s*Margin)",
    "revenue":       r"Revenue from operations",
    "net_worth":     r"Net\s*[Ww]orth(?!\s*means)",
    "borrowings":    r"Total [Bb]orrowings|Gross [Dd]ebt",
    "cash":          r"Cash and cash equivalents",
    "net_debt_eq":   r"Net [Dd]ebt to (?:Total )?Equity",
    "debt_eq":       r"(?<!Net )(?<!net )\bDebt to (?:Total )?Equity",
    "roce":          r"(?:Adjusted )?RoCE|Return on Capital Employed",
}
KPI_UNIT_RX = re.compile(
    r"\s*((?:Rs\.?|INR|₹)\s*(?:million|mn|crore|cr|lakh)?|%|times|x)\b", re.I)


def _unit_scale(unit_text: str) -> float | None:
    """Convert a units cell into a multiplier onto Rs crore."""
    u = (unit_text or "").lower()
    if "million" in u or re.search(r"\bmn\b", u):
        return 0.1                      # 10 million = 1 crore
    if "lakh" in u:
        return 0.01
    if "crore" in u or re.search(r"\bcr\b", u):
        return 1.0
    return None                         # unknown; caller assumes Rs million


def kpi_row(seg: str, label_rx: str, want: int = 3) -> tuple[list[float], str]:
    m = re.search(label_rx, seg)
    if not m:
        return [], ""
    tail = FOOTNOTE.sub(" ", seg[m.end(): m.end() + 300])
    um = KPI_UNIT_RX.match(tail)
    unit = um.group(1).strip() if um else ""
    rest = tail[um.end():] if um else tail
    vals: list[float] = []
    for tok in re.finditer(NUM, rest):
        if tok.start() > 120:
            break
        v = _n(tok.group())
        if v is None:
            break
        vals.append(v)
        if len(vals) >= want:
            break
    return vals, unit


def _kpi_yield_score(parsed: dict[str, Any]) -> int:
    """
    Score a candidate window by what it YIELDS, not by how many labels appear.

    Every RHP has a glossary defining EBITDA, PAT and RoCE, so label-count
    scoring reliably picks the glossary over the actual table. Scoring on
    extracted values fixes that.
    """
    score = 0
    for v in parsed.values():
        vals = v["values"]
        if len(vals) >= 2 and any(abs(x) >= 0.01 for x in vals):
            score += 2
            if v["unit"]:
                score += 1
            if len(set(vals)) > 1:      # a real series moves between years
                score += 1
    return score


def extract_kpis(pages: list[str]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score, best_i = 0, None
    for i, t in enumerate(pages):
        if sum(1 for rx in KPI_LABELS.values() if re.search(rx, t)) < 3:
            continue
        if not re.search(r"(?:Fiscal|Financial Year|March 31,|Units?)", t):
            continue
        seg = " ".join(pages[i:i + 2])
        parsed: dict[str, Any] = {}
        for key, rx in KPI_LABELS.items():
            vals, unit = kpi_row(seg, rx)
            if vals:
                parsed[key] = {"values": vals, "unit": unit,
                               "scale": _unit_scale(unit)}
        # a bare fiscal year in a value cell is never a real KPI
        for v in parsed.values():
            v["values"] = [x for x in v["values"] if not _looks_like_year(x)]
        parsed = {k: v for k, v in parsed.items() if v["values"]}
        sc = _kpi_yield_score(parsed)
        if sc > best_score:
            best, best_score, best_i = parsed, sc, i
    if best_score < 6:
        return {}
    best["_kpi_page"] = best_i
    best["_kpi_score"] = best_score
    return best


# =====================================================================
# Offer split — fresh issue versus offer for sale
# =====================================================================
# The RHP cover states these asymmetrically, and consistently so across every
# issue examined: the FRESH issue carries a rupee amount with the share count
# left as "[<bullet>]", while the OFFER FOR SALE carries a share count with the
# rupee amount left blank. Both are pending the price band. So the fresh leg
# is read in rupees and the OFS leg is read in shares and priced at the cap.
#
# This matters more than it looks. Without it `IPO.fresh_issue_cr` and
# `IPO.ofs_cr` stay None forever, which silently disables three separate
# signals: the EVA post-issue dilution warning, the offer-for-sale penalty in
# the structure score, and the OFS polarity rule in the LLM guardrails.
_OFFER_MARKER = re.compile(
    r"(?:SIZE OF THE FRESH ISSUE|FRESH ISSUE SIZE|"
    r"Fresh [Ii]ssue and Offer for Sale)")
_AGGREGATING = re.compile(
    r"aggregating up to\s*(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*"
    r"(million|mn|crore|cr|lakh)", re.I)
_UPTO_SHARES = re.compile(r"[Uu]p to\s*([\d,]{7,})\s*[Ee]quity [Ss]hares")


def extract_offer_split(pages: list[str], cap_price: float | None = None
                        ) -> dict[str, Any]:
    """Fresh-issue value and offer-for-sale size from the RHP cover."""
    out: dict[str, Any] = {}
    window = ""
    for i, t in enumerate(pages[:60]):
        if _OFFER_MARKER.search(t) and re.search(r"[Oo]ffer for [Ss]ale", t):
            window = " ".join(pages[i:i + 2])
            break
    if not window:
        return out

    am = _AGGREGATING.search(window)
    if not am:
        return out
    scale = {"million": 0.1, "mn": 0.1, "crore": 1.0, "cr": 1.0,
             "lakh": 0.01}.get(am.group(2).lower(), 0.1)
    fresh = _n(am.group(1))
    if fresh is None or fresh <= 0:
        return out
    out["fresh_issue_cr"] = round(fresh * scale, 2)

    # The OFS share count on the cover is a CEILING set at filing, not the
    # final number - Rentomojo's RHP says "up to 27,365,529" against a final
    # offer of 21,772,311 shares in total. Naively pricing it produced an
    # offer 43% larger than the one NSE is actually running, so it is kept
    # only as an upper bound and the real split is reconciled by the caller
    # against NSE's authoritative issue size.
    sm = _UPTO_SHARES.search(window, am.end())
    if sm:
        shares = _n(sm.group(1))
        if shares and shares > 0:
            out["ofs_shares_ceiling"] = shares
            if cap_price and cap_price > 0:
                out["ofs_cr_ceiling"] = round(shares * cap_price / 1e7, 2)
    return out


def reconcile_offer_split(fresh_issue_cr: float | None,
                          issue_size_cr: float | None) -> dict[str, Any]:
    """
    Split the offer using the one figure from each source that is actually
    fixed: the fresh-issue rupee amount from the RHP, and the total issue
    size from NSE.

    Everything else on the RHP cover is a "[<bullet>]" placeholder or an
    "up to" ceiling. A fresh amount exceeding the total simply means the
    ceiling was trimmed before pricing and the offer is effectively all-fresh.
    """
    out: dict[str, Any] = {}
    if not (isinstance(fresh_issue_cr, (int, float)) and fresh_issue_cr > 0):
        return out
    if not (isinstance(issue_size_cr, (int, float)) and issue_size_cr > 0):
        out["fresh_issue_cr"] = round(fresh_issue_cr, 2)
        return out
    fresh = min(float(fresh_issue_cr), float(issue_size_cr))
    ofs = max(0.0, float(issue_size_cr) - fresh)
    out["fresh_issue_cr"] = round(fresh, 2)
    out["ofs_cr"] = round(ofs, 2)
    out["ofs_share_pct"] = round(ofs / issue_size_cr * 100.0, 1)
    if fresh_issue_cr > issue_size_cr * 1.01:
        out["offer_split_note"] = (
            "RHP fresh-issue ceiling exceeds the final issue size - treated "
            "as an all-fresh offer")
    return out


# =====================================================================
# Cash flow from operations
# =====================================================================
# Two shapes appear in practice and both are handled, narrative first because
# it names its own fiscal year and unit and is therefore unambiguous:
#
#   narrative  "Net cash generated from operating activities was Rs 1,938.27
#               million in Fiscal 2026"
#   tabular    "Net cash flows from operating activities  339.49  1.79  85.94"
#
# CFO is the accrual check on reported profit: a company can book revenue it
# has not collected, but it cannot fabricate cash in the bank.
# The direction word is CAPTURED, not just matched. "Net cash USED IN
# operating activities was Rs 500 million" states a positive magnitude for a
# NEGATIVE cash flow; recording +500 there turns a cash burner into a cash
# generator and suppresses the exact accruals warning this feeds.
_CFO_LABEL = (r"[Nn]et cash (?P<dir>generated|used|flows?|outflows?)?\s*"
              r"(?:from|in|used in) operating activities")
_CFO_NARRATIVE = re.compile(
    _CFO_LABEL +
    r"\s+was\s*(?:INR|Rs\.?|₹)?\s*(?P<val>\(?-?[\d,]+(?:\.\d+)?\)?)\s*"
    r"(?P<unit>million|mn|crore|cr|lakh)?[^.]{0,60}?"
    r"(?:Fiscal|Financial Year|FY)\s*(?P<fy>20\d\d)", re.I)
_CFO_TABLE = re.compile(_CFO_LABEL + r"\s*(?:\(\d{1,2}\))?\s*"
                        r"((?:\(?-?[\d,]+(?:\.\d+)?\)?\s+){1,4})")


def _looks_like_year(v: float | None) -> bool:
    """A bare 19xx/20xx integer in a numeric cell is a stray fiscal year."""
    return (isinstance(v, (int, float)) and float(v).is_integer()
            and 1990 <= v <= 2100)


def extract_cfo(pages: list[str]) -> dict[str, Any]:
    """Cash flow from operations for the latest fiscal year, in Rs crore."""
    out: dict[str, Any] = {}
    best: tuple[str, float, float] | None = None      # (fy, value, scale)

    for t in pages:
        for m in _CFO_NARRATIVE.finditer(t):
            val = _n(m.group("val"))
            if val is None:
                continue
            direction = (m.group("dir") or "").lower()
            if direction.startswith(("used", "outflow")) and val > 0:
                val = -val          # "used in" states a magnitude, not a sign
            scale = {"million": 0.1, "mn": 0.1, "crore": 1.0, "cr": 1.0,
                     "lakh": 0.01}.get((m.group("unit") or "million").lower(), 0.1)
            fy = m.group("fy")
            if best is None or fy > best[0]:
                best = (fy, val, scale)
    if best:
        out["cfo_fy"] = best[0]
        out["kpi_cfo"] = best[1]
        out["kpi_cfo_scale"] = best[2]
        out["cfo_source"] = "narrative"
        return out

    # tabular fallback: columns run newest-first like every other RHP table
    for t in pages:
        m = _CFO_TABLE.search(t)
        if not m:
            continue
        negate = (m.group("dir") or "").lower().startswith(("used", "outflow"))
        vals = [v for v in (_n(x) for x in m.group(1).split()) if v is not None]
        vals = [v for v in vals if not _looks_like_year(v)]
        if not vals:
            continue
        # Consolidation tables list parent + subsidiaries and then the group
        # total, e.g. "339.49  1.79  85.94  427.22" where the first three sum
        # to the fourth. Taking vals[0] there reports the parent alone and
        # understates group cash flow by a quarter.
        if len(vals) >= 3 and abs(sum(vals[:-1]) - vals[-1]) <= abs(vals[-1]) * 0.01:
            out["kpi_cfo"] = vals[-1]
            out["cfo_note"] = "group total (components summed)"
        else:
            out["kpi_cfo"] = vals[0]
        if negate and out["kpi_cfo"] > 0:
            out["kpi_cfo"] = -out["kpi_cfo"]
        out["kpi_cfo_series"] = vals[:4]
        out["kpi_cfo_scale"] = 0.1        # Rs million is the RHP convention
        out["cfo_source"] = "table"
        return out
    return out
