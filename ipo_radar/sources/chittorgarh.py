"""
Chittorgarh failover: a non-exchange copy of everything NSE gives us.

NSE's Akamai edge routinely refuses datacenter IPs (GitHub Actions, most
clouds), and when `all-upcoming-issues` fails the engine has no universe at
all - every downstream layer, BSE included, starts from an IPO record. This
module rebuilds that record from chittorgarh.com, which aggregates the same
exchange data and serves it from an ordinary web host:

  * universe      - `cloud/index/ipo-read/<board>` JSON (name, dates, lot)
  * issue detail  - the IPO page's HTML tables (price band, face value, shares
                    offered per category, SEBI filing links)
  * live demand   - report 21 JSON: subscription multiple per category
  * RHP           - resolved from the SEBI filing page to the full PDF on
                    sebi.gov.in, so DocumentSource works without nsearchives

Two honest limitations:

  * Chittorgarh has no NSE symbol until listing. A record discovered here
    reuses the symbol of a stored IPO with a matching name, and otherwise
    gets a placeholder derived from the name (meta.source = "chittorgarh").
  * Report 21 carries multiples, not share counts. Bids are reconstructed as
    multiple x shares offered (from the issue page), which is exactly how the
    multiple is defined, but it is an aggregator's number, not the exchange's.
"""
from __future__ import annotations

import html
import logging
import re
from datetime import date
from typing import Any

from ..models import (BNII, CategoryBid, EMPLOYEE, IPO, NII, QIB, RETAIL,
                      SHAREHOLDER, SNII, SubscriptionSnapshot)
from ..util import Http, name_similarity, now_iso, parse_band, parse_money, slugify

log = logging.getLogger("ipo_radar.chittorgarh")

SITE = "https://www.chittorgarh.com"
API = "https://webnodejs.chittorgarh.com/cloud"
HEADERS = {"Accept": "application/json, text/plain, */*",
           "Referer": f"{SITE}/", "Origin": SITE}

STATUS = {"current": "Active", "upcoming": "Forthcoming", "closed": "Closed"}

# report-21 column -> canonical category
DEMAND_COLS = {"QIB (x)": QIB, "NII (x)": NII, "bNII (x)": BNII,
               "sNII (x)": SNII, "Retail (x)": RETAIL,
               "Employee (x)": EMPLOYEE, "Shareholder (x)": SHAREHOLDER}

MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def _cells(row_html: str) -> list[str]:
    return [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S)]


def _int(s: Any) -> int | None:
    m = re.search(r"\d[\d,]*", str(s or ""))
    return int(m.group().replace(",", "")) if m else None


def _num(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _offered_category(label: str) -> str | None:
    """Map a row of the issue page's reservation table onto a category.

    QIB demand is quoted against the ex-anchor portion, so that row (not the
    gross QIB row, which includes anchors) is the denominator NSE uses too.
    """
    low = label.lower().lstrip("−- ").strip()
    if low.startswith("qib (ex"):
        return QIB
    if low.startswith("bnii"):
        return BNII
    if low.startswith("snii"):
        return SNII
    if low.startswith("nii"):
        return NII
    if low.startswith("retail"):
        return RETAIL
    if low.startswith("employee"):
        return EMPLOYEE
    if low.startswith("shareholder"):
        return SHAREHOLDER
    return None


def _issue_dates(text: str) -> tuple[str | None, str | None]:
    """'25 to 29 Sep, 2026' / '29 Sep to 1 Oct, 2026' -> ISO open, close."""
    m = re.search(r"(\d{1,2})(?:\s+([A-Za-z]{3}))?\s+to\s+(\d{1,2})\s+([A-Za-z]{3})"
                  r"[a-z]*,?\s+(\d{4})", text)
    if not m:
        return None, None
    d1, m1, d2, m2, yr = m.groups()
    mo2 = MONTHS.get(m2.lower())
    mo1 = MONTHS.get((m1 or m2).lower())
    if not (mo1 and mo2):
        return None, None
    y2 = int(yr)
    y1 = y2 - 1 if mo1 > mo2 else y2           # Dec -> Jan issues
    try:
        return date(y1, mo1, int(d1)).isoformat(), date(y2, mo2, int(d2)).isoformat()
    except ValueError:
        return None, None


class ChittorgarhSource:
    def __init__(self, http: Http) -> None:
        self.http = http

    async def _json(self, path: str) -> Any | None:
        return await self.http.json(f"{API}/{path}", headers=HEADERS)

    # ------------------------------------------------------------ listing
    async def list_ipos(self, include_sme: bool = False,
                        known: list[IPO] | None = None) -> list[IPO]:
        """The universe, each record already enriched from its issue page."""
        known = known or []
        out: list[IPO] = []
        boards = [("mainline", "MAINBOARD")] + ([("sme", "SME")] if include_sme else [])
        for path, board in boards:
            data = await self._json(f"index/ipo-read/{path}")
            rows = (data or {}).get("ipoList") or []
            for row in rows:
                if row.get("issue_withdraw") or row.get("issue_type") != "IPO":
                    continue
                name = str(row.get("company_name") or "").strip()
                if not name or not row.get("id"):
                    continue
                ipo = IPO(
                    symbol=self._symbol(name, known),
                    name=name,
                    status=STATUS.get(str(row.get("ipo_status")).lower(), "Unknown"),
                    board=board,
                    lot_size=_int(row.get("market_lot_size")),
                    meta={"source": "chittorgarh", "series": "EQ",
                          "cg_id": row["id"],
                          "cg_slug": row.get("urlrewrite_folder_name")},
                )
                out.append(await self.enrich(ipo))
        log.info("Chittorgarh listed %d issues", len(out))
        return out

    @staticmethod
    def _symbol(name: str, known: list[IPO]) -> str:
        best = max(known, key=lambda k: name_similarity(name, k.name), default=None)
        if best and name_similarity(name, best.name) >= 0.62:
            return best.symbol
        return slugify(name).upper()[:20]

    async def _find(self, ipo: IPO) -> tuple[int, str] | None:
        """Chittorgarh id + slug for an IPO first seen on NSE."""
        if ipo.meta.get("cg_id") and ipo.meta.get("cg_slug"):
            return ipo.meta["cg_id"], ipo.meta["cg_slug"]
        path = "sme" if ipo.board == "SME" else "mainline"
        data = await self._json(f"index/ipo-read/{path}")
        for row in (data or {}).get("ipoList") or []:
            if name_similarity(str(row.get("company_name") or ""), ipo.name) >= 0.62:
                ipo.meta["cg_id"] = row["id"]
                ipo.meta["cg_slug"] = row.get("urlrewrite_folder_name")
                return row["id"], row.get("urlrewrite_folder_name")
        return None

    # ------------------------------------------------------ issue detail
    async def enrich(self, ipo: IPO) -> IPO:
        """Fold the issue page into the record, filling only what is missing."""
        found = await self._find(ipo)
        if not found:
            return ipo
        cg_id, slug = found
        page = await self.http.text(f"{SITE}/ipo/{slug}/{cg_id}/")
        if not page:
            return ipo

        pairs: dict[str, str] = {}
        offered: dict[str, float] = {}
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
            c = _cells(tr)
            if len(c) == 2 and c[0]:
                pairs.setdefault(c[0], c[1])
            # reservation rows: category | shares | % [| % | max allottees].
            # The '%' test excludes the lot table's "Retail (Min) | 1 | 468".
            elif len(c) >= 3 and any("%" in x for x in c[2:]):
                cat = _offered_category(c[0])
                shares = _int(c[1])
                if cat and shares and cat not in offered:
                    offered[cat] = float(shares)

        o, cl = _issue_dates(pairs.get("IPO Date", ""))
        ipo.open_date = ipo.open_date or o
        ipo.close_date = ipo.close_date or cl
        lo, hi = parse_band(pairs.get("Price Band"))
        ipo.price_low = ipo.price_low or lo
        ipo.price_high = ipo.price_high or hi
        ipo.lot_size = ipo.lot_size or _int(pairs.get("Lot Size"))
        if ipo.face_value is None and pairs.get("Face Value"):
            ipo.face_value = parse_money(pairs["Face Value"])
        ipo.issue_size_shares = ipo.issue_size_shares or _int(pairs.get("Total Issue Size"))
        if ipo.issue_size_shares and ipo.cap_price:
            ipo.issue_size_cr = ipo.issue_size_shares * ipo.cap_price / 1e7
        if pairs.get("Employee Discount"):
            ipo.meta.setdefault("discount", pairs["Employee Discount"])
            ipo.meta.setdefault("discount_amount", parse_money(pairs["Employee Discount"]))
        if offered:
            ipo.meta["cg_offered"] = offered

        if not (ipo.meta.get("documents") or {}).get("rhp"):
            rhp = await self._sebi_rhp(page)
            if rhp:
                ipo.meta["documents"] = {**(ipo.meta.get("documents") or {}), "rhp": rhp}
        return ipo

    async def _sebi_rhp(self, page: str) -> str | None:
        """Issue page -> SEBI RHP filing page -> the full PDF it embeds."""
        m = re.search(r"https://www\.sebi\.gov\.in/filings/public-issues/[^\"\\]*-rhp_\d+\.html",
                      page)
        if not m:
            return None
        filing = await self.http.text(m.group())
        pdf = re.search(r"file=(https://www\.sebi\.gov\.in/sebi_data/attachdocs/[^\"'&\s]+\.pdf)",
                        filing or "")
        return pdf.group(1) if pdf else None

    # ------------------------------------------------------ subscription
    async def subscription(self, ipo: IPO) -> SubscriptionSnapshot | None:
        found = await self._find(ipo)
        if not found:
            return None
        cg_id = found[0]
        today = date.today()
        fy = today.year if today.month >= 4 else today.year - 1
        board = "sme" if ipo.board == "SME" else "mainboard"
        data = await self._json(
            f"report/data-read/21/1/{today.month}/{today.year}/"
            f"{fy}-{str(fy + 1)[2:]}/0/{board}/0?search=")
        row = next((r for r in (data or {}).get("reportTableData") or []
                    if r.get("~id") == cg_id), None)
        if not row:
            return None

        # Chittorgarh sizes each category at the cap price; NSE sizes it at the
        # floor, so NSE's offered is larger and its multiple smaller by
        # cap/floor. The bids agree. Restate in NSE's convention, otherwise a
        # flip between sources mid-issue reads as a jump in demand.
        scale = (ipo.price_high / ipo.price_low
                 if ipo.price_high and ipo.price_low else 1.0)
        offered: dict[str, float] = ipo.meta.get("cg_offered") or {}
        cats: dict[str, CategoryBid] = {}
        for col, canon in DEMAND_COLS.items():
            times = _num(row.get(col))
            if times is None:
                continue
            off = offered.get(canon)
            bid = off * times if off else None
            if off:
                off *= scale
            cats[canon] = CategoryBid(canon, off, bid, times / scale)
        if not cats:
            return None
        total = _num(row.get("Total (x)"))
        return SubscriptionSnapshot(symbol=ipo.symbol, ts=now_iso(), categories=cats,
                                    total_times=total / scale if total is not None else None,
                                    source="chittorgarh")
