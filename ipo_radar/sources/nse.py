"""
NSE is the authoritative source: issue master, bid lot, and the live
category-wise demand book. Everything else in this package is corroboration.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..models import (BNII, CategoryBid, EMPLOYEE, IPO, NII, QIB, RETAIL,
                      SHAREHOLDER, SNII, SubscriptionSnapshot)
from ..util import Http, now_iso, parse_band, parse_money, parse_nse_date

log = logging.getLogger("ipo_radar.nse")

# NSE keys its demand book by Sr.No, which is far more stable than the
# free-text category labels. Sub-rows (1(a), 2.1(b)...) are breakdowns.
SR_MAP = {"1": QIB, "2": NII, "2.1": BNII, "2.2": SNII, "3": RETAIL,
          "4": EMPLOYEE, "5": SHAREHOLDER}


def _canon_category(sr_no: str | None, label: str) -> str | None:
    sr = (sr_no or "").strip()
    if sr in SR_MAP:
        return SR_MAP[sr]
    low = (label or "").lower()
    if "total" in low and not sr:
        return "TOTAL"
    if "qualified institutional" in low:
        return QIB
    if "more than ten lakh" in low:
        return BNII
    if "two lakh" in low and "upto ten lakh" in low:
        return SNII
    if "non institutional" in low:
        return NII
    if "retail individual" in low:
        return RETAIL
    if "employee" in low:
        return EMPLOYEE
    if "shareholder" in low:
        return SHAREHOLDER
    return None


def _num(v: Any) -> float | None:
    """NSE emits '2.5414937E7' and '' interchangeably."""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return parse_money(s)


class NSESource:
    def __init__(self, http: Http) -> None:
        self.http = http

    # ------------------------------------------------------------ listing
    async def list_ipos(self, include_sme: bool = False) -> list[IPO]:
        out: list[IPO] = []
        cats = ["ipo"] + (["sme"] if include_sme else [])
        for cat in cats:
            data = await self.http.nse_json(f"all-upcoming-issues?category={cat}")
            if not isinstance(data, list):
                log.warning("all-upcoming-issues(%s) returned %s", cat, type(data))
                continue
            for row in data:
                ipo = self._row_to_ipo(row, "SME" if cat == "sme" else "MAINBOARD")
                if ipo:
                    out.append(ipo)
        log.info("NSE listed %d issues", len(out))
        return out

    @staticmethod
    def _row_to_ipo(row: dict[str, Any], board: str) -> IPO | None:
        sym = (row.get("symbol") or "").strip()
        if not sym:
            return None
        lo, hi = parse_band(row.get("issuePrice"))
        shares = _num(row.get("issueSize"))
        size_cr = (shares * hi / 1e7) if (shares and hi) else None
        return IPO(
            symbol=sym,
            name=(row.get("companyName") or sym).strip(),
            status=(row.get("status") or "Unknown").strip(),
            board=board,
            open_date=parse_nse_date(row.get("issueStartDate")),
            close_date=parse_nse_date(row.get("issueEndDate")),
            price_low=lo, price_high=hi,
            issue_size_shares=shares, issue_size_cr=size_cr,
            meta={"series": row.get("series") or "EQ"},
        )

    # ------------------------------------------- one-shot totals for all
    async def current_totals(self) -> dict[str, float]:
        """`ipo-current-issue` returns total subscription for every live issue
        in a single request — cheap enough to poll every minute."""
        data = await self.http.nse_json("ipo-current-issue")
        out: dict[str, float] = {}
        if isinstance(data, list):
            for row in data:
                sym = (row.get("symbol") or "").strip()
                t = _num(row.get("noOfTime"))
                if sym and t is not None:
                    out[sym] = t
        return out

    # ------------------------------------------------------ issue detail
    async def issue_detail(self, symbol: str, series: str = "EQ") -> dict[str, Any]:
        """Bid lot, face value, registrar, BRLMs and document URLs."""
        data = await self.http.nse_json(f"ipo-detail?symbol={symbol}&series={series}")
        if not isinstance(data, dict):
            return {}
        info: dict[str, Any] = {}
        pairs: dict[str, str] = {}
        for item in (data.get("issueInfo") or {}).get("dataList", []) or []:
            t = (item.get("title") or "").strip()
            v = str(item.get("value") or "").strip().strip('"')
            if t:
                pairs[t] = v
        info["raw_info"] = pairs

        def find(*needles: str) -> str | None:
            for k, v in pairs.items():
                kl = k.lower()
                if all(n in kl for n in needles):
                    return v
            return None

        lot = find("bid", "lot") or find("minimum", "order", "quantity")
        if lot:
            m = re.search(r"([\d,]+)", lot)
            if m:
                info["lot_size"] = int(m.group(1).replace(",", ""))
        fv = find("face", "value")
        if fv:
            info["face_value"] = parse_money(fv)
        band = find("price", "range") or find("price", "band")
        if band:
            lo, hi = parse_band(band)
            info["price_low"], info["price_high"] = lo, hi
        reg = find("name", "registrar")
        if reg:
            info["registrar"] = reg
        brlm = find("book", "running", "lead")
        if brlm:
            info["brlms"] = [b.strip() for b in re.split(r",| and ", brlm) if b.strip()]
        disc = find("discount")
        if disc:
            info["discount"] = disc
            info["discount_amount"] = parse_money(disc)
        period = find("issue", "period")
        if period:
            info["issue_period"] = period

        # document links (zip archives on nsearchives)
        docs: dict[str, str] = {}
        for k, v in pairs.items():
            if not v.startswith("http"):
                continue
            kl = k.lower()
            if "red herring" in kl:
                docs["rhp"] = v
            elif "anchor" in kl:
                docs["anchor"] = v
            elif "ratio" in kl or "basis of issue" in kl:
                docs["ratios"] = v
        if docs:
            info["documents"] = docs

        info["_bid_details"] = data.get("bidDetails") or []
        info["_active_cat"] = data.get("activeCat") or {}
        return info

    async def enrich(self, ipo: IPO) -> IPO:
        """Fold issue_detail into the IPO record."""
        d = await self.issue_detail(ipo.symbol, ipo.meta.get("series", "EQ"))
        if not d:
            return ipo
        if d.get("lot_size"):
            ipo.lot_size = d["lot_size"]
        if d.get("face_value") is not None:
            ipo.face_value = d["face_value"]
        if d.get("price_low"):
            ipo.price_low = d["price_low"]
        if d.get("price_high"):
            ipo.price_high = d["price_high"]
        if d.get("registrar"):
            ipo.registrar = d["registrar"]
        for k in ("brlms", "discount", "discount_amount", "documents",
                  "issue_period", "raw_info"):
            if d.get(k) is not None:
                ipo.meta[k] = d[k]
        if ipo.issue_size_shares and ipo.cap_price:
            ipo.issue_size_cr = ipo.issue_size_shares * ipo.cap_price / 1e7
        return ipo

    # ------------------------------------------------------ subscription
    async def subscription(self, symbol: str, series: str = "EQ"
                           ) -> SubscriptionSnapshot | None:
        """
        Live demand book. `ipo-active-category` is the freshest during the
        bid window; `ipo-detail.bidDetails` is the fallback / post-close copy.
        """
        rows: list[dict[str, Any]] = []
        active = await self.http.nse_json(f"ipo-active-category?symbol={symbol}")
        if isinstance(active, dict) and active.get("dataList"):
            for r in active["dataList"]:
                rows.append({"srNo": r.get("srNo"),
                             "category": r.get("category"),
                             "offered": r.get("noOfShareOffered"),
                             "bid": r.get("noOfSharesBid"),
                             "times": r.get("noOfTotalMeant")})
        if not rows:
            detail = await self.http.nse_json(
                f"ipo-detail?symbol={symbol}&series={series}")
            for r in (detail or {}).get("bidDetails", []) or []:
                rows.append({"srNo": r.get("srNo"),
                             "category": r.get("category"),
                             "offered": r.get("noOfSharesOffered"),
                             "bid": r.get("noOfsharesBid"),
                             "times": r.get("noOfTime")})
        if not rows:
            return None

        cats: dict[str, CategoryBid] = {}
        total: float | None = None
        for r in rows:
            label = str(r.get("category") or "")
            if label.strip().lower() == "category":   # header row
                continue
            canon = _canon_category(r.get("srNo"), label)
            if not canon:
                continue
            offered, bid, times = _num(r["offered"]), _num(r["bid"]), _num(r["times"])
            if canon == "TOTAL":
                total = times
                continue
            # keep the row with real numbers; skip empty breakdown sub-rows
            if offered is None and bid is None and times is None:
                continue
            if canon in cats and cats[canon].offered is not None and offered is None:
                continue
            if times is None and offered and bid:
                times = bid / offered
            cats[canon] = CategoryBid(canon, offered, bid, times,
                                      str(r.get("srNo") or ""))

        # NSE reports NII as parent + two children; derive whichever is missing
        if NII in cats and SNII in cats and BNII not in cats:
            p, s = cats[NII], cats[SNII]
            if p.offered and s.offered and p.bid is not None and s.bid is not None:
                off, bid = p.offered - s.offered, p.bid - s.bid
                if off > 0:
                    cats[BNII] = CategoryBid(BNII, off, bid, bid / off, "2.1")

        if not cats:
            return None
        return SubscriptionSnapshot(symbol=symbol, ts=now_iso(),
                                    categories=cats, total_times=total)
