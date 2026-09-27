"""
BSE demand failover, and the layered feed that uses it.

Every mainboard IPO is bid on BOTH exchanges and the registrar consolidates
them, so BSE carries the same demand book NSE does. That makes it a genuine
redundancy rather than a substitute dataset — which matters between 14:30 and
16:30 IST on the closing day, when NSE's Akamai edge starts returning 403 to
non-browser TLS sessions and the T-day engine is making its only irreversible
decision of the cycle.

An honest note on verification. BSE's API host answers and serves JSON for
other endpoints (`getScripHeaderData` returns live quotes), but every
IPO-specific path tried from this environment 302s to an error page. The
endpoint names below are therefore CANDIDATES, tried in order, and the module
is built so that none of them working costs nothing: `DemandFeed` degrades
through NSE's own redundant endpoints and finally to the last good snapshot in
the database. Set `bse_endpoint` in config.json to pin a known-good path.

The design rule is that the decision engine must never starve. A stale number
that says it is stale beats no number at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..models import (BNII, CategoryBid, EMPLOYEE, IPO, NII, QIB, RETAIL,
                      SHAREHOLDER, SNII, SubscriptionSnapshot)
from ..util import Http, now_iso

log = logging.getLogger("ipo_radar.bse")

BSE_API = "https://api.bseindia.com/BseIndiaAPI/api"
BSE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
    "sec-fetch-dest": "empty", "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
}

# Tried in order. BSE renames these between site revisions, so the chain is
# deliberately broad and a miss is cheap (one short request).
DEMAND_ENDPOINTS = (
    "/IPODemandSchedule/w?scripcode={code}",
    "/GetIPODemandSchedule/w?scripcode={code}",
    "/DemandSchedule/w?scripcode={code}",
    "/ipo/demandschedule/w?scripcode={code}",
)
ISSUE_ENDPOINTS = (
    "/GetIPOIssuesData/w?Type=IPO&status=Active",
    "/DisplayIPO/w?type=IPO&status=A",
    "/ipo/w?type=IPO",
)

# BSE labels differ from NSE's; map onto the same canonical categories.
BSE_CATEGORY = {
    "qib": QIB, "qualified institutional": QIB,
    "nii": NII, "non institutional": NII, "hni": NII,
    "bnii": BNII, "nii above": BNII, "b-hni": BNII,
    "snii": SNII, "nii below": SNII, "s-hni": SNII,
    "rii": RETAIL, "retail": RETAIL, "individual": RETAIL,
    "employee": EMPLOYEE, "shareholder": SHAREHOLDER,
}


def _canon(label: str) -> str | None:
    low = (label or "").strip().lower()
    if not low:
        return None
    for key, canon in BSE_CATEGORY.items():
        if key in low:
            return canon
    return None


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


class BSESource:
    def __init__(self, http: Http, endpoint_override: str = "") -> None:
        self.http = http
        self.endpoint_override = endpoint_override
        self._scrip_cache: dict[str, str] = {}

    async def _get(self, path: str) -> Any | None:
        try:
            r = await self.http._client.get(f"{BSE_API}{path}",
                                            headers=BSE_HEADERS, timeout=12.0)
            if r.status_code != 200 or not r.text.strip():
                return None
            if "error_Bse" in r.text or r.text.lstrip().startswith("<"):
                return None                      # the bot-block error page
            return r.json()
        except Exception as exc:
            log.debug("bse %s: %s", path, exc)
            return None

    async def scrip_code(self, ipo: IPO) -> str | None:
        """BSE keys IPOs by numeric scrip code, not by NSE symbol."""
        cached = self._scrip_cache.get(ipo.symbol)
        if cached:
            return cached
        code = str((ipo.meta or {}).get("bse_scrip_code") or "").strip()
        if code:
            self._scrip_cache[ipo.symbol] = code
            return code
        for ep in ISSUE_ENDPOINTS:
            data = await self._get(ep)
            for row in _rows(data):
                name = str(row.get("Company_Name") or row.get("CompanyName")
                           or row.get("scrip_name") or "")
                found = str(row.get("scrip_cd") or row.get("SCRIP_CD")
                            or row.get("Scripcode") or "").strip()
                if not (name and found):
                    continue
                from ..util import name_similarity
                if name_similarity(name, ipo.name) >= 0.62:
                    self._scrip_cache[ipo.symbol] = found
                    return found
        return None

    async def subscription(self, ipo: IPO) -> SubscriptionSnapshot | None:
        code = await self.scrip_code(ipo)
        if not code:
            return None
        paths = ((self.endpoint_override,) if self.endpoint_override
                 else DEMAND_ENDPOINTS)
        for template in paths:
            data = await self._get(template.format(code=code))
            snap = self._parse(ipo, data)
            if snap:
                log.info("%s demand served by BSE (%s)", ipo.symbol, template)
                return snap
        return None

    @staticmethod
    def _parse(ipo: IPO, data: Any) -> SubscriptionSnapshot | None:
        cats: dict[str, CategoryBid] = {}
        total: float | None = None
        for row in _rows(data):
            label = str(row.get("Category") or row.get("CATEGORY")
                        or row.get("category") or "")
            offered = _num(row.get("OfferedQty") or row.get("OFFERED")
                           or row.get("offered") or row.get("NoOfshareOffered"))
            bid = _num(row.get("BidQty") or row.get("BID") or row.get("bid")
                       or row.get("NoOfshareBid"))
            times = _num(row.get("NoOfTimes") or row.get("Subscription")
                         or row.get("noOfTime"))
            if "total" in label.lower():
                total = times
                continue
            canon = _canon(label)
            if not canon:
                continue
            if times is None and offered and bid:
                times = bid / offered
            if offered is None and bid is None and times is None:
                continue
            cats[canon] = CategoryBid(canon, offered, bid, times)
        if not cats:
            return None
        return SubscriptionSnapshot(symbol=ipo.symbol, ts=now_iso(),
                                    categories=cats, total_times=total,
                                    source="bse")


def _rows(data: Any) -> list[dict[str, Any]]:
    """BSE wraps payloads inconsistently: bare list, {Table:[...]}, {data:[...]}."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in ("Table", "table", "data", "Data", "dataList"):
            v = data.get(key)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
    return []


# ============================================================ layered feed
@dataclass
class FeedResult:
    snapshot: SubscriptionSnapshot | None
    source: str                      # nse | nse-detail | nse-totals | bse | cache
    degraded: bool = False
    stale_seconds: float | None = None
    note: str = ""


class DemandFeed:
    """
    Resilient demand provider.

    Layers, in order, each independently verified except BSE:

      1. NSE `ipo-active-category`  - the live book, richest form
      2. NSE `ipo-detail.bidDetails` - same data, different handler
      3. NSE `ipo-current-issue`     - totals only for every live issue at once
      4. BSE demand schedule         - different exchange, different edge
      5. Chittorgarh report          - aggregator on an ordinary web host;
                                       the layer that survives a cloud IP
      6. last good snapshot from SQLite, explicitly marked stale

    Layers 2 and 3 matter more than they look: an Akamai block is usually
    path-scoped, so a sibling NSE endpoint often still answers when the
    primary is throttled. Dropping straight to BSE would give up redundancy
    that is still there.
    """

    TRIP_AFTER_BLOCKS = 2            # consecutive NSE blocks before trying BSE
    FRESH_CACHE_SECONDS = 5400.0     # 90 min: still better than no breakdown

    def __init__(self, http: Http, nse: Any, store: Any = None,
                 bse: BSESource | None = None, cg: Any = None) -> None:
        self.http = http
        self.nse = nse
        self.store = store
        self.bse = bse or BSESource(http)
        self.cg = cg
        self.stats: dict[str, int] = {}

    def _record(self, source: str) -> None:
        self.stats[source] = self.stats.get(source, 0) + 1

    def _cached(self, ipo: IPO, max_age_s: float | None = None
                ) -> FeedResult | None:
        cached = self.store.latest_subscription(ipo.symbol) if self.store else None
        if not cached or not cached.categories:
            return None
        age = None
        try:
            age = (datetime.now()
                   - datetime.fromisoformat(cached.ts)).total_seconds()
        except Exception:
            pass
        if max_age_s is not None and (age is None or age > max_age_s):
            return None
        self._record("cache")
        return FeedResult(cached, "cache", degraded=True, stale_seconds=age,
                          note=(f"live sources unavailable; last full snapshot "
                                f"is {age / 60:.0f} minutes old" if age else
                                "live sources unavailable; using last snapshot"))

    async def fetch(self, ipo: IPO) -> FeedResult:
        series = (ipo.meta or {}).get("series", "EQ")

        snap = await self.nse.subscription(ipo.symbol, series)
        if snap and snap.categories:
            self._record("nse")
            return FeedResult(snap, "nse")

        blocked = self.http.nse_consecutive_blocks >= self.TRIP_AFTER_BLOCKS
        if blocked:
            log.warning("%s: NSE blocked %d times consecutively (last status "
                        "%s) - failing over", ipo.symbol,
                        self.http.nse_consecutive_blocks,
                        self.http.last_nse_status)

        # BSE before totals. A totals-only snapshot carries no category
        # breakdown, so nothing downstream can consume it: the projector
        # returns no posteriors and add_subscription writes no rows. Serving
        # it first short-circuited the two layers that actually work.
        snap = await self.bse.subscription(ipo)
        if snap and snap.categories:
            self._record("bse")
            return FeedResult(snap, "bse", degraded=True,
                              note="served by BSE after NSE failure")

        if self.cg:
            snap = await self.cg.subscription(ipo)
            if snap and snap.categories:
                self._record("chittorgarh")
                return FeedResult(snap, "chittorgarh", degraded=True,
                                  note="served by Chittorgarh after NSE and BSE "
                                       "failed - bids reconstructed from multiples")

        # A recent full snapshot beats a live number with no breakdown.
        if self.store:
            fresh = self._cached(ipo, max_age_s=self.FRESH_CACHE_SECONDS)
            if fresh:
                return fresh

        try:
            totals = await self.nse.current_totals()
        except Exception:
            totals = {}
        if ipo.symbol in totals:
            coarse = SubscriptionSnapshot(
                symbol=ipo.symbol, ts=now_iso(), categories={},
                total_times=totals[ipo.symbol], source="nse-totals")
            self._record("nse-totals")
            return FeedResult(coarse, "nse-totals", degraded=True,
                              note=("headline subscription only - no category "
                                    "breakdown, so allotment odds are unavailable"))

        if self.store:
            cached = self.store.latest_subscription(ipo.symbol)
            if cached:
                age = None
                try:
                    age = (datetime.now()
                           - datetime.fromisoformat(cached.ts)).total_seconds()
                except Exception:
                    pass
                self._record("cache")
                return FeedResult(
                    cached, "cache", degraded=True, stale_seconds=age,
                    note=(f"every live source failed; last good snapshot is "
                          f"{age / 60:.0f} minutes old" if age else
                          "every live source failed; using the last snapshot"))
        self._record("none")
        return FeedResult(None, "none", degraded=True,
                          note="no demand data from any source")
