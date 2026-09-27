"""
Post-listing prices from Yahoo Finance: what an IPO actually listed at, and
where it trades now.

Yahoo's chart endpoint needs no key, but its edge returns 429 to a request
that claims to be Chrome without Chrome's TLS fingerprint - which is exactly
what the shared client's full browser User-Agent looks like. A bare
"Mozilla/5.0" is served normally, so this module overrides the header.

The listing price is the OPEN of the first daily candle: the price discovered
in the listing-day pre-open auction, which is how listing gains are quoted.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from ..models import IPO
from ..util import Http, now_iso

log = logging.getLogger("ipo_radar.prices")

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}.NS"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


class YahooPrices:
    def __init__(self, http: Http) -> None:
        self.http = http

    async def listing(self, ipo: IPO) -> dict[str, Any] | None:
        """{listing_date, listing_price, cmp, cmp_ts} once the issue trades,
        else None (not listed yet, or a symbol Yahoo does not know)."""
        if not ipo.close_date:
            return None
        start = datetime.combine(date.fromisoformat(ipo.close_date), datetime.min.time())
        if start.date() >= date.today():
            return None                      # Yahoo 400s on a future window
        data = await self.http.json(
            CHART.format(sym=ipo.symbol), headers=HEADERS,
            params={"period1": int(start.timestamp()),
                    "period2": int((datetime.now() + timedelta(days=1)).timestamp()),
                    "interval": "1d"})
        res = ((data or {}).get("chart") or {}).get("result") or []
        if not res or not res[0].get("timestamp"):
            return None
        r = res[0]
        opens = (r.get("indicators", {}).get("quote") or [{}])[0].get("open") or []
        first = next(((t, o) for t, o in zip(r["timestamp"], opens) if o), None)
        if not first:
            return None
        cmp = r.get("meta", {}).get("regularMarketPrice")
        return {"listing_date": date.fromtimestamp(first[0]).isoformat(),
                "listing_price": round(first[1], 2),
                "cmp": round(cmp, 2) if cmp else None,
                "cmp_ts": now_iso()}
