"""
Macro conditions: the liquidity backdrop an IPO prices into.

The India 10-year yield is not scraped from a headline number. NSE publishes
every centrally-issued G-Sec traded on the cash market, and the symbol encodes
the coupon and maturity (`754GS2036` is a 7.54% bond maturing 2036). Given the
traded price, the yield to maturity is solved directly. That produces a curve
built from the actual market rather than a single quoted figure, and it comes
from a source this system already authenticates against.

Credit spread is harder. NSE's corporate-bond feed returns empty, and FRED is
not reachable from every environment, so the spread resolves through a chain:
an explicit override, then any optional adapter that happens to be reachable,
then the G-Sec term spread as a clearly-labelled proxy. A proxy that says it
is a proxy beats a precise-looking number nobody can source.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import DATA_DIR
from ..util import Http

log = logging.getLogger("ipo_radar.macro")

MANUAL_PATH = DATA_DIR / "macro_manual.json"
GSEC_PATH = "liveBonds-traded-on-cm?type=gsec"
FRED_HY_OAS = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"

# `824GS2033` -> 8.24% maturing 2033. Two-digit coupons are tenths (`92` = 9.2%).
GSEC_SYM = re.compile(r"^(\d{2,4})GS(\d{4})[A-Z]?$")


def coupon_from_symbol(raw: str) -> float:
    return int(raw) / 10 if len(raw) == 2 else int(raw) / 100


def ytm(price: float, coupon_pct: float, years: float, freq: int = 2) -> float | None:
    """
    Yield to maturity by bisection on the semi-annual coupon bond price.

    Bisection rather than Newton: it cannot diverge on a stale or crossed
    price, and 120 iterations on a monotone function is instant.
    """
    if price <= 0 or years <= 0:
        return None
    n = max(1, int(round(years * freq)))
    cf = coupon_pct / freq

    def pv(y: float) -> float:
        per = y / freq
        return sum(cf / (1 + per) ** i for i in range(1, n + 1)) + 100 / (1 + per) ** n

    lo, hi = 1e-4, 0.60
    if pv(lo) < price or pv(hi) > price:
        return None                        # price outside a sane yield range
    for _ in range(120):
        mid = (lo + hi) / 2
        if pv(mid) > price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * 100


@dataclass
class MacroSnapshot:
    ts: str
    india_2y: float | None = None
    india_5y: float | None = None
    india_10y: float | None = None
    india_30y: float | None = None
    term_spread: float | None = None       # 10y minus 2y, in percentage points
    credit_spread: float | None = None
    credit_spread_source: str = "unavailable"
    index_change_pct: float | None = None
    curve_points: int = 0
    source: str = "nse-gsec"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MacroSource:
    def __init__(self, http: Http) -> None:
        self.http = http

    # ------------------------------------------------------------- curve
    async def yield_curve(self) -> tuple[dict[float, float], int]:
        """{years_to_maturity: ytm_pct} for central government securities."""
        data = await self.http.nse_json(GSEC_PATH)
        rows = (data or {}).get("data") or []
        today = dt.date.today()
        points: list[tuple[float, float, float]] = []      # (years, ytm, volume)
        for r in rows:
            name = str(r.get("companyName") or "").upper()
            if not name.startswith("GOVERNMENT OF INDIA"):
                continue                                   # exclude state loans
            m = GSEC_SYM.match(str(r.get("symbol") or ""))
            if not m:
                continue
            price = r.get("lastPrice") or r.get("closePrice")
            try:
                price = float(price)
            except (TypeError, ValueError):
                continue
            if price <= 1:
                continue
            years = (dt.date(int(m.group(2)), 6, 30) - today).days / 365.25
            if years <= 0.5:
                continue
            y = ytm(price, coupon_from_symbol(m.group(1)), years)
            if y is None or not (3.0 < y < 15.0):
                continue                                   # reject stale prints
            vol = float(r.get("totalTradedVolume") or 0)
            points.append((years, y, vol))
        if not points:
            log.warning("no usable G-Sec quotes for the curve")
            return {}, 0
        return {p[0]: p[1] for p in points}, len(points)

    async def _tenor(self, target: float, band: float = 1.5) -> float | None:
        raise NotImplementedError      # replaced by snapshot(); kept explicit

    @staticmethod
    def _pick(points: list[tuple[float, float, float]], target: float,
              band: float, max_dev: float = 1.2) -> float | None:
        """
        Outlier-rejected, volume-weighted median near a tenor.

        Two hazards, both observed live. Most G-Sec turnover happens on
        NDS-OM, so NSE cash-market prints are thin and some are stale. And
        maturity is approximated as mid-year, which is a rounding error at ten
        years but a large one at two - a single heavily-traded 2027 bond
        priced off a mis-assumed maturity dragged the 2y point to 4.28% while
        its neighbours sat near 5.7%.

        So: take the band median first, discard anything more than `max_dev`
        percentage points away from it, and only then weight by volume.
        """
        near = [(y, v) for yrs, y, v in points if abs(yrs - target) <= band]
        if not near:
            return None
        med = statistics.median([y for y, _ in near])
        kept = [(y, v) for y, v in near if abs(y - med) <= max_dev]
        if not kept:
            return med
        traded = [(y, v) for y, v in kept if v > 0]
        if traded:
            total = sum(v for _, v in traded)
            if total > 0:
                traded.sort()
                acc = 0.0
                for y, v in traded:
                    acc += v
                    if acc >= total / 2:
                        return y
        return statistics.median([y for y, _ in kept])

    # ---------------------------------------------------------- snapshot
    async def snapshot(self) -> MacroSnapshot:
        data = await self.http.nse_json(GSEC_PATH)
        rows = (data or {}).get("data") or []
        today = dt.date.today()
        points: list[tuple[float, float, float]] = []
        for r in rows:
            if not str(r.get("companyName") or "").upper().startswith(
                    "GOVERNMENT OF INDIA"):
                continue
            m = GSEC_SYM.match(str(r.get("symbol") or ""))
            if not m:
                continue
            try:
                price = float(r.get("lastPrice") or r.get("closePrice") or 0)
            except (TypeError, ValueError):
                continue
            if price <= 1:
                continue
            years = (dt.date(int(m.group(2)), 6, 30) - today).days / 365.25
            # Below ~1.5y the mid-year maturity assumption swamps the yield,
            # so those bonds are excluded from curve construction entirely.
            if years <= 1.5:
                continue
            y = ytm(price, coupon_from_symbol(m.group(1)), years)
            if y is None or not (3.0 < y < 15.0):
                continue
            points.append((years, y, float(r.get("totalTradedVolume") or 0)))

        snap = MacroSnapshot(ts=dt.datetime.now().isoformat(timespec="seconds"),
                             curve_points=len(points))
        if points:
            snap.india_2y = self._pick(points, 2.5, 1.2)
            snap.india_5y = self._pick(points, 5.0, 1.5)
            snap.india_10y = self._pick(points, 10.0, 2.0)
            snap.india_30y = self._pick(points, 30.0, 6.0)
            if snap.india_10y is not None and snap.india_2y is not None:
                snap.term_spread = snap.india_10y - snap.india_2y

        cs, src = await self._credit_spread(snap)
        snap.credit_spread, snap.credit_spread_source = cs, src

        try:
            idx = await self.http.nse_json("allIndices")
            for row in (idx or {}).get("data", []):
                if (row.get("index") or "").upper() == "NIFTY 50":
                    snap.index_change_pct = float(row.get("percentChange") or 0)
                    break
        except Exception as exc:
            log.debug("index fetch failed: %s", exc)
        return snap

    # ----------------------------------------------------- credit spread
    async def _credit_spread(self, snap: MacroSnapshot) -> tuple[float | None, str]:
        manual = self._manual()
        if isinstance(manual.get("credit_spread"), (int, float)):
            return float(manual["credit_spread"]), "manual override"

        # Optional adapter. FRED is unreachable from many environments, so it
        # gets ONE short attempt - the shared retry policy would otherwise
        # spend fifteen seconds backing off against a dead host on every
        # single macro refresh.
        txt = await self._try_once(FRED_HY_OAS, timeout=4.0)
        if txt and "," in txt:
            for line in reversed(txt.strip().splitlines()):
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        return float(parts[-1]), "FRED BAMLH0A0HYM2 (US HY OAS)"
                    except ValueError:
                        continue

        if snap.term_spread is not None:
            # Not a credit spread. A flattening or inverting curve signals
            # tightening liquidity, which is the property we actually use.
            return snap.term_spread, "G-Sec term spread 10y-2y (proxy)"
        return None, "unavailable"

    async def _try_once(self, url: str, timeout: float = 4.0) -> str | None:
        """Single attempt, short timeout, never raises."""
        try:
            r = await self.http._client.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            log.debug("optional macro source unavailable (%s): %s", url, exc)
            return None

    @staticmethod
    def _manual() -> dict[str, Any]:
        if not MANUAL_PATH.exists():
            return {}
        try:
            return json.loads(MANUAL_PATH.read_text())
        except Exception as exc:
            log.warning("bad macro_manual.json: %s", exc)
            return {}
