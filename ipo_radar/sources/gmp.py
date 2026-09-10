"""
Grey-market premium.

GMP is unofficial, unregulated and thinly traded — treat it as sentiment, not
truth. Sources rotate their endpoints constantly, so this is deliberately a
best-effort chain with a manual override at the end:

    data/gmp_manual.json   {"RENTOMOJO": 45, "KANOHAR": 120}

If every scraper breaks, drop numbers in that file and the engine keeps
working with full fidelity.
"""
from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path

from ..config import DATA_DIR
from ..models import GMPSnapshot, IPO
from ..util import Http, name_similarity, now_iso, parse_money

log = logging.getLogger("ipo_radar.gmp")

IPOWATCH_URL = "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/"
MANUAL_PATH = DATA_DIR / "gmp_manual.json"
MATCH_THRESHOLD = 0.62


def _cells(row_html: str) -> list[str]:
    raw = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S)
    return [html.unescape(re.sub(r"<[^>]+>", " ", c)).strip() for c in raw]


class GMPSource:
    def __init__(self, http: Http) -> None:
        self.http = http

    async def fetch_all(self) -> list[dict]:
        """Return raw rows: {name, gmp, price, est_listing, est_gain_pct, source}."""
        rows = await self._ipowatch()
        if not rows:
            log.warning("no GMP rows scraped; relying on manual overrides")
        rows.extend(self._manual())
        return rows

    async def _ipowatch(self) -> list[dict]:
        page = await self.http.text(IPOWATCH_URL)
        if not page:
            return []
        out: list[dict] = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
            c = _cells(tr)
            if len(c) < 5:
                continue
            name = c[0]
            if not name or name.lower().startswith("ipo name"):
                continue
            gmp = parse_money(c[1])
            price = parse_money(c[3]) if len(c) > 3 else None
            est_listing = parse_money(c[4]) if len(c) > 4 else None
            gain = None
            if len(c) > 4:
                m = re.search(r"\(([-+]?\d+(?:\.\d+)?)\s*%\)", c[4])
                if m:
                    gain = float(m.group(1))
            if gain is None and gmp is not None and price:
                gain = gmp / price * 100.0
            out.append({"name": name, "gmp": gmp, "price": price,
                        "est_listing": est_listing, "est_gain_pct": gain,
                        "source": "ipowatch"})
        log.info("ipowatch: %d GMP rows", len(out))
        return out

    @staticmethod
    def _manual() -> list[dict]:
        if not MANUAL_PATH.exists():
            return []
        try:
            data = json.loads(MANUAL_PATH.read_text())
        except Exception as exc:
            log.warning("bad gmp_manual.json: %s", exc)
            return []
        out = []
        for k, v in data.items():
            if isinstance(v, dict):
                out.append({"name": k, "source": "manual", **v})
            else:
                out.append({"name": k, "gmp": float(v), "price": None,
                            "est_listing": None, "est_gain_pct": None,
                            "source": "manual"})
        return out

    def match(self, ipos: list[IPO], rows: list[dict]) -> dict[str, GMPSnapshot]:
        """
        Map scraped rows onto NSE symbols. Manual entries may name the symbol
        directly; scraped rows are fuzzy-matched on company name and must
        clear MATCH_THRESHOLD so we never attach one IPO's GMP to another.
        """
        out: dict[str, GMPSnapshot] = {}
        by_symbol = {i.symbol.upper(): i for i in ipos}
        for row in rows:
            raw_name = str(row.get("name") or "")
            target: IPO | None = by_symbol.get(raw_name.upper())
            score = 1.0
            if target is None:
                best, best_s = None, 0.0
                for ipo in ipos:
                    s = max(name_similarity(raw_name, ipo.name),
                            name_similarity(raw_name, ipo.symbol))
                    if s > best_s:
                        best, best_s = ipo, s
                if not best or best_s < MATCH_THRESHOLD:
                    continue
                target, score = best, best_s
            gmp = row.get("gmp")
            price = row.get("price") or target.cap_price
            gain = row.get("est_gain_pct")
            if gain is None and gmp is not None and price:
                gain = gmp / price * 100.0
            snap = GMPSnapshot(symbol=target.symbol, ts=now_iso(), gmp=gmp,
                               price=price, est_listing=row.get("est_listing"),
                               est_gain_pct=gain,
                               source=str(row.get("source") or "?"),
                               raw_name=raw_name)
            prev = out.get(target.symbol)
            # manual always wins; otherwise keep the better name match
            if prev is None or snap.source == "manual" or score >= 0.99:
                out[target.symbol] = snap
        log.info("GMP matched %d/%d IPOs", len(out), len(ipos))
        return out
