"""HTTP plumbing, logging, and small parsing helpers."""
from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from datetime import date, datetime
from typing import Any

from zoneinfo import ZoneInfo

import httpx

from .config import UA

# Single source of truth for the market timezone. Everything in this system is
# denominated in Indian market time - bid windows, mandate cut-offs, clearing
# days - so a host clock is never the right reference. Defined here because
# util imports nothing from the package and can therefore be imported by
# anything without a cycle.
IST = ZoneInfo("Asia/Kolkata")


def today_ist() -> "date":
    """Today's date in IST, regardless of where the process is running."""
    return datetime.now(IST).date()

log = logging.getLogger("ipo_radar")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Http:
    """
    Shared async client. NSE hands out data only to sessions that carry the
    cookies it sets on a real page view, so we prime (and re-prime on 401/403)
    before hitting /api/*.
    """

    NSE_HOME = "https://www.nseindia.com"
    NSE_REFERER = "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"

    def __init__(self, timeout: float = 25.0, retries: int = 3) -> None:
        self.retries = retries
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": UA,
                "Accept-Language": "en-US,en;q=0.9",
                # Deliberately NOT setting Accept-Encoding: httpx advertises
                # only the codecs it can actually decode. Hard-coding "br"
                # without the brotli package yields undecodable bytes.
            },
        )
        self._nse_primed = False
        self._lock = asyncio.Lock()
        # Failover needs to tell "throttled" apart from "nothing there".
        self.last_nse_status: int | None = None
        self.nse_consecutive_blocks = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _prime_nse(self, force: bool = False) -> None:
        async with self._lock:
            if self._nse_primed and not force:
                return
            try:
                await self._client.get(
                    self.NSE_HOME,
                    headers={"Accept": "text/html,application/xhtml+xml"},
                )
                await self._client.get(
                    self.NSE_REFERER,
                    headers={"Accept": "text/html,application/xhtml+xml"},
                )
                self._nse_primed = True
                log.debug("NSE session primed (%d cookies)", len(self._client.cookies))
            except Exception as exc:  # network flake — try again next call
                log.warning("NSE priming failed: %s", exc)

    async def nse_json(self, path: str) -> Any | None:
        """GET https://www.nseindia.com/api/<path> and parse JSON."""
        url = f"{self.NSE_HOME}/api/{path.lstrip('/')}"
        headers = {"Accept": "*/*", "Referer": self.NSE_REFERER,
                   "X-Requested-With": "XMLHttpRequest"}
        for attempt in range(self.retries):
            await self._prime_nse(force=attempt > 0)
            try:
                r = await self._client.get(url, headers=headers)
                self.last_nse_status = r.status_code
                if r.status_code in (401, 403, 429):
                    self._nse_primed = False
                    await asyncio.sleep(1.0 + attempt)
                    continue
                r.raise_for_status()
                self.nse_consecutive_blocks = 0
                if not r.text.strip():
                    return None
                return r.json()
            except Exception as exc:
                self.last_nse_status = self.last_nse_status or -1
                log.debug("nse_json %s attempt %d: %s", path, attempt + 1, exc)
                await asyncio.sleep(1.0 + attempt * 1.5)
        if self.last_nse_status in (401, 403, 429, -1, None):
            self.nse_consecutive_blocks += 1
        log.warning("nse_json gave up on %s (last status %s, %d consecutive)",
                    path, self.last_nse_status, self.nse_consecutive_blocks)
        return None

    async def text(self, url: str, **kw: Any) -> str | None:
        for attempt in range(self.retries):
            try:
                r = await self._client.get(url, **kw)
                r.raise_for_status()
                return r.text
            except Exception as exc:
                log.debug("GET %s attempt %d: %s", url, attempt + 1, exc)
                await asyncio.sleep(1.0 + attempt * 1.5)
        return None

    async def json(self, url: str, **kw: Any) -> Any | None:
        txt = await self.text(url, **kw)
        if not txt:
            return None
        try:
            import json as _json
            return _json.loads(txt)
        except Exception:
            return None


# ------------------------------------------------------------------ parsing
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_nse_date(s: str | None) -> str | None:
    """'10-Sep-2026' -> '2026-09-10'."""
    if not s:
        return None
    m = re.match(r"(\d{1,2})[-\s]([A-Za-z]{3})[a-z]*[-\s](\d{4})", s.strip())
    if not m:
        return None
    d, mon, y = m.groups()
    mi = _MONTHS.get(mon.lower())
    if not mi:
        return None
    try:
        return date(int(y), mi, int(d)).isoformat()
    except ValueError:
        return None


def parse_money(s: Any) -> float | None:
    """'Rs.130 to Rs.140' / '₹255' / '1,234.5' -> float (first number found)."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = unicodedata.normalize("NFKC", str(s)).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", txt)
    return float(m.group()) if m else None


def parse_band(s: Any) -> tuple[float | None, float | None]:
    """'Rs.130 to Rs.140' -> (130.0, 140.0). Single value -> (v, v)."""
    if s is None:
        return None, None
    txt = unicodedata.normalize("NFKC", str(s)).replace(",", "")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums), max(nums)


def slugify(name: str) -> str:
    """Normalise a company name for fuzzy cross-source matching."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = n.lower()
    for junk in (" limited", " ltd", " private", " pvt", " india", " (india)",
                 " ipo", " corporation", " company", " & co", " co."):
        n = n.replace(junk, " ")
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n


def name_similarity(a: str, b: str) -> float:
    """Cheap token/prefix similarity for matching GMP rows to NSE symbols."""
    sa, sb = slugify(a), slugify(b)
    if not sa or not sb:
        return 0.0
    if sa == sb:
        return 1.0
    if sa.startswith(sb) or sb.startswith(sa):
        return 0.93
    if sa in sb or sb in sa:
        return 0.85
    # character bigram Jaccard
    ga = {sa[i:i + 2] for i in range(len(sa) - 1)}
    gb = {sb[i:i + 2] for i in range(len(sb) - 1)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def fmt_cr(v: float | None) -> str:
    return "—" if v is None else f"₹{v:,.0f} Cr"


def fmt_pct(v: float | None, dp: int = 1) -> str:
    return "—" if v is None else f"{v:+.{dp}f}%"


def strip_html(html_text: str) -> str:
    txt = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html_text, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    import html as _html
    return re.sub(r"\s+", " ", _html.unescape(txt)).strip()
