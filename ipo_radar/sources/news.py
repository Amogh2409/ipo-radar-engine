"""Google News RSS per company. Free, no key, good enough for sentiment."""
from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

from ..models import IPO
from ..util import Http, strip_html

log = logging.getLogger("ipo_radar.news")

RSS = ("https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en")


class NewsSource:
    def __init__(self, http: Http) -> None:
        self.http = http

    async def fetch(self, ipo: IPO, limit: int = 12) -> list[dict]:
        base = re.sub(r"\s+(Limited|Ltd)\.?$", "", ipo.name, flags=re.I)
        query = quote_plus(f'"{base}" IPO')
        xml = await self.http.text(RSS.format(q=query))
        if not xml:
            return []
        items = []
        for block in re.findall(r"<item>(.*?)</item>", xml, re.S)[:limit]:
            def tag(t: str) -> str:
                m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", block, re.S)
                return strip_html(m.group(1)) if m else ""
            title = tag("title")
            if not title:
                continue
            items.append({"title": title, "url": tag("link"),
                          "published": tag("pubDate"), "source": tag("source")})
        log.debug("%s: %d news items", ipo.symbol, len(items))
        return items
