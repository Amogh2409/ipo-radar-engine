"""
RHP / anchor-book parsing.

NSE publishes the Red Herring Prospectus and the Anchor Allocation letter as
zipped PDFs. The RHP is the single richest source that exists for an Indian
IPO — restated financials, "Basis for the Offer Price", objects of the offer,
risk factors, and what insiders originally paid per share.

Three realities this module handles explicitly:

  * Section HEADINGS often don't survive PDF text extraction (they're styled
    or split across text runs), so sections are located by distinctive body
    CONTENT instead — "Basic and Diluted Earnings Per Equity Share" always
    appears, "BASIS FOR THE OFFER PRICE" frequently does not.
  * The RHP is filed BEFORE the price band is fixed, so every valuation
    multiple inside it reads "[<bullet>]". We extract the price-independent
    inputs (EPS, RoNW, NAV, WACA) and compute the multiples ourselves.
  * Some documents (notably the price-band newspaper advert) are scanned
    images. We score text quality and refuse to guess from OCR-less garbage.
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any

from ..config import DATA_DIR
from ..models import IPO
from ..util import Http
from . import rhp_tables

log = logging.getLogger("ipo_radar.docs")

DOC_DIR = DATA_DIR / "documents"
MIN_ASCII_RATIO = 0.85          # below this the PDF is a scan, not text

# a number token, tolerating thousands separators and (1.23) for negatives
NUM = r"\(?-?[\d,]+(?:\.\d+)?\)?"


def _n(tok: str | None) -> float | None:
    if not tok:
        return None
    t = tok.strip().replace(",", "")
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


# key -> (content anchors, page span, which hit to prefer)
SECTION_ANCHORS: dict[str, tuple[list[str], int, str]] = {
    "basis_for_price": ([r"Basic and Diluted Earnings Per Equity Share",
                         r"Price/Earning\s*\(.{0,6}P/E.{0,6}\)\s*ratio in relation"],
                        3, "first"),
    "waca": ([r"[Ww]eighted average cost of acquisition of all Equity Shares",
              r"Cap Price is .{0,3}x.{0,3} times the weighted average cost"],
             2, "first"),
    "objects": ([r"OBJECTS OF THE OFFER", r"Objects of the Offer"], 4, "last"),
    "risk_factors": ([r"INTERNAL RISK FACTORS", r"RISK FACTORS"], 4, "first"),
    "kpi": ([r"KEY PERFORMANCE INDICATORS",
             r"Key Performance Indicators of our Company"], 3, "first"),
    "summary_financials": ([r"RESTATED CONSOLIDATED SUMMARY",
                            r"Restated Consolidated Statement of Profit and Loss",
                            r"RESTATED CONSOLIDATED"], 3, "first"),
    "capital_structure": ([r"CAPITAL STRUCTURE", r"Share Capital History"], 3, "first"),
}


@dataclass
class DocBundle:
    symbol: str
    rhp_pages: list[str] = field(default_factory=list)
    anchor_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    quality: float = 0.0
    available: bool = False

    def section(self, *names: str) -> str:
        for n in names:
            if self.sections.get(n):
                return self.sections[n]
        return ""


class DocumentSource:
    """Downloads are slow (10-40 MB) but happen once per IPO and cache to disk."""

    def __init__(self, http: Http, fetch_rhp: bool = True) -> None:
        self.http = http
        self.fetch_rhp = fetch_rhp
        # Extraction runs in a worker thread, and asyncio waits for the default
        # executor on shutdown - so a 600-page parse would hold the process
        # open well after Ctrl-C. The worker polls this and bails out.
        self.abort = threading.Event()
        DOC_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ download
    async def _zip_pdfs(self, url: str, cache_name: str) -> list[tuple[str, bytes]]:
        # NSE ships zips; SEBI (the Chittorgarh failover) links the bare PDF.
        is_pdf = url.lower().endswith(".pdf")
        cache = DOC_DIR / f"{cache_name}.{'pdf' if is_pdf else 'zip'}"
        blob: bytes | None = None
        if cache.exists() and cache.stat().st_size > 1024:
            blob = cache.read_bytes()
        else:
            log.info("downloading %s", url)
            try:
                r = await self.http._client.get(
                    url, timeout=600.0,
                    headers={"Referer": "https://www.nseindia.com/"})
                r.raise_for_status()
                blob = r.content
                cache.write_bytes(blob)
            except Exception as exc:
                log.warning("doc download failed %s: %s", url, exc)
                return []
        if blob[:5] == b"%PDF-":
            return [(url.rsplit("/", 1)[-1], blob)]
        try:
            zf = await asyncio.to_thread(zipfile.ZipFile, io.BytesIO(blob))
        except Exception as exc:
            log.warning("bad zip %s: %s (%d bytes) - will retry next cycle",
                        cache_name, exc, len(blob or b""))
            cache.unlink(missing_ok=True)     # truncated download
            return []
        out = []
        for info in zf.infolist():
            if info.filename.lower().endswith(".pdf") and info.file_size > 0:
                try:
                    out.append((info.filename, zf.read(info)))
                except Exception:
                    pass
        return out

    async def _pdf_pages_async(self, blob: bytes,
                               max_pages: int | None = None) -> list[str]:
        """
        Off-thread PDF extraction.

        A 600-page RHP takes 10-20s of pure CPU. Running that inline in a
        coroutine blocks the event loop, which stalls subscription polling and
        makes the daemon deaf to Ctrl-C. Hand it to a worker thread instead -
        pypdf releases the GIL on I/O and this keeps the loop responsive.
        """
        return await asyncio.to_thread(self._pdf_pages, blob, max_pages,
                                       self.abort)

    @staticmethod
    def _pdf_pages(blob: bytes, max_pages: int | None = None,
                   abort: threading.Event | None = None) -> list[str]:
        try:
            from pypdf import PdfReader
        except ImportError:
            log.warning("pypdf not installed - RHP parsing disabled "
                        "(pip install pypdf)")
            return []
        try:
            reader = PdfReader(io.BytesIO(blob))
        except Exception as exc:
            log.warning("unreadable pdf: %s", exc)
            return []
        n = len(reader.pages)
        if max_pages:
            n = min(n, max_pages)
        pages = []
        for i in range(n):
            if abort is not None and i % 20 == 0 and abort.is_set():
                log.info("pdf extraction aborted at page %d/%d", i, n)
                return []
            try:
                pages.append(re.sub(r"\s+", " ", reader.pages[i].extract_text() or ""))
            except Exception:
                pages.append("")
        return pages

    @staticmethod
    def _quality(text: str) -> float:
        if not text:
            return 0.0
        sample = text[:200_000]
        return sum(c.isascii() for c in sample) / len(sample)

    # -------------------------------------------------------------- public
    async def fetch(self, ipo: IPO) -> DocBundle:
        bundle = DocBundle(symbol=ipo.symbol)
        docs: dict[str, str] = ipo.meta.get("documents") or {}

        if docs.get("anchor"):
            for _, blob in await self._zip_pdfs(docs["anchor"], f"{ipo.symbol}_ANCHOR"):
                pages = await self._pdf_pages_async(blob)
                bundle.anchor_text += " ".join(pages) + " "

        if self.fetch_rhp and docs.get("rhp"):
            pdfs = await self._zip_pdfs(docs["rhp"], f"{ipo.symbol}_RHP")
            if pdfs:   # the RHP proper is the largest PDF (the GID is smaller)
                name, blob = max(pdfs, key=lambda kv: len(kv[1]))
                bundle.rhp_pages = await self._pdf_pages_async(blob)
                log.info("%s RHP: %s (%d pages)", ipo.symbol,
                         name.split("/")[-1], len(bundle.rhp_pages))

        full = " ".join(bundle.rhp_pages)
        bundle.quality = self._quality(full)
        if full and bundle.quality < MIN_ASCII_RATIO:
            log.warning("%s RHP looks scanned (ascii=%.2f) - skipping text pass",
                        ipo.symbol, bundle.quality)
            bundle.rhp_pages, full = [], ""

        bundle.available = bool(full or bundle.anchor_text)
        if full:
            bundle.sections = self._split_sections(bundle.rhp_pages)
            bundle.metrics = self.extract_metrics(bundle, ipo)
            if bundle.metrics.get("_missing_metrics"):
                bundle.sections["ratios_raw"] = rhp_tables.ratios_context(
                    bundle.rhp_pages)
        if bundle.anchor_text:
            bundle.metrics.update(self.parse_anchor(bundle.anchor_text))
        return bundle

    # ----------------------------------------------------------- sections
    @staticmethod
    def _split_sections(pages: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, (anchors, span, prefer) in SECTION_ANCHORS.items():
            hit: int | None = None
            for rx in anchors:
                pat = re.compile(rx)
                hits = [i for i, t in enumerate(pages) if pat.search(t)]
                if hits:
                    hit = hits[-1] if prefer == "last" else hits[0]
                    break
            if hit is not None:
                out[key] = " ".join(pages[hit:hit + span])
        return out

    # ------------------------------------------------------------ metrics
    def extract_metrics(self, bundle: DocBundle, ipo: IPO) -> dict[str, Any]:
        """Deterministic table parsing, then the price-dependent multiples."""
        m: dict[str, Any] = {}
        if bundle.rhp_pages:
            try:
                m = rhp_tables.extract(bundle.rhp_pages, ipo.name)
            except Exception as exc:
                log.warning("%s table parsing failed: %s", ipo.symbol, exc)
                m = {}

        # --- multiples the RHP itself leaves as "[<bullet>]" placeholders
        cap, floor = ipo.price_high, ipo.price_low
        if cap:
            for src, dst in (("eps_basic", "pe_cap_basic"),
                             ("eps_diluted", "pe_cap_diluted")):
                v = m.get(src)
                if isinstance(v, (int, float)) and v > 0:
                    m[dst] = round(cap / v, 2)
            nav = m.get("nav_pre_issue")
            if isinstance(nav, (int, float)) and nav > 0:
                m["pb_cap_pre_issue"] = round(cap / nav, 2)
            # the 1-year WACA is the fairest "what did recent insiders pay" anchor
            waca_ref = m.get("waca_1y") or m.get("waca_18m")
            if isinstance(waca_ref, (int, float)) and waca_ref > 0:
                m["insider_exit_multiple"] = round(cap / waca_ref, 2)
        if floor and isinstance(m.get("eps_basic"), (int, float)) and m["eps_basic"] > 0:
            m["pe_floor_basic"] = round(floor / m["eps_basic"], 2)
        if isinstance(m.get("eps_basic"), (int, float)) and m["eps_basic"] <= 0:
            m["loss_making"] = True

        missing = [k for k in ("eps_basic", "ronw_pct", "nav_pre_issue",
                               "kpi_pat", "kpi_ebitda")
                   if m.get(k) is None]
        # net debt needs either the ratio or both sides of it
        if m.get("kpi_net_debt_eq") is None and m.get("kpi_borrowings") is None:
            missing.append("net debt (borrowings and cash)")
        if missing:
            m["_missing_metrics"] = missing
            log.info("%s: %s not found by table parsing - LLM fallback will try",
                     ipo.symbol, ", ".join(missing))
        return m

    # ------------------------------------------------------------- anchor
    @staticmethod
    def parse_anchor(text: str) -> dict[str, Any]:
        """Anchor book size, price, and who actually showed up."""
        out: dict[str, Any] = {}
        flat = re.sub(r"\s+", " ", text)
        am = re.search(r"allocation of\s+([\d,]+)\s+Equity Shares", flat, re.I)
        if am:
            out["anchor_shares"] = int(am.group(1).replace(",", ""))
        pm = re.search(r"Anchor Investor Allocation Price of\s*(?:INR|Rs\.?|₹)?\s*([\d,.]+)",
                       flat, re.I)
        if pm:
            out["anchor_price"] = _n(pm.group(1))
        names = re.findall(
            r"([A-Z][A-Za-z&.,'\- ]{6,70}?(?:Fund|Trust|LLP|Limited|Ltd|PLC|Inc|"
            r"Company|Corporation|Insurance|AMC|Capital|Partners|Investments?))"
            r"(?=\s*[\d,]{4,}|\s*\|)", flat)
        clean, seen = [], set()
        for n in names:
            n = n.strip(" ,.")
            k = n.lower()
            if len(n) > 6 and k not in seen:
                seen.add(k)
                clean.append(n)
        if clean:
            out["anchor_investors"] = clean[:40]
            out["anchor_investor_count"] = len(clean)
        # Book quality proxy: long-only domestic MFs and sovereign funds are a
        # much better signal than a book padded with small NBFCs and AIFs.
        marquee = ("government of singapore", "monetary authority", "abu dhabi",
                   "nomura", "goldman", "morgan stanley", "blackrock", "fidelity",
                   "sbi mutual", "sbi life", "hdfc mutual", "hdfc life",
                   "icici prudential", "nippon", "axis mutual", "kotak",
                   "aditya birla sun life", "mirae", "franklin", "invesco",
                   "schroder", "eastspring", "societe generale", "citigroup",
                   "jpmorgan", "j.p. morgan", "edelweiss", "dsp", "uti mutual",
                   "tata mutual", "bandhan", "whiteoak", "motilal oswal")
        joined = " ".join(clean).lower()
        out["anchor_marquee_hits"] = sum(1 for b in marquee if b in joined)
        return out
