#!/usr/bin/env python3
"""
Weekly IPO intelligence briefing.

Stitches the engine's markdown output into one PDF and appends a system-health
audit read straight from the database. Deliberately standalone: it imports
nothing from the daemon, touches no network, and opens the database read-only,
so it is safe to run against a live engine mid-cycle.

    python3 generate_briefing.py
    python3 generate_briefing.py --output-dir briefings --days 14
    python3 generate_briefing.py --engine chrome --keep-html

PDF ENGINES
-----------
There is no single PDF library that is present everywhere, so the script tries
a chain and uses the first that works. All four consume the same generated
HTML, so the output is consistent whichever one runs:

  1. weasyprint      best CSS fidelity, pure Python, needs cairo/pango
  2. chrome          headless Chrome/Chromium/Edge --print-to-pdf; no Python
                     dependency at all and renders exactly like the browser
  3. wkhtmltopdf     via pdfkit
  4. reportlab       pure-Python fallback that renders the parsed document
                     directly, so a PDF is produced even with no HTML engine

The HTML is always written alongside the PDF, so if every engine is missing
you can still open it and print to PDF from any browser.

INSTALLING AN ENGINE (all optional - reportlab alone is enough)

    # macOS, best fidelity
    brew install cairo pango gdk-pixbuf libffi
    pip install weasyprint

    # or the zero-config route: headless Chrome, already present on most Macs
    #   (nothing to install if you have Chrome/Chromium/Edge)

    # or wkhtmltopdf
    brew install --cask wkhtmltopdf && pip install pdfkit

    # pure-Python fallback (usually already installed)
    pip install reportlab

If pip refuses with an "externally-managed-environment" error (PEP 668), use a
virtualenv rather than --break-system-packages:

    python3 -m venv .venv && source .venv/bin/activate && pip install weasyprint
"""
from __future__ import annotations

import argparse
import html
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
DB_PATH = ROOT / "data" / "ipo_radar.db"

log = logging.getLogger("briefing")

# Calibration thresholds live in the engine. They are mirrored here with a
# best-effort import so the briefing cannot drift from what `calibrate`
# actually requires; the literals are the fallback when the package is absent.
GMP_MIN_SAMPLES = 12        # brain/calibration -> ListingModel.calibrate
PROJECTION_MIN_IPOS = 4     # analytics/subscription -> fit_exponents
RETAIL_K_MIN_SAMPLES = 6    # brain/calibration -> calibrate_all


def _sync_thresholds() -> None:
    """Read the real defaults if the package imports; ignore failure."""
    global GMP_MIN_SAMPLES, PROJECTION_MIN_IPOS
    try:
        import inspect
        sys.path.insert(0, str(ROOT))
        from ipo_radar.analytics.listing import ListingModel
        from ipo_radar.analytics.subscription import SubscriptionProjector
        GMP_MIN_SAMPLES = inspect.signature(
            ListingModel.calibrate).parameters["min_samples"].default
        PROJECTION_MIN_IPOS = inspect.signature(
            SubscriptionProjector.fit_exponents).parameters["min_ipos"].default
    except Exception as exc:                    # standalone by design
        log.debug("using literal thresholds (%s)", exc)


# =====================================================================
# Markdown -> blocks
# =====================================================================
# A focused parser, not a general one. The input is this engine's own report
# writer, whose vocabulary is known and small: headings, pipe tables, bullets,
# blockquotes, rules, and inline bold/italic/links. Parsing exactly that is
# more reliable than pulling in a full CommonMark implementation, and it means
# the briefing has no hard dependency to install.
@dataclass
class Block:
    kind: str                                   # h1..h3 | p | table | ul | quote | hr | pagebreak
    text: str = ""
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    header: list[str] = field(default_factory=list)


_TABLE_SEP = re.compile(r"^\s*\|?[\s:.-]*-{2,}[\s:|.-]*\|?\s*$")


def _split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def parse_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    lines = text.replace("\r\n", "\n").split("\n")
    i, n = 0, len(lines)
    para: list[str] = []
    bullets: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(Block("p", " ".join(para).strip()))
            para.clear()

    def flush_bullets() -> None:
        if bullets:
            blocks.append(Block("ul", items=list(bullets)))
            bullets.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_para(); flush_bullets(); i += 1; continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para(); flush_bullets()
            level = min(len(m.group(1)), 3)
            blocks.append(Block(f"h{level}", m.group(2).strip()))
            i += 1; continue

        if re.match(r"^-{3,}\s*$", stripped) or re.match(r"^\*{3,}\s*$", stripped):
            flush_para(); flush_bullets()
            blocks.append(Block("hr")); i += 1; continue

        # pipe table: a header row followed by a separator row
        if stripped.startswith("|") and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            flush_para(); flush_bullets()
            header = _split_row(stripped)
            rows: list[list[str]] = []
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            width = len(header)
            rows = [(r + [""] * width)[:width] for r in rows]
            blocks.append(Block("table", header=header, rows=rows))
            continue

        m = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if m:
            flush_para()
            bullets.append(m.group(1).strip())
            i += 1; continue

        if stripped.startswith(">"):
            flush_para(); flush_bullets()
            quote = [stripped.lstrip("> ").strip()]
            i += 1
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip("> ").strip())
                i += 1
            blocks.append(Block("quote", " ".join(quote)))
            continue

        flush_bullets()
        para.append(stripped)
        i += 1

    flush_para(); flush_bullets()
    return blocks


# =====================================================================
# HTML rendering
# =====================================================================
CSS = """
@page { size: A4; margin: 18mm 14mm 16mm 14mm;
        @bottom-center { content: counter(page) " / " counter(pages);
                         font-family: Helvetica, Arial, sans-serif;
                         font-size: 8pt; color: #8a8f98; } }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
       font-size: 9.5pt; line-height: 1.45; color: #1c2330; margin: 0; }
h1 { font-size: 21pt; margin: 0 0 4pt; color: #0f1826; letter-spacing: -0.4pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 6pt; color: #0f1826;
     border-bottom: 1px solid #d7dce4; padding-bottom: 3pt; }
h3 { font-size: 10.5pt; margin: 12pt 0 4pt; color: #33405a; }
p  { margin: 0 0 6pt; }
ul { margin: 0 0 8pt; padding-left: 15pt; }
li { margin: 0 0 2.5pt; }
strong { color: #0f1826; }
em { color: #4a5468; }
a { color: #1f4f8f; text-decoration: none; }
hr { border: 0; border-top: 1px solid #e2e6ec; margin: 12pt 0; }
blockquote { margin: 8pt 0; padding: 7pt 10pt; background: #f3f6fa;
             border-left: 3px solid #4a7fc1; color: #24304a; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt;
        font-size: 8.4pt; page-break-inside: auto; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th { background: #eef2f7; text-align: left; font-weight: 600; color: #223047;
     border-bottom: 1.2px solid #c3ccd9; padding: 3.5pt 5pt; }
td { border-bottom: 1px solid #e8ecf1; padding: 3.5pt 5pt;
     vertical-align: top; }
tbody tr:nth-child(even) td { background: #fbfcfd; }
.cover { text-align: center; padding-top: 58mm; }
.cover .title { font-size: 30pt; font-weight: 700; letter-spacing: -1pt; }
.cover .sub { font-size: 12pt; color: #55607a; margin-top: 6pt; }
.cover .meta { font-size: 9pt; color: #8a8f98; margin-top: 26pt; line-height: 1.7; }
.pagebreak { page-break-before: always; }
.tag { display: inline-block; padding: 1pt 6pt; border-radius: 3pt;
       font-size: 7.5pt; font-weight: 600; }
.tag-ok   { background: #e3f4e8; color: #1d6b33; }
.tag-warn { background: #fdf1dc; color: #8a5a09; }
.tag-bad  { background: #fbe4e4; color: #8f1f1f; }
.footnote { font-size: 8pt; color: #8a8f98; margin-top: 14pt;
            border-top: 1px solid #e2e6ec; padding-top: 6pt; }
"""

# Underscore emphasis is what report.py actually emits ("_ASBA blocks..._"),
# so it has to be handled or those lines render with visible underscores. The
# lookarounds keep snake_case identifiers intact.
_ITALIC_UNDERSCORE = r"(?<![A-Za-z0-9_])_([^_\n]+)_(?![A-Za-z0-9_])"

_INLINE = (
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"<em>\1</em>"),
    (re.compile(_ITALIC_UNDERSCORE), r"<em>\1</em>"),
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
)


def inline_html(text: str) -> str:
    out = html.escape(text, quote=False)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def blocks_to_html(blocks: Iterable[Block]) -> str:
    parts: list[str] = []
    for b in blocks:
        if b.kind == "pagebreak":
            parts.append('<div class="pagebreak"></div>')
        elif b.kind == "hr":
            parts.append("<hr/>")
        elif b.kind.startswith("h"):
            parts.append(f"<{b.kind}>{inline_html(b.text)}</{b.kind}>")
        elif b.kind == "p":
            parts.append(f"<p>{inline_html(b.text)}</p>")
        elif b.kind == "quote":
            parts.append(f"<blockquote>{inline_html(b.text)}</blockquote>")
        elif b.kind == "ul":
            lis = "".join(f"<li>{inline_html(i)}</li>" for i in b.items)
            parts.append(f"<ul>{lis}</ul>")
        elif b.kind == "table":
            head = "".join(f"<th>{inline_html(c)}</th>" for c in b.header)
            body = "".join(
                "<tr>" + "".join(f"<td>{inline_html(c)}</td>" for c in r) + "</tr>"
                for r in b.rows)
            parts.append(f"<table><thead><tr>{head}</tr></thead>"
                         f"<tbody>{body}</tbody></table>")
        elif b.kind == "raw":
            parts.append(b.text)
    return "\n".join(parts)


def build_html(blocks: Sequence[Block], title: str) -> str:
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{blocks_to_html(blocks)}</body></html>")


# =====================================================================
# PDF engines
# =====================================================================
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
)


def _find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if os.path.isabs(c) and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def render_weasyprint(html_text: str, out: Path) -> bool:
    try:
        from weasyprint import HTML                      # noqa: PLC0415
    except Exception:
        return False
    try:
        HTML(string=html_text, base_url=str(ROOT)).write_pdf(str(out))
        return out.exists() and out.stat().st_size > 0
    except Exception as exc:
        log.warning("weasyprint failed: %s", exc)
        return False


def render_chrome(html_text: str, out: Path, timeout: float = 60.0) -> bool:
    exe = _find_chrome()
    if not exe:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "briefing.html"
        src.write_text(html_text, encoding="utf-8")
        profile = Path(tmp) / "profile"
        base = [exe, f"--user-data-dir={profile}", "--disable-gpu",
                "--no-sandbox", "--no-first-run", "--disable-extensions",
                f"--print-to-pdf={out}", "--no-pdf-header-footer",
                f"file://{src}"]
        # Headless Chrome frequently writes the PDF and then lingers instead of
        # exiting, so waiting on the process can hang for the whole timeout.
        # Poll for the artefact and stop the browser as soon as it lands.
        for flag in ("--headless=new", "--headless"):
            if out.exists():
                out.unlink()
            try:
                proc = subprocess.Popen([base[0], flag] + base[1:],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
            except Exception as exc:
                log.debug("chrome %s: %s", flag, exc)
                continue
            try:
                deadline = time.monotonic() + timeout
                size, stable = -1, 0
                while time.monotonic() < deadline:
                    if proc.poll() is not None and out.exists():
                        break
                    if out.exists():
                        cur = out.stat().st_size
                        # two consecutive equal sizes means the write finished
                        stable = stable + 1 if cur == size and cur > 0 else 0
                        size = cur
                        if stable >= 2:
                            break
                    time.sleep(0.25)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
            if out.exists() and out.stat().st_size > 0:
                return True
            log.debug("chrome %s produced nothing", flag)
    return False


def render_wkhtmltopdf(html_text: str, out: Path) -> bool:
    exe = shutil.which("wkhtmltopdf")
    if not exe:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "briefing.html"
        src.write_text(html_text, encoding="utf-8")
        try:
            subprocess.run([exe, "--enable-local-file-access", "--quiet",
                            str(src), str(out)],
                           capture_output=True, timeout=180)
        except Exception as exc:
            log.warning("wkhtmltopdf failed: %s", exc)
            return False
    return out.exists() and out.stat().st_size > 0


def render_reportlab(blocks: Sequence[Block], out: Path, title: str) -> bool:
    """
    Pure-Python fallback. Renders the parsed blocks directly rather than the
    HTML, so it needs no rendering engine at all - the guarantee that a PDF
    always gets produced.
    """
    try:
        from reportlab.lib import colors                  # noqa: PLC0415
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (HRFlowable, ListFlowable, ListItem,
                                        PageBreak, Paragraph, SimpleDocTemplate,
                                        Spacer, Table, TableStyle)
    except Exception as exc:
        log.warning("reportlab unavailable: %s", exc)
        return False

    # ReportLab's built-in fonts are Latin-1 and cannot encode the rupee sign.
    # A Unicode TTF is used only if it actually CONTAINS the glyphs we need -
    # checking that a font merely registers is not enough. Arial Unicode, the
    # obvious macOS candidate, loads fine but has no U+20B9 (it predates the
    # 2010 rupee sign), so every amount rendered as a tofu box while text
    # extraction still reported the character as present.
    NEEDED = (0x20B9,)                      # rupee sign; the rest are Latin-1
    body_font, bold_font = "Helvetica", "Helvetica-Bold"
    unicode_ok = False
    for path in ("/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Geneva.ttf",
                 "/System/Library/Fonts/NewYork.ttf",
                 "/Library/Fonts/Arial Unicode.ttf"):
        if not os.path.exists(path):
            continue
        try:
            probe = TTFont("BriefProbe", path)
            if not all(cp in probe.face.charToGlyph for cp in NEEDED):
                log.debug("%s lacks the rupee glyph - skipping",
                          os.path.basename(path))
                continue
            pdfmetrics.registerFont(TTFont("BriefUni", path))
            body_font = bold_font = "BriefUni"
            unicode_ok = True
            break
        except Exception as exc:
            log.debug("font %s unusable: %s", path, exc)
            continue
    if not unicode_ok:
        log.debug("no font with U+20B9 found - transliterating currency marks")

    def safe(text: str) -> str:
        if not unicode_ok:
            text = text.replace("₹", "Rs ").replace("→", "->")
            text = text.replace("–", "-").replace("—", "-")
            text = text.replace("‘", "'").replace("’", "'")
            text = text.replace("“", '"').replace("”", '"')
            text = text.encode("latin-1", "replace").decode("latin-1")
        return text

    ss = getSampleStyleSheet()
    def style(name: str, **kw: Any) -> ParagraphStyle:
        base = dict(fontName=body_font, alignment=TA_LEFT, leading=12.5)
        base.update(kw)
        return ParagraphStyle(name, parent=ss["Normal"], **base)

    st_h1 = style("bh1", fontSize=19, leading=23, spaceAfter=8,
                  textColor=colors.HexColor("#0f1826"))
    st_h2 = style("bh2", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5,
                  textColor=colors.HexColor("#0f1826"))
    st_h3 = style("bh3", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3,
                  textColor=colors.HexColor("#33405a"))
    st_p = style("bp", fontSize=9.2, spaceAfter=5)
    st_q = style("bq", fontSize=9, leftIndent=9, spaceBefore=4, spaceAfter=6,
                 textColor=colors.HexColor("#24304a"),
                 backColor=colors.HexColor("#f3f6fa"), borderPadding=5)
    st_cell = style("bc", fontSize=7.6, leading=9.6)
    st_head = style("bhd", fontSize=7.6, leading=9.6, fontName=bold_font,
                    textColor=colors.HexColor("#223047"))

    def rich(text: str) -> str:
        """Markdown inline -> ReportLab's mini-HTML."""
        t = html.escape(safe(text), quote=False)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
        t = re.sub(r"\*\*(.+?)\*\*", rf'<font name="{bold_font}">\1</font>', t)
        t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", t)
        t = re.sub(_ITALIC_UNDERSCORE, r"<i>\1</i>", t)
        t = re.sub(r"`([^`]+)`", r"\1", t)
        return t

    story: list[Any] = []
    for b in blocks:
        try:
            if b.kind == "pagebreak":
                story.append(PageBreak())
            elif b.kind == "hr":
                story.append(Spacer(1, 4))
                story.append(HRFlowable(width="100%", thickness=0.6,
                                        color=colors.HexColor("#e2e6ec")))
                story.append(Spacer(1, 6))
            elif b.kind == "h1":
                story.append(Paragraph(rich(b.text), st_h1))
            elif b.kind == "h2":
                story.append(Paragraph(rich(b.text), st_h2))
            elif b.kind == "h3":
                story.append(Paragraph(rich(b.text), st_h3))
            elif b.kind == "p":
                story.append(Paragraph(rich(b.text), st_p))
            elif b.kind == "quote":
                story.append(Paragraph(rich(b.text), st_q))
            elif b.kind == "ul":
                story.append(ListFlowable(
                    [ListItem(Paragraph(rich(i), st_p), leftIndent=12)
                     for i in b.items],
                    bulletType="bullet", start="•" if unicode_ok else "-",
                    leftIndent=12))
            elif b.kind == "table" and b.header:
                data = [[Paragraph(rich(c), st_head) for c in b.header]]
                data += [[Paragraph(rich(c), st_cell) for c in r] for r in b.rows]
                avail = A4[0] - 28 * mm
                tbl = Table(data, repeatRows=1,
                            colWidths=[avail / len(b.header)] * len(b.header))
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.9, colors.HexColor("#c3ccd9")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e8ecf1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#fbfcfd")]),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 6))
        except Exception as exc:                # one bad block must not abort
            log.debug("reportlab skipped a %s block: %s", b.kind, exc)

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(body_font, 7.5)
        canvas.setFillColor(colors.HexColor("#8a8f98"))
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    try:
        SimpleDocTemplate(
            str(out), pagesize=A4, title=title, author="IPO Radar",
            leftMargin=14 * mm, rightMargin=14 * mm,
            topMargin=16 * mm, bottomMargin=16 * mm,
        ).build(story, onFirstPage=footer, onLaterPages=footer)
    except Exception as exc:
        log.warning("reportlab build failed: %s", exc)
        return False
    return out.exists() and out.stat().st_size > 0


ENGINES = ("weasyprint", "chrome", "wkhtmltopdf", "reportlab")


def render_pdf(blocks: Sequence[Block], html_text: str, out: Path,
               title: str, preferred: str = "auto") -> str | None:
    order = ENGINES if preferred in ("auto", "", None) else (preferred,)
    for name in order:
        try:
            if name == "weasyprint" and render_weasyprint(html_text, out):
                return name
            if name == "chrome" and render_chrome(html_text, out):
                return name
            if name == "wkhtmltopdf" and render_wkhtmltopdf(html_text, out):
                return name
            if name == "reportlab" and render_reportlab(blocks, out, title):
                return name
        except Exception as exc:
            log.warning("engine %s raised: %s", name, exc)
    return None


# =====================================================================
# System health & calibration audit
# =====================================================================
@dataclass
class HealthReport:
    ok: bool = True
    error: str = ""
    window_days: int = 7
    level_counts: dict[str, int] = field(default_factory=dict)
    kind_counts: list[tuple[str, str, int]] = field(default_factory=list)
    bse_failover: dict[str, Any] = field(default_factory=dict)
    degraded: list[tuple[str, str, str]] = field(default_factory=list)
    outcomes: int = 0
    predictions: int = 0
    ipos: int = 0
    subs: int = 0
    macro_days: int = 0
    last_event: str = ""
    recent_errors: list[tuple[str, str, str]] = field(default_factory=list)


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (name,))
    return cur.fetchone() is not None


def _count(cur: sqlite3.Cursor, table: str) -> int:
    if not _table_exists(cur, table):
        return 0
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])
    except sqlite3.Error:
        return 0


def collect_health(db_path: Path, days: int = 7) -> HealthReport:
    """Read-only audit. Any failure degrades to a reported error, never a raise."""
    rep = HealthReport(window_days=days)
    if not db_path.exists():
        rep.ok = False
        rep.error = f"database not found at {db_path}"
        return rep

    conn = None
    try:
        # read-only URI so a running daemon is never blocked or corrupted
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        cur = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

        if _table_exists(cur, "events"):
            cur.execute("SELECT level, COUNT(*) FROM events WHERE ts >= ? "
                        "GROUP BY level ORDER BY 2 DESC", (cutoff,))
            rep.level_counts = {str(r[0] or "UNKNOWN"): int(r[1])
                                for r in cur.fetchall()}

            cur.execute("SELECT level, kind, COUNT(*) FROM events WHERE ts >= ? "
                        "GROUP BY level, kind ORDER BY 3 DESC LIMIT 12", (cutoff,))
            rep.kind_counts = [(str(a or "?"), str(b or "?"), int(c))
                               for a, b, c in cur.fetchall()]

            cur.execute("SELECT ts, symbol, message FROM events "
                        "WHERE ts >= ? AND level IN ('ERROR','CRITICAL') "
                        "ORDER BY ts DESC LIMIT 10", (cutoff,))
            rep.recent_errors = [(str(a), str(b or "-"), str(c or ""))
                                 for a, b, c in cur.fetchall()]

            cur.execute("SELECT ts, symbol, message FROM events WHERE ts >= ? "
                        "AND kind='demand_degraded' ORDER BY ts DESC LIMIT 12",
                        (cutoff,))
            rep.degraded = [(str(a), str(b or "-"), str(c or ""))
                            for a, b, c in cur.fetchall()]

            cur.execute("SELECT MAX(ts) FROM events")
            rep.last_event = str((cur.fetchone() or [""])[0] or "")

        # BSE failover: the authoritative signal is a stored snapshot whose
        # source is BSE. The degraded-event log is corroboration, not proof.
        bse_rows = 0
        if _table_exists(cur, "subscription"):
            try:
                cur.execute("SELECT COUNT(*) FROM subscription "
                            "WHERE source='bse' AND ts >= ?", (cutoff,))
                bse_rows = int(cur.fetchone()[0])
            except sqlite3.Error:
                bse_rows = 0
        mentions = [d for d in rep.degraded if "bse" in d[2].lower()]
        rep.bse_failover = {
            "triggered": bool(bse_rows or mentions),
            "snapshots": bse_rows,
            "events": len(mentions),
            "last": mentions[0][0] if mentions else "",
        }

        rep.outcomes = _count(cur, "outcomes")
        rep.predictions = _count(cur, "predictions")
        rep.ipos = _count(cur, "ipos")
        rep.subs = _count(cur, "subscription")
        if _table_exists(cur, "macro"):
            try:
                cur.execute("SELECT COUNT(DISTINCT substr(ts,1,10)) FROM macro")
                rep.macro_days = int(cur.fetchone()[0])
            except sqlite3.Error:
                rep.macro_days = 0
    except Exception as exc:
        rep.ok = False
        rep.error = f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return rep


def health_blocks(rep: HealthReport) -> list[Block]:
    """Render the audit as document blocks."""
    out: list[Block] = [Block("pagebreak"),
                        Block("h1", "System Health & Calibration Audit")]

    if not rep.ok:
        out.append(Block("quote", f"Audit unavailable: {rep.error}. The "
                                  f"briefing above is unaffected."))
        return out

    out.append(Block("p", f"*Database telemetry for the {rep.window_days} days "
                          f"to {datetime.now():%d %b %Y %H:%M}. "
                          f"Read-only; the live engine is not interrupted.*"))

    # ---- event log
    out.append(Block("h2", "Event log"))
    warn = rep.level_counts.get("WARNING", 0)
    err = rep.level_counts.get("ERROR", 0) + rep.level_counts.get("CRITICAL", 0)
    total = sum(rep.level_counts.values())
    rows = [[lvl, f"{cnt:,}"] for lvl, cnt in sorted(
        rep.level_counts.items(), key=lambda kv: -kv[1])] or [["(none)", "0"]]
    out.append(Block("table", header=["Level", "Events"], rows=rows))
    verdict = ("No warnings or errors in the window."
               if warn == 0 and err == 0 else
               f"{warn:,} warning(s) and {err:,} error(s) out of {total:,} events.")
    out.append(Block("p", f"**{verdict}**"))

    if rep.kind_counts:
        out.append(Block("h3", "By category"))
        out.append(Block("table", header=["Level", "Kind", "Count"],
                         rows=[[l, k, f"{c:,}"] for l, k, c in rep.kind_counts]))

    if rep.recent_errors:
        out.append(Block("h3", "Recent errors"))
        out.append(Block("table", header=["When", "Symbol", "Message"],
                         rows=[[t[:16].replace("T", " "), s, m[:110]]
                               for t, s, m in rep.recent_errors]))

    # ---- failover
    out.append(Block("h2", "Demand feed & BSE failover"))
    bse = rep.bse_failover
    if bse.get("triggered"):
        out.append(Block("quote",
                         f"**BSE failover was triggered.** "
                         f"{bse.get('snapshots', 0)} subscription snapshot(s) "
                         f"were served by BSE and {bse.get('events', 0)} "
                         f"degraded-feed event(s) reference it"
                         + (f", most recently {bse['last'][:16].replace('T', ' ')}"
                            if bse.get("last") else "") + "."))
    else:
        out.append(Block("p", "**BSE failover was not triggered.** Every "
                              "subscription snapshot in the window came from NSE."))
    if rep.degraded:
        out.append(Block("h3", "Degraded-feed events"))
        out.append(Block("table", header=["When", "Symbol", "Detail"],
                         rows=[[t[:16].replace("T", " "), s, m[:110]]
                               for t, s, m in rep.degraded]))
    else:
        out.append(Block("p", "_No degraded-feed events recorded._"))

    # ---- calibration
    out.append(Block("h2", "Calibration status"))
    n = rep.outcomes
    def status(have: int, need: int) -> str:
        return ("**Ready to refit**" if have >= need
                else f"Needs {need - have} more (have {have} of {need})")
    out.append(Block("table",
        header=["Model", "Requires", "Recorded", "Status"],
        rows=[
            ["GMP realisation factor", f"{GMP_MIN_SAMPLES} outcomes",
             str(n), status(n, GMP_MIN_SAMPLES)],
            ["Projection exponents (alpha)", f"{PROJECTION_MIN_IPOS} completed IPOs",
             str(n), status(n, PROJECTION_MIN_IPOS)],
            ["Retail k (lots/application)", f"{RETAIL_K_MIN_SAMPLES} allotment results",
             str(n), status(n, RETAIL_K_MIN_SAMPLES)],
        ]))
    if n == 0:
        out.append(Block("quote",
            f"**Outcomes recorded: 0. Requires {GMP_MIN_SAMPLES} to refit GMP "
            f"realisation.** No self-calibration has run yet, so every model is "
            f"still on its seeded prior. Record a listing with "
            f"`python3 run.py outcome SYMBOL --listing-price P` and then run "
            f"`python3 run.py calibrate`."))
    else:
        out.append(Block("p", f"**Outcomes recorded: {n}.** "
                              f"{status(n, GMP_MIN_SAMPLES)} for GMP realisation."))

    # ---- corpus
    out.append(Block("h2", "Data corpus"))
    out.append(Block("table", header=["Table", "Rows"], rows=[
        ["IPOs tracked", f"{rep.ipos:,}"],
        ["Subscription snapshots", f"{rep.subs:,}"],
        ["Forecasts recorded", f"{rep.predictions:,}"],
        ["Outcomes (ground truth)", f"{rep.outcomes:,}"],
        ["Macro snapshot days", f"{rep.macro_days:,}"],
    ]))
    if rep.macro_days < 3:
        out.append(Block("p", "_Macro baseline is thin, so yield-spike penalties "
                              "read near-neutral by design until more days accumulate._"))
    if rep.last_event:
        out.append(Block("p", f"_Most recent engine event: "
                              f"{rep.last_event[:19].replace('T', ' ')}._"))
    return out


# =====================================================================
# Assembly
# =====================================================================
def _rank_from_index(index_md: str) -> list[str]:
    """Symbol order as ranked on the dashboard, so the PDF matches the board."""
    order: list[str] = []
    for m in re.finditer(r"\|\s*\[([A-Z0-9&.\-]+)\]\(([^)]+\.md)\)", index_md):
        sym = m.group(1)
        if sym not in order:
            order.append(sym)
    return order


def cover_blocks(when: date, n_sheets: int, engine_note: str) -> list[Block]:
    return [Block("raw", (
        f'<div class="cover">'
        f'<div class="title">IPO Intelligence Briefing</div>'
        f'<div class="sub">Indian primary market &mdash; weekly review</div>'
        f'<div class="meta">{when:%d %B %Y}<br/>'
        f'{n_sheets} issue tear sheet{"s" if n_sheets != 1 else ""} '
        f'&middot; dashboard &middot; system audit<br/>'
        f'Generated by IPO Radar{engine_note}</div>'
        f'</div>'))]


def load_documents(report_dir: Path) -> tuple[list[Block], int, list[str]]:
    """Dashboard first, then tear sheets in dashboard rank order."""
    blocks: list[Block] = []
    problems: list[str] = []
    index_path = report_dir / "index.md"

    index_md = ""
    try:
        index_md = index_path.read_text(encoding="utf-8")
    except Exception as exc:
        problems.append(f"could not read {index_path.name}: {exc}")

    if index_md:
        blocks.append(Block("pagebreak"))
        try:
            blocks.extend(parse_markdown(index_md))
        except Exception as exc:
            problems.append(f"could not parse {index_path.name}: {exc}")

    try:
        sheets = sorted(p for p in report_dir.glob("*.md")
                        if p.name.lower() != "index.md")
    except Exception as exc:
        problems.append(f"could not list {report_dir}: {exc}")
        sheets = []

    ranked = _rank_from_index(index_md)
    by_stem = {p.stem: p for p in sheets}
    ordered = [by_stem[s] for s in ranked if s in by_stem]
    ordered += [p for p in sheets if p.stem not in ranked]

    count = 0
    for path in ordered:
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                problems.append(f"{path.name} is empty")
                continue
            blocks.append(Block("pagebreak"))
            blocks.extend(parse_markdown(text))
            count += 1
        except Exception as exc:                # one bad sheet must not abort
            problems.append(f"skipped {path.name}: {exc}")
    return blocks, count, problems


def build_briefing(report_dir: Path, db_path: Path, out_dir: Path,
                   days: int, engine: str, keep_html: bool) -> int:
    _sync_thresholds()
    today = date.today()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"IPO_Intelligence_Briefing_{today:%Y-%m-%d}"
    pdf_path = out_dir / f"{stem}.pdf"
    html_path = out_dir / f"{stem}.html"

    body, n_sheets, problems = load_documents(report_dir)
    if not body:
        log.error("no readable markdown in %s - nothing to compile", report_dir)
        print(f"\n  No reports found in {report_dir}. "
              f"Run `python3 run.py once` first.\n", file=sys.stderr)
        return 2

    health = collect_health(db_path, days)
    try:
        audit = health_blocks(health)
    except Exception as exc:                    # audit must never sink the PDF
        log.warning("audit rendering failed: %s", exc)
        audit = [Block("pagebreak"), Block("h1", "System Health & Calibration Audit"),
                 Block("quote", f"Audit could not be rendered: {exc}")]

    blocks = cover_blocks(today, n_sheets, "") + body + audit
    if problems:
        blocks.append(Block("h3", "Compilation notes"))
        blocks.append(Block("ul", items=problems))

    title = f"IPO Intelligence Briefing {today:%d %b %Y}"
    html_text = build_html(blocks, title)
    try:
        html_path.write_text(html_text, encoding="utf-8")
    except Exception as exc:
        log.warning("could not write HTML: %s", exc)

    used = render_pdf(blocks, html_text, pdf_path, title, engine)

    print()
    if used:
        size = pdf_path.stat().st_size / 1024
        print(f"  Briefing : {pdf_path}")
        print(f"  Engine   : {used}   ({size:,.0f} KB, "
              f"{n_sheets} tear sheet{'s' if n_sheets != 1 else ''})")
    else:
        print(f"  No PDF engine available - wrote HTML instead:")
        print(f"  {html_path}")
        print(f"  Open it in a browser and print to PDF, or install an engine "
              f"(see the header of this script).")
    if health.ok:
        w = health.level_counts.get("WARNING", 0)
        e = health.level_counts.get("ERROR", 0) + health.level_counts.get("CRITICAL", 0)
        print(f"  Audit    : {w} warning(s), {e} error(s) in {days}d; "
              f"BSE failover {'TRIGGERED' if health.bse_failover.get('triggered') else 'not triggered'}; "
              f"{health.outcomes} outcome(s) recorded")
    else:
        print(f"  Audit    : unavailable ({health.error})")
    for p in problems:
        print(f"  Note     : {p}")
    if not keep_html and used and html_path.exists():
        try:
            html_path.unlink()
        except Exception:
            pass
    print()
    return 0 if used else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="generate_briefing",
        description="Compile IPO Radar's reports into a single PDF briefing "
                    "with a system-health appendix.")
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "briefings")
    ap.add_argument("--days", type=int, default=7,
                    help="event-log window for the audit (default 7)")
    ap.add_argument("--engine", default="auto",
                    choices=("auto",) + ENGINES,
                    help="force a PDF engine instead of trying the chain")
    ap.add_argument("--keep-html", action="store_true",
                    help="keep the intermediate HTML next to the PDF")
    ap.add_argument("--log", default="WARNING")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.WARNING),
                        format="%(levelname)-7s %(name)s  %(message)s")
    try:
        return build_briefing(args.report_dir, args.db, args.output_dir,
                              max(1, args.days), args.engine, args.keep_html)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:                    # last resort: never traceback
        logging.getLogger("briefing").exception("briefing failed: %s", exc)
        print(f"\n  Briefing failed: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
