"""
Anchor-book profiling: who is holding, and for how long.

SEBI locks anchor allocations in two tranches — 50% of the shares for 30 days
from allotment and the remaining 50% for 90 days (the April 2022 regime). So
the anchor book is a schedule of future supply, and its composition tells you
what that supply will do when it unlocks.

A book of sovereign wealth funds and large domestic mutual funds is patient
capital that mostly stays. A book of small AIFs, Category III long-short
sleeves, broker proprietary accounts and HNI family trusts is fast money that
was allocated to be sold, and the 30-day unlock lands as an overhang on a
stock with no float and no research coverage.

Classification is deliberately structural rather than reputational. A
long-short sleeve run under a large AMC's brand is still tactical money: the
mandate decides the holding period, not the letterhead.

One practical hazard drives the whole normalisation step. Anchor names are
regex-scraped from a PDF, so the raw list contains fragments ("Long-Short
Fund"), overlapping duplicates of the same fund, and occasionally the issuer's
own name. Ratios computed over that list are meaningless, so names are
deduplicated, fragments absorbed and the issuer removed before anything is
counted.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ..util import clamp, slugify

log = logging.getLogger("ipo_radar.anchors")

AnchorClass = Literal["compounder", "flipper", "neutral"]

# ---------------------------------------------------------------- patterns
# Structural markers. These decide the holding period regardless of whose
# brand is on the fund, so they are tested FIRST.
STRUCTURAL_FLIPPER = (
    r"\baif\b", r"alternative investment fund", r"category\s*(?:iii|3)\b",
    r"\bcat[\s\-]*(?:iii|3)\b", r"long[\s\-]?short", r"absolute return",
    r"\bhuf\b", r"hindu undivided", r"family (?:trust|office)",
    r"portfolio management", r"\bpms\b", r"arbitrage",
    r"opportunit(?:y|ies) fund", r"alpha fund", r"multi[\s\-]?strategy",
    r"special situations",
    # broker proprietary books: tested here so a brand match cannot rescue
    # "<Big AMC brand> Securities Limited", which is a trading desk, not a fund
    r"\bbroking\b", r"stock brok", r"securities (?:ltd|limited|pvt|private)",
)

# Self-identifying institutions: the name alone settles it.
COMPOUNDER_SOVEREIGN = (
    r"government of singapore", r"\bgic\b", r"monetary authority of singapore",
    r"abu dhabi investment", r"\badia\b", r"kuwait investment", r"temasek",
    r"norges", r"government pension fund", r"provident fund", r"\bepfo\b",
    r"national pension", r"pension fund",
    r"life insurance corporation", r"\blic\b", r"sbi life", r"hdfc life",
    r"icici prudential life", r"max life", r"bajaj allianz", r"kotak life",
    r"tata aia",
)

# A fund/vehicle marker. Brand patterns below require one of these, because a
# brand on its own says nothing about what the entity IS: "Tata Motors
# Limited" and "Tata Mutual Fund" share a brand and nothing else, and both
# were previously classified as patient capital.
FUND_MARKER = re.compile(
    r"\b(mutual funds?|\bmf\b|amc|asset management|trustee(?:ship)?|funds?|"
    r"schemes?|portfolio|sicav|ucits|investment trust|life insurance|"
    r"insurance|pension)\b", re.I)

# Brand families. Only decisive when a FUND_MARKER is also present.
COMPOUNDER_FAMILIES = (
    # large domestic asset managers (brand match; fund names vary wildly)
    r"\bsbi\b", r"\bhdfc\b", r"icici prudential", r"\bnippon\b",
    r"\bkotak\b", r"aditya birla sun life", r"\baxis\b", r"\buti\b",
    r"\bdsp\b", r"mirae asset", r"franklin templeton",
    r"\btata\b", r"canara robeco", r"invesco",
    r"edelweiss (?:mutual|mf|trusteeship)", r"bandhan (?:mutual|amc)",
    r"whiteoak", r"motilal oswal (?:mutual|amc)", r"quant (?:mutual|amc)",
    r"hsbc (?:mutual|asset)", r"baroda bnp", r"sundaram (?:mutual|amc)",
    # marquee foreign long-only
    r"fidelity", r"capital group", r"blackrock", r"vanguard", r"schroder",
    r"eastspring", r"abrdn", r"aberdeen", r"nomura", r"goldman sachs",
    r"morgan stanley", r"jpmorgan", r"j\.?p\.? morgan", r"societe generale",
    r"citigroup", r"wellington", r"t\.? rowe", r"amundi", r"pinebridge",
    r"ashmore", r"fil investment", r"robeco", r"allianz global",
)

# Weaker signals: fast money that lacks a structural marker.
WEAK_FLIPPER = (
    r"\bfinvest\b", r"\bfincap\b", r"finserv", r"capital services",
    r"\bnbfc\b", r"ventures? llp", r"capital llp", r"partners llp",
    r"\btrading\b", r"proprietary", r"\bprop\b",
)

MIN_NAME_LEN = 8
# Trailing series markers distinguish sibling vehicles. "Alpha Growth Fund I"
# is a substring of "Alpha Growth Fund II" but is not a fragment of it.
SERIES_SUFFIX = re.compile(r"(?:i{1,3}|iv|vi{0,3}|ix|xi{0,2}|\d{1,2})$")
MIN_ISSUER_SLUG = 8
# A genuine scraper fragment is a near-duplicate of its parent - a truncated
# leading slice of the same name. A DIFFERENT entity that merely shares a
# prefix is much longer than the name it contains. Absorb only when the
# candidate parent is not materially longer than the fragment.
MAX_ABSORB_RATIO = 1.6
_TOKEN_JUNK = {"limited", "ltd", "private", "pvt", "the", "and", "co",
               "company", "india", "indian"}


def _tokens(name: str) -> list[str]:
    """
    Word tokens for containment testing.

    slugify() destroys word boundaries, which makes character containment
    unsafe: "envisionfund" literally ends with "visionfund", so two unrelated
    funds looked like a fragment and its parent. Comparing token SEQUENCES
    keeps "vision" distinct from "envision".
    """
    raw = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower())
    return [t for t in raw.split() if t and t not in _TOKEN_JUNK]


def _is_token_slice(short: list[str], long: list[str]) -> bool:
    """True when `short` appears as a contiguous run inside `long`."""
    if not short or len(short) >= len(long):
        return False
    for i in range(len(long) - len(short) + 1):
        if long[i:i + len(short)] == short:
            return True
    return False
GENERIC_FRAGMENTS = (
    "fund", "trust", "limited", "ltd", "the fund", "trusteeship",
    "mutual fund", "co ltd", "india fund",
)


@dataclass
class AnchorProfile:
    total: int = 0
    compounders: list[str] = field(default_factory=list)
    flippers: list[str] = field(default_factory=list)
    neutral: list[str] = field(default_factory=list)
    compounder_ratio: float = 0.0
    flipper_ratio: float = 0.0
    quality_score: float = 50.0            # 0-100
    overhang_risk: str = "UNKNOWN"         # LOW | MEDIUM | HIGH | UNKNOWN
    flags: list[str] = field(default_factory=list)
    dropped: int = 0                       # fragments / duplicates removed
    day30: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total,
                "compounders": self.compounders, "flippers": self.flippers,
                "neutral": self.neutral,
                "compounder_ratio": round(self.compounder_ratio, 3),
                "flipper_ratio": round(self.flipper_ratio, 3),
                "quality_score": round(self.quality_score, 1),
                "overhang_risk": self.overhang_risk,
                "flags": self.flags, "dropped": self.dropped,
                "day30": self.day30}


# ------------------------------------------------------------ normalisation
def normalise_names(raw: list[str], issuer_name: str = "") -> tuple[list[str], int]:
    """
    Clean the regex-scraped list before anything is counted.

    Drops the issuer's own name, generic fragments, and any entry wholly
    contained in a longer entry (the scraper emits overlapping slices of the
    same fund). Returns the cleaned list and how many were removed.
    """
    seen: dict[str, str] = {}
    issuer_slug = slugify(issuer_name) if issuer_name else ""
    candidates: list[str] = []
    for name in raw or []:
        n = re.sub(r"\s+", " ", str(name)).strip(" ,.-")
        if len(n) < MIN_NAME_LEN:
            continue
        low = n.lower().strip()
        if low in GENERIC_FRAGMENTS:
            continue
        slug = slugify(n)
        if not slug or len(slug) < 6:
            continue
        # The issuer is not its own anchor - but match strictly. An
        # unbounded substring test on a short slug deleted real allocations
        # ("Vision Limited" as issuer removed an anchor called "Vision Fund").
        if issuer_slug and (slug == issuer_slug
                            or (len(issuer_slug) >= MIN_ISSUER_SLUG
                                and slug.startswith(issuer_slug))):
            continue
        candidates.append(n)

    # absorb fragments: keep the longest surviving form of each entity
    candidates.sort(key=len, reverse=True)
    kept: list[str] = []
    kept_tokens: list[list[str]] = []
    for n in candidates:
        toks = _tokens(n)
        slug = slugify(n)
        absorbed = False
        for parent in kept_tokens:
            if not _is_token_slice(toks, parent):
                continue
            # a distinguishing series suffix means siblings, not a fragment
            if SERIES_SUFFIX.search(slug) and parent[-len(toks):] != toks:
                continue
            # A parent carrying materially more tokens is a different entity
            # that merely shares a leading name - "Goldman Sachs India" and
            # "Goldman Sachs Singapore" both reduce to ["goldman","sachs"]
            # once the junk tokens go, and are two allocations, not one.
            # Short fragments get the tightest allowance, since two shared
            # tokens is weak evidence of identity.
            if len(parent) > len(toks) + max(1, int(len(toks) * 0.5)):
                continue
            absorbed = True
            break
        if absorbed:
            continue
        kept.append(n)
        kept_tokens.append(toks)
    dropped = len([c for c in (raw or []) if c]) - len(kept)
    return kept, max(0, dropped)


def classify_anchor(name: str) -> AnchorClass:
    """
    Structural markers first, then brand families, then weak signals.

    A Category III long-short sleeve under a large AMC's brand is tactical
    money: the mandate sets the holding period, not the letterhead.
    """
    low = name.lower()
    for rx in STRUCTURAL_FLIPPER:
        if re.search(rx, low):
            return "flipper"
    for rx in COMPOUNDER_SOVEREIGN:
        if re.search(rx, low):
            return "compounder"
    # A brand is only patient capital when the entity is actually a fund.
    if FUND_MARKER.search(low):
        for rx in COMPOUNDER_FAMILIES:
            if re.search(rx, low):
                return "compounder"
    for rx in WEAK_FLIPPER:
        if re.search(rx, low):
            return "flipper"
    return "neutral"


# -------------------------------------------------------------- profiling
# --------------------------------------------------- day-30 supply shock
# SEBI releases 50% of the anchor allocation at 30 days and the rest at 90.
# The first tranche is the one that hurts: it lands on a stock with no
# research coverage, a thin float, and - if the book was allocated to fast
# money - holders who were always going to sell it.
# The two float bases measure different quantities and read ~2.7x apart, so
# one threshold cannot serve both.
#
#   spec basis    total shares less promoter lock-in. Includes pre-IPO
#                 investor stock that is itself locked for six months, so it
#                 OVERSTATES what can trade at day 30.
#   listing basis offer less anchor allocation - what genuinely changes hands
#                 in the first month. A typical issue lands near 21% on this
#                 basis simply because anchors take about 30% of the offer,
#                 so a 15% trigger here would fire on almost everything.
DAY30_IMPACT_THRESHOLD = 0.15          # spec basis
DAY30_IMPACT_THRESHOLD_FLOAT = 0.30    # listing-float basis
DAY30_FLIPPER_THRESHOLD = 0.35     # only alarming if the book will actually sell
ANCHOR_TRANCHE_AT_30D = 0.50


def day30_supply_impact(anchor_shares: float | None,
                        total_shares: float | None = None,
                        promoter_locked_shares: float | None = None,
                        offer_shares: float | None = None) -> dict[str, Any]:
    """
    Free-float expansion when the 30-day anchor tranche unlocks.

        impact = (0.5 x anchor_shares) / tradable_float

    The denominator is the float that can actually change hands. Preferred
    basis is `total_shares - promoter_locked_shares` as specified. When
    promoter lock-in data is not parseable - it is buried in the capital
    structure tables and frequently left as "[<bullet>]" pending the price band -
    the listing float is used instead: the offer itself, minus the anchor
    allocation, since anchors are inside the offer and locked at listing.

    That fallback is the more conservative reading (smaller denominator, larger
    impact), so the basis used is always reported alongside the number and the
    caller can decide how much weight it deserves.
    """
    out: dict[str, Any] = {}
    if not (isinstance(anchor_shares, (int, float)) and anchor_shares > 0):
        out["day30_status"] = "anchor allocation not parsed"
        return out
    released = ANCHOR_TRANCHE_AT_30D * float(anchor_shares)
    out["day30_released_shares"] = round(released)

    float_shares: float | None = None
    basis = ""
    if (isinstance(total_shares, (int, float)) and total_shares > 0
            and isinstance(promoter_locked_shares, (int, float))
            and promoter_locked_shares >= 0
            and total_shares > promoter_locked_shares):
        float_shares = float(total_shares) - float(promoter_locked_shares)
        basis = "total shares less promoter lock-in"
    elif isinstance(offer_shares, (int, float)) and offer_shares > anchor_shares:
        float_shares = float(offer_shares) - float(anchor_shares)
        basis = "listing float (offer less anchor allocation)"

    if not float_shares or float_shares <= 0:
        out["day30_status"] = "tradable float unknown"
        return out

    impact = released / float_shares
    out["day30_float_shares"] = round(float_shares)
    out["day30_impact_pct"] = round(impact * 100.0, 1)
    out["day30_basis"] = basis
    out["day30_status"] = "ok"
    return out


FLIPPER_DOMINANT = 0.50        # flippers are the plurality of the book
FLIPPER_ELEVATED = 0.34
COMPOUNDER_STRONG = 0.50
MIN_NAMES_FOR_RATIO = 3        # below this the extraction is too thin to judge


def profile_anchors(raw_names: list[str] | None,
                    issuer_name: str = "",
                    anchor_shares: float | None = None,
                    total_shares: float | None = None,
                    promoter_locked_shares: float | None = None,
                    offer_shares: float | None = None) -> AnchorProfile:
    names, dropped = normalise_names(raw_names or [], issuer_name)
    prof = AnchorProfile(total=len(names), dropped=dropped)
    if not names:
        prof.overhang_risk = "UNKNOWN"
        prof.flags.append("anchor book could not be read from the RHP")
        return prof

    for n in names:
        getattr(prof, {"compounder": "compounders", "flipper": "flippers",
                       "neutral": "neutral"}[classify_anchor(n)]).append(n)

    prof.compounder_ratio = len(prof.compounders) / prof.total
    prof.flipper_ratio = len(prof.flippers) / prof.total

    if prof.total < MIN_NAMES_FOR_RATIO:
        prof.overhang_risk = "UNKNOWN"
        prof.quality_score = 50.0
        prof.flags.append(
            f"only {prof.total} anchor name(s) parsed - too few to judge the book")
        return prof

    # 50 is a neutral book; patient capital lifts it, fast money sinks it
    score = 50.0 + 55.0 * prof.compounder_ratio - 60.0 * prof.flipper_ratio
    prof.quality_score = clamp(score, 0.0, 100.0)

    # --- day-30 supply shock, judged jointly with who is holding
    prof.day30 = day30_supply_impact(anchor_shares, total_shares,
                                     promoter_locked_shares, offer_shares)
    impact = prof.day30.get("day30_impact_pct")
    on_listing_basis = "listing float" in str(prof.day30.get("day30_basis", ""))
    threshold = (DAY30_IMPACT_THRESHOLD_FLOAT if on_listing_basis
                 else DAY30_IMPACT_THRESHOLD)
    prof.day30["day30_threshold_pct"] = round(threshold * 100.0, 1)
    if (isinstance(impact, (int, float))
            and impact > threshold * 100.0
            and prof.flipper_ratio > DAY30_FLIPPER_THRESHOLD):
        prof.day30["day30_warning"] = True
        prof.flags.insert(0, (
            f"HIGH DAY-30 DUMP RISK: 30-day lock-in release expands free "
            f"float by {impact:.1f}% with high flipper participation "
            f"({prof.flipper_ratio:.0%} of the book)."))
        prof.quality_score = clamp(prof.quality_score - 15.0, 0.0, 100.0)

    if prof.flipper_ratio >= FLIPPER_DOMINANT:
        prof.overhang_risk = "HIGH"
        prof.flags.append(
            f"High risk of 30-day lock-in dump due to low-quality anchor "
            f"allocations ({len(prof.flippers)} of {prof.total} allocations are "
            f"AIFs, long-short sleeves, broker books or family trusts)")
    elif prof.flipper_ratio >= FLIPPER_ELEVATED:
        prof.overhang_risk = "MEDIUM"
        prof.flags.append(
            f"{prof.flipper_ratio:.0%} of the anchor book is tactical money - "
            f"expect supply pressure when 50% unlocks at 30 days")
    elif prof.compounder_ratio >= COMPOUNDER_STRONG:
        prof.overhang_risk = "LOW"
        prof.flags.append(
            f"anchor book is dominated by patient capital "
            f"({len(prof.compounders)} of {prof.total} are sovereign funds, "
            f"large AMCs or insurers)")
    else:
        prof.overhang_risk = "MEDIUM"
        if not prof.compounders:
            prof.flags.append(
                "no recognisable long-only institution in the anchor book")
    return prof
