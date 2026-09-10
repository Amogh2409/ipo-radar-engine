"""
The analyst loop: what the local model actually does with the data.

Four passes, deliberately separated so each one is a small, checkable job for
a small model rather than one giant "analyse this IPO" prompt:

  1. business   - read the RHP's objects/risk sections, produce structure
  2. news       - score recent headlines for relevance and sentiment
  3. critique   - argue AGAINST the draft verdict (models are sycophantic by
                  default; asking for the bear case explicitly is the cheapest
                  correction available)
  4. synthesис  - one paragraph a human can act on

Memory: past IPOs with known outcomes are retrieved and injected as few-shot
context, so the analyst's priors come from this market rather than from
whatever was in the base model's training data.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..models import IPO
from ..store import Store
from ..util import clamp
from .ollama_client import OllamaClient
from .schemas import (BUSINESS_ANALYSIS, CRITIQUE, METRIC_EXTRACTION,
                      NEWS_SENTIMENT, SYNTHESIS)

log = logging.getLogger("ipo_radar.analyst")

MAX_CHARS = 7000        # keep well inside num_ctx after the system prompt


# Small models routinely file a favourable metric under red_flags and then
# annotate it as fine ("...which is reassuring"). Rather than fight the model
# size, drop entries that disown themselves.
_SELF_NEGATING = (
    "not a red flag", "not a concern", "reassuring", "is a positive",
    "which is good", "is good", "not necessarily negative", "positive sign",
    "no red flag", "favourable", "favorable",
)


def _clean_flags(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        text = str(it).strip()
        if not text:
            continue
        low = text.lower()
        if any(p in low for p in _SELF_NEGATING):
            continue
        out.append(text)
    return out


def _norm_score(v: Any, default: float = 50.0) -> float:
    """Models answer 0-100 or 0-1 unpredictably. Accept both."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if 0.0 <= f <= 1.0:
        f *= 100.0
    return clamp(f, 0.0, 100.0)


class Analyst:
    def __init__(self, client: OllamaClient, store: Store) -> None:
        self.llm = client
        self.store = store

    # ----------------------------------------------------------- memory
    def _memory_block(self, limit: int = 5) -> str:
        """Few-shot from our OWN observed outcomes, not the model's priors."""
        rows = self.store.training_rows()
        if not rows:
            return ""
        seen, lines = set(), []
        for r in rows:
            sym = r["symbol"]
            if sym in seen:
                continue
            seen.add(sym)
            f = r.get("features") or {}
            lines.append(
                f"- {sym}: QIB {f.get('qib_x', '?')}x, retail {f.get('retail_x', '?')}x, "
                f"GMP {f.get('gmp_pct', '?')}%, P/E {f.get('pe', '?')} "
                f"-> listed {r['listing_gain_pct']:+.1f}%")
            if len(lines) >= limit:
                break
        if not lines:
            return ""
        return ("\nOutcomes actually observed in this market recently:\n"
                + "\n".join(lines) + "\n")

    # ------------------------------------------------- 0. metric fallback
    async def extract_metrics(self, ipo: IPO, ratios_text: str,
                              missing: list[str]) -> dict[str, Any]:
        """
        Read the ratios pages when the table parsers could not.

        RHP layouts vary without limit, so rather than chase every variant with
        another regex, the deterministic parsers handle the common shapes and
        the model reads whatever is left. Results are range-checked before use:
        a small model asked for a number will always produce one.
        """
        if not ratios_text.strip():
            return {}
        prompt = f"""Extract the reported figures for {ipo.name} from this excerpt of
its Red Herring Prospectus. These specifically could not be parsed
automatically: {', '.join(missing)}.

Rules:
- Report the MOST RECENT fiscal year's figure.
- pat_latest_fy, ebitda_latest_fy, total_borrowings and cash_and_equivalents
  are ABSOLUTE amounts. Report them exactly as printed and set amounts_unit
  to the unit the table declares ("million", "crore" or "lakh"). Do not
  convert between units yourself.
- These are the ISSUER's own figures, not a listed peer's. Peer rows belong
  only in peer_pe_average.
- "[<bullet>]" means the figure is not yet fixed. Return null for it.
- Return null for anything not clearly stated. Do not calculate, infer or
  estimate. A null is a correct answer.
- List in `found` only the fields you actually read off the text.

{ratios_text[:MAX_CHARS]}"""
        out = await self.llm.structured(
            prompt, METRIC_EXTRACTION, cache_key=f"metrics:{ipo.symbol}",
            cache_ttl=86400.0, temperature=0.0)
        if not out:
            return {}

        # A model asked for a number will always give one; bound it.
        def ok(v: Any, lo: float, hi: float) -> float | None:
            if not isinstance(v, (int, float)):
                return None
            v = float(v)
            if not (lo <= abs(v) <= hi):
                return None
            if 1990 <= v <= 2100 and v.is_integer():   # a fiscal year, not a value
                return None
            return v

        cap = ipo.cap_price or 0.0
        cleaned: dict[str, Any] = {}
        eps = ok(out.get("eps_basic"), 0.01, 100000)
        if eps is not None and (not cap or eps < cap * 3):
            cleaned["eps_basic"] = eps
            cleaned["eps_diluted"] = ok(out.get("eps_diluted"), 0.01, 100000) or eps
            if out.get("eps_fiscal_year"):
                cleaned["eps_fy"] = str(out["eps_fiscal_year"])
        ronw = out.get("ronw_pct")
        if isinstance(ronw, (int, float)) and -200 <= ronw <= 300:
            cleaned["ronw_pct"] = float(ronw)
        nav = ok(out.get("nav_per_share"), 0.1, 200000)
        if nav is not None:
            cleaned["nav_pre_issue"] = nav
        peer = ok(out.get("peer_pe_average"), 0.5, 300)
        if peer is not None:
            cleaned["peer_pe_avg"] = peer

        # --- enterprise-value inputs, with the unit the RHP declared
        unit = str(out.get("amounts_unit") or "").lower()
        scale = {"million": 0.1, "crore": 1.0, "lakh": 0.01}.get(unit)
        for src, dst in (("pat_latest_fy", "kpi_pat"),
                         ("ebitda_latest_fy", "kpi_ebitda"),
                         ("total_borrowings", "kpi_borrowings"),
                         ("cash_and_equivalents", "kpi_cash")):
            v = out.get(src)
            if isinstance(v, (int, float)) and abs(v) < 1e9:
                cleaned[dst] = float(v)
                if scale is not None:
                    cleaned[f"{dst}_scale"] = scale
        if cleaned:
            cleaned["_source"] = "llm_fallback"
            log.info("%s: LLM recovered %s", ipo.symbol,
                     ", ".join(k for k in cleaned if not k.startswith("_")))
        return cleaned

    # --------------------------------------------------------- 1. business
    async def business(self, ipo: IPO, sections: dict[str, str],
                       metrics: dict[str, Any]) -> dict[str, Any] | None:
        objects = (sections.get("objects") or "")[:MAX_CHARS // 2]
        risks = (sections.get("risk_factors") or "")[:MAX_CHARS // 2]
        if not objects and not risks:
            return None
        facts = {k: metrics.get(k) for k in
                 ("eps_by_fy", "ronw_pct", "ronw_weighted_pct", "nav_pre_issue",
                  "pe_cap_diluted", "pb_cap_pre_issue", "insider_exit_multiple",
                  "peer_note", "anchor_marquee_hits", "anchor_investor_count")
                 if metrics.get(k) is not None}
        size = (f"\nISSUE SIZE: Rs {ipo.issue_size_cr:,.0f} crore"
                if ipo.issue_size_cr else "")
        prompt = f"""Analyse this Indian IPO for a retail investor.

COMPANY: {ipo.name} ({ipo.symbol})
PRICE BAND: Rs {ipo.price_low}-{ipo.price_high}, lot {ipo.lot_size} shares{size}

NUMBERS EXTRACTED FROM THE RHP:
{json.dumps(facts, indent=2, default=str)}

OBJECTS OF THE OFFER (verbatim from the RHP):
{objects}

RISK FACTORS (verbatim from the RHP):
{risks}

Give qualitative_score on a 0-100 scale where 50 is an average mainboard IPO.
Base every claim on the text above. Distinguish boilerplate risk factors from
the two or three that genuinely matter.

litigation_severity: judge the OUTSTANDING LEGAL PROCEEDINGS disclosed above.
  LOW = routine tax and consumer matters
  MEDIUM = material claims, or regulatory action against the company
  CRITICAL = proceedings that could impair the business, or criminal matters
             against promoters or directors
governance_flag: true only for a concrete governance concern evidenced in the
  text - related-party dependence, auditor qualifications, promoter pledging,
  or regulatory censure. Absence of discussion is not evidence; default false."""
        out = await self.llm.structured(
            prompt, BUSINESS_ANALYSIS, cache_key=f"business:{ipo.symbol}",
            cache_ttl=86400.0)
        if out:
            out["qualitative_score"] = _norm_score(out.get("qualitative_score"))
            out["red_flags"] = _clean_flags(out.get("red_flags"))
        return out

    # ------------------------------------------------------------ 2. news
    async def news(self, ipo: IPO, items: list[dict[str, Any]]
                   ) -> dict[str, Any] | None:
        if not items:
            return None
        listed = "\n".join(f"{i}. {it['title']}" for i, it in enumerate(items[:15]))
        prompt = f"""Headlines mentioning {ipo.name} IPO. For each, judge whether it is
genuinely about THIS company's IPO (relevant) and its sentiment from -1
(clearly negative) to +1 (clearly positive). Generic market roundups that
merely list the IPO are relevance=false.

{listed}

Then give an overall_sentiment weighted by relevance, and a one-sentence summary."""
        out = await self.llm.structured(
            prompt, NEWS_SENTIMENT, model=self.llm.cfg.fast_model,
            cache_key=f"news:{ipo.symbol}:{len(items)}", cache_ttl=3600.0)
        return out

    # -------------------------------------------------------- 3. critique
    async def critique(self, ipo: IPO, draft: dict[str, Any]
                       ) -> dict[str, Any] | None:
        prompt = f"""Here is a draft verdict on the {ipo.name} IPO. Your job is to argue
AGAINST it. Find what the analysis is glossing over. Be specific and
quantitative. If the score is genuinely fair, say so rather than inventing
objections.

{json.dumps(draft, indent=2, default=str)[:MAX_CHARS]}
{self._memory_block()}
suggested_score_adjustment is how many points the composite score should move
(negative to lower it)."""
        return await self.llm.structured(
            prompt, CRITIQUE, cache_key=f"critique:{ipo.symbol}:"
                                        f"{round(draft.get('score', 0))}",
            cache_ttl=7200.0, temperature=0.35)

    # ------------------------------------------------------- 4. synthesis
    async def synthesise(self, ipo: IPO, bundle: dict[str, Any]
                         ) -> dict[str, Any] | None:
        prompt = f"""Write the final call on this IPO for a retail investor who must
decide today whether to apply, and in which category.

{json.dumps(bundle, indent=2, default=str)[:MAX_CHARS]}
{self._memory_block()}
Remember: when retail is heavily oversubscribed every winner gets exactly one
lot, so "retail max lots" is only sensible when retail demand is light.
Be decisive but honest about uncertainty."""
        return await self.llm.structured(
            prompt, SYNTHESIS, cache_key=f"synth:{ipo.symbol}:"
                                         f"{round(bundle.get('score', 0))}",
            cache_ttl=3600.0)

    # ------------------------------------------------------------- driver
    async def full_pass(self, ipo: IPO, *, sections: dict[str, str],
                        metrics: dict[str, Any], news_items: list[dict[str, Any]],
                        draft: dict[str, Any]) -> dict[str, Any]:
        """Runs every pass that has enough input, tolerating partial failure."""
        result: dict[str, Any] = {}
        try:
            biz = await self.business(ipo, sections, metrics)
            if biz:
                result.update(biz)
        except Exception as exc:
            log.warning("%s business pass failed: %s", ipo.symbol, exc)
        try:
            nws = await self.news(ipo, news_items)
            if nws:
                result["news"] = nws
                result["news_sentiment"] = nws.get("overall_sentiment")
        except Exception as exc:
            log.warning("%s news pass failed: %s", ipo.symbol, exc)

        merged = {**draft, "qualitative": {k: v for k, v in result.items()
                                           if k != "news"}}
        try:
            crit = await self.critique(ipo, merged)
            if crit:
                result["critique"] = crit
                adj = crit.get("suggested_score_adjustment")
                if isinstance(adj, (int, float)):
                    result["score_adjustment"] = clamp(float(adj), -20, 10)
        except Exception as exc:
            log.warning("%s critique failed: %s", ipo.symbol, exc)
        try:
            syn = await self.synthesise(ipo, {**merged, "critique":
                                              result.get("critique")})
            if syn:
                result["synthesis"] = syn
        except Exception as exc:
            log.warning("%s synthesis failed: %s", ipo.symbol, exc)
        return result
