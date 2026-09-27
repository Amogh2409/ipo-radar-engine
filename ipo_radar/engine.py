"""
The orchestrator: one object that owns every source, model and store, and
exposes the handful of operations the daemon and CLI actually need.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any

from .analytics.allotment import AllotmentEngine, demand_heat
from .analytics.bayesian_projector import (BayesianProjectionEngine,
                                           demand_weight, dynamic_weights)
from .analytics.listing import ListingModel
from .analytics.portfolio import PortfolioOptimizer
from .analytics.regime import assess as assess_regime
from .analytics.scoring import Scorer
from .analytics.subscription import SubscriptionProjector
from .analytics.tday_signals import TDaySignalEngine
from .analytics.valuation import ValuationEngine
from .brain.analyst import Analyst
from .alerts import AlertDispatcher, SignalWatcher
from .brain.guardrails import FactChecker, GroundTruth
from .brain.ollama_client import OllamaClient
from .config import Settings
from .models import IPO, QIB, RETAIL, Verdict
from .sources.documents import DocumentSource
from .sources.bse import BSESource, DemandFeed
from .sources.chittorgarh import ChittorgarhSource
from .sources.gmp import GMPSource
from .sources.macro import MacroSource
from .sources.news import NewsSource
from .sources.nse import NSESource
from .store import Store
from .util import Http, clamp, now_iso

log = logging.getLogger("ipo_radar.engine")

ACTIVE = ("Active", "Forthcoming")


class Engine:
    def __init__(self, settings: Settings | None = None,
                 store: Store | None = None) -> None:
        self.s = settings or Settings.load()
        self.store = store or Store()
        self.http = Http(self.s.http_timeout, self.s.http_retries)
        self.nse = NSESource(self.http)
        self.gmp = GMPSource(self.http)
        self.news = NewsSource(self.http)
        self.docs = DocumentSource(self.http)
        self.macro = MacroSource(self.http)
        self.bse = BSESource(self.http, getattr(self.s, "bse_endpoint", ""))
        self.cg = ChittorgarhSource(self.http)
        self.feed = DemandFeed(self.http, self.nse, self.store, self.bse, self.cg)
        self.valuation = ValuationEngine()
        self.projector = SubscriptionProjector(self.store)      # legacy fallback
        self.bayes = BayesianProjectionEngine(self.store)
        self.tday = TDaySignalEngine(self.store)
        self.factcheck = FactChecker()
        self.alerts = AlertDispatcher(self.s.alerts)
        self.watcher = SignalWatcher()
        self.allotment = AllotmentEngine(self.s.allotment)
        self.listing = ListingModel(self.store)
        self.scorer = Scorer(self.s.weights)
        self.llm: OllamaClient | None = None
        self.analyst: Analyst | None = None
        self._doc_cache: dict[str, Any] = {}

    async def aclose(self) -> None:
        await self.alerts.aclose()
        await self.http.aclose()
        if self.llm:
            await self.llm.aclose()
        self.store.close()

    async def ensure_llm(self) -> bool:
        if not self.s.llm.enabled:
            return False
        if self.analyst:
            return True
        self.llm = OllamaClient(self.s.llm, self.store)
        if not await self.llm.ensure_ready():
            log.warning("Ollama unavailable - continuing with quant-only analysis")
            self.s.llm.enabled = False
            return False
        self.analyst = Analyst(self.llm, self.store)
        return True

    # ------------------------------------------------------------ intake
    async def refresh_universe(self) -> list[IPO]:
        ipos = await self.nse.list_ipos(self.s.include_sme)
        if not ipos:
            log.warning("NSE returned no issues (last status %s) - falling back "
                        "to Chittorgarh", self.http.last_nse_status)
            ipos = await self.cg.list_ipos(self.s.include_sme, self.store.get_ipos())
        else:
            self._retire_placeholders(ipos)
        for ipo in ipos:
            existing = self.store.get_ipo(ipo.symbol)
            if existing and existing.lot_size:
                # keep enrichment we already paid for
                ipo.lot_size = ipo.lot_size or existing.lot_size
                ipo.face_value = ipo.face_value or existing.face_value
                ipo.registrar = ipo.registrar or existing.registrar
                ipo.meta = {**existing.meta, **ipo.meta}
            elif ipo.meta.get("source") != "chittorgarh":
                try:
                    ipo = await self.nse.enrich(ipo)
                except Exception as exc:
                    log.warning("enrich %s failed: %s", ipo.symbol, exc)
                if not ipo.lot_size:          # NSE listed it but detail was blocked
                    try:
                        ipo = await self.cg.enrich(ipo)
                    except Exception as exc:
                        log.warning("chittorgarh enrich %s failed: %s", ipo.symbol, exc)
            self.store.upsert_ipo(ipo)
        self.store.log_event("discover", f"{len(ipos)} issues in universe")
        return ipos

    def _retire_placeholders(self, nse_ipos: list[IPO]) -> None:
        """A Chittorgarh record stored under a name-derived symbol while NSE
        was blocked would otherwise sit beside the real NSE record forever."""
        from .util import name_similarity
        for old in self.store.get_ipos(ACTIVE):
            if old.meta.get("source") != "chittorgarh":
                continue
            if any(n.symbol != old.symbol and name_similarity(n.name, old.name) >= 0.62
                   for n in nse_ipos):
                old.status = "Superseded"
                self.store.upsert_ipo(old)
                log.info("%s superseded by its NSE record", old.symbol)

    def live_ipos(self) -> list[IPO]:
        return [i for i in self.store.get_ipos() if i.status in ACTIVE]

    async def poll_subscription(self, ipos: list[IPO] | None = None) -> int:
        ipos = ipos or [i for i in self.live_ipos() if i.status == "Active"]
        n, degraded = 0, 0
        for ipo in ipos:
            try:
                result = await self.feed.fetch(ipo)
            except Exception as exc:
                log.warning("subscription %s failed: %s", ipo.symbol, exc)
                continue
            snap = result.snapshot
            # Only persist a genuinely new reading. The cache layer returns a
            # row we already stored, and re-inserting it would fabricate a
            # fresh timestamp on stale data - which the velocity and
            # spike detectors would then read as real movement.
            if snap and snap.categories and result.source != "cache":
                self.store.add_subscription(snap)
                n += 1
            if result.degraded:
                degraded += 1
                self.store.log_event("demand_degraded",
                                     f"{ipo.symbol}: {result.source} - {result.note}",
                                     symbol=ipo.symbol, level="WARNING")
            await asyncio.sleep(0.4)          # be polite to the exchange
        if degraded:
            log.warning("%d of %d books served from a degraded source (%s)",
                        degraded, len(ipos), self.feed.stats)
        return n

    async def poll_gmp(self, ipos: list[IPO] | None = None) -> int:
        ipos = ipos or self.live_ipos()
        if not ipos:
            return 0
        rows = await self.gmp.fetch_all()
        matched = self.gmp.match(ipos, rows)
        for snap in matched.values():
            self.store.add_gmp(snap)
        return len(matched)

    async def poll_news(self, ipos: list[IPO] | None = None) -> int:
        ipos = ipos or self.live_ipos()
        n = 0
        for ipo in ipos:
            try:
                for it in await self.news.fetch(ipo):
                    if self.store.add_news(ipo.symbol, it["published"],
                                           it["title"], it["url"], it["source"]):
                        n += 1
            except Exception as exc:
                log.warning("news %s failed: %s", ipo.symbol, exc)
            await asyncio.sleep(0.3)
        return n

    async def fetch_documents(self, ipo: IPO) -> Any:
        """RHP + anchor letter. Slow, cached on disk, once per IPO."""
        if ipo.symbol in self._doc_cache:
            return self._doc_cache[ipo.symbol]
        try:
            bundle = await self.docs.fetch(ipo)
        except Exception as exc:
            log.warning("documents %s failed: %s", ipo.symbol, exc)
            return None
        self._doc_cache[ipo.symbol] = bundle
        if bundle and bundle.metrics:
            # set_financials is INSERT OR REPLACE, so writing a partial parse
            # straight in would wipe a previously-complete extraction. A run
            # with --no-docs parses only the anchor letter, and that alone once
            # erased every financial figure on record.
            merged = self.store.get_financials(ipo.symbol).get("rhp", {}) or {}
            merged.update({k: v for k, v in bundle.metrics.items()
                           if v is not None})
            self.store.set_financials(ipo.symbol, "rhp", merged)
        return bundle

    # ------------------------------------------------------------ regime
    # Below this the macro backdrop is judged hostile enough that the demand
    # book stops being a reliable guide to the closing book - institutions
    # pull or trim late - so the Bayesian observation noise is widened.
    MACRO_STRESS_THRESHOLD = 40.0

    async def market_regime(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            snap = await self.macro.snapshot()
            history = self.store.macro_history(limit=90)
            self.store.add_macro(snap)
            macro = assess_regime(snap, history)
            out["macro"] = macro.to_dict()
            out["index_change_pct"] = snap.index_change_pct
            out["index"] = "NIFTY 50"

            # The gate the spec asks for: only a genuinely hostile regime
            # widens the posterior, and it widens in proportion to how far
            # below the threshold we are.
            if macro.score < self.MACRO_STRESS_THRESHOLD:
                gated = clamp((self.MACRO_STRESS_THRESHOLD - macro.score)
                              / self.MACRO_STRESS_THRESHOLD, 0.0, 1.0)
                effective = max(gated, macro.stress * gated)
                self.bayes.set_macro_stress(effective)
                out["macro"]["bayes_noise_gain"] = round(effective, 3)
                log.info("macro regime %.0f below %.0f - widening subscription "
                         "posteriors (stress gain %.2f)",
                         macro.score, self.MACRO_STRESS_THRESHOLD, effective)
                self.store.log_event(
                    "macro_stress",
                    f"regime {macro.score:.0f}: {'; '.join(macro.drivers[:2])}",
                    level="WARNING")
            else:
                self.bayes.set_macro_stress(0.0)
                out["macro"]["bayes_noise_gain"] = 0.0
        except Exception as exc:
            log.warning("macro snapshot failed: %s", exc)
        rows = [r for r in self.store.training_rows()
                if r.get("listing_gain_pct") is not None]
        if rows:
            seen, gains = set(), []
            for r in rows:
                if r["symbol"] not in seen:
                    seen.add(r["symbol"])
                    gains.append(r["listing_gain_pct"])
            if gains:
                out["recent_listing_avg_pct"] = round(sum(gains) / len(gains), 1)
                out["recent_listing_n"] = len(gains)
        return out

    # ---------------------------------------------------------- analysis
    async def analyse(self, ipo: IPO, *, use_llm: bool = True,
                      regime: dict[str, Any] | None = None) -> Verdict:
        sub = self.store.latest_subscription(ipo.symbol)
        gmp_snap = self.store.latest_gmp(ipo.symbol)
        gmp_pct = gmp_snap.est_gain_pct if gmp_snap else None
        gmp_trend = self._gmp_trend(ipo.symbol)

        bundle = await self.fetch_documents(ipo)
        fresh = dict(bundle.metrics) if bundle else {}
        sections = bundle.sections if bundle else {}
        # Merge rather than replace. A partial load - the anchor letter parses
        # while the RHP is skipped or still downloading - leaves `fresh`
        # non-empty but without financials, and an "only if empty" fallback
        # would silently discard everything already extracted.
        stored = self.store.get_financials(ipo.symbol).get("rhp", {}) or {}
        metrics = dict(stored)
        metrics.update({k: v for k, v in fresh.items() if v is not None})

        # Fill any core metric the table parsers missed BEFORE valuing the
        # issue - a P/E computed from a recovered EPS is worth far more than
        # a qualitative paragraph appended after the fact.
        if metrics.get("_missing_metrics") and use_llm and await self.ensure_llm():
            raw = sections.get("ratios_raw", "")
            try:
                recovered = await self.analyst.extract_metrics(
                    ipo, raw, metrics["_missing_metrics"])
            except Exception as exc:
                log.warning("%s metric fallback failed: %s", ipo.symbol, exc)
                recovered = {}
            if recovered:
                metrics.update({k: v for k, v in recovered.items()
                                if k != "_source" and metrics.get(k) is None})
                metrics["metrics_source"] = "tables+llm"
                self._derive_multiples(ipo, metrics)
                self.store.set_financials(ipo.symbol, "rhp", metrics)

        # Posterior over final subscription, not a point on a fitted curve.
        # Medians feed the downstream maths; the spread feeds confidence.
        # Structural-break test BEFORE projecting: a dislocation changes how
        # the observations themselves should be weighted, not just the output.
        shock = self.bayes.assess_shock(ipo)
        if sub:
            posteriors = self.bayes.posteriors(ipo, sub, metrics=metrics)
        else:
            # forthcoming issue: no book yet, but capital may still need to be
            # reserved for it, so fall back to the structural prior
            posteriors = self.bayes.prior_projection(ipo, metrics)
        projected = {c: p.median for c, p in posteriors.items()}
        pc = self.bayes.confidence(ipo)

        # Feed the sector classifier the issuer's own description of itself
        # rather than guessing from the company name alone.
        # Populate the fresh/OFS split. Without this, IPO.fresh_issue_cr stays
        # None and three signals silently never fire: the EVA dilution
        # warning, the offer-for-sale structure penalty, and the OFS polarity
        # rule in the guardrails.
        if ipo.fresh_issue_cr is None and bundle and bundle.rhp_pages:
            try:
                from .sources.rhp_tables import (extract_offer_split,
                                                 reconcile_offer_split)
                raw_split = extract_offer_split(bundle.rhp_pages, ipo.cap_price)
                split = reconcile_offer_split(raw_split.get("fresh_issue_cr"),
                                              ipo.issue_size_cr)
                if split.get("fresh_issue_cr") is not None:
                    ipo.fresh_issue_cr = split["fresh_issue_cr"]
                    ipo.ofs_cr = split.get("ofs_cr")
                    ipo.meta["offer_split"] = {**raw_split, **split}
                    self.store.upsert_ipo(ipo)
                    log.info("%s offer split: fresh Rs %.0f Cr, OFS Rs %.0f Cr "
                             "(%.0f%% OFS)", ipo.symbol, ipo.fresh_issue_cr,
                             ipo.ofs_cr or 0.0, split.get("ofs_share_pct") or 0.0)
            except Exception as exc:
                log.warning("%s offer split failed: %s", ipo.symbol, exc)

        sector_context = (sections.get("objects", "") or "")[:1200]
        # The EVA gate needs a real risk-free rate; the live 10y G-Sec yield
        # is exactly that, so Ke moves with the actual bond market.
        rf = ((regime or {}).get("macro") or {}).get("india_10y")
        valuation = self.valuation.analyse(ipo, metrics, context=sector_context,
                                           risk_free_pct=rf)
        # Derived, not NSE's "Total" row - that field is absent more often
        # than not, and both gates below must never read missing as failing.
        total_x = sub.total_subscription() if sub else None
        listing = self.listing.predict(ipo, projected, gmp_pct, valuation,
                                       projection_confidence=pc,
                                       total_x=total_x)

        heat = demand_heat(sub, gmp_pct)
        allot = self.allotment.analyse(
            ipo, sub, projected,
            num_applications=self.s.portfolio.num_pans, gmp_pct=gmp_pct,
            allow_snii=self.s.portfolio.allow_snii,
            allow_bnii=self.s.portfolio.allow_bnii)

        # The day-30 float calculation wants the post-issue share count, which
        # only enterprise_value() derives. Without this hand-off it read None
        # from `metrics` every time and silently fell back to the listing-float
        # basis, so the specified denominator was unreachable in practice.
        for key in ("shares_outstanding", "net_worth_cr"):
            if valuation.get(key) is not None and metrics.get(key) is None:
                metrics[key] = valuation[key]

        dyn_weights, demand_w = dynamic_weights(self.s.weights, ipo)
        verdict = self.scorer.score(
            ipo, valuation=valuation, projected=projected, gmp_pct=gmp_pct,
            gmp_trend=gmp_trend, metrics=metrics, listing=listing,
            regime=regime, llm=None, allotment=allot,
            projection_confidence=pc, weights=dyn_weights, total_x=total_x)
        verdict.allotment = allot

        # --- LLM layer (optional; quant verdict stands on its own)
        if use_llm and await self.ensure_llm() and self.analyst:
            draft = {"symbol": ipo.symbol, "name": ipo.name,
                     "score": verdict.score, "grade": verdict.grade,
                     "valuation": valuation, "projected_subscription": projected,
                     "gmp_pct": gmp_pct, "listing": listing,
                     "reasons": verdict.reasons, "risks": verdict.risks}
            try:
                llm_out = await self.analyst.full_pass(
                    ipo, sections=sections, metrics=metrics,
                    news_items=self.store.recent_news(ipo.symbol), draft=draft)
            except Exception as exc:
                log.warning("%s LLM pass failed: %s", ipo.symbol, exc)
                llm_out = {}
            if llm_out:
                # NOTHING the model says about a number reaches the score
                # without being checked against the number Python computed.
                gt = GroundTruth.build(ipo, metrics, valuation, gmp_pct)
                checked = self.factcheck.check(llm_out, gt)
                guard_summary = checked.summary()
                safe = dict(llm_out)
                safe.update(checked.cleaned)
                if checked.used_fallback:
                    # the model's judgement is discarded, but its critique and
                    # synthesis text are kept out of the score entirely
                    safe.pop("score_adjustment", None)
                    self.store.log_event(
                        "guardrail_fallback",
                        f"{ipo.symbol}: {checked.fallback_reason}",
                        symbol=ipo.symbol, level="WARNING")
                safe["_guardrails"] = guard_summary
                llm_out = safe

                verdict = self.scorer.score(
                    ipo, valuation=valuation, projected=projected,
                    gmp_pct=gmp_pct, gmp_trend=gmp_trend, metrics=metrics,
                    listing=listing, regime=regime, llm=llm_out,
                    allotment=allot, projection_confidence=pc,
                    weights=dyn_weights, total_x=total_x)
                verdict.allotment = allot
                verdict.guardrails = guard_summary
                adj = llm_out.get("score_adjustment")
                if isinstance(adj, (int, float)) and not checked.used_fallback:
                    # Re-apply the terminal veto after the critique nudge - a
                    # positive adjustment must never lift a Value Trap back
                    # into the APPLY band, nor rename a SPECULATIVE FLIP.
                    v = verdict.veto or {}
                    trap = bool(v.get("value_trap"))
                    flip = bool(v.get("speculative_flip"))
                    adjusted = clamp(verdict.score + adj, 0, 100)
                    if trap and not flip:
                        adjusted = min(adjusted, float(v.get("ceiling", 59.0)))
                    verdict.score = round(adjusted, 1)
                    verdict.grade = self.scorer._grade(
                        adjusted, value_trap=trap, speculative_flip=flip)

        verdict.valuation = valuation
        verdict.regime = regime or {}
        if shock.active:
            verdict.regime = {**verdict.regime, "structural_break": shock.to_dict()}
            self.store.log_event("structural_break", shock.reason,
                                 symbol=ipo.symbol, level="WARNING")
        verdict.posteriors = self.bayes.describe(ipo)
        verdict.weights_used = {"demand": round(demand_w, 1),
                                "valuation": round(dyn_weights.valuation, 1),
                                "financials": round(dyn_weights.financials, 1)}
        try:
            signal = self.tday.evaluate(
                ipo, allotment=allot,
                expected_gain_pct=listing.get("expected_gain_pct"),
                score=verdict.score, grade=verdict.grade, veto=verdict.veto,
                capital=self.s.portfolio.total_capital)
            verdict.tday = signal.to_dict()
            for alert in signal.alerts:
                if alert.level.value in ("HIGH", "CRITICAL"):
                    self.store.log_event("tday_alert", alert.message,
                                         symbol=ipo.symbol, level=alert.level.value)
        except Exception as exc:
            log.warning("%s T-day signal failed: %s", ipo.symbol, exc)

        self.store.add_verdict(verdict)
        self.store.add_prediction(
            ipo.symbol, "listing_gain_pct",
            listing.get("expected_gain_pct", 0.0),
            *(listing.get("range") or (None, None)),
            features={"gmp_pct": gmp_pct, "qib_x": projected.get(QIB),
                      "retail_x": projected.get(RETAIL),
                      "pe": valuation.get("pe"),
                      "ronw": valuation.get("ronw_pct"),
                      "issue_size_cr": ipo.issue_size_cr,
                      "score": verdict.score, "heat": round(heat, 3),
                      "projection_confidence": round(pc, 3)},
            model=("quant+llm" if verdict.llm else "quant"))
        return verdict

    @staticmethod
    def _derive_multiples(ipo: IPO, m: dict[str, Any]) -> None:
        """Recompute price-dependent ratios after metrics change."""
        cap, floor = ipo.price_high, ipo.price_low
        if not cap:
            return
        for src, dst in (("eps_basic", "pe_cap_basic"),
                         ("eps_diluted", "pe_cap_diluted")):
            v = m.get(src)
            if isinstance(v, (int, float)) and v > 0:
                m[dst] = round(cap / v, 2)
        nav = m.get("nav_pre_issue")
        if isinstance(nav, (int, float)) and nav > 0:
            m["pb_cap_pre_issue"] = round(cap / nav, 2)
        waca = m.get("waca_1y") or m.get("waca_18m")
        if isinstance(waca, (int, float)) and waca > 0:
            m["insider_exit_multiple"] = round(cap / waca, 2)
        if floor and isinstance(m.get("eps_basic"), (int, float)) and m["eps_basic"] > 0:
            m["pe_floor_basic"] = round(floor / m["eps_basic"], 2)

    def _gmp_trend(self, symbol: str) -> float | None:
        series = self.store.gmp_series(symbol, limit=12)
        if len(series) < 3:
            return None
        head = series[0][1]
        tail = series[-1][1]
        return tail - head

    async def analyse_all(self, use_llm: bool = True,
                          should_stop: Any = None) -> list[Verdict]:
        regime = await self.market_regime()
        out = []
        for ipo in self.live_ipos():
            # a sweep over eleven issues is minutes long; check between each
            # so shutdown does not have to wait for the whole pass
            if should_stop is not None and should_stop():
                log.info("analysis sweep stopping early at %s", ipo.symbol)
                break
            try:
                out.append(await self.analyse(ipo, use_llm=use_llm, regime=regime))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("analysis failed for %s: %s", ipo.symbol, exc)
        out.sort(key=lambda v: v.score, reverse=True)
        return out

    # --------------------------------------------------------- portfolio
    def allocate(self, verdicts: list[Verdict]) -> Any:
        # Date filtering (already-closed, and the forward lookahead horizon)
        # now lives in the optimiser, which needs the calendar anyway.
        candidates = []
        for v in verdicts:
            ipo = self.store.get_ipo(v.symbol)
            if not ipo:
                continue
            candidates.append({
                "ipo": ipo, "verdict": v, "allotment": v.allotment,
                "expected_gain_pct": v.expected_listing_gain_pct,
                "score": v.score, "veto": v.veto})
        pf = self.s.portfolio
        opt = PortfolioOptimizer(
            pf.total_capital, pf.num_pans,
            min_score=pf.min_score,
            objective=pf.objective,
            lookahead_days=pf.lookahead_days,
            settlement_days=pf.asba_settlement_days,
            reserve_min_score=pf.reserve_min_score)
        return opt.optimise(candidates)
