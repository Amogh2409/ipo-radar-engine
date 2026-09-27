"""
The always-on loop.

Each job has its own cadence and they are checked independently, so a slow
RHP download never blocks a subscription poll. The subscription job
self-accelerates during the closing hours of a bid window, which is exactly
when the book moves and when your decision is still reversible.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Awaitable, Callable

from .analytics.tday_signals import DECISION_WINDOW_START, IST, Phase
from .config import Settings
from .engine import Engine
from .report import console, print_dashboard, write_reports
from .util import now_iso

log = logging.getLogger("ipo_radar.daemon")


@dataclass
class Job:
    name: str
    interval: Callable[[], float]
    run: Callable[[], Awaitable[Any]]
    last: float = 0.0
    runs: int = 0
    errors: int = 0
    last_result: Any = None

    def due(self, now: float) -> bool:
        return (now - self.last) >= self.interval()


class Daemon:
    def __init__(self, engine: Engine | None = None,
                 settings: Settings | None = None,
                 show_dashboard: bool = True) -> None:
        self.s = settings or Settings.load()
        self.engine = engine or Engine(self.s)
        self.show = show_dashboard
        self._stop = asyncio.Event()
        self.jobs: list[Job] = []
        self.verdicts: list[Any] = []
        self.allocation: Any = None
        self.regime: dict[str, Any] = {}
        self.cycle = 0
        self.signals: dict[str, Any] = {}
        self._alerted: set[str] = set()

    # ------------------------------------------------------------ cadence
    def _closing_soon(self) -> bool:
        """True inside the final 3 hours of any live issue's bid window.

        IST, not host-local: the 10:00-17:00 bounds below are Indian trading
        hours, so comparing them against a host clock in another zone would
        accelerate or suppress polling at exactly the wrong moment.
        """
        now = datetime.now(IST)
        for ipo in self.engine.live_ipos():
            _, close = ipo.days()
            if close == now.date() and 10 <= now.hour < 17:
                return (17 - now.hour) <= 3
        return False

    def _sub_interval(self) -> float:
        c = self.s.cadence
        return float(c.subscription_hot if self._closing_soon() else c.subscription)

    def _in_decision_window(self) -> bool:
        """14:00 IST onwards on any live issue's closing day."""
        now = datetime.now(IST)
        for ipo in self.engine.live_ipos():
            _, close = ipo.days()
            if close == now.date() and now.time() >= DECISION_WINDOW_START:
                return True
        return False

    def _tday_interval(self) -> float:
        c = self.s.cadence
        return float(c.tday_hot if self._in_decision_window() else c.tday)

    # --------------------------------------------------------------- jobs
    def build_jobs(self) -> None:
        c = self.s.cadence
        self.jobs = [
            Job("discover", lambda: float(c.discover), self._job_discover),
            Job("subscription", self._sub_interval, self._job_subscription),
            Job("gmp", lambda: float(c.gmp), self._job_gmp),
            Job("news", lambda: float(c.news), self._job_news),
            Job("analyse", lambda: float(c.analyse), self._job_analyse),
            Job("tday", self._tday_interval, self._job_tday),
            Job("report", lambda: float(c.report), self._job_report),
        ]

    async def _job_discover(self) -> str:
        ipos = await self.engine.refresh_universe()
        self.regime = await self.engine.market_regime()
        listed = await self.engine.poll_listings()
        return f"{len(ipos)} issues, {listed} listed prices"

    async def _job_subscription(self) -> str:
        n = await self.engine.poll_subscription()
        return f"{n} books updated"

    async def _job_gmp(self) -> str:
        return f"{await self.engine.poll_gmp()} GMP matched"

    async def _job_news(self) -> str:
        return f"{await self.engine.poll_news()} new headlines"

    async def _job_analyse(self) -> str:
        self.cycle += 1
        # the LLM pass is expensive; run it every Nth quant cycle
        every = max(1, round(self.s.cadence.llm / max(self.s.cadence.analyse, 1)))
        use_llm = self.s.llm.enabled and (self.cycle % every == 1 or every == 1)
        self.verdicts = await self.engine.analyse_all(
            use_llm=use_llm, should_stop=self._stop.is_set)
        self.allocation = self.engine.allocate(self.verdicts)
        return f"{len(self.verdicts)} scored{' (+LLM)' if use_llm else ''}"

    async def _job_tday(self) -> str:
        """
        Execution signals only - clock, spike, advice. Deliberately does no
        LLM or PDF work so it can run every minute in the closing window,
        which is exactly when a 30-minute analysis cadence is useless.
        """
        fired = pushed = 0
        for verdict in self.verdicts:
            ipo = self.engine.store.get_ipo(verdict.symbol)
            if not ipo:
                continue
            try:
                sig = self.engine.tday.evaluate(
                    ipo, allotment=verdict.allotment,
                    expected_gain_pct=verdict.expected_listing_gain_pct,
                    score=verdict.score, grade=verdict.grade,
                    veto=verdict.veto,
                    capital=self.s.portfolio.total_capital)
            except Exception as exc:
                log.warning("tday %s failed: %s", verdict.symbol, exc)
                continue
            self.signals[verdict.symbol] = sig
            verdict.tday = sig.to_dict()

            # Edge-triggered push notifications. dispatch() is synchronous and
            # schedules the POST, so a dead webhook cannot stall this loop.
            try:
                plan = None
                if self.allocation and self.allocation.applications:
                    mine = [a for a in self.allocation.applications
                            if a.symbol == verdict.symbol]
                    if mine:
                        plan = {"summary": "; ".join(
                            f"PAN#{a.pan_index} {a.category} x{a.lots} "
                            f"(Rs {a.capital:,.0f})" for a in mine[:3])}
                pushed += self.engine.alerts.dispatch_many(
                    self.engine.watcher.observe(sig, plan))
            except Exception as exc:
                log.warning("alert dispatch for %s failed: %s", verdict.symbol, exc)
            for alert in sig.alerts:
                if alert.level.value not in ("HIGH", "CRITICAL"):
                    continue
                key = f"{verdict.symbol}:{alert.kind}:{sig.clock.phase.value}"
                if key in self._alerted:
                    continue
                self._alerted.add(key)
                fired += 1
                style = ("bold red" if alert.level.value == "CRITICAL"
                         else "bold yellow")
                console.print(f"[{style}]{alert.level.value:<8}[/{style}] "
                              f"{alert.message}")
                self.engine.store.log_event("tday_alert", alert.message,
                                            symbol=verdict.symbol,
                                            level=alert.level.value)
        live = sum(1 for s in self.signals.values()
                   if s.clock.phase in (Phase.DECISION_WINDOW, Phase.FINAL_CALL))
        return (f"{len(self.signals)} tracked, {live} in window, "
                f"{fired} new alerts, {pushed} pushed")

    async def _job_report(self) -> str:
        if not self.verdicts:
            return "nothing to write"
        paths = write_reports(self.engine.store, self.verdicts, self.allocation)
        return f"{len(paths)} files"

    # --------------------------------------------------------------- loop
    async def run(self) -> None:
        self.build_jobs()
        self._install_signals()
        console.print("[bold cyan]IPO Radar[/bold cyan] starting — "
                      "Ctrl-C to stop\n")
        await self.engine.ensure_llm()

        # prime everything once so the first dashboard is not empty
        for job in self.jobs:
            if self._stop.is_set():
                break
            await self._run_job(job)

        if self.show and self.verdicts:
            print_dashboard(self.engine.store, self.verdicts,
                            self.allocation, self.regime, self.signals)

        while not self._stop.is_set():
            loop_now = asyncio.get_event_loop().time()
            ran = False
            for job in self.jobs:
                if self._stop.is_set():
                    break
                if job.due(loop_now):
                    await self._run_job(job)
                    ran = True
            if ran and self.show and self.verdicts:
                print_dashboard(self.engine.store, self.verdicts,
                                self.allocation, self.regime, self.signals)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

        console.print("\n[dim]shutting down…[/dim]")
        # tell any in-flight PDF worker to stop so the executor can drain
        self.engine.docs.abort.set()
        # let queued notifications land, but never hang shutdown on them
        await self.engine.alerts.drain(timeout=5.0)
        await self.engine.aclose()

    async def _run_job(self, job: Job) -> None:
        """
        Run one job, but stay interruptible while it does.

        A full analysis pass takes minutes (PDF extraction, LLM passes). If
        shutdown simply awaited the current job, Ctrl-C would appear to hang
        for the length of that pass. Instead the job races the stop event and
        is cancelled if the operator wins.
        """
        job.last = asyncio.get_event_loop().time()
        task = asyncio.create_task(job.run(), name=f"job:{job.name}")
        stopper = asyncio.create_task(self._stop.wait(), name="stop")
        try:
            done, _ = await asyncio.wait({task, stopper},
                                         return_when=asyncio.FIRST_COMPLETED)
            if task not in done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                log.info("%-13s cancelled on shutdown", job.name)
                return
            job.last_result = task.result()
            job.runs += 1
            log.info("%-13s %s", job.name, job.last_result)
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            job.errors += 1
            log.exception("job %s failed: %s", job.name, exc)
            self.engine.store.log_event("job_error", f"{job.name}: {exc}",
                                        level="ERROR")
        finally:
            stopper.cancel()

    def _install_signals(self) -> None:
        loop = asyncio.get_event_loop()
        def _request_stop() -> None:
            self._stop.set()
            self.engine.docs.abort.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except NotImplementedError:      # not available on some platforms
                signal.signal(sig, lambda *_: _request_stop())

    def status(self) -> dict[str, Any]:
        return {"ts": now_iso(), "cycle": self.cycle,
                "jobs": [{"name": j.name, "runs": j.runs, "errors": j.errors,
                          "last": j.last_result} for j in self.jobs]}
