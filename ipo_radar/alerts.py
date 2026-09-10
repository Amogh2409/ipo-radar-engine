"""
Fire-and-forget webhook alerts.

Design constraint above all others: this must never be able to stall the
daemon. The T-day job runs every sixty seconds inside the decision window and
shares an event loop with subscription polling and PDF extraction, so a
webhook against a dead host must cost that loop nothing.

Three things follow from that.

  * Sending is scheduled, never awaited. `dispatch()` is a synchronous call
    that returns immediately.
  * Every scheduled task is held in a strong reference set. `asyncio` keeps
    only a weak reference to running tasks, so a task nobody holds can be
    garbage-collected mid-flight - the alert simply vanishes, intermittently
    and unreproducibly.
  * Nothing raises. Network errors are logged and dropped; a missed
    notification must never become a failed analysis cycle.

The second constraint is state. A sixty-second loop would otherwise emit the
same "decision window open" alert sixty times an hour, so `SignalWatcher`
converts a stream of T-day snapshots into edges: alerts fire on TRANSITIONS,
not on conditions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import httpx

from .analytics.tday_signals import Action, AlertLevel, Phase, TDaySignal
from .config import AlertConfig

log = logging.getLogger("ipo_radar.alerts")

LEVEL_ORDER = {AlertLevel.INFO.value: 0, AlertLevel.WATCH.value: 1,
               AlertLevel.HIGH.value: 2, AlertLevel.CRITICAL.value: 3}
LEVEL_EMOJI = {"INFO": "🔵", "WATCH": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
LEVEL_COLOUR = {"INFO": 0x3498DB, "WATCH": 0xF1C40F,
                "HIGH": 0xE67E22, "CRITICAL": 0xE74C3C}


class AlertKind(str, Enum):
    PHASE_CHANGE = "phase_change"
    QIB_SPIKE = "qib_spike"
    ACTION_REQUIRED = "action_required"
    ACTION_FLIP = "action_flip"
    WINDOW_CLOSING = "window_closing"


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    level: str
    symbol: str
    title: str
    body: str
    fields: tuple[tuple[str, str], ...] = ()
    dedupe_key: str = ""

    def key(self) -> str:
        return self.dedupe_key or f"{self.symbol}:{self.kind.value}"

    def as_text(self) -> str:
        emoji = LEVEL_EMOJI.get(self.level, "")
        lines = [f"{emoji} *{self.title}*", self.body]
        lines += [f"• {k}: {v}" for k, v in self.fields]
        return "\n".join(x for x in lines if x)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "level": self.level,
                "symbol": self.symbol, "title": self.title, "body": self.body,
                "fields": dict(self.fields)}


# ============================================================== dispatcher
class AlertDispatcher:
    def __init__(self, cfg: AlertConfig,
                 client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self._client = client
        self._owns_client = client is None
        # Strong references. Without this set, asyncio's weak references let
        # the GC collect an in-flight task and the alert silently disappears.
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._last_sent: dict[str, float] = {}
        self._hour_bucket: list[float] = []
        self.sent = 0
        self.failed = 0
        self.suppressed = 0

    # ------------------------------------------------------------- client
    def _get_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("AlertDispatcher is closed")
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.cfg.timeout)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        await self.drain()
        self._closed = True                 # a later dispatch must not revive us
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
            self._owns_client = False       # idempotent: safe to call twice

    # -------------------------------------------------------------- gates
    def _should_send(self, alert: Alert) -> bool:
        if not self.cfg.enabled or not self.cfg.any_configured():
            return False
        if LEVEL_ORDER.get(alert.level, 0) < LEVEL_ORDER.get(self.cfg.min_level, 2):
            return False
        now = time.monotonic()
        last = self._last_sent.get(alert.key())
        if last is not None and (now - last) < self.cfg.min_repeat_seconds:
            self.suppressed += 1
            return False
        self._hour_bucket = [t for t in self._hour_bucket if now - t < 3600.0]
        if len(self._last_sent) > 512:      # bounded: keys are symbol x kind
            cutoff = now - max(self.cfg.min_repeat_seconds * 4, 3600.0)
            self._last_sent = {k: v for k, v in self._last_sent.items()
                               if v >= cutoff}
        if len(self._hour_bucket) >= self.cfg.max_per_hour:
            self.suppressed += 1
            log.warning("alert budget exhausted (%d/hour) - dropping %s",
                        self.cfg.max_per_hour, alert.key())
            return False
        return True

    # ----------------------------------------------------------- dispatch
    def dispatch(self, alert: Alert) -> bool:
        """
        Schedule an alert. Synchronous, returns immediately, never raises.

        Returns True if the send was scheduled, False if it was gated.
        """
        try:
            if self._closed or not self._should_send(alert):
                return False
            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.debug("no running loop - alert %s not sent", alert.key())
            return False
        except Exception as exc:
            log.warning("alert gating failed: %s", exc)
            return False

        now = time.monotonic()
        self._last_sent[alert.key()] = now
        self._hour_bucket.append(now)

        task = loop.create_task(self._send_all(alert), name=f"alert:{alert.key()}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    def dispatch_many(self, alerts: Iterable[Alert]) -> int:
        return sum(1 for a in alerts if self.dispatch(a))

    async def drain(self, timeout: float = 10.0) -> None:
        """
        Let in-flight posts finish on shutdown, but never hang on them.

        Two subtleties. `dispatch()` is synchronous and public, so tasks can
        appear while this coroutine is suspended - hence the re-check loop.
        And a cancelled task must still be awaited, or asyncio emits
        "Task was destroyed but it is pending" at interpreter shutdown.
        """
        deadline = time.monotonic() + timeout
        while self._tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            pending = list(self._tasks)
            await asyncio.wait(pending, timeout=remaining)
            # loop again: dispatch() may have added more while we waited
        if self._tasks:
            stragglers = list(self._tasks)
            for t in stragglers:
                t.cancel()
            # awaiting after cancel is what actually reaps them
            await asyncio.gather(*stragglers, return_exceptions=True)
            log.debug("cancelled %d in-flight alert(s) at shutdown",
                      len(stragglers))

    # --------------------------------------------------------- transports
    async def _send_all(self, alert: Alert) -> None:
        targets: list[tuple[str, str, dict[str, Any]]] = []
        c = self.cfg
        if c.discord_webhook:
            targets.append(("discord", c.discord_webhook, _discord_payload(alert)))
        if c.slack_webhook:
            targets.append(("slack", c.slack_webhook, _slack_payload(alert)))
        if c.telegram_bot_token and c.telegram_chat_id:
            targets.append(("telegram",
                            f"https://api.telegram.org/bot{c.telegram_bot_token}"
                            f"/sendMessage",
                            _telegram_payload(alert, c.telegram_chat_id)))
        if c.generic_webhook:
            targets.append(("generic", c.generic_webhook, alert.to_dict()))

        # Concurrently, not sequentially: a black-holed Discord webhook must
        # not delay Slack by its entire retry budget.
        if targets:
            await asyncio.gather(
                *(self._post(name, url, payload, alert)
                  for name, url, payload in targets),
                return_exceptions=True)

    async def _post(self, name: str, url: str, payload: dict[str, Any],
                    alert: Alert) -> None:
        client = self._get_client()
        attempts = max(1, self.cfg.max_retries)
        for attempt in range(attempts):
            last = attempt + 1 >= attempts
            try:
                r = await client.post(url, json=payload, timeout=self.cfg.timeout)
                if r.status_code == 429:            # provider rate limit
                    retry_after = 2.0
                    try:
                        retry_after = float(r.headers.get("Retry-After", 2.0))
                    except (TypeError, ValueError):
                        pass
                    if last:
                        # `continue` on the final pass fell out of the loop and
                        # returned silently - the alert vanished and the failure
                        # counter never moved.
                        self.failed += 1
                        log.warning("alert via %s rate-limited on the final "
                                    "attempt; dropped: %s", name, alert.title)
                        return
                    await asyncio.sleep(min(retry_after, 10.0))
                    continue
                r.raise_for_status()
                self.sent += 1
                log.info("alert sent via %s: %s", name, alert.title)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if last:
                    self.failed += 1
                    # A missed notification must never become a failed cycle.
                    log.warning("alert via %s failed (%s): %s", name,
                                type(exc).__name__, str(exc)[:160])
                    return
                await asyncio.sleep(1.0 + attempt)

    def stats(self) -> dict[str, int]:
        return {"sent": self.sent, "failed": self.failed,
                "suppressed": self.suppressed, "in_flight": len(self._tasks)}


# ------------------------------------------------------------- payloads
def _discord_payload(alert: Alert) -> dict[str, Any]:
    return {"username": "IPO Radar",
            "embeds": [{"title": alert.title[:250],
                        "description": alert.body[:3900],
                        "color": LEVEL_COLOUR.get(alert.level, 0x95A5A6),
                        "fields": [{"name": k[:250], "value": str(v)[:1000],
                                    "inline": True} for k, v in alert.fields[:8]],
                        "footer": {"text": f"{alert.symbol} · {alert.level}"}}]}


def _slack_payload(alert: Alert) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = [
        {"type": "header",
         "text": {"type": "plain_text",
                  "text": f"{LEVEL_EMOJI.get(alert.level, '')} {alert.title}"[:150]}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": alert.body[:2900]}}]
    if alert.fields:
        blocks.append({"type": "section",
                       "fields": [{"type": "mrkdwn", "text": f"*{k}*\n{v}"[:2000]}
                                  for k, v in alert.fields[:8]]})
    return {"text": f"{alert.title} - {alert.symbol}", "blocks": blocks}


def _telegram_payload(alert: Alert, chat_id: str) -> dict[str, Any]:
    return {"chat_id": chat_id, "parse_mode": "Markdown",
            "disable_web_page_preview": True,
            "text": alert.as_text()[:4000]}


# ========================================================= state watcher
@dataclass
class _SymbolState:
    phase: str | None = None
    action: str | None = None
    spiked: dict[str, bool] = field(default_factory=dict)
    shadow_fired: bool = False


class SignalWatcher:
    """
    Converts a stream of T-day snapshots into edge-triggered alerts.

    The T-day job runs every sixty seconds; conditions like "the decision
    window is open" stay true for hours. Alerting on the condition would emit
    the same message all afternoon, so this tracks the previous state per
    symbol and emits only on a transition.
    """

    # Inside this many minutes of the practical (shadow) cut-off, an APPLY
    # call becomes an instruction rather than an observation.
    SHADOW_WINDOW_MIN = 75.0

    def __init__(self) -> None:
        self._state: dict[str, _SymbolState] = {}

    def observe(self, signal: TDaySignal,
                plan: dict[str, Any] | None = None) -> list[Alert]:
        prev = self._state.setdefault(signal.symbol, _SymbolState())
        out: list[Alert] = []
        clock = signal.clock
        phase = clock.phase.value
        advice = signal.advice
        action = advice.action.value if advice else None

        # --- 1. phase transitions
        if prev.phase != phase:
            if clock.phase is Phase.DECISION_WINDOW:
                out.append(Alert(
                    kind=AlertKind.PHASE_CHANGE, level="HIGH",
                    symbol=signal.symbol,
                    title=f"PHASE CHANGE: Decision Window Open - {signal.symbol}",
                    body=(f"{signal.name} enters its decision window. The book "
                          f"is now informative and there is still time to act."),
                    fields=(("Time to cut-off", clock.countdown()),
                            ("Mandate cut-off",
                             clock.cutoff_at.strftime("%d %b %H:%M IST")
                             if clock.cutoff_at else "—"),
                            ("Current call", action or "—")),
                    dedupe_key=f"{signal.symbol}:phase:{phase}"))
            elif clock.phase is Phase.FINAL_CALL:
                out.append(Alert(
                    kind=AlertKind.WINDOW_CLOSING, level="CRITICAL",
                    symbol=signal.symbol,
                    title=f"FINAL CALL - {signal.symbol}",
                    body=(f"{clock.countdown()} to the UPI mandate cut-off. "
                          f"Bids placed without an accepted mandate are not "
                          f"applications."),
                    fields=(("Call", action or "—"),
                            ("Cut-off",
                             clock.cutoff_at.strftime("%H:%M IST")
                             if clock.cutoff_at else "—")),
                    dedupe_key=f"{signal.symbol}:phase:{phase}"))
            prev.phase = phase

        # --- 2. institutional spike, on the rising edge only
        for cat, spike in (signal.spikes or {}).items():
            was = prev.spiked.get(cat, False)
            if spike.is_spike and not was:
                out.append(Alert(
                    kind=AlertKind.QIB_SPIKE, level="HIGH",
                    symbol=signal.symbol,
                    title=f"{cat} SPIKE DETECTED - {signal.symbol}",
                    body=(f"{cat} demand is accelerating at "
                          f"{spike.current_rate:.2f}x/hour against a "
                          f"{spike.baseline_mean:.2f}x/hour baseline "
                          f"({spike.z_score:.1f} sigma). Institutions are "
                          f"committing late."),
                    fields=(("Z-score", f"{spike.z_score:.2f}"),
                            ("Baseline samples", str(spike.samples)),
                            ("Time to cut-off", clock.countdown())),
                    dedupe_key=f"{signal.symbol}:spike:{cat}"))
            prev.spiked[cat] = bool(spike.is_spike)

        # --- 3. the shadow cut-off: act now or lose the option
        mins = clock.minutes_to_cutoff
        in_shadow = (mins is not None and 0 < mins <= self.SHADOW_WINDOW_MIN)
        if in_shadow and advice and advice.action is Action.APPLY \
                and not prev.shadow_fired:
            fields = [("Category", advice.category or "—"),
                      ("Lots", str(advice.lots or "—")),
                      ("Capital blocked", f"Rs {advice.capital:,.0f}"
                       if advice.capital else "—"),
                      ("P(allotment)", f"{advice.p_allot:.0%}"
                       if advice.p_allot is not None else "—"),
                      ("Time left", clock.countdown())]
            if plan:
                fields.append(("Portfolio plan", str(plan.get("summary", ""))[:180]))
            out.append(Alert(
                kind=AlertKind.ACTION_REQUIRED, level="CRITICAL",
                symbol=signal.symbol,
                title=f"ACTION REQUIRED: Shadow Cut-Off approaching - {signal.symbol}",
                body=(f"Optimal allocation: {advice.category} x "
                      f"{advice.lots} lot(s).\n{advice.bid_price_instruction}\n"
                      f"{advice.deadline_text}"),
                fields=tuple(fields),
                dedupe_key=f"{signal.symbol}:shadow"))
            prev.shadow_fired = True

        # --- 4. the call itself flipped after we had already advised
        if prev.action is not None and action is not None and action != prev.action:
            if clock.phase in (Phase.DECISION_WINDOW, Phase.FINAL_CALL):
                out.append(Alert(
                    kind=AlertKind.ACTION_FLIP, level="HIGH",
                    symbol=signal.symbol,
                    title=f"CALL CHANGED: {prev.action} -> {action} - {signal.symbol}",
                    body=(f"The recommendation moved from {prev.action} to "
                          f"{action} with {clock.countdown()} left."),
                    fields=(("New call", action),
                            ("Reason", "; ".join(advice.rationale[:2])
                             if advice and advice.rationale else "—")),
                    dedupe_key=f"{signal.symbol}:flip:{prev.action}->{action}"))
        prev.action = action
        return out

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._state.clear()
        else:
            self._state.pop(symbol, None)
