"""Central configuration. Everything tunable lives here or in config.json."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
DB_PATH = DATA_DIR / "ipo_radar.db"
CONFIG_PATH = ROOT / "config.json"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class Cadence:
    """How often each job runs, in seconds. Bid-window jobs auto-accelerate."""
    discover: int = 900           # find new / retired IPOs
    subscription: int = 300       # NSE category-wise demand
    subscription_hot: int = 60    # last 3 hours of the final bid day
    gmp: int = 900                # grey market premium
    news: int = 1800              # news + sentiment
    analyse: int = 1800           # full quant pass
    llm: int = 3600               # deep LLM pass (expensive)
    report: int = 1800            # write reports to disk
    # T-day execution signals are cheap (no LLM, no PDF) so they can run far
    # more often than a full analysis - and must, inside the decision window.
    tday: int = 600
    tday_hot: int = 60            # 14:00 IST onwards on the closing day


@dataclass
class AllotmentParams:
    """
    Priors for allotment math. These are LEARNED over time: every time an
    IPO's real allotment basis lands, calibration.py nudges them and writes
    the updated values back to config.json.
    """
    # Mean lots per retail application. Retail P(allot) ~= avg_lots / sub_x,
    # so this is the single most sensitive knob in the whole engine.
    avg_lots_retail_cold: float = 1.20   # weak demand -> most people apply 1 lot
    avg_lots_retail_hot: float = 1.95    # hot IPO -> many apply near the 2L cap
    # sNII / bNII applicants overwhelmingly bid the minimum ticket.
    snii_min_ticket: float = 200_000.0
    bnii_min_ticket: float = 1_000_001.0
    retail_max_ticket: float = 200_000.0
    # Fraction of QIB reserved for anchors (locked, not available to bid).
    anchor_share_of_qib: float = 0.60


@dataclass
class ScoreWeights:
    """Composite conviction score. Must sum to 100."""
    valuation: float = 18.0
    financials: float = 20.0
    demand: float = 20.0
    gmp: float = 14.0
    structure: float = 10.0
    regime: float = 6.0
    qualitative: float = 12.0

    def total(self) -> float:
        return sum(v for v in asdict(self).values())


@dataclass
class LLMConfig:
    host: str = "http://127.0.0.1:11434"
    # Deep reasoning pass. Auto-falls back to whatever is installed.
    model: str = "ipo-analyst"
    base_model: str = "qwen3:4b"       # what ipo-analyst is built FROM
    fast_model: str = "qwen3:4b"       # triage / sentiment
    temperature: float = 0.2
    num_ctx: int = 8192
    timeout: float = 300.0
    enabled: bool = True
    auto_pull: bool = True             # pull base_model if missing
    auto_build: bool = True            # build ipo-analyst from Modelfile


@dataclass
class PortfolioConfig:
    """Your capital reality — drives the allocation optimizer."""
    total_capital: float = 500_000.0   # rupees available to block in ASBA
    num_pans: int = 1                  # family PANs you can apply from
    allow_snii: bool = True
    allow_bnii: bool = False
    risk_appetite: str = "balanced"    # conservative | balanced | aggressive

    # ------------------------------------------------------------ objective
    # "maximize_roi"              rank plans by return on blocked capital.
    #                             Favours spraying single retail lots, which is
    #                             right when capital is the binding constraint.
    # "maximize_absolute_profit"  rank by expected rupees. Naturally promotes
    #                             sNII on hot issues, where the sheer share
    #                             count outweighs the worse capital efficiency.
    objective: str = "maximize_roi"

    # ------------------------------------------------- temporal optimisation
    # ASBA blocks cash from application until the mandate is released. Under
    # the T+3 listing regime that is the close date plus three business days.
    lookahead_days: int = 14           # horizon for forthcoming issues
    asba_settlement_days: int = 3      # business days from close to unblock
    # An issue scoring above this is worth holding cash back for rather than
    # spending that cash on a mediocre issue closing earlier.
    reserve_min_score: float = 65.0
    # Execution floor. 61 is the APPLY band boundary in analytics/scoring.py,
    # so NEUTRAL issues are starved of capital entirely. Set to 50 to allow
    # NEUTRAL again, but understand what that re-admits: a weakly-subscribed
    # issue has HIGH allotment odds precisely because the market declined it,
    # and expected-value maths alone will rank that above a sought-after book.
    min_score: float = 61.0


@dataclass
class AlertConfig:
    """
    Webhook destinations. Prefer the environment for these - a webhook URL is
    a bearer credential and does not belong in a file you might commit.

        IPO_DISCORD_WEBHOOK, IPO_SLACK_WEBHOOK,
        IPO_TELEGRAM_BOT_TOKEN + IPO_TELEGRAM_CHAT_ID, IPO_WEBHOOK_URL
    """
    enabled: bool = True
    discord_webhook: str = ""
    slack_webhook: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    generic_webhook: str = ""          # any endpoint accepting a JSON POST
    timeout: float = 8.0
    max_retries: int = 2
    # A 60-second decision-window loop must never turn into a 60-second
    # notification loop; the same alert cannot repeat inside this window.
    min_repeat_seconds: float = 900.0
    max_per_hour: int = 40
    min_level: str = "HIGH"            # INFO | WATCH | HIGH | CRITICAL

    def any_configured(self) -> bool:
        return bool(self.discord_webhook or self.slack_webhook
                    or self.generic_webhook
                    or (self.telegram_bot_token and self.telegram_chat_id))


@dataclass
class Settings:
    cadence: Cadence = field(default_factory=Cadence)
    allotment: AllotmentParams = field(default_factory=AllotmentParams)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    llm: LLMConfig = field(default_factory=LLMConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    include_sme: bool = False
    # Pin a known-good BSE demand path if you have one, e.g.
    # "/IPODemandSchedule/w?scripcode={code}". Empty = try the built-in chain.
    bse_endpoint: str = ""
    http_timeout: float = 25.0
    http_retries: int = 3
    log_level: str = "INFO"

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        if CONFIG_PATH.exists():
            try:
                raw = json.loads(CONFIG_PATH.read_text())
            except Exception:
                raw = {}
            for key, val in raw.items():
                if not hasattr(s, key):
                    continue
                cur = getattr(s, key)
                if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
                    for k2, v2 in val.items():
                        if hasattr(cur, k2):
                            setattr(cur, k2, v2)
                else:
                    setattr(s, key, val)
        # env overrides win (handy for cron / containers)
        if os.getenv("IPO_OLLAMA_HOST"):
            s.llm.host = os.environ["IPO_OLLAMA_HOST"]
        if os.getenv("IPO_OLLAMA_MODEL"):
            s.llm.model = os.environ["IPO_OLLAMA_MODEL"]
        if os.getenv("IPO_CAPITAL"):
            s.portfolio.total_capital = float(os.environ["IPO_CAPITAL"])
        if os.getenv("IPO_NO_LLM"):
            s.llm.enabled = False
        for env, attr in (("IPO_DISCORD_WEBHOOK", "discord_webhook"),
                          ("IPO_SLACK_WEBHOOK", "slack_webhook"),
                          ("IPO_TELEGRAM_BOT_TOKEN", "telegram_bot_token"),
                          ("IPO_TELEGRAM_CHAT_ID", "telegram_chat_id"),
                          ("IPO_WEBHOOK_URL", "generic_webhook")):
            if os.getenv(env):
                setattr(s.alerts, attr, os.environ[env])
        if os.getenv("IPO_NO_ALERTS"):
            s.alerts.enabled = False
        if os.getenv("IPO_OBJECTIVE"):
            s.portfolio.objective = os.environ["IPO_OBJECTIVE"]
        return s

    def save(self) -> None:
        """Persist config, with webhook credentials redacted out of the file."""
        data = asdict(self)
        secrets = ("discord_webhook", "slack_webhook", "telegram_bot_token",
                   "telegram_chat_id", "generic_webhook")
        for key in secrets:
            if data.get("alerts", {}).get(key):
                data["alerts"][key] = ""      # keep credentials in the env only
        CONFIG_PATH.write_text(json.dumps(data, indent=2))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
