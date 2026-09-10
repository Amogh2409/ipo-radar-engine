# IPO Radar — handoff notes

Everything needed to rerun the engine and backtest its forecasts. Read this
before `README.md`; the README is the design reference, this is the runbook.

## What is in the archive

```
ipo_radar/            the package (26 modules)
run.py                engine entry point
generate_briefing.py  weekly PDF briefing compiler (standalone)
generate_tactical_memo.py  dense 1-2 page execution memo (standalone)
tests/                regression suite (stdlib only, no pytest required)
requirements.txt      httpx, rich, pypdf  (pydantic optional)
Modelfile             builds the local `ipo-analyst` model
config.example.json   every tunable, at its default
data/ipo_radar.db     SQLite with the captured history  <-- the backtest material
reports/              last rendered board and per-IPO write-ups
README.md             design notes and the reasoning behind each model
```

**Not included in the zip archive: `data/documents/`** — ~125 MB of cached
RHP and anchor PDFs. They re-download automatically on first run (roughly
45–90 s per issue) and cache to disk permanently. Excluded purely for archive
size; nothing depends on shipping them.

**The git repository does include them,** along with `data/ipo_radar.db`. A
clone is therefore self-contained and reproducible: the exchange URLs the
downloader uses expire, so a fresh clone months from now would otherwise
analyse a different document set — or none — and quietly produce different
numbers from the same code.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pydantic          # optional: stricter guardrail validation

ollama serve &                # optional; without it use --no-llm
python3 run.py once --no-docs  # fastest first run, skips RHP downloads
```

Python 3.11+. Built and tested on 3.14.7.

Ollama is optional. Every number is computed in Python; the local model only
contributes qualitative text and a metric-recovery fallback, and it is capped
at ~12% of the composite score. `--no-llm` produces a complete analysis.

## Running

| Command | Purpose |
|---|---|
| `python3 run.py once` | one full cycle, prints the board, writes `reports/` |
| `python3 run.py run` | the daemon (self-paces; 60 s cadence in the closing window) |
| `python3 run.py analyse SYM --markdown` | one issue, full write-up |
| `python3 run.py allot SYM --applications 4` | allotment odds across categories |
| `python3 run.py report` | re-render from the DB, no network |
| `python3 run.py config` | effective configuration |

Useful flags: `--no-llm`, `--no-docs`, `--capital 500000`, `--pans 4`,
`--objective maximize_absolute_profit`, `--log DEBUG`.

## Weekly briefing PDF

`generate_briefing.py` compiles `reports/` into one PDF and appends a system
health and calibration audit read straight from the database. It is standalone
- imports nothing from the daemon, touches no network, and opens the database
**read-only**, so it is safe to run against a live engine mid-cycle.

```bash
python3 generate_briefing.py                      # -> briefings/IPO_Intelligence_Briefing_YYYY-MM-DD.pdf
python3 generate_briefing.py --days 14 --keep-html
python3 generate_briefing.py --engine reportlab   # force a specific engine
```

**No install is required.** It tries four PDF engines in order and uses the
first that works, all consuming the same HTML:

| Engine | Needs | Notes |
|---|---|---|
| `weasyprint` | `pip install weasyprint` + cairo/pango | best CSS fidelity |
| `chrome` | nothing, if Chrome/Chromium/Edge is installed | renders like the browser |
| `wkhtmltopdf` | `brew install --cask wkhtmltopdf` + `pip install pdfkit` | |
| `reportlab` | usually already present | pure Python; always produces a PDF |

The HTML is written alongside the PDF, so if every engine is missing you can
open it and print from a browser. Exit codes: `0` success, `1` HTML only (no
engine), `2` nothing to compile.

Two rendering notes worth knowing. Headless Chrome often writes the PDF and
then lingers instead of exiting, so the script polls for the artefact and
terminates the browser rather than waiting on the process — that alone took a
run from over two minutes to under three seconds. And the reportlab path
checks that a font actually contains U+20B9 before using it: Arial Unicode
loads fine but predates the 2010 rupee sign, so every amount rendered as a
tofu box while text extraction still reported the character as present.

## Tactical action memo

Where the weekly briefing is a document you read, the memo is a sheet you act
from. No history, no narrative - only issues you can still bid on and the
numbers that change what you do in the next few hours.

```bash
python3 run.py memo                     # -> briefings/Tactical_Action_Memo_YYYY-MM-DD.pdf
python3 run.py memo --hours 48
python3 run.py --capital 1000000 --objective maximize_absolute_profit memo
python3 generate_tactical_memo.py --engine reportlab   # same thing, standalone
```

Scope is every `Active`/`Forthcoming` issue whose close date falls inside the
horizon (default 96h). The hour figure becomes whole days by floor division,
so `--hours 1` means today only and `--hours 96` covers the next four close
dates. Closed issues are excluded.

**Page 1 - the execution desk.** Capital tiles (budget, committed today,
liquid reserve, peak block, expected profit), an ACT NOW table for issues
closing today or tomorrow (cut-off time, category, lots, price rule, odds,
expected gain, capital, net profit), and the deployment roadmap showing when
blocked ASBA funds unlock and what they can then fund.

**Page 2 - forensic due diligence.** A red hard-veto box for Value Traps and
accruals warnings explaining why capital must not go there, the fundamental
scorecard (P/E vs benchmark, EV/EBITDA for capital-intensive names, RoNW, Ke,
% OFS), and the anchor lock-in table with day-30 float expansion against its
basis-specific trigger.

It reuses the rendering chain from `generate_briefing.py`, so it adds no
dependency. The two-page budget is **enforced, not hoped for**: the memo is
rendered, its page count measured, and re-rendered at a tighter type scale
until it fits. Verified holding at 2 pages with all 11 live issues in scope.

Note the price-rule column encodes a SEBI rule worth remembering: cut-off
pricing is a retail and employee privilege. An NII bid must name a price, and
naming below the cap risks exclusion if the issue prices at the cap - which
oversubscribed issues almost always do.

## The backtest loop

The engine records every forecast with the features behind it, so it can be
scored against reality afterwards.

```bash
# 1. once an IPO lists, record what actually happened
python3 run.py outcome RENTOMOJO --listing-price 465 --allotment-rate 0.33

# 2. refit the models against accumulated outcomes
python3 run.py calibrate
```

`calibrate` reports `self_score` — median and mean absolute error of the
listing-gain forecast in percentage points — and refits three things once
there is enough data:

| Parameter | Needs | Method |
|---|---|---|
| GMP realisation factor | 12 outcomes | robust median of actual/GMP |
| Projection exponents (α per category) | 4 completed IPOs | median of `log(x/X)/log(t)` |
| Retail `k` (mean lots per application) | 6 allotment results | median by demand regime |

Below those thresholds it reports `insufficient` and changes nothing. That is
deliberate: refitting nine coefficients on six observations is worse than the
prior.

`python3 run.py export-finetune` writes a chat-format JSONL corpus from the
same (features → outcome) pairs.

## What is already in the DB

| Table | Rows | Use |
|---|---|---|
| `subscription` | 1,315 | category-wise book, time series — drives projection backtests |
| `gmp` | 303 | grey-market premium history |
| `verdicts` | 284 | every scored verdict with full payload |
| `predictions` | 284 | forecasts with the features behind them |
| `macro` | 35 | India yield curve snapshots |
| `financials` | 11 | RHP-extracted metrics per issuer |
| `outcomes` | 0 | **empty — this is what the backtester fills** |

The 12 issues captured are the September 2026 mainboard cohort. `outcomes` is
empty because none had listed at capture time; populating it is the first step
of any backtest.

## Timezone, calendar and settlement

Everything is denominated in **Indian market time**. `ipo_radar/util.py` holds
the single `IST` definition and `today_ist()`; the optimizer plans against
that, never `date.today()`. This matters on a UTC host: between 18:30 and
23:59 UTC it is already tomorrow in India, and planning against the host date
there would fund an issue whose bidding had already closed.

`analytics/calendar_in.py` resolves clearing holidays from
`data/holidays.json` (operator override, always wins), a seeded festival
table, and computable rules. **It is deliberately uncached** — drop an ad-hoc
clearing holiday into the JSON and a running daemon picks it up on the next
call, no restart needed.

A year with no seed entry falls back to rules only, which errs toward an
**early** unlock — the unsafe direction, since a shorter blocked window lets
the optimizer commit capital that has not cleared. Such years are therefore
padded by an extra clearing day, and `holidays_seeded(year)` exposes the
condition. **Verify the seed table against the annual NSE/RBI circular**;
lunar-calendar festivals move.

The reports show a **Deployable** date, not the mandate release date. Sponsor
banks push ASBA revocations in an evening batch, so money released on day T
cannot back a bid for an issue closing on day T — the UPI cut-off is 17:00,
hours earlier.

## The Winner's Curse defence

The single most important behavioural fix in this build, and the one a
backtester is most likely to trip over when comparing against older runs.

**The failure.** Allotment odds are highest exactly when the market has
declined an issue. The optimiser ranks by expected profit — odds x gain — so
78.8% of a small gain beat 6% of a large one, and capital was routed to the
weakest book on the board. Observed live: Prasol Chemicals opened its final
day at QIB 0.05x and was funded ahead of a 215x-subscribed peer.

**Why the pre-existing guard missed it.** `listing.py` already collapsed its
estimate when *every* category was under 1.0x. The real book never qualified:

```
QIB 0.05x   NII 0.66x   RETAIL 1.14x   (derived total 0.72x)
```

Retail alone was above 1.0x — the loudest and least informed category — so
`max(...) < 1.0` was False and the guard stayed silent while informed money
refused the issue.

**Four layered defences**, deliberately independent so no single threshold is
load-bearing:

1. `analytics/listing.py` — `adverse_selection()`. Past 75% of the bid window,
   a *present* Total or QIB figure below 1.0x forces `expected_gain_pct` to
   `min(0, central)` and `p_positive` to 0. A **missing** figure never
   condemns an issue: NSE omits the Total row on most issues, including fully
   subscribed ones, so `SubscriptionSnapshot.total_subscription()` derives it
   from category bid/offered and unknown stays unknown.
2. `analytics/scoring.py` — `_demand_score()` caps the demand factor at 10.0
   on the same condition and surfaces it as a **risk**, not a reason. The cap
   is re-applied *after* confidence damping: damping exists to soften
   uncertain projections, and an observably empty book at 90% elapsed is not a
   projection.
3. `analytics/portfolio.py` — the execution floor, raised 50 → 61 (the APPLY
   band boundary). NEUTRAL issues are starved entirely.
4. `analytics/tday_signals.py` — the alert floor. This kept its own literal
   `MIN_SCORE = 50.0`; when the optimiser was raised to 61, Prasol at score 57
   was refused capital by the plan while still firing two CRITICAL "the call
   is APPLY" pushes two minutes before cut-off. An alert is the most
   actionable surface in the system and must never contradict the plan.

**Single source of truth.** Both score thresholds now live only in
`PortfolioConfig` (`config.py`). `PortfolioOptimizer` takes `None` and
resolves per call rather than freezing a literal at import; `tday_signals`
reads the same field; `generate_tactical_memo.py` passes the configured
value. `grep -rn min_score` should show literals in `config.py` and nowhere
else — if it ever shows a second literal, that is the drift that caused the
alert contradiction above.

**Effect on the board.** The three highest-odds issues are now refused:
ARCIL (score 59, 81% odds), PRASOLCHEM (57, 81%), MPIMANIPAL (54, 63%).
Every funded name scores 63 or better. If you are backtesting against runs
from before this change, expect the older allocations to include these names
and to show a higher headline P(allot) with worse realised returns.

## Regression suite

```bash
python3 tests/test_winners_curse.py     # 12/12, stdlib only
```

Runs standalone or under pytest. It asserts the live Prasol book (not a
sanitised fixture) against all four defences, and includes a control that
disables the rule to prove the bug reproduces: **+3.30% expected gain with
64.6% probability of a positive listing before, 0.00% after**.

Two of the twelve exist purely to stop an over-correction: a healthy book and
an issue with a missing Total must both pass through untouched, and a
76-scorer must still be funded. A floor that starves everything is not a fix.

## Assumptions worth challenging

These are judgement calls, not facts. A backtester should test them.

- **Application timing.** Capital is assumed blocked from the *close date*,
  matching the T-day engine's advice to decide at 14:00 on the final day. If
  you apply on day one, funds block up to two days earlier and the cash-flow
  timeline is optimistic. `PortfolioConfig.asba_settlement_days` tunes the
  release side only.
- **Retail `k` = 1.20 (cold) to 1.95 (hot).** Mean lots per retail
  application, unobservable from NSE data since they publish shares, not
  application counts. `P(allot) ≈ k / subscription_x`, so this is the single
  most sensitive input in the allotment engine.
- **Clearing holidays are seeded, not authoritative.** Fixed-date holidays and
  Good Friday are computed; lunar festivals (Holi, Diwali, Eid, Mahashivratri)
  come from a table in `calendar_in.py` covering 2025–2027. Verify against the
  official circular, or override via `data/holidays.json`.
- **BSE failover is unverified.** BSE's API host answers and serves JSON for
  other endpoints, but every IPO-specific path tried returned a bot-block
  redirect. The endpoint chain in `sources/bse.py` is a set of candidates; pin
  a known-good one with `bse_endpoint` in config. Layers 1, 2, 3 and 5 of
  `DemandFeed` are verified, so the redundancy is real without it.
- **The 75% late-window threshold and the 1.0x subscription bar** are round
  numbers, not calibrated ones. A book at 0.98x with 74% of the window gone is
  treated as healthy. Sweep both against recorded outcomes before trusting the
  exact boundary; the direction of the rule is better supported than its
  precise location.
- **`promoter_locked_shares` has no producer.** The day-30 float calculation
  therefore always uses the listing-float basis (offer less anchor
  allocation), which is disclosed as `day30_basis` and carries its own
  threshold (30% vs 15%), because the two bases read roughly 2.7x apart.
- **Equity risk premium 6.5%**, beta from a sector table with a
  capital-intensity fallback. Ke moves with the live 10Y G-Sec.
- **Credit spread is a proxy.** NSE's corporate-bond feed returns empty and
  FRED was unreachable, so it falls back to the G-Sec 10y–2y term spread,
  labelled as such in every output. Drop a real series into
  `data/macro_manual.json` and it takes priority.
- **`macro.baseline_days` is small.** Yield-spike penalties need a trailing
  baseline; with few snapshots the regime reads near-neutral by design.

## If a scraper breaks

Sources rotate endpoints. NSE is the backbone and is stable; the grey-market
scraper is the fragile one. Drop values into `data/gmp_manual.json` and
everything downstream keeps working:

```json
{ "RENTOMOJO": 135, "KANOHAR": 218 }
```

## Alerts

Off unless configured. Webhook URLs are bearer credentials, so they are read
from the environment and `Settings.save()` redacts them out of `config.json`:

```bash
export IPO_DISCORD_WEBHOOK="..."   # or IPO_SLACK_WEBHOOK / IPO_WEBHOOK_URL
export IPO_TELEGRAM_BOT_TOKEN="..." IPO_TELEGRAM_CHAT_ID="..."
export IPO_NO_ALERTS=1             # disable entirely
```

## Caveat

Grey-market premium is unofficial, thinly traded and easily manipulated; it is
treated as sentiment, not truth. Projections before the final day are wide by
construction. This is analysis tooling, not investment advice.
