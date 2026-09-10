# IPO Radar

A continuously-running analyst for live Indian IPOs. It pulls the real order
book from NSE, parses the actual Red Herring Prospectus, computes allotment
probabilities under SEBI's real allotment rules, estimates listing gains with
an explicit uncertainty band, and tells you where to put a fixed amount of
capital across every IPO open at once.

> ### Disclaimer
>
> **This is a personal research tool, not investment advice.** The author is
> not a SEBI-registered investment adviser or research analyst, and nothing
> this software produces is a recommendation to buy or sell any security.
>
> The engine prints words like `STRONG APPLY` and `AVOID`. They are the
> output of a scoring model built on public data and stated assumptions —
> several of which are explicitly uncalibrated guesses, documented in
> *Assumptions worth challenging* in `HANDOFF.md`. Treat them as one input to
> your own judgement, never as a signal to act on.
>
> IPO investing carries risk of permanent capital loss. Listing gains are not
> guaranteed, allotment is not guaranteed, and the grey-market premium this
> model consumes is an unofficial, unregulated and easily manipulated number.
> Past behaviour of any model here says nothing about future results.
>
> Provided as-is, without warranty of any kind. You are responsible for
> everything you do with it.

A local Ollama model handles the qualitative half — reading the RHP's objects
and risk-factor sections, scoring news, and arguing against its own verdict.
Everything numeric is computed in Python, so the analysis stands up even with
the LLM switched off.

## Quick start

```bash
pip install -r requirements.txt      # httpx, rich, pypdf
ollama serve &                       # optional but recommended

python3 run.py once                  # one full pass, prints a board + writes reports/
python3 run.py run                   # run continuously
```

First run downloads each RHP (10–40 MB) and caches it under `data/documents/`,
so it is slow once and fast forever after. `python3 run.py once --no-docs`
skips that if you want an immediate answer.

## Commands

| Command | What it does |
|---|---|
| `run` | The daemon. Polls, analyses and rewrites reports on a schedule. |
| `once` | A single full cycle. |
| `list` | The live IPO universe from NSE. |
| `analyse SYM` | Deep dive on one IPO (`--markdown` for the raw report). |
| `allot SYM --applications N` | Allotment odds across categories for N PANs. |
| `report` | Rewrite reports from stored data, no network. |
| `outcome SYM --listing-price P` | Record what actually happened. |
| `calibrate` | Refit the models against recorded outcomes. |
| `export-finetune` | Write a fine-tune corpus from observed outcomes. |
| `config` | Print effective configuration. |

Useful flags: `--no-llm`, `--capital 500000`, `--pans 4`, `--log DEBUG`.

## The allotment maths

This is the part most GMP sites get wrong, so it is worth being precise.

When retail is oversubscribed, SEBI requires the **minimum bid lot** to be
allotted to as many applicants as possible, **by lottery**. Past that point
every winner receives exactly one lot whether they bid 1 lot or 13. With `A`
shares offered to retail, `L` the lot size, `x` the subscription multiple and
`k` the mean lots per application:

```
applications   = (A·x / L) / k
max allottees  = A / L
P(allotment)   = max_allottees / applications = k / x
```

So **bidding more lots on one PAN does not improve your odds — it only blocks
more cash.** More PANs do: `P(at least one) = 1 − (1 − p)^n`.

`k` cannot be observed from NSE data (they publish shares bid, not application
counts), so it is estimated from demand heat and then calibrated against real
allotment results you feed back in via `outcome --allotment-rate`.

sNII (₹2–10L) and bNII (>₹10L) are proportionate with a one-lot floor since
the April 2022 circular, which degenerates into a lottery once the
proportionate entitlement drops below one lot. Both are modelled explicitly,
and because those applicants bid the minimum ticket, `k` there is known rather
than estimated — which is why sNII odds are usually the most reliable number
on the board.

## Honesty about projections

Allotment is decided on the **closing** book, but you must decide before it
closes — and Indian bidding is violently back-loaded, especially QIB. The
projector models cumulative demand as `f(t) = t^α` with a different α per
category, blended with observed run-rate as the window closes.

A day-one QIB reading of 1.2x is genuinely compatible with a final book
anywhere between 5x and 80x. Rather than hide that behind a confident point
estimate, every projection carries a confidence that scales with elapsed
window time, and that confidence:

- widens the listing-gain interval,
- pulls the demand factor toward neutral in the composite score,
- is shown as `Conf` on the dashboard.

Both the α values and the GMP realisation factor are **re-fitted from this
database's own completed IPOs** once enough have been observed. Until then
they report `insufficient` rather than fitting noise.

## Scoring

Seven weighted factors, each 0–100, each retained so any score can be
explained:

| Factor | Weight | Source |
|---|---|---|
| Financial quality | 20 | RoNW, EPS CAGR and consistency from the RHP |
| Demand | 20 | Projected QIB / NII / retail subscription |
| Valuation | 18 | P/E and P/B vs RHP peer group or sector anchor |
| GMP | 14 | Grey market premium and its trend |
| Structure | 10 | OFS share, anchor book quality, insider exit multiple |
| Qualitative | 12 | The local model's read of the RHP |
| Market regime | 6 | NIFTY 50 and recent listing performance |

## What it reads

- **NSE** — issue master, bid lot, face value, registrar, lead managers, and
  the live category-wise demand book. Authoritative.
- **RHP** (from `nsearchives`) — restated EPS/RoNW/NAV, objects of the offer,
  risk factors, and the weighted average cost of acquisition, i.e. what recent
  insiders paid versus what you are being asked to pay.
- **Anchor allocation letter** — book size, anchor price, and which
  institutions actually showed up.
- **ipowatch** — grey market premium, fuzzy-matched to NSE symbols.
- **Google News RSS** — headlines, scored for relevance and sentiment.

The RHP's own P/E and NAV tables read `[●]` because it is filed before the
price band is fixed, so multiples are computed here from EPS and NAV against
the actual band. Documents that turn out to be scans (the price-band newspaper
advert usually is) are detected by text-quality score and skipped rather than
guessed at.

## The local model

`ipo-analyst` is built automatically from a base model (default `qwen3:4b`)
with a system prompt encoding SEBI mechanics and Indian-market priors. Every
call uses Ollama's JSON-schema-constrained decoding, so output is parsed, not
regexed.

Four passes: read the RHP → score news → **argue against the draft verdict** →
synthesise. The critique pass exists because models are sycophantic by
default; asking explicitly for the bear case is the cheapest correction
available.

A 4B model is the floor. If you have the RAM, the qualitative half improves
markedly with a larger one:

```json
{ "llm": { "base_model": "qwen3:8b" } }
```

in `config.json`, then delete the `ipo-analyst` model so it rebuilds. The
quantitative analysis is unaffected either way.

## The three execution upgrades

### Bayesian projection (`analytics/bayesian_projector.py`)

Final subscription is a posterior, not a curve reading. `log X ~ Normal(m, P)`,
because subscription is strictly positive and heavily right-skewed.

The **prior** is built from three things known before a single bid lands:
issue-size bracket (`<₹500 Cr` / `₹500–1500 Cr` / `>₹1500 Cr`), anchor-book
quality (marquee institutions as a share of the book), and day-1 retail
momentum as an early proxy for general demand.

Each hourly snapshot then updates it through a scalar Kalman recursion. The
observation model is the part that matters: under `x(t) = X·t^α`, one snapshot
implies `z(t) = log x(t) − α·log t` with variance

```
Var[z(t)] ≈ (log t)² · Var[α] + floor
```

which collapses to the noise floor as `t → 1`. That single line *derives*
"day-one readings are nearly worthless, closing readings are nearly exact"
from the geometry of the curve, instead of asserting it with a tuned ramp.
Readings are bucketed to one per hour and the observation noise is inflated,
because successive cumulative readings are not independent samples.

**Dynamic weighting.** `demand_weight()` ramps the demand factor from **5% on
day 1 to 25% from 14:00 IST on the closing day**. Whatever demand gives up
early is handed to valuation and financials — the factors that are fully
knowable on day one because they come out of the RHP, not the order book.
Totals always sum to 100.

| Moment | demand | valuation | financials |
|---|---|---|---|
| Day 1, 10:00 | 5.0 | 25.1 | 27.9 |
| Day 1 close | 12.0 | 21.8 | 24.2 |
| Day 2, 13:00 | 15.5 | 20.1 | 22.4 |
| Day 3, 11:00 | 21.0 | 17.5 | 19.5 |
| Day 3, 14:00+ | 25.0 | 15.6 | 17.4 |

### Guardrails (`brain/guardrails.py`)

Nothing the model says about a number reaches the score without being checked
against the number Python already computed. The rule is: **the LLM may
contribute judgement, never facts, and never the sign of a fact.**

1. **Polarity/inversion.** `POLARITY_RULES` pairs a computed fact with the
   directions Python knows. If `insider_exit_multiple < 2.0` the fact is
   favourable, and any entry matching it in `risks`/`red_flags` is deleted.
   The reverse also holds — an adverse fact filed under `positives` is
   deleted. Covers insider exit multiple, RoNW, EPS growth and OFS share,
   plus a profitability check that strips "loss-making" claims from a
   profitable issuer.
2. **Ground-truth metadata.** `board == "MAINBOARD"` strips every SME claim
   from lists and prose. No observed GMP strips grey-market claims.
3. **Enums, ranges and types.** `litigation_severity: LOW|MEDIUM|CRITICAL`,
   `governance_flag: bool`, scores bounded to 0–100 and confidence to 0–1.
   Pydantic enforces this when installed; an equivalent stdlib validator with
   the same enums and bounds runs when it is not.
4. **Numeric claim verification.** A quoted P/E or RoNW must match the
   computed figure within tolerance, or the claim is dropped.
5. **Contradiction detection.** A high score alongside CRITICAL litigation, a
   governance flag with a 75+ score, or the same item on both sides of the
   ledger.

Past `FALLBACK_THRESHOLD` violations — or any critical one, or invalid schema
— the model's judgement is discarded entirely and `rule_based_assessment()`
produces a **zero-hallucination score from the computed figures alone**. Every
correction is recorded on the verdict and printed in the report.

### T-Day execution (`analytics/tday_signals.py`)

IST-aware throughout. The mandate deadline is **parsed from NSE's own issue
circular** (`"10-Sep-2026 (upto 5:00 PM)"`) rather than assumed.

- **Phases**: `pre_open → early → final_day_morning → decision_window
  (14:00–16:30) → final_call (16:30–17:00) → closed`.
- **Countdown** to the UPI mandate cut-off, plus derived `submit_by`
  (cut-off − 60 min) and `mandate_by` (cut-off − 20 min). The instruction text
  ages with the clock: past `submit_by` it becomes "PLACE THE BID NOW".
- **Institutional spike detector**: QIB fill-rate in the final 120 minutes
  against its own baseline, firing at **≥2σ**. Overnight gaps are excluded so
  the 17:00→10:00 dead zone cannot masquerade as a collapse in demand. A
  degenerate (flat) baseline falls back to a documented relative test rather
  than dividing by zero.
- **Bidding advice**: ranks the concrete plans by return on blocked capital
  and states the category, lot count, cash blocked, and the price instruction —
  correctly distinguishing that **cut-off pricing is available only to retail
  and employees**; an NII bid must name the cap price or risk exclusion from
  allotment entirely.

### Wiring

All three are already wired. There is no `main.py` in this project — the
entry point is `run.py`, and the loop lives in `ipo_radar/daemon.py`.

| Where | What changed |
|---|---|
| `engine.py` | `self.bayes`, `self.tday`, `self.factcheck` constructed; `analyse()` takes posteriors from the Bayesian engine, applies `dynamic_weights()`, passes LLM output through `FactChecker.check()` before scoring, and attaches the T-day signal |
| `daemon.py` | new `tday` job on its own cadence, plus `_in_decision_window()` |
| `config.py` | `cadence.tday = 600`, `cadence.tday_hot = 60` |
| `scoring.py` | `score(..., weights=)` override for the ramped weights |
| `models.py` | `Verdict.posteriors`, `.guardrails`, `.tday`, `.weights_used` |
| `report.py` | `Cut-off` and `Do` columns, execution alerts, posterior table, execution and guardrail report sections |

The T-day job deliberately does no LLM or PDF work, so it can run **every 60
seconds inside the decision window** while the full analysis stays on its
30-minute cadence. That split is the whole point: a 30-minute cadence is
useless at 16:45.

## Valuation: EV/EBITDA and sector weighting

P/E is the wrong lens for an asset-heavy issuer. Reported earnings there are
largely an artefact of depreciation policy and leverage, both of which vary
wildly between otherwise identical businesses. EV/EBITDA is neutral to both,
so it takes the larger share of the valuation score exactly where P/E is least
trustworthy.

**Enterprise value** is built from RHP figures, not scraped:

```
EV = cap_price x shares_outstanding + net_debt
shares_outstanding = PAT / EPS
net_debt = net_debt_to_equity x net_worth      (preferred: one number, one unit)
         = total_borrowings - cash             (fallback)
```

Share count is derived rather than parsed because the capital-structure tables
are far messier than the KPI table. Two independent gates catch a bad
extraction before it can produce a confident-looking wrong multiple:

1. **RoNW reconciliation** — `EPS/NAV` must equal the separately-extracted
   RoNW, since both reduce to PAT/net-worth. On GLASSWALL this predicts 32.0%
   against an extracted 32.03%.
2. **Scale sanity** — the offer must land at 2–65% of implied market cap. This
   caught two live extractions claiming an offer worth 101% and 2,510% of the
   company.

**Sector weighting.** `classify_sector()` reads the issuer's name first and its
RHP prose second — Indian issuers say what they do ("LCC **Projects**",
"Kanohar **Electricals**", "Manika **Plastech**"), and a passing mention of
"rental" in an objects section once reclassified a construction company as
internet. Capital intensity then sets the blend:

```
w_ev = 0.25 + 0.50 x capital_intensity        (0.70 minimum if loss-making)
valuation_score = (1 - w_ev) x pe_score + w_ev x ev_score
```

Power, telecom, infrastructure and renewables sit at 0.90–0.95 intensity, so
EV/EBITDA carries ~70–75% of the score. Software and internet sit at 0.10–0.15,
so P/E dominates. Lenders (banks, NBFCs, ARCs) are excluded from EV entirely —
borrowings are their raw material, not their financing, so EBITDA and
enterprise value are meaningless for them.

## Macro regime

**India 10-year yield is computed, not scraped.** NSE publishes every centrally
issued G-Sec traded on the cash market, and the symbol encodes coupon and
maturity (`754GS2036` = 7.54% maturing 2036). Given the traded price, YTM is
solved by bisection, producing a curve from the actual market:

| Tenor | Yield |
|---|---|
| 2y | 6.20% |
| 5y | 6.65% |
| 10y | **6.72%** |
| 30y | 7.52% |

Two hazards are handled explicitly. Most G-Sec turnover happens on NDS-OM, so
NSE prints are thin and some are stale — hence outlier rejection (band median
first, discard anything >1.2pp away, then weight by volume). And maturity is
approximated as mid-year, which is a rounding error at ten years but a large
one at two, so bonds under 1.5 years are excluded entirely. Without those two
guards the 2y point read 4.28% against neighbours near 5.7%.

Credit spread resolves through a chain: explicit override → any reachable
optional adapter (FRED HY OAS) → the G-Sec term spread as a **clearly labelled
proxy**. A proxy that admits it is a proxy beats a precise-looking number
nobody can source. The optional adapter gets one short attempt, not the shared
retry policy — backing off against an unreachable host on every refresh cost
fifteen seconds a cycle.

### Non-linear stress

Small rate moves are noise; large ones are repricings. A linear penalty cannot
express both, so stress rises with an exponent above one:

| 10y move | stress contribution |
|---|---|
| 10bp | 0.00 |
| 25bp | 0.21 |
| 40bp | 0.76 |
| 60bp+ | 1.00 |

Components (yield change, spread change, absolute level, equity tape) are
weighted to sum to exactly 1.0, so a fully stressed regime lands near 8 rather
than saturating — a regime that always reads zero cannot rank anything.

### Duration-scaled, and asymmetric

`issue_sensitivity()` reads how much of a valuation rests on distant growth —
multiple versus benchmark, absolute P/E, P/B, loss-making status, growth rate,
RoNW — and returns 0.50 (cheap and cash-generating) to 1.65 (expensive growth
story). The regime penalty is multiplied by it:

| Issue | sensitivity | base 70 | base 35 | base 12 |
|---|---|---|---|---|
| cheap cash-flowing | 0.50 | 70 | 42 | 31 |
| average mainboard | 1.05 | 70 | 34 | 10 |
| expensive growth | 1.65 | 70 | 25 | 3 |
| loss-making story | 1.62 | 70 | 26 | 3 |

**The application is asymmetric on purpose.** A hostile regime is amplified by
sensitivity; a benign one passes through unchanged. The symmetric version of
this scored an expensive loss-making issue 83 against a base of 70, purely
because easy money flatters long-duration assets — which is exactly the
reasoning that loses money at the top of a cycle. Cheap credit is a reason to
be less afraid, never a reason to pay more.

### Feeding back into the posterior

When the regime score drops below **40**, the Bayesian projector's observation
noise is inflated in proportion to how far below it sits. The rationale is
behavioural: in a deteriorating macro tape institutions pull or trim bids in
the final hour, so the shape of the book so far becomes a weaker guide to the
closing book. The central estimate is not nudged — only the confidence in it —
and the inflation is per-category, because retail behaviour is sticky under
stress while institutional behaviour is not:

```
QIB 1.8x   bNII 1.5x   NII 1.4x   sNII 1.1x   RETAIL 0.35x   EMPLOYEE 0.2x
```

Macro snapshots are stored in the `macro` table so changes are measured
against a real trailing baseline. With no history yet the module reports "no
macro baseline" and stays neutral rather than inventing a trend.

## Economic value added (EVA) gate

A live risk-free rate makes a real test possible: does this business earn more
on equity than equity costs?

```
beta = SECTOR_BETA[sector]  or  1.30 - 0.50 x capital_intensity
Ke   = Rf(live 10y G-Sec) + beta x 6.5%
```

Beta **falls** as capital intensity rises — the opposite of the intuition that
heavy balance sheets are risky. Regulated, asset-heavy businesses have stable
cash flows and low beta; asset-light growth is far more volatile. So a power
utility clears a ~11.9% bar while an internet issuer must clear ~15.8%.

| Verdict | Condition | Effect |
|---|---|---|
| **Value destroyer** | RoNW < Ke | quality penalty scaled by the shortfall (18–34 pts) + risk flag |
| Marginal | 0 ≤ spread < 6pp | −4 and a thin-spread note |
| **High-spread compounder** | RoNW ≥ Ke + 6pp | +10 |

Live, this flags **ARCIL as a Value Trap** — a 13.9% RoNW against a 15.8% cost
of equity for an NBFC beta of 1.40. A percentile-based RoNW score would have
called that "average"; against its actual cost of capital it is destroying
value.

A post-issue RoNW is reported alongside where the fresh-issue size is known,
because raising equity dilutes return on equity mechanically — an issuer
sitting just above its cost of capital before the offer can fall below it the
moment the money lands.

Scores are **soft-capped** rather than clipped. An additive factor model
overshoots, and a hard clamp tied five of eleven live issues at exactly 100.0,
destroying the ranking precisely where it matters. Compressing above the knee
keeps the ordering; `quality_score_raw` is exposed alongside.

## Clearing calendar and the evening-release lock

ASBA settles through the banking system, so "T+3 business days" means three
**clearing** days. `analytics/calendar_in.py` resolves holidays from three
sources in priority order: `data/holidays.json` (operator override), a seeded
table of variable-date festivals, and rules that are computable for any year —
the fixed national holidays plus Good Friday via the Gregorian Easter
algorithm.

A mid-week holiday genuinely moves settlement: a 10 Sep close now releases on
**16 Sep**, not 15 Sep, because Ganesh Chaturthi falls in the window.

**The seed table is seed data, not gospel.** Exchange and clearing holidays are
published annually by NSE and RBI and occasionally move. Verify against the
official circular before trusting a live settlement date.

A year with no seed entry falls back to rules only, and that errs in the
**unsafe** direction — fewer known holidays means fewer skipped days, an
earlier assumed unlock, a shorter blocked window, and therefore an optimiser
that commits *more* capital than has actually cleared. (An earlier version of
this section claimed the opposite; that was wrong.) Unseeded years are
therefore padded by an extra clearing day, and `holidays_seeded(year)` exposes
the condition so callers can disclose that the date is an estimate.

**Evening-release lock.** Sponsor banks push ASBA revocations in a batch
between roughly 18:00 and 23:59 IST. Money released on the evening of day T is
useless for an issue *closing* on day T — the UPI mandate cut-off is 17:00,
hours earlier. So `capital_available_from()` returns the next clearing day, and
the timeline blocks through the release date rather than up to it. Verified: an
issue closing 10 Sep frees capital that can fund a 17 Sep close but **not** a
16 Sep one.

## Accruals gate

Reported profit is an opinion; cash is a fact. An issuer polishing its numbers
before listing does it by letting receivables and inventory build, which shows
as profit rising while operating cash flow does not follow.

```
cash_conversion = CFO / EBITDA        (both in Rs crore)
```

Fires only when conversion < 0.35 **or** CFO < 0, **and** PAT is positive
**and** growing — a loss-maker with negative CFO is not an accruals problem,
it is simply a loss-maker, and the EVA gate already covers it. On trigger:
a critical risk flag, a −12 quality penalty, and any EVA compounder bonus is
**halved**.

CFO is parsed in two shapes, narrative first because it names its own fiscal
year and unit. Two guards proved necessary on live data:

- **Consolidation tables.** `339.49 1.79 85.94 427.22` is parent plus
  subsidiaries plus the group total, not a three-year series. Taking the first
  column reported the parent alone and understated group cash flow by a
  quarter — enough to trip a false accruals warning on GLASSWALL. The parser
  now detects that the components sum to the last value and takes the total.
- **Implausible ratios.** A conversion above 10× means the EBITDA denominator
  is mis-parsed, not that the company is a cash machine. The gate reports
  "extraction suspect, not assessed" rather than firing. This is what stops
  KARAMTARA's broken 750× from producing a confident wrong verdict.

Live: GLASSWALL 0.41, PRASOLCHEM 0.36, RENTOMOJO 1.48, STEAMHOUSE 1.20 — all
healthy, no false positives.

## Day-30 lock-in float expansion

SEBI releases 50% of the anchor allocation at 30 days and the rest at 90. The
first tranche lands on a stock with no research coverage and a thin float:

```
day30_impact = (0.5 x anchor_shares) / tradable_float
```

The denominator prefers `total_shares − promoter_locked_shares` as specified.
Promoter lock-in is rarely parseable — it sits in capital-structure tables that
are largely `[●]` until the price band is fixed — so the fallback is the
listing float: the offer itself minus the anchor allocation, since anchors are
*inside* the offer and locked at listing. That is the more conservative reading
(smaller denominator, larger impact), and the basis used is reported alongside
the number.

The warning needs **both** legs: impact > 15% AND flipper ratio > 35%. A large
float expansion into patient hands is not a dump risk — verified, a book with
identical 37.3% expansion and 0% flippers stays silent.

## Structural break filter

The `t^α` model assumes continuous back-loaded bidding. It is not robust to a
regime break: when the macro tape turns and the grey market gaps down
mid-window, the curve stops describing the process, and a Kalman filter left
alone will keep tightening around a trajectory the market has abandoned.

Detection needs both legs — GMP down ≥10% **and** macro stress ≥0.35 — and
refuses to call a break before 30% of the window has elapsed. On detection:

- observation covariance `P ← P × 3`
- the mean is pulled 45% toward the **size-bracket prior** — the only prior
  component a demand shock does not invalidate, since anchor quality and
  momentum are both readings of the enthusiasm that just evaporated
- the pull is **downward only**; a dislocation is never a reason to raise an
  estimate

One interaction worth understanding. The committed book is a hard floor that
outranks the pull, because **SEBI does not permit QIB or NII bids to be
withdrawn or revised down** once submitted. So on an issue whose book is
already large the pull is fully absorbed and only the variance inflation
survives — which is correct: the central estimate is pinned by committed bids
while uncertainty about the remaining flow widens. Measured: KARAMTARA
(mid-window) 19.71 → 13.14; KANOHAR (book near-final) pinned at 63.79 with the
interval still widening ×√3.

## Demand failover

Every mainboard IPO is bid on both exchanges, so BSE is genuine redundancy.
`DemandFeed` degrades through five layers:

1. NSE `ipo-active-category` — the live book
2. NSE `ipo-detail.bidDetails` — same data, different handler
3. NSE `ipo-current-issue` — totals only, every live issue in one call
4. BSE demand schedule — different exchange, different edge
5. last good snapshot from SQLite, explicitly marked stale

Layers 2 and 3 matter more than they look: an Akamai block is usually
path-scoped, so a sibling NSE endpoint often answers when the primary is
throttled. Dropping straight to BSE would abandon redundancy that is still
there. The cache layer never re-persists — writing a cached row back would
stamp a fresh timestamp on stale data, and the velocity and spike detectors
would read that as real movement.

**Honest limitation.** BSE's API host answers and serves JSON for other
endpoints, but every IPO-specific path tried 302s to a bot-block page. The
endpoint names are therefore *candidates* tried in order, and the module is
built so none of them working costs nothing. Set `bse_endpoint` in
`config.json` to pin a known-good path. This layer is **unverified against
live BSE**; layers 1, 2, 3 and 5 are verified.

## Temporal capital optimisation

ASBA does not *spend* capital, it sterilises it. Under the T+3 listing regime
the mandate holds funds from the close date until roughly three business days
later, so an issue closing on the 11th locks cash through the 16th — and if a
far better issue closes on the 15th, spending everything on the 11th means
missing it entirely.

The optimiser therefore models a calendar, not just a budget:

- `asba_unlock_date(close, 3)` advances three **business** days. Exchange
  holidays are deliberately not modelled: a stale holiday list would unlock
  capital a day *early*, which is the dangerous direction. A real holiday only
  ever means funds free up later than projected.
- `CapitalTimeline` tracks committed capital per day and checks feasibility
  across an application's whole blocked window, not against a running total.
- **Conviction claims its window first.** Issues scoring above
  `reserve_min_score` (65) are allocated before anything else, so a 74-scorer
  closing on the 15th cannot be starved by a 56-scorer closing on the 11th
  whose funds are still locked. Capital is reserved *only where the windows
  genuinely overlap*, so a good issue closing after everything else has
  already unlocked costs the earlier applications nothing.
- Forthcoming issues within `lookahead_days` (14) are candidates too. They
  have no order book, so their odds come from the Bayesian **prior** — issue
  size, anchor quality, comparable momentum — with correspondingly low
  confidence.

`capital_used` reports **peak simultaneous commitment**, not the sum. Cash
released on the 15th and redeployed on the 16th is the same cash twice, and
summing it would overstate what the account actually needs.

## Objective toggle

`PortfolioConfig.objective`, or `--objective` / `IPO_OBJECTIVE`:

| Mode | Behaviour |
|---|---|
| `maximize_roi` (default) | Stop once density is exhausted. Idle cash beats a poor return. |
| `maximize_absolute_profit` | Keep upgrading to larger tickets while capital would otherwise sit unused. |

**Both modes select by profit density.** Ranking by raw rupees is the classic
knapsack error — a big sNII ticket wins the comparison, sterilises the
capital, and crowds out several cheaper applications worth more in total.
Measured on the live board, naive rupee-ranking produced **₹9,445** of expected
profit against **₹12,853** for density ranking: the absolute-profit objective
was *losing* absolute profit.

So the objective decides what happens to LEFTOVER capital instead. In absolute
mode an upgrade pass walks the chosen applications and swaps each to the
largest plan for the same issue that still fits its own window, richest
upgrade first. Same live board:

| Mode | Peak blocked | Expected profit | ROI on blocked |
|---|---|---|---|
| `maximize_roi` | ₹312,967 | ₹12,863 | 4.11% |
| `maximize_absolute_profit` | ₹478,355 | **₹15,694** | 3.28% |

Lower percentage, ~22% more rupees — the right trade for a ₹5L account, and
it naturally promotes sNII exactly where the capital would otherwise be idle.

## Cash flow projection

The dashboard and `reports/index.md` carry a day-by-day ledger of when money
comes back:

```
  Date           Blocked   Released   Outstanding   Free to deploy  Movements
  Thu 10 Sep     ₹44,332          —       ₹44,332         ₹455,668  +GLASSWALL +KANOHAR +PRASOLCHEM
  Fri 11 Sep    ₹268,635          —      ₹312,967         ₹187,033  +KARAMTARA +LCCPROJECT +MPIMANIPAL ...
  Sat 12 Sep           —          —      ₹312,967         ₹187,033  —
  Tue 15 Sep           —    ₹29,352      ₹283,615         ₹216,385  +VEEGALAND -GLASSWALL -KANOHAR ...
  Wed 16 Sep           —   ₹253,671       ₹29,944         ₹470,056  +MANIKA -KARAMTARA -LCCPROJECT ...
  Mon 21 Sep           —    ₹14,964            ₹0         ₹500,000  -MANIKA
```

Weekends show no releases, which is the business-day rule made visible.
"Free to deploy" turns red below 15% headroom.

## Terminal veto: the Value Trap ceiling

The EVA gate docks points, but points are not enough. On live data ARCIL
carried a Value Trap flag and still reached **67 / APPLY**, because quality is
only about a fifth of the composite and its demand and GMP factors were
strong. A wealth-destroying business should never earn a blind APPLY for
long-term capital, however hot the book is.

So the veto is terminal, not a penalty:

```
if RoNW < Ke:                      composite = min(composite, 59.0)   # NEUTRAL
long-term stance                := "AVOID - Value Trap: Fails to earn its
                                     Cost of Capital. Do not hold post-listing."
```

### The speculative-flip exception

A listing pop is a liquidity event, not an endorsement. If the expected gain
is large **and institutions are genuinely crowding the book**, the trade is
still available — but it is renamed so nobody mistakes it for an investment:

| Condition | Threshold |
|---|---|
| Expected listing gain | `> 20%` |
| Projected QIB subscription | `> 5x` |
| Composite before the veto | `>= 61` (the APPLY threshold) |

All three must hold. Retail enthusiasm alone never qualifies — it is the least
informed capital in the offer and the most likely to be exiting into itself.

That third condition is not in the original brief and was added deliberately.
The exception is meant to lift the **veto ceiling**, not the ordinary bands;
without it a hot grey market could promote a business scoring 45 straight out
of AVOID, which is precisely backwards for a safety mechanism.

When it applies the grade becomes **`SPECULATIVE FLIP`** — styled magenta, not
green, because it is a trade to exit rather than a holding. The long-term
stance is *unchanged*: still the fixed Value Trap string. The T-day execution
call carries its own exit instruction ("Sell on listing day; do not hold"), and
the flip lifts the T-day score floor so the veto does not cancel the very trade
the exception permits.

### Why the veto lives in `_grade()`

`engine.analyse` applies the LLM critique's `score_adjustment` *after* scoring
and then re-derives the grade. Putting the veto only at the original call site
would let a `+10` nudge quietly restore an APPLY on a wealth-destroying
business. Instead `_grade()` takes the veto flags, `Verdict.veto` carries the
state, and the engine re-applies the ceiling after any adjustment. Verified:

| Case | Score | +10 nudge | Grade after |
|---|---|---|---|
| Value Trap, capped | 59.0 | 59.0 | NEUTRAL |
| Value Trap, flip allowed | 62.8 | 72.8 | SPECULATIVE FLIP |

Even at 72.8 — deep in STRONG APPLY territory — a flip stays a flip.

## Anchor "flipper" profiling

SEBI unlocks **50% of anchor shares at 30 days and the rest at 90**, so the
anchor book is a schedule of future supply. Its composition tells you what
that supply will do.

Classification is **structural, not reputational** — a Category III long-short
sleeve run under a large AMC's brand is still tactical money, because the
mandate sets the holding period, not the letterhead. Precedence:

1. **Structural flipper** — AIF, Category III, long-short, family trust, HUF,
   PMS, arbitrage, opportunities/alpha funds, **and broker proprietary books**
   (tested here so `<Big AMC> Securities Ltd` cannot be rescued by a brand match)
2. **Compounder family** — sovereign wealth, large domestic AMCs (matched on
   bare brand, since funds appear as "Kotak ELSS Tax Saver Fund"), insurers,
   marquee foreign long-only
3. **Weak flipper** — finvest/fincap/finserv, ventures LLP, prop desks
4. Otherwise neutral

Names are **normalised before anything is counted**. The list is regex-scraped
from a PDF and contains fragments ("Long-Short Fund"), overlapping duplicates
of the same fund, and occasionally the issuer's own name — MPIMANIPAL's sole
"anchor" was itself. Ratios over that raw list would be meaningless.

Live results: RENTOMOJO scores 86/100 with **LOW** overhang (Edelweiss MF,
Mirae, Kotak); VEEGALAND scores 26/100 with **MEDIUM** overhang and a risk
flag, its book being Navbharat Opportunities, Swyorn Alpha and similar.

## Push alerts (`ipo_radar/alerts.py`)

Zero new dependencies — `httpx` was already in the stack. Discord, Slack,
Telegram and any generic JSON endpoint.

**The design constraint is that this can never stall the daemon.** The T-day
job runs every sixty seconds and shares an event loop with subscription
polling and PDF extraction, so a webhook against a dead host must cost that
loop nothing:

- `dispatch()` is **synchronous** — it schedules and returns. Measured at
  **0.03ms against an unroutable host**, with the loop running 84 ticks
  unblocked during the timeout.
- Every scheduled task is held in a **strong reference set**. `asyncio` keeps
  only weak references to running tasks, so a task nobody holds can be
  garbage-collected mid-flight and the alert vanishes intermittently.
- **Nothing raises.** A missed notification must never become a failed cycle.
- `drain(timeout)` lets queued posts land at shutdown without hanging on them.

**Alerts are edge-triggered.** `SignalWatcher` tracks previous state per symbol
and fires on transitions, not conditions — a 60-second loop would otherwise
emit "decision window open" sixty times an hour. Over a simulated closing day,
**195 ticks produced 4 alerts**:

| Time | Alert |
|---|---|
| 14:00 | PHASE CHANGE: Decision Window Open |
| 14:00 | CALL CHANGED: WAIT → APPLY |
| 15:45 | ACTION REQUIRED: Shadow Cut-Off approaching (+ optimal allocation) |
| 16:30 | FINAL CALL |

Plus `QIB SPIKE DETECTED` on the rising edge of a ≥2σ acceleration.

Configure via environment (webhook URLs are bearer credentials, and
`Settings.save()` redacts them out of `config.json`):

```bash
export IPO_DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
export IPO_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export IPO_TELEGRAM_BOT_TOKEN="..." IPO_TELEGRAM_CHAT_ID="..."
export IPO_WEBHOOK_URL="https://your.endpoint/hook"
```

Gates: `min_level` (default HIGH), `min_repeat_seconds` (900) and
`max_per_hour` (40). Set `IPO_NO_ALERTS=1` to disable.

## Closing the loop

```bash
python3 run.py outcome RENTOMOJO --listing-price 465 --allotment-rate 0.33
python3 run.py calibrate
```

Every forecast is stored with its features. Recording outcomes lets
`calibrate` refit the projection exponents, the GMP realisation factor, and
the retail `k` — and `export-finetune` turns the accumulated pairs into a
training corpus once you have enough of them.

## Configuration

`config.json` in the project root, created by editing the defaults printed by
`python3 run.py config`. Cadences, score weights, allotment priors, LLM host
and model, and your capital and PAN count all live there. Environment
overrides: `IPO_CAPITAL`, `IPO_OLLAMA_HOST`, `IPO_OLLAMA_MODEL`, `IPO_NO_LLM`.

## Layout

```
ipo_radar/
  config.py models.py store.py util.py engine.py daemon.py cli.py report.py
  sources/     nse.py gmp.py news.py documents.py
  analytics/   allotment.py subscription.py valuation.py listing.py
               scoring.py portfolio.py
  brain/       ollama_client.py analyst.py schemas.py modelfile.py calibration.py
data/     ipo_radar.db, cached RHPs, gmp_manual.json
reports/  index.md, <SYMBOL>.md, latest.json
```

## If the GMP scraper breaks

Grey market sources rotate endpoints constantly. Drop numbers into
`data/gmp_manual.json` and everything downstream keeps working:

```json
{ "RENTOMOJO": 135, "KANOHAR": 218 }
```

## Caveat

GMP is unofficial, thinly traded and easily manipulated; it is treated as
sentiment, not truth. Projections before the final day are wide by
construction. This is analysis, not investment advice.
