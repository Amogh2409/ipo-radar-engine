# Changelog

## Board split: live issues on top, track record below

- **Closed issues leave the live board.** NSE drops an issue from its list once
  it closes, so its stored status stayed `Active` forever: the board showed
  September issues weeks later and every cycle re-analysed them (19 issues
  instead of 8). `refresh_universe` now closes anything past its close date
  that the source no longer lists.
- **Listing and current prices from Yahoo Finance** (`sources/prices.py`).
  `poll_listings` records the listing-day open in `outcomes` (only when no
  outcome exists, so a hand-entered one is never overwritten) and marks the
  issue `Listed`. That is the same row calibration trains on, so the model now
  learns from every listing without the manual `outcome` command.
- **"Closed and listed" section** under the board: the engine's call and
  predicted gain beside the actual listing gain and the current price.

## Chittorgarh failover

NSE's Akamai edge refuses most datacenter IPs, and when `all-upcoming-issues`
fails the engine had no universe at all. `sources/chittorgarh.py` rebuilds
every NSE input from chittorgarh.com, so the engine can run on a cloud host.

- **Universe**: used only when NSE lists nothing. Records reuse a stored NSE
  symbol when the name matches, otherwise get a name-derived placeholder
  (`meta.source = "chittorgarh"`), retired as `Superseded` once NSE returns.
- **Issue detail**: fills lot, band, face value and shares offered when NSE's
  `ipo-detail` is blocked.
- **Demand**: a `DemandFeed` layer after BSE, before the cache. Chittorgarh
  sizes categories at the cap price and NSE at the floor; the layer restates
  in NSE's convention. Checked live against NSE: shares offered match to the
  share, multiples to Chittorgarh's two-decimal rounding.
- **RHP**: resolved to the full PDF on sebi.gov.in. `DocumentSource` now
  accepts a bare PDF as well as NSE's zips.

## Winner's Curse defence

The most consequential behavioural change in the engine. If you are comparing
against output from before it, read this first — allocations legitimately
differ.

**The failure.** Allotment odds peak exactly when the market has declined an
issue. The optimiser ranked by expected profit — odds × gain — so 78.8% of a
small gain beat 6% of a large one, and capital was routed to the weakest book
on the board. Observed live: Prasol Chemicals opened its final day at QIB
0.05x and was funded ahead of a 215x-subscribed peer.

**Why the existing guard missed it.** `listing.py` collapsed its estimate only
when *every* category was under 1.0x. The real book read QIB 0.05x, NII 0.66x,
**RETAIL 1.14x** — retail alone above 1.0x, the loudest and least informed
category — so the guard stayed silent while institutions refused the issue.

**Four independent defences.**

1. `listing.py` — past 75% of the bid window, a *present* Total or QIB below
   1.0x forces expected gain to `min(0, central)` and `p_positive` to 0. A
   **missing** figure never condemns: the exchange omits the Total row on most
   issues, so unknown stays unknown.
2. `scoring.py` — demand factor capped at 10.0 on the same condition, surfaced
   as a risk rather than a reason. Re-applied *after* confidence damping,
   which exists to soften uncertain projections; an observably empty book at
   90% elapsed is not a projection.
3. `portfolio.py` — execution floor raised 50 → 61, the APPLY band boundary.
4. `tday_signals.py` — the alert floor, which had drifted (see below).

**Effect.** The three highest-odds issues on the live board are now refused:
ARCIL (59, 81% odds), PRASOLCHEM (57, 81%), MPIMANIPAL (54, 63%). Every funded
name scores 63+. Expect older runs to include these and to show a *higher*
headline P(allot) with worse realised returns.

### Threshold drift, and the alert that contradicted the plan
`tday_signals.py` kept a private `MIN_SCORE = 50.0`. When the optimiser rose to
61, Prasol at score 57 was refused capital by the plan while still firing two
CRITICAL *"the call is APPLY"* pushes two minutes before cut-off — the most
actionable surface in the system contradicting the least. Both thresholds now
resolve from `PortfolioConfig`, and a test asserts they cannot diverge.

`PortfolioOptimizer` resolves them per call rather than freezing a literal in a
default argument, so a reloaded config is honoured.

## Tactical action memo
`generate_tactical_memo.py` and `run.py memo`: a dense two-page A4 PDF filtered
to issues closing within 96 hours. Page 1 is the execution desk (capital
status, ACT NOW table, deployment roadmap); page 2 is forensic due diligence —
hard vetoes, fundamental scorecard, anchor lock-in dump risk. Auto-scales
through five type sizes to hold two pages. Shares the cascading render chain
with the briefing: WeasyPrint → headless Chrome → wkhtmltopdf → ReportLab.

## Weekly briefing
`generate_briefing.py`: stitches the board and every tear sheet into a single
PDF, with a system health and calibration appendix drawn from the `events` and
`outcomes` tables.

## Earlier
- **Edge cases** — clearing-holiday calendar with same-day unblock lock;
  accruals and cash-conversion gate; day-30 anchor lock-in float expansion;
  jump-diffusion structural break filter; BSE demand failover.
- **Terminal vetoes** — Value Trap (RoNW below cost of equity) capped at 59.0,
  with a narrow speculative-flip exception carrying a mandatory exit
  instruction.
- **EV/EBITDA and macro regime** — enterprise value from RHP data with
  sector-weighted multiples; regime scorecard on the live 10Y G-Sec, harsher
  on high-multiple growth.
- **Bayesian projection** — log-normal posterior on subscription with Kalman
  updating; confidence scaling with elapsed bid window; deterministic
  guardrails against small-model hallucination.

### Bugs worth remembering
- **CFO sign inversion.** *"Net cash used in operating activities"* was parsed
  as **+500** instead of −500, turning cash burners into cash generators.
- **Structural break made the optimistic case *more* likely.** Tripling
  variance raises the mean of a log-normal (`E[X] = exp(m + v/2)`), so a shock
  intended as a warning lifted the forecast. Fixed with a quantile-preserving
  pull.
- **Calendar fallback ran in the unsafe direction** — and the docstring claimed
  the opposite. Fewer holidays means an earlier unlock, a shorter block, and an
  optimiser committing capital that has not cleared.
- **Rupee glyph rendered as tofu.** Arial Unicode registers but has no U+20B9;
  `pypdf` reported the character present either way. Caught only by rendering
  to PNG and looking at it.
- **`.gitignore`'s bare `*.zip` silently swallowed the RHP cache** — the
  document archives are zips too. Now root-anchored.
