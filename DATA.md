# The dataset

Everything the engine has observed, and exactly how to read it.

`data/` is tracked in this repository on purpose. The engine downloads
prospectuses from exchange URLs that expire and polls an order book that only
exists while an issue is open — neither is reconstructable after the fact. A
clone is therefore self-contained: you can re-run every model against the same
inputs that produced the committed reports, and get the same numbers.

```
data/
├── ipo_radar.db        3.8 MB   SQLite, WAL-checkpointed, 12 tables
└── documents/          125 MB   22 archives: one RHP + one anchor filing per issue
```

---

## Coverage, stated honestly

| | |
|---|---|
| Issues | 12 Indian mainboard IPOs |
| Bid windows spanned | 2026-09-07 → 2026-09-16 |
| **Observation period** | **2026-09-10, 05:08 → 17:14 IST — a single trading day** |
| Subscription snapshots | 1,774 rows (~148 per issue) |
| Verdicts / predictions | 383 each |
| Realised outcomes | **0** |

**Two limits that matter more than the row counts.**

*The observation window is one day.* Every timestamp in every table falls on
2026-09-10. It is a dense day — 148 order-book snapshots per issue, capturing
several books going from undersubscribed to covered — but it is one day. Any
claim about model accuracy *across* issues, sectors or market regimes is not
supported by this data.

*The `outcomes` table is empty.* Nothing here has listed yet, so no forecast in
`predictions` has been scored against reality. The calibration machinery
(`run.py calibrate`) is implemented and tested but has never had anything to
learn from. Treat every published accuracy figure as a model property, not a
measured result.

---

## Tables

### `ipos` — the universe
One row per issue, upserted on each discovery pass.

| Column | Notes |
|---|---|
| `symbol` | NSE symbol, primary key |
| `board` | `MAINBOARD` or `SME` (SME off by default) |
| `open_date` / `close_date` / `listing_date` | ISO dates |
| `price_low` / `price_high` | band floor and cap |
| `lot_size` | shares per lot — the retail allotment quantum |
| `issue_size_cr`, `fresh_issue_cr`, `ofs_cr` | ₹ crore. OFS is money to selling shareholders, not the company |
| `meta` | JSON: registrar, lead managers, raw NSE fields |

### `subscription` — the order book *(1,774 rows)*
The core time series. One row per issue **per category** per poll, every ~60–300s.

| Column | Notes |
|---|---|
| `category` | `QIB`, `NII`, `RETAIL`, `EMPLOYEE`, `sNII`, `bNII` |
| `offered` / `bid` | shares. `bid/offered` is the true subscription multiple |
| `times` | as reported by the exchange |
| `source` | `nse`, or `bse` when failover fired |

> **`sNII` and `bNII` are subdivisions of `NII`, not siblings.** Summing all six
> categories double-counts the non-institutional book. Use the `TOP_LEVEL`
> constant in `ipo_radar/models.py`.
>
> **The exchange's own total is usually absent.** It reads `NULL` even on a 215x
> book, so `SubscriptionSnapshot.total_subscription()` derives it from
> `bid/offered`. Code that treats a missing total as *undersubscribed* will
> condemn almost every issue — see the Winner's Curse notes in `HANDOFF.md`.

### `gmp` — grey market premium *(402 rows)*
Scraped from `ipowatch`. `gmp` in ₹ over the cap price; `est_gain_pct` derived.

> Unofficial, unregulated, thin, and trivially manipulated — quoted by dealers
> with a position. The scoring model caps its weight and applies a realisation
> haircut precisely because it is the most predictive *and* least trustworthy
> input available.

### `verdicts` — scored analysis *(383 rows)*
`score` (0–100), `grade` (`STRONG APPLY` … `AVOID`), and a `payload` JSON
carrying the full factor breakdown, allotment odds, valuation, regime, vetoes
and the listing-gain distribution. This is the richest table in the file.

### `predictions` — forecasts *(383 rows)*
`metric` with `value`/`lo`/`hi`, plus the `features` JSON the forecast was made
from. Designed to be joined against `outcomes` — which is empty. **This join is
the backtest, and it has never been run.**

### `financials` — parsed from the RHP *(11 rows)*
Revenue, EBITDA, PAT, EPS, NAV, RoNW, borrowings, cash, CFO, per fiscal year.

> Extracted from 400–600 page PDFs by table parsing with an LLM fallback.
> Roughly 91% field coverage; the rest read `NULL`. Sign errors on cash-flow
> figures were a recurring bug class — `"net cash used in operating
> activities"` means **negative** CFO. Spot-check before trusting.

### `macro` — rate environment *(53 rows)*
G-Sec yields (2y/5y/10y/30y) bootstrapped from live bond prices, term spread,
credit spread, index move.

> `credit_spread_source` is load-bearing. The corporate-bond feed returns empty,
> so this usually falls back to the 10y–2y **term** spread — a different
> quantity wearing the same name. Check the column before using it.

### `news` *(265 rows)* · `llm_cache` *(86)* · `events` *(254)* · `kv` *(0)* · `outcomes` *(0)*
Headlines with sentiment; cached LLM responses keyed by prompt hash; the
structured run log (`WARNING`/`ERROR` rows drive the briefing's health audit);
a general key-value store; and the empty realised-outcomes table.

---

## `data/documents/`

22 archives, `{SYMBOL}_RHP.zip` and `{SYMBOL}_ANCHOR.zip`. RHPs run 436–661
pages (5–19 MB); anchor filings are small.

These are **public regulatory filings** — prospectuses filed with SEBI and
published by the exchanges, mirrored here so the parse is reproducible after
the source URLs expire. Copyright rests with the respective issuers. They are
redistributed unmodified for research reproducibility, and the MIT licence on
this repository covers the code, not these documents.

Delete the directory and the engine re-downloads on next run (~45–90 s each).

---

## Querying it

```bash
sqlite3 data/ipo_radar.db
```

```sql
-- how a book filled through the day
SELECT ts, category, ROUND(times,2)
FROM subscription
WHERE symbol='PRASOLCHEM' AND category='QIB'
ORDER BY ts;

-- score trajectory
SELECT ts, ROUND(score,1), grade FROM verdicts
WHERE symbol='KANOHAR' ORDER BY ts;

-- the full factor breakdown
SELECT json_extract(payload,'$.factors') FROM verdicts
WHERE symbol='ARCIL' ORDER BY ts DESC LIMIT 1;
```

The Winner's Curse case study is in here directly:

```sql
SELECT ts, ROUND(times,2) AS qib FROM subscription
WHERE symbol='PRASOLCHEM' AND category='QIB' ORDER BY ts LIMIT 3;
-- 05:08 → 0.05x, while RETAIL already read 1.14x.
-- Allotment odds of 78.8% on a book institutions had refused.
```

## Regenerating

```bash
python3 run.py once      # one full cycle
python3 run.py run       # continuous
```

Writes are additive — history accumulates rather than overwriting, which is
what makes the time series usable. To start clean, delete `data/ipo_radar.db`.
