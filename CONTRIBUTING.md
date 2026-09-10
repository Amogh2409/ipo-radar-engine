# Contributing

## Running it

```bash
pip install -r requirements.txt   # httpx, rich, pypdf
python3 run.py once               # one cycle
python3 tests/test_winners_curse.py
```

The regression suite needs no pytest - it runs under plain `python3` - though
it works under pytest if you prefer. It does import the engine, so the
runtime dependencies in `requirements.txt` must be present.

## Two hard constraints

**Never hard-require a third-party package.** PEP 668 blocks `pip` on a
Homebrew Python, so pydantic, weasyprint and yfinance are genuinely
uninstallable for many users. Every optional dependency must degrade: pydantic
falls back to stdlib validation, the PDF renderer cascades WeasyPrint →
headless Chrome → wkhtmltopdf → ReportLab. If you add a dependency, add its
fallback in the same commit.

**Keep `ipo_radar.db` backward compatible.** Existing databases carry captured
history that cannot be re-collected — an order book only exists while an issue
is open. Migrations must be additive.

## Thresholds live in exactly one place

Score floors belong in `PortfolioConfig` (`ipo_radar/config.py`) and nowhere
else. This is not style. `tday_signals.py` once kept its own `MIN_SCORE = 50.0`
while the optimiser was raised to 61; the result was an issue refused capital
by the plan that still fired a CRITICAL *"the call is APPLY"* alert two minutes
before the bid cut-off. A test now asserts the two cannot diverge.

```bash
grep -rn "min_score" --include=*.py .   # literals should appear only in config.py
```

## What a change needs

- **A model change needs a test that fails without it.** The Winner's Curse
  suite is the template: it reproduces the bug from real captured data, proves
  the fix, and includes counter-tests so the fix cannot over-correct.
- **Say what the data does not support.** Claims about accuracy need an entry
  in `outcomes`, and that table is empty.
- **Comments explain *why*.** The code is dense with market mechanics that are
  not self-evident — why NII cannot revise a bid down, why a missing total is
  not a zero. Preserve that reasoning.

## Style

Standard library first. Type hints on public functions. No emoji in code or
terminal output. Match the surrounding density; this codebase comments the
non-obvious heavily and the obvious not at all.
