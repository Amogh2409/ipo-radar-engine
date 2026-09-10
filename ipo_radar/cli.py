"""Command line entry points."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .brain.calibration import calibrate_all, record_allotment_outcome
from .brain.modelfile import export_finetune_dataset
from .config import CONFIG_PATH, Settings, ensure_dirs
from .daemon import Daemon
from .engine import Engine
from .report import console, detail, print_dashboard, write_reports
from .store import Store
from .util import setup_logging


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ipo-radar",
        description="Continuous analysis of live Indian IPOs: fundamentals, "
                    "demand, allotment odds and where to put your money.")
    p.add_argument("--log", default=None, help="DEBUG|INFO|WARNING")
    p.add_argument("--no-llm", action="store_true",
                   help="skip the Ollama passes (quant only)")
    p.add_argument("--capital", type=float, default=None,
                   help="rupees available to block in ASBA")
    p.add_argument("--pans", type=int, default=None,
                   help="how many PANs you can apply from")
    p.add_argument("--objective", choices=("maximize_roi",
                                           "maximize_absolute_profit"),
                   default=None,
                   help="rank plans by return on blocked capital (default) or "
                        "by absolute rupees (promotes sNII on hot issues)")
    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("run", help="run continuously (default)")
    o = sub.add_parser("once", help="one full cycle, print and write reports")
    o.add_argument("--no-docs", action="store_true",
                   help="skip RHP downloads (much faster first run)")
    sub.add_parser("list", help="show the live IPO universe")
    a = sub.add_parser("analyse", help="deep-dive one symbol")
    a.add_argument("symbol")
    a.add_argument("--markdown", action="store_true")
    al = sub.add_parser("allot", help="allotment odds for one symbol")
    al.add_argument("symbol")
    al.add_argument("--applications", type=int, default=1)
    sub.add_parser("report", help="rewrite reports from stored data")
    mm = sub.add_parser("memo", help="dense 1-2 page tactical action memo (PDF)")
    mm.add_argument("--hours", type=int, default=96,
                    help="close-date horizon in hours (default 96)")
    mm.add_argument("--output-dir", type=Path, default=None)
    mm.add_argument("--engine", default="auto",
                    help="force a PDF engine: weasyprint|chrome|wkhtmltopdf|reportlab")
    mm.add_argument("--max-pages", type=int, default=2)
    mm.add_argument("--keep-html", action="store_true")
    sub.add_parser("calibrate", help="refit models from recorded outcomes")
    oc = sub.add_parser("outcome", help="record what actually happened")
    oc.add_argument("symbol")
    oc.add_argument("--listing-price", type=float, required=True)
    oc.add_argument("--retail-subscription", type=float, default=None)
    oc.add_argument("--allotment-rate", type=float, default=None,
                    help="applications allotted / applications made, 0-1")
    sub.add_parser("export-finetune", help="write a fine-tune JSONL corpus")
    sub.add_parser("config", help="print effective configuration")
    return p


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.load()
    if args.no_llm:
        s.llm.enabled = False
    if args.capital is not None:
        s.portfolio.total_capital = args.capital
    if args.pans is not None:
        s.portfolio.num_pans = args.pans
    if args.objective:
        s.portfolio.objective = args.objective
    if args.log:
        s.log_level = args.log
    return s


async def _once(engine: Engine, no_docs: bool = False) -> None:
    if no_docs:
        engine.docs.fetch_rhp = False
    await engine.refresh_universe()
    regime = await engine.market_regime()
    await engine.poll_subscription()
    await engine.poll_gmp()
    await engine.poll_news()
    verdicts = await engine.analyse_all(use_llm=engine.s.llm.enabled)
    allocation = engine.allocate(verdicts)
    print_dashboard(engine.store, verdicts, allocation, regime)
    paths = write_reports(engine.store, verdicts, allocation)
    console.print(f"\n[dim]wrote {len(paths)} files to reports/[/dim]")


async def _amain(args: argparse.Namespace) -> int:
    ensure_dirs()
    s = _settings(args)
    setup_logging(s.log_level)
    cmd = args.cmd or "run"

    if cmd == "config":
        from dataclasses import asdict
        console.print_json(json.dumps(asdict(s), default=str))
        console.print(f"[dim]config file: {CONFIG_PATH} "
                      f"({'present' if CONFIG_PATH.exists() else 'not created yet'})[/dim]")
        return 0

    if cmd == "memo":
        # The memo generator lives at the repository root beside run.py, not
        # inside the package - it is an operator tool, not part of the engine.
        # Imported lazily so a missing PDF stack never affects other commands.
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from generate_tactical_memo import build_memo
        except Exception as exc:
            console.print(f"[red]tactical memo generator unavailable: {exc}[/red]")
            return 1
        out_dir = args.output_dir or (root / "briefings")
        return build_memo(
            db_path=root / "data" / "ipo_radar.db", out_dir=out_dir,
            hours=max(1, args.hours), capital=s.portfolio.total_capital,
            pans=s.portfolio.num_pans, objective=s.portfolio.objective,
            engine=args.engine, keep_html=args.keep_html,
            max_pages=max(1, args.max_pages))

    if cmd == "calibrate":
        store = Store()
        console.print_json(json.dumps(calibrate_all(store, s), default=str))
        store.close()
        return 0

    if cmd == "export-finetune":
        store = Store()
        console.print_json(json.dumps(export_finetune_dataset(store), default=str))
        store.close()
        return 0

    if cmd == "outcome":
        store = Store()
        ipo = store.get_ipo(args.symbol.upper())
        if not ipo or not ipo.cap_price:
            console.print(f"[red]unknown symbol {args.symbol}[/red]")
            store.close()
            return 1
        gain = (args.listing_price - ipo.cap_price) / ipo.cap_price * 100.0
        store.set_outcome(args.symbol.upper(), listing_price=args.listing_price,
                          listing_gain_pct=gain, listing_date=ipo.listing_date)
        if args.allotment_rate is not None:
            x = args.retail_subscription
            if x is None:
                snap = store.latest_subscription(args.symbol.upper())
                x = snap.times("RETAIL") if snap else 0.0
            record_allotment_outcome(store, args.symbol.upper(), x,
                                     args.allotment_rate)
        console.print(f"recorded {args.symbol.upper()}: listed "
                      f"{gain:+.1f}% vs issue price")
        store.close()
        return 0

    engine = Engine(s)
    try:
        if cmd == "run":
            await Daemon(engine, s).run()
            return 0

        if cmd == "once":
            await _once(engine, no_docs=getattr(args, "no_docs", False))
            return 0

        if cmd == "list":
            ipos = await engine.refresh_universe()
            for i in sorted(ipos, key=lambda z: (z.status, z.close_date or "")):
                console.print(
                    f"{i.symbol:<14}{i.status:<12}"
                    f"₹{i.price_low or 0:.0f}-{i.price_high or 0:.0f}  "
                    f"lot {i.lot_size or '?':<5} "
                    f"{i.open_date}→{i.close_date}  "
                    f"₹{i.issue_size_cr or 0:,.0f} Cr   {i.name}")
            return 0

        if cmd == "analyse":
            await engine.refresh_universe()
            ipo = engine.store.get_ipo(args.symbol.upper())
            if not ipo:
                console.print(f"[red]unknown symbol {args.symbol}[/red]")
                return 1
            await engine.poll_subscription([ipo])
            await engine.poll_gmp([ipo])
            await engine.poll_news([ipo])
            regime = await engine.market_regime()
            v = await engine.analyse(ipo, use_llm=s.llm.enabled, regime=regime)
            if args.markdown:
                print(detail(engine.store, v))
            else:
                print_dashboard(engine.store, [v], None, regime)
                console.print(detail(engine.store, v))
            return 0

        if cmd == "allot":
            await engine.refresh_universe()
            ipo = engine.store.get_ipo(args.symbol.upper())
            if not ipo:
                console.print(f"[red]unknown symbol {args.symbol}[/red]")
                return 1
            await engine.poll_subscription([ipo])
            snap = engine.store.latest_subscription(ipo.symbol)
            proj = engine.projector.project(ipo, snap)
            gmp = engine.store.latest_gmp(ipo.symbol)
            odds = engine.allotment.analyse(
                ipo, snap, proj, num_applications=args.applications,
                gmp_pct=gmp.est_gain_pct if gmp else None,
                allow_snii=s.portfolio.allow_snii,
                allow_bnii=s.portfolio.allow_bnii)
            console.print(f"\n[bold]{ipo.name}[/bold]  lot {ipo.lot_size} "
                          f"@ ₹{ipo.cap_price:.0f} = ₹{ipo.lot_value:,.0f}\n")
            for key, o in odds.items():
                console.print(
                    f"  [cyan]{o.category:<7}[/cyan] {o.lots_applied:>3} lots "
                    f"₹{o.ticket_value:>10,.0f}  {o.subscription_x:>7.2f}x  "
                    f"[bold]{o.p_single:>6.1%}[/bold] "
                    f"({args.applications} PAN → {o.p_any_multi:.1%})  "
                    f"exp.shares={o.expected_lots * (ipo.lot_size or 0):>6.0f}  "
                    f"[dim]{o.mechanism}[/dim]")
            return 0

        if cmd == "report":
            verdicts = []
            for ipo in engine.live_ipos():
                p = engine.store.latest_verdict(ipo.symbol)
                if p:
                    from .models import AllotmentOdds, Verdict
                    p["allotment"] = {k: AllotmentOdds(**v)
                                      for k, v in (p.get("allotment") or {}).items()}
                    # JSON has no tuple type; restore it so regenerated reports
                    # carry the same range row that `once` prints.
                    rng = p.get("listing_gain_range")
                    p["listing_gain_range"] = tuple(rng) if isinstance(rng, list) else rng
                    verdicts.append(Verdict(**p))
            if not verdicts:
                console.print("[yellow]no stored verdicts — run `once` first[/yellow]")
                return 1
            verdicts.sort(key=lambda v: v.score, reverse=True)
            paths = write_reports(engine.store, verdicts)
            console.print(f"wrote {len(paths)} files")
            return 0
    finally:
        if cmd != "run":
            await engine.aclose()
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
