"""CLI for running backtests.

Usage:
    backtester run US --years 5
    backtester run ASHARE --years 3
    backtester compare US
    backtester list
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Backtester CLI")
    sub = parser.add_subparsers(dest="command")

    # run command
    run_p = sub.add_parser("run", help="Run a backtest")
    run_p.add_argument("market", choices=["US", "HK", "ASHARE"])
    run_p.add_argument("--years", type=int, default=5)
    run_p.add_argument("--strategy", default="strategist_regime",
                       choices=["strategist_regime", "buy_hold"])
    run_p.add_argument("--save", action="store_true", help="Save results to disk")
    # FORK EXTENSION: cost + notes flags
    run_p.add_argument("--cost-bps", type=float, default=10.0,
                       help="Transaction cost in basis points (per-side, default 10bp)")
    run_p.add_argument("--notes", type=str, default="",
                       help="Free-form note attached to the saved run")

    # compare command
    cmp_p = sub.add_parser("compare", help="Compare strategy vs buy-and-hold")

    # compare-strategies command (multi-strategy)
    cmps_p = sub.add_parser("compare-strategies", help="Compare all strategies (buy_hold, strategist_regime, hmm_strategist)")
    cmps_p.add_argument("market", choices=["US", "HK", "ASHARE"])
    cmps_p.add_argument("--years", type=int, default=5)

    cmp_p.add_argument("market", choices=["US", "HK", "ASHARE"])
    cmp_p.add_argument("--years", type=int, default=5)

    # FORK EXTENSION: list subcommand
    list_p = sub.add_parser("list", help="List recent saved backtest runs")
    list_p.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    if args.command == "run":
        _run_backtest(args.market, args.years, args.strategy, args.save,
                      cost_bps=args.cost_bps, notes=args.notes)
    elif args.command == "compare":
        _compare(args.market, args.years)
    elif args.command == "compare-strategies":
        _compare_strategies(args.market, args.years)
    elif args.command == "list":
        _list_runs(args.limit)
    else:
        parser.print_help()


def _run_backtest(market: str, years: int, strategy_name: str, save: bool,
                  cost_bps: float = 10.0, notes: str = ""):
    from backtester.core import BacktestEngine
    from backtester.data import fetch_etf_basket
    from backtester.strategies.buy_hold import BuyAndHold
    from backtester.strategies.strategist_regime import StrategistRegimeStrategy
    from backtester import storage

    print(f"[backtester] Fetching {market} data ({years}y)...")
    prices = fetch_etf_basket(market=market)
    print(f"[backtester] Prices: {prices.shape}")

    if strategy_name == "strategist_regime":
        strategy = StrategistRegimeStrategy(market=market)
    else:
        strategy = BuyAndHold()

    # FORK EXTENSION: pass transaction_cost_bps to engine
    engine = BacktestEngine(prices, transaction_cost_bps=cost_bps)
    result = engine.run(strategy, benchmark=BuyAndHold())
    print(result.summary())

    # FORK EXTENSION: always persist to SQLite history
    run_id = storage.save_run(
        market=market,
        strategy=strategy.name if hasattr(strategy, "name") else strategy_name,
        years=years,
        metrics=result.metrics,
        equity_curve=result.equity_curve,
        notes=notes,
    )
    print(f"[backtester] Saved run id={run_id} (cost_bps={cost_bps}, notes='{notes}')")

    if save:
        path = engine.save_result(result)
        print(f"[backtester] CSV saved to {path}")


def _list_runs(limit: int = 20):
    """Print a table of recent backtest runs."""
    from backtester import storage

    runs = storage.list_runs(limit=limit)
    if not runs:
        print("(no saved runs)")
        return
    cols = ["id", "ts", "market", "strategy", "total_return", "sharpe",
            "max_drawdown", "n_trades", "transaction_costs_total", "notes"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in runs), default=0))
              for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in runs:
        row = []
        for c in cols:
            v = r.get(c, "")
            if c in ("total_return", "max_drawdown") and v != "":
                try:
                    v = f"{float(v):.4f}"
                except Exception:
                    pass
            elif c in ("sharpe", "transaction_costs_total") and v != "":
                try:
                    v = f"{float(v):.4f}"
                except Exception:
                    pass
            row.append(str(v).ljust(widths[c]))
        print("  ".join(row))


def _compare(market: str, years: int):
    from backtester.core import BacktestEngine
    from backtester.data import fetch_etf_basket
    from backtester.strategies.buy_hold import BuyAndHold
    from backtester.strategies.strategist_regime import StrategistRegimeStrategy

    print(f"[backtester] Fetching {market} data ({years}y)...")
    prices = fetch_etf_basket(market=market)
    print(f"[backtester] Prices: {prices.shape}")

    engine = BacktestEngine(prices)
    strat = StrategistRegimeStrategy(market=market)
    bh = BuyAndHold()

    strat_result = engine.run(strat, benchmark=bh)
    bh_result = engine.run(bh)

    print(f"\n{'='*60}")
    print(f"BACKTEST COMPARISON — {market}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Buy & Hold':<15} {'Strategy':<15}")
    print(f"{'-'*50}")
    for k in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "volatility"):
        print(f"{k:<20} {bh_result.metrics[k]:>14.4f} {strat_result.metrics[k]:>14.4f}")
    print(f"\nExcess Return:  {strat_result.metrics.get('excess_return', 0):.4f}")
    print(f"Excess Sharpe:  {strat_result.metrics.get('excess_sharpe', 0):.4f}")




def _compare_strategies(market: str, years: int):
    """Run all three strategies and print a comparison table."""
    from backtester.core import BacktestEngine
    from backtester.data import fetch_etf_basket
    from backtester.strategies.buy_hold import BuyAndHold
    from backtester.strategies.strategist_regime import StrategistRegimeStrategy
    from backtester.strategies.hmm_strategist import HMMStrategistStrategy

    print(f"[backtester] Fetching {market} data ({years}y)...")
    prices = fetch_etf_basket(market=market)
    print(f"[backtester] Prices: {prices.shape}")

    engine = BacktestEngine(prices)
    strats = [
        ("buy_hold", BuyAndHold()),
        ("strategist_regime", StrategistRegimeStrategy(market=market)),
        ("hmm_strategist", HMMStrategistStrategy(market=market)),
    ]
    results = []
    for name, strat in strats:
        try:
            res = engine.run(strat, benchmark=BuyAndHold())
            results.append((name, res))
        except Exception as e:
            print(f"  {name} failed: {e}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON — {market} ({years}y)")
    print(f"{'='*80}")
    metrics = ["total_return", "cagr", "sharpe", "sortino", "max_drawdown", "volatility", "win_rate", "n_trades"]
    header = f"{'Strategy':<22}" + "".join(f"{m:>14}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, res in results:
        m = res.metrics
        row = f"{name:<22}" + "".join(f"{m.get(k, 0):>14.4f}" for k in metrics)
        print(row)
    print()

if __name__ == "__main__":
    main()
