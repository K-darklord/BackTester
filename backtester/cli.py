"""CLI for running backtests.

Usage:
    backtester run US --years 5
    backtester run ASHARE --years 3
    backtester compare US
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

    # compare command
    cmp_p = sub.add_parser("compare", help="Compare strategy vs buy-and-hold")
    cmp_p.add_argument("market", choices=["US", "HK", "ASHARE"])
    cmp_p.add_argument("--years", type=int, default=5)

    args = parser.parse_args()

    if args.command == "run":
        _run_backtest(args.market, args.years, args.strategy, args.save)
    elif args.command == "compare":
        _compare(args.market, args.years)
    else:
        parser.print_help()


def _run_backtest(market: str, years: int, strategy_name: str, save: bool):
    from backtester.core import BacktestEngine
    from backtester.data import fetch_etf_basket
    from backtester.strategies.buy_hold import BuyAndHold
    from backtester.strategies.strategist_regime import StrategistRegimeStrategy

    print(f"[backtester] Fetching {market} data ({years}y)...")
    prices = fetch_etf_basket(market=market)
    print(f"[backtester] Prices: {prices.shape}")

    if strategy_name == "strategist_regime":
        strategy = StrategistRegimeStrategy(market=market)
    else:
        strategy = BuyAndHold()

    engine = BacktestEngine(prices)
    result = engine.run(strategy, benchmark=BuyAndHold())
    print(result.summary())
    if save:
        path = engine.save_result(result)
        print(f"[backtester] Saved to {path}")


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


if __name__ == "__main__":
    main()
