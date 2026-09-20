"""Backtest engine and performance metrics.

The engine walks through price history, applying strategy signals as target
positions. Signals are applied at the NEXT day's open (no look-ahead bias).

Usage:
    from backtester.core import BacktestEngine, compute_metrics
    from backtester.strategies.buy_hold import BuyAndHold

    engine = BacktestEngine(prices)
    result = engine.run(BuyAndHold())
    print(result.metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .signal import Signal


class Strategy(Protocol):
    """Strategy interface: generate target-position signals from prices."""

    name: str

    def generate_signals(self, prices: pd.DataFrame) -> list[Signal]:
        """Return a list of Signal objects (one per rebalance date)."""
        ...


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    strategy_name: str
    metrics: dict
    equity_curve: pd.Series
    signals: list[Signal]
    trades: pd.DataFrame

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"=== {self.strategy_name} ===",
            f"Total Return:    {m['total_return']:>10.2%}",
            f"CAGR:            {m['cagr']:>10.2%}",
            f"Sharpe:          {m['sharpe']:>10.3f}",
            f"Sortino:         {m['sortino']:>10.3f}",
            f"Max Drawdown:    {m['max_drawdown']:>10.2%}",
            f"Volatility:      {m['volatility']:>10.2%}",
            f"Win Rate:        {m['win_rate']:>10.2%}",
            f"# Trades:        {m['n_trades']:>10d}",
            f"TxCost (bps):    {m.get('transaction_cost_bps', 0):>10.2f}",
            f"TxCost Total:    {m.get('transaction_costs_total', 0):>10.4f}",
        ]
        return "\n".join(lines)


def compute_metrics(returns: pd.Series, n_trades: int = 0) -> dict:
    """Compute standard performance metrics from a daily returns series."""
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "max_drawdown": 0.0, "volatility": 0.0,
            "win_rate": 0.0, "n_trades": n_trades,
        }

    total_return = float(np.exp(returns.sum()) - 1.0)
    n_years = len(returns) / 252.0
    cagr = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1)
    vol = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252)) if returns.std(ddof=0) > 0 else 0.0
    downside = returns[returns < 0]
    sortino = (
        float(returns.mean() / downside.std(ddof=0) * np.sqrt(252))
        if len(downside) > 1 and downside.std(ddof=0) > 0 else 0.0
    )

    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = float(-drawdown.min())

    win_rate = float((returns > 0).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "volatility": vol,
        "win_rate": win_rate,
        "n_trades": n_trades,
    }


class BacktestEngine:
    """Walk-forward backtest engine.

    Converts strategy target-positions into daily returns.
    A signal generated on date T applies to the return on date T+1
    (next-day execution, no look-ahead bias).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        transaction_cost_bps: float = 10.0,
    ):
        """prices: DataFrame of prices indexed by date, columns = assets.

        transaction_cost_bps: per-side cost in basis points (10bp = 0.1%).
        A round-trip (0 -> 1 -> 0) therefore costs 2 * transaction_cost_bps.
        """
        self.prices = prices.sort_index()
        self.initial_capital = initial_capital
        self.transaction_cost_bps = float(transaction_cost_bps)

    def run(self, strategy: Strategy, benchmark: Strategy | None = None) -> BacktestResult:
        """Run the strategy and return metrics + equity curve.

        If `benchmark` is provided, computes excess return vs benchmark.
        """
        # Equal-weight basket returns (or single asset)
        if isinstance(self.prices, pd.Series):
            basket_returns = self.prices.pct_change().dropna()
        else:
            basket_returns = self.prices.pct_change().dropna().mean(axis=1)

        # Generate signals (reindex to full price index first, then to returns)
        signals = strategy.generate_signals(self.prices)
        sig_full = pd.Series(
            {s.date: s.target_position for s in signals}
        ).reindex(self.prices.index).ffill().fillna(0.0)
        sig_series = sig_full.reindex(basket_returns.index).ffill().fillna(0.0)

        # Shift signal by 1 day (signal at T applies to T+1 return)
        position = sig_series.shift(1).fillna(0.0)

        # Strategy returns = position * basket_return
        strat_returns = basket_returns * position

        # Count trades (position changes)
        pos_diff = position.diff().abs()
        n_trades = int((pos_diff > 0.001).sum())

        # FORK EXTENSION: deduct transaction costs on position changes.
        # cost_bps is per-side; a round-trip (e.g., 0->1->0) costs 2*cost_bps.
        # Apply cost as a daily return drag proportional to abs(position change).
        cost_per_unit = self.transaction_cost_bps / 10000.0
        trade_costs = pos_diff.fillna(0.0) * cost_per_unit
        strat_returns = strat_returns - trade_costs
        transaction_costs_total = float(trade_costs.sum())

        metrics = compute_metrics(strat_returns, n_trades=n_trades)
        metrics["transaction_costs_total"] = transaction_costs_total
        metrics["transaction_cost_bps"] = self.transaction_cost_bps

        # Equity curve
        equity = self.initial_capital * (1 + strat_returns).cumprod()

        # Trades log (position changes only)
        trade_mask = (pos_diff > 0.001).fillna(False)
        trades = pd.DataFrame({
            "date": position.index,
            "position": position.values,
            "basket_return": basket_returns.values,
            "strategy_return": strat_returns.values,
        })
        trades = trades[trade_mask.values]

        result = BacktestResult(
            strategy_name=strategy.name,
            metrics=metrics,
            equity_curve=equity,
            signals=signals,
            trades=trades,
        )

        # Benchmark comparison
        if benchmark is not None:
            bench_signals = benchmark.generate_signals(self.prices)
            bench_full = pd.Series(
                {s.date: s.target_position for s in bench_signals}
            ).reindex(self.prices.index).ffill().fillna(0.0)
            bench_sig = bench_full.reindex(basket_returns.index).ffill().fillna(0.0).shift(1).fillna(0.0)
            bench_returns = basket_returns * bench_sig
            bench_metrics = compute_metrics(bench_returns)
            metrics["benchmark_return"] = bench_metrics["total_return"]
            metrics["benchmark_sharpe"] = bench_metrics["sharpe"]
            metrics["excess_return"] = metrics["total_return"] - bench_metrics["total_return"]
            metrics["excess_sharpe"] = metrics["sharpe"] - bench_metrics["sharpe"]

        return result

    def save_result(self, result: BacktestResult, output_dir: str | Path = "results") -> Path:
        """Save backtest result to disk (equity curve + metrics)."""
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{result.strategy_name}_{ts}.csv"
        result.equity_curve.to_csv(out / fname)
        # Also save metrics as JSON
        import json
        with open(out / f"{result.strategy_name}_{ts}_metrics.json", "w") as f:
            json.dump(result.metrics, f, indent=2)
        return out / fname
