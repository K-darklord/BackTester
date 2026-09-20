"""Analyst stock-picking strategy.

Backtests the Analyst agent's BUY/SELL/HOLD decisions on individual stocks.
Currently uses a simple rule: hold stocks rated BUY, flat on SELL.

This is a placeholder for the full Analyst decision backtesting pipeline.
"""

from __future__ import annotations

import pandas as pd

from backtester.signal import Signal
from .base import BaseStrategy


class AnalystPicksStrategy(BaseStrategy):
    """Position sizing based on Analyst ratings.

    Parameters
    ----------
    ratings : pd.Series
        Series indexed by date with rating values ("BUY", "HOLD", "SELL").
    """

    name = "analyst_picks"

    def __init__(self, ratings: pd.Series | None = None):
        self.ratings = ratings

    def generate_signals(self, prices: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []
        if self.ratings is None:
            return signals

        for date, rating in self.ratings.items():
            if date not in prices.index:
                continue
            if rating == "BUY":
                pos = 1.0
            elif rating == "SELL":
                pos = 0.0
            else:  # HOLD / neutral
                pos = 0.5
            signals.append(Signal(
                date=date,
                target_position=pos,
                metadata={"rating": rating},
            ))
        return signals
