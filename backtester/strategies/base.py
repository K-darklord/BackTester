"""Base strategy class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from backtester.signal import Signal


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> list[Signal]:
        """Generate target-position signals from price data."""
        ...
