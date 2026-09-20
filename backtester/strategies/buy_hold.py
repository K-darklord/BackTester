"""Buy & Hold benchmark strategy (always fully invested)."""

from __future__ import annotations

import pandas as pd

from backtester.signal import Signal
from .base import BaseStrategy


class BuyAndHold(BaseStrategy):
    """Always hold 100% of the basket."""

    name = "buy_and_hold"

    def generate_signals(self, prices: pd.DataFrame) -> list[Signal]:
        return [Signal(date=prices.index[0], target_position=1.0, metadata={})]
