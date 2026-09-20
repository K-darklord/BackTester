"""Strategist regime-based position sizing strategy.

Walks through price history, recomputes the Strategist's regime signal
every N days, and converts position_override (0.0-1.2) into a target
position (clamped to [0, 1]).

This is the core strategy for backtesting the Strategist agent's decisions.
"""

from __future__ import annotations

import pandas as pd

# Add Strategist repo to path
from backtester._paths import ensure_repos
ensure_repos("Strategist")

from backtester.signal import Signal
from .base import BaseStrategy


class StrategistRegimeStrategy(BaseStrategy):
    """Position sizing based on Strategist regime detection.

    Parameters
    ----------
    market : str
        "US", "HK", or "ASHARE".
    rebalance_freq : int
        Re-evaluate regime every N trading days.
    lookback_window : int
        Rolling window for spectral features (default 90).
    """

    name = "strategist_regime"

    def __init__(
        self,
        market: str = "US",
        rebalance_freq: int = 5,
        lookback_window: int = 90,
    ):
        self.market = market
        self.rebalance_freq = rebalance_freq
        self.lookback_window = lookback_window
        self.name = f"strategist_regime_{market}"

    def generate_signals(self, prices: pd.DataFrame) -> list[Signal]:
        from strategist.spectral_features import rolling_spectral_features
        from strategist.orca_detector import detect_trend
        from strategist.omd_detector import detect_collapse
        from strategist.signal_combiner import combine_signals
        from strategist.config import StrategistConfig

        config = StrategistConfig()
        window = self.lookback_window

        # Compute log returns
        import numpy as np
        returns = np.log(prices / prices.shift(1)).iloc[1:]
        returns = returns.dropna(how="all")

        signals: list[Signal] = []
        for end in range(window, len(returns) + 1, self.rebalance_freq):
            sub = returns.iloc[end - window : end]
            if sub.isna().any().any():
                continue
            try:
                spectral = rolling_spectral_features(
                    sub, window=window, top_k=config.absorption_top_k
                )
                if spectral.empty:
                    continue
                orca = detect_trend(sub, spectral, config, market=self.market)
                omd = detect_collapse(spectral, config)
                combined = combine_signals(orca, omd, config)
                # position_override can be 0.0-1.2; clamp to [0, 1] for backtest
                target_pos = max(0.0, min(1.0, combined.position_override))
                signals.append(Signal(
                    date=returns.index[end - 1],
                    target_position=target_pos,
                    metadata={
                        "regime": combined.regime,
                        "position_override": combined.position_override,
                        "rally_prob": orca.rally_prob,
                        "crash_prob": orca.crash_prob,
                        "omd_state": omd.state,
                    },
                ))
            except Exception:
                continue

        return signals
