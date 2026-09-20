"""HMM-based Strategist strategy for the Backtester.

Uses the trained 7-state HMM (strategist.hmm_regime) to detect market
regime every N days, then maps regime -> target position via the same
position_multipliers table the Strategist uses in production.

Mirrors strategist_regime.py but with HMM as the regime detector (no ORCA
thresholds, no spectral recomputation per step) so it is much faster and
reflects the production HMM-first code path.
"""

from __future__ import annotations

import pandas as pd

# Strategist repo on the same machine
from backtester._paths import ensure_repos
ensure_repos("Strategist")

from backtester.signal import Signal
from .base import BaseStrategy


class HMMStrategistStrategy(BaseStrategy):
    """Position sizing based on the trained 7-state HMM regime detector.

    Parameters
    ----------
    market : str
        "US", "HK", or "ASHARE".
    rebalance_freq : int
        Re-evaluate regime every N trading days (default 5).
    lookback_window : int
        Rolling window of daily returns fed to detect_regime (default 60).
    """

    name = "hmm_strategist"

    def __init__(
        self,
        market: str = "US",
        rebalance_freq: int = 5,
        lookback_window: int = 60,
    ):
        self.market = market
        self.rebalance_freq = rebalance_freq
        self.lookback_window = lookback_window
        self.name = f"hmm_strategist_{market}"

    def generate_signals(self, prices: pd.DataFrame) -> list[Signal]:
        import numpy as np
        from strategist.hmm_regime import detect_regime
        from strategist.config import StrategistConfig

        config = StrategistConfig()
        window = self.lookback_window

        # Daily log returns, drop the first NaN row
        returns = np.log(prices / prices.shift(1)).iloc[1:]
        # For multi-asset baskets, returns is a DataFrame; HMM expects a
        # DataFrame of asset returns (it computes the basket mean internally
        # via _build_emission_features).
        returns = returns.dropna(how="all")

        signals: list[Signal] = []
        for end in range(window, len(returns) + 1, self.rebalance_freq):
            sub = returns.iloc[end - window: end]
            if sub.isna().any().any():
                # Skip windows with missing data
                continue
            try:
                sig = detect_regime(sub, market=self.market)
                # Look up the same position multiplier the Strategist uses.
                pos_override = float(
                    config.position_multipliers.get(sig.regime, 0.8)
                )
                # position_override can be 0.0-1.2; clamp to [0, 1] for backtest
                target_pos = max(0.0, min(1.0, pos_override))
                signals.append(Signal(
                    date=returns.index[end - 1],
                    target_position=target_pos,
                    metadata={
                        "regime": sig.regime,
                        "state_idx": sig.state_idx,
                        "state_prob": float(sig.state_prob),
                        "expected_return": float(sig.expected_return),
                        "expected_vol": float(sig.expected_vol),
                        "position_override": pos_override,
                    },
                ))
            except Exception:
                # Skip this rebalance — strategy is robust to isolated failures
                continue
        return signals
