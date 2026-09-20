"""Signal contract for the Backtester.

CRITICAL DESIGN DECISION (learned from experience):
  A strategy's `signal` is the TARGET POSITION (float in [0, 1]), NOT a
  trade instruction (1 = buy, -1 = sell). The backtester converts target
  positions into trades internally. This avoids the classic bug where -1
  gets misinterpreted as "short" instead of "flat".

  - signal = 1.0  → fully invested
  - signal = 0.0  → flat (no position)
  - signal = 0.5  → half invested
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    """A strategy signal for a single date.

    Attributes
    ----------
    date : pd.Timestamp
        The date the signal is generated (for the NEXT trading day's open).
    target_position : float
        Target position as a fraction of capital in [0.0, 1.0].
    metadata : dict
        Strategy-specific details (e.g., regime, probabilities, reason).
    """
    date: object  # pd.Timestamp
    target_position: float
    metadata: dict

    def __post_init__(self):
        self.target_position = max(0.0, min(1.0, float(self.target_position)))
