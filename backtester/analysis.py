"""DDTW event-evaluation scoring for the Backtester.

This module formalises the shape-similarity scoring rule (agreed with kevin,
2026-09-21) for evaluating event-driven probability predictions (e.g. ORCA
rally/crash curves) against realised returns. It is INTENDED FOR EVENT STUDIES
only -- see the two caveats at the bottom of this docstring.

Rule overview
-------------
1. Inputs are two curves over the SAME event window (+-30 calendar days):
   - a probability curve (e.g. ORCA ``rally_prob``), and
   - a realised-return curve (e.g. 10-day forward return of the tech sector).

2. Both curves are z-score standardised internally (mean=0, std=1), then
   differentiated (DDTW = Derivative Dynamic Time Warping) so the metric is
   scale-invariant and shape-focused, tolerating a +-20-day timing error via a
   Sakoe-Chiba band.

3. ``shape_score = max(0, 1 - DDTW_pred / MAX)`` where ``MAX`` is a
   PREDICTION-SPECIFIC random baseline: the mean DDTW between the prediction
   and the time-shuffled true curve (200 trials). This anchors 0 = random,
   1 = perfect shape, <0 = worse than random (clamped to 0).

4. Direction is checked with ``aligned Pearson`` -- the Pearson correlation of
   the two derivative sequences computed along the DDTW warping path (so a
   valid timing offset is NOT penalised, unlike synchronous Pearson).

5. Final score uses a SIGN GATE (not magnitude multiplication):
   ``final = shape_score if aligned_corr > 0 else 0.0``. This keeps the shape
   score intact while zeroing out direction-reversed predictions.

Dual-curve scoring: ORCA emits ``(rally_prob, crash_prob)``. These are scored
separately:
   - ``rally_score = DDTW(rally_prob, +forward_return)``
   - ``crash_score = DDTW(crash_prob, -forward_return)``

CAVEAT 1 -- crash_prob aligned Pearson is non-monotonic, KEEP INVESTIGATING:
   crash_prob does NOT relate monotonically to realised return. It rises both
   during a fragile bull top (anticipating a crash) AND during the crash
   itself. The aligned Pearson therefore cannot be read as a clean
   "right/wrong direction" sign for crash the way it can for rally. Do not
   over-interpret the crash aligned Pearson until this is resolved.

CAVEAT 2 -- event studies only:
   This rule compares a probability curve against a realised move over a
   bounded event window. It is meaningless where there is no event (e.g. the
   US market did not react to the 2024-09-24 A-share stimulus, so there is no
   rally/crash signal to score). Only apply it to markets/events that actually
   moved.

The core implementation lives in Strategist (``strategist.ddtw_eval``); this
module is a thin, Backtester-importable wrapper so the rule and its caveats
live next to the rest of the evaluation tooling.
"""

from __future__ import annotations

import numpy as np

from backtester._paths import ensure_repos
ensure_repos("Strategist")

from strategist.ddtw_eval import score  # noqa: E402


def score_event_probabilities(
    rally_prob: np.ndarray,
    crash_prob: np.ndarray,
    forward_return: np.ndarray,
    n_trials: int = 200,
    window: int = 20,
) -> dict:
    """Score an event window's ORCA-style probability curves via DDTW.

    Parameters
    ----------
    rally_prob, crash_prob:
        Probability series over the event window (same length as returns).
    forward_return:
        Realised forward-return curve (e.g. 10-day) over the same window.
        NaN entries are dropped together with the matching probability
        entries before scoring.
    n_trials:
        Number of shuffles for the random baseline (default 200).
    window:
        Sakoe-Chiba band width in days (default 20, >= the 10-day horizon).

    Returns
    -------
    dict with, for each of rally and crash:
        shape  : shape_score = max(0, 1 - DDTW/MAX)
        aligned: DDTW-aligned Pearson correlation (direction check)
        final  : shape if aligned > 0 else 0.0
    """
    rally_prob = np.asarray(rally_prob, dtype=float)
    crash_prob = np.asarray(crash_prob, dtype=float)
    forward_return = np.asarray(forward_return, dtype=float)

    valid = ~np.isnan(forward_return)
    r, c, f = rally_prob[valid], crash_prob[valid], forward_return[valid]

    def _one(pred: np.ndarray, true: np.ndarray) -> dict:
        final, ddtw, mx, aligned = score(pred, true, n_trials=n_trials, window=window)
        shape = max(0.0, 1.0 - ddtw / mx) if mx > 1e-10 else 0.0
        return {"shape": float(shape), "aligned": float(aligned), "final": float(final)}

    return {
        "rally": _one(r, f),
        "crash": _one(c, -f),
    }
