"""Centralised sys.path management for Fund subrepos.

Avoids scattered ``sys.path.insert`` calls across strategy modules.
Mirrors ``dashboard.app._ensure_repos`` but kept local to the Backtester
package so we do not create a cross-repo import dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_REPO_PATHS = {
    "Strategist":  f"{_REPO_ROOT}/Strategist",
    "Backtester":  f"{_REPO_ROOT}/Backtester",
    "Analyst":     f"{_REPO_ROOT}/Analyst",
    "Auditor":     f"{_REPO_ROOT}/Auditor",
    "Researcher":  f"{_REPO_ROOT}/Researcher",
    "Trader":      f"{_REPO_ROOT}/Trader",
    "Manager":     f"{_REPO_ROOT}/Manager",
    # FundTeam is the orchestration layer (not an agent repo) but lives
    # alongside them and is imported for shared helpers (decision_log, etc.).
    "FundTeam":    f"{_REPO_ROOT}/FundTeam",
}


def ensure_repos(*names: str) -> None:
    """Add named Fund subrepos to ``sys.path`` if missing (idempotent)."""
    for _n in names:
        _p = _REPO_PATHS.get(_n)
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)
