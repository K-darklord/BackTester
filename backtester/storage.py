"""SQLite-backed history of backtest runs.

Stores one row per run with metrics + downsampled equity curve so the
dashboard can render a scrollable history table without re-reading CSVs.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    market TEXT NOT NULL,
    strategy TEXT NOT NULL,
    years INTEGER,
    total_return REAL,
    cagr REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown REAL,
    volatility REAL,
    win_rate REAL,
    n_trades INTEGER,
    benchmark_return REAL,
    excess_return REAL,
    excess_sharpe REAL,
    transaction_costs_total REAL,
    transaction_cost_bps REAL,
    notes TEXT,
    metrics_json TEXT,
    equity_curve_json TEXT
);
"""


def get_db_path() -> Path:
    """Return path to the SQLite database file (creating parent dir)."""
    p = Path(__file__).parent.parent / "data" / "backtest_runs.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Create the backtest_runs table if it does not exist."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def _downsample_equity(equity_curve: Any, target_points: int = 60) -> list[dict]:
    """Downsample an equity curve (pandas Series or list) to ~target_points."""
    try:
        import pandas as pd  # local import keeps module import cheap
        if isinstance(equity_curve, pd.Series):
            if len(equity_curve) <= target_points:
                idx = equity_curve.index
                vals = equity_curve.values
            else:
                step = max(1, len(equity_curve) // target_points)
                idx = equity_curve.index[::step]
                vals = equity_curve.values[::step]
            return [
                {"t": str(i), "v": float(v)} for i, v in zip(idx, vals)
            ]
    except Exception:
        pass
    if isinstance(equity_curve, (list, tuple)):
        return [{"t": str(i), "v": float(v)} for i, v in enumerate(equity_curve)]
    return []


def save_run(
    market: str,
    strategy: str,
    years: int,
    metrics: dict,
    equity_curve: Any,
    notes: str = "",
) -> int:
    """Insert a new backtest run row and return its id."""
    init_db()
    eq_json = json.dumps(_downsample_equity(equity_curve))
    metrics_json = json.dumps(metrics, default=float)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO backtest_runs (
                ts, market, strategy, years,
                total_return, cagr, sharpe, sortino,
                max_drawdown, volatility, win_rate, n_trades,
                benchmark_return, excess_return, excess_sharpe,
                transaction_costs_total, transaction_cost_bps,
                notes, metrics_json, equity_curve_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ts, market, strategy, years,
                float(metrics.get("total_return", 0.0)),
                float(metrics.get("cagr", 0.0)),
                float(metrics.get("sharpe", 0.0)),
                float(metrics.get("sortino", 0.0)),
                float(metrics.get("max_drawdown", 0.0)),
                float(metrics.get("volatility", 0.0)),
                float(metrics.get("win_rate", 0.0)),
                int(metrics.get("n_trades", 0) or 0),
                float(metrics.get("benchmark_return", 0.0) or 0.0),
                float(metrics.get("excess_return", 0.0) or 0.0),
                float(metrics.get("excess_sharpe", 0.0) or 0.0),
                float(metrics.get("transaction_costs_total", 0.0) or 0.0),
                float(metrics.get("transaction_cost_bps", 0.0) or 0.0),
                notes,
                metrics_json,
                eq_json,
            ),
        )
        rid = cur.lastrowid
        conn.commit()
    return int(rid)


def list_runs(limit: int = 100) -> list[dict]:
    """Return recent runs, newest first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, ts, market, strategy, years,
                   total_return, cagr, sharpe, sortino,
                   max_drawdown, volatility, win_rate, n_trades,
                   benchmark_return, excess_return, excess_sharpe,
                   transaction_costs_total, transaction_cost_bps,
                   notes
            FROM backtest_runs
            ORDER BY ts DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_notes(run_id: int, notes: str) -> bool:
    """Update the notes column of a run. Returns True if a row was updated."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE backtest_runs SET notes = ? WHERE id = ?",
            (notes, int(run_id)),
        )
        conn.commit()
        return cur.rowcount > 0


def get_run(run_id: int) -> dict | None:
    """Return a single run including the full metrics + equity JSON."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM backtest_runs WHERE id = ?",
            (int(run_id),),
        ).fetchone()
    return dict(row) if row else None
