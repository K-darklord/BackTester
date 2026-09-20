"""Agent signal backfiller — compute realised PnL for logged decisions.

Pulls decisions from FundTeam's ``decision_log`` SQLite DB (those that have
no outcome row yet), fetches prices for each decision's ticker, computes
forward returns over 1d / 5d / 10d / 20d windows, and writes them back
via ``decision_log.update_outcome``.

For market-level decisions with no ticker (e.g. Strategist regime calls),
a representative benchmark ETF per market is used:

  US      -> SPY
  ASHARE  -> 510300.SH  (CSI300 ETF via akshare)
  HK      -> FXI

Run::

    python -m backtester.agent_signal_backtester            # backfill all
    python -m backtester.agent_signal_backtester --dry-run  # preview only
    python -m backtester.agent_signal_backtester --limit 20
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Make FundTeam (decision_log) and Strategist (price fetchers) importable.
# Both repos live on the same machine as the Backtester.
from backtester._paths import ensure_repos
ensure_repos("FundTeam", "Strategist")
os.environ.setdefault("NO_PROXY", "*")

from dashboard.decision_log import (  # noqa: E402
    list_pending_outcomes,
    update_outcome,
)
from strategist.data_fetcher import (  # noqa: E402
    _fetch_akshare_etf,
    _fetch_tushare_ashare,
    _fetch_tushare_sw,
    _fetch_tushare_us,
    _fetch_yfinance_etf,
)


# Representative benchmark per market (used when ticker is None).
_MARKET_PROXY: dict[str, str] = {
    "US": "SPY",
    "ASHARE": "510300.SH",
    "HK": "FXI",
}

_WINDOWS: tuple[int, ...] = (1, 5, 10, 20)


def _fetch_ashare_stock(ticker: str, start: str, end: str) -> pd.Series:
    """A-share individual stock via akshare (qfq adjusted)."""
    try:
        import akshare as ak
    except Exception:
        return pd.Series(dtype=float)
    code = ticker.upper().replace(".SH", "").replace(".SZ", "")
    try:
        raw = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
    except Exception:
        return pd.Series(dtype=float)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    raw["日期"] = pd.to_datetime(raw["日期"])
    return raw.set_index("日期")["收盘"].astype(float).rename(ticker).sort_index()


def _fetch_hk_stock(ticker: str, start: str, end: str) -> pd.Series:
    """HK individual stock via akshare (qfq adjusted)."""
    try:
        import akshare as ak
    except Exception:
        return pd.Series(dtype=float)
    code = ticker.upper().replace(".HK", "")
    try:
        raw = ak.stock_hk_hist(
            symbol=code, period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
    except Exception:
        return pd.Series(dtype=float)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    raw["日期"] = pd.to_datetime(raw["日期"])
    return raw.set_index("日期")["收盘"].astype(float).rename(ticker).sort_index()


def fetch_close(ticker: str, start: str, end: str) -> pd.Series:
    """Dispatch to the right fetcher based on ticker format.

    Order of preference matches the project's data-source configuration:
    tushare first (most stable), akshare second, yfinance last.
    """
    if not ticker:
        return pd.Series(dtype=float)
    up = ticker.upper()

    # A-share: .SH / .SZ suffix
    if up.endswith((".SH", ".SZ")):
        # 申万 sector indices (801xxx.SI handled separately by _fetch_tushare_sw)
        if up.endswith(".SI"):
            return _fetch_tushare_sw(up, start, end)
        # A-share ETFs (510xxx, 159xxx) -> akshare fund_etf_hist_em
        if up.startswith(("510", "511", "512", "513", "515", "516", "159", "501")):
            return _fetch_akshare_etf(up, start, end)
        # Individual A-share stocks: tushare first (stable), akshare fallback
        s = _fetch_tushare_ashare(up, start, end)
        if s.empty:
            s = _fetch_ashare_stock(up, start, end)
        return s

    # HK: .HK suffix or pure digit
    if up.endswith(".HK") or (up.isdigit() and len(up) <= 5):
        return _fetch_hk_stock(up, start, end)

    # US: tushare us_daily first (stable), yfinance as fallback
    s = _fetch_tushare_us(up, start, end)
    if s.empty:
        s = _fetch_yfinance_etf(up, start, end)
    return s


def compute_forward_pnls(
    prices: pd.Series,
    decision_ts: pd.Timestamp,
    windows: tuple[int, ...] = _WINDOWS,
) -> dict:
    """Compute forward PnLs from decision_ts.

    Entry: close on the first trading day strictly after decision_ts.
    Exit : close N trading days later (close-to-close return).

    Returns ``{"pnl_1d": .., "pnl_5d": .., ...}`` with ``None`` for windows
    where there isn't enough forward data yet (decision too recent).
    """
    out = {f"pnl_{w}d": None for w in windows}
    if prices is None or prices.empty:
        return out
    px = prices.sort_index()
    future = px[px.index > decision_ts]
    if future.empty:
        return out
    entry_date = future.index[0]
    entry_px = float(future.iloc[0])

    # Position of entry_date in the full sorted series
    loc = px.index.get_loc(entry_date)
    if isinstance(loc, slice):  # shouldn't happen with unique dates
        loc = loc.start
    for w in windows:
        target = loc + w
        if target >= len(px):
            continue
        exit_px = float(px.iloc[target])
        out[f"pnl_{w}d"] = (exit_px - entry_px) / entry_px
    return out


def _resolve_ticker_for_decision(d: dict) -> str | None:
    """Pick the ticker whose prices we should measure PnL against."""
    ticker = d.get("ticker")
    if ticker:
        return ticker
    # Market-level decision (e.g. Strategist regime): use the market proxy.
    market = (d.get("market") or "").upper()
    return _MARKET_PROXY.get(market)


def run(dry_run: bool = False, limit: int | None = None) -> dict:
    """Process all pending decisions, backfill PnLs."""
    pending = list_pending_outcomes()
    if limit:
        pending = pending[:limit]
    print(
        f"[agent_signal_backtester] {len(pending)} pending decisions "
        f"to backfill{' (dry-run)' if dry_run else ''}"
    )

    stats = {"processed": 0, "skipped": 0, "filled": 0, "errors": 0}
    for d in pending:
        stats["processed"] += 1
        try:
            ticker = _resolve_ticker_for_decision(d)
            ts_str = d.get("timestamp")
            if not ticker or not ts_str:
                stats["skipped"] += 1
                continue
            decision_ts = pd.Timestamp(ts_str)
            # 10 calendar days before (to validate entry liquidity) and
            # 60 calendar days after (covers 20 trading days + slack).
            start = (decision_ts - timedelta(days=10)).strftime("%Y-%m-%d")
            end = (decision_ts + timedelta(days=60)).strftime("%Y-%m-%d")
            prices = fetch_close(ticker, start, end)
            if prices is None or prices.empty:
                stats["skipped"] += 1
                print(f"  decision {d['id']} ({d.get('agent_name')}/{ticker}): no prices")
                continue
            pnls = compute_forward_pnls(prices, decision_ts)
            if all(v is None for v in pnls.values()):
                stats["skipped"] += 1
                print(f"  decision {d['id']} ({d.get('agent_name')}/{ticker}): too recent, no forward data")
                continue
            if dry_run:
                stats["filled"] += 1
                print(f"  [dry-run] decision {d['id']} ({d.get('agent_name')}/{ticker}): {pnls}")
                continue
            res = update_outcome(
                d["id"],
                pnls["pnl_1d"], pnls["pnl_5d"],
                pnls["pnl_10d"], pnls["pnl_20d"],
            )
            stats["filled"] += 1
            print(
                f"  decision {d['id']} ({d.get('agent_name')}/{ticker}): "
                f"1d={pnls['pnl_1d']}, 10d={pnls['pnl_10d']} -> {res.get('hit_status')}"
            )
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            print(f"  decision {d.get('id')} error: {exc}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Backfill agent decision PnLs from FundTeam decision_log",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be filled without writing")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max decisions to process")
    args = parser.parse_args()
    stats = run(dry_run=args.dry_run, limit=args.limit)
    print(f"\n[agent_signal_backtester] done: {stats}")


if __name__ == "__main__":
    main()
