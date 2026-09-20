"""Data fetching layer for the Backtester.

Supports:
  - FundTeam HTTP API (ETF basket via /api/data/etf_basket)
  - Local CSV files (for offline backtesting)
  - Direct Strategist import (for regime signals)
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

FUNDTEAM_BASE_URL = os.environ.get("FUNDTEAM_URL", "http://127.0.0.1:8080")


def fetch_etf_basket(
    market: str = "US",
    start: str = "",
    end: str = "",
) -> pd.DataFrame:
    """Fetch ETF basket prices from FundTeam's data API.

    Returns a DataFrame of close prices indexed by date.
    Falls back to local Strategist data if API is unavailable.
    """
    import requests

    params = {"market": market}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    try:
        r = requests.get(f"{FUNDTEAM_BASE_URL}/api/data/etf_basket", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("ok") and data.get("data"):
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            return df
    except Exception as e:
        print(f"[data] FundTeam API failed: {e}, falling back to Strategist repo")

    # Fallback: import Strategist's data fetcher directly
    try:
        import sys
        sys.path.insert(0, "/Users/kevin/PycharmProjects/Fund/Strategist")
        from strategist.data_fetcher import fetch_etf_basket as fetch_basket
        from strategist.config import US_ETFS, HK_ETFS, ASHARE_ETFS

        etfs = {"US": US_ETFS, "HK": HK_ETFS, "ASHARE": ASHARE_ETFS}[market]
        end_dt = pd.Timestamp(end) if end else pd.Timestamp.now()
        start_dt = pd.Timestamp(start) if start else end_dt - timedelta(days=365 * 5)
        prices = fetch_basket(etfs, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        return prices
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {market} data: {e}")


def fetch_strategist_signal(market: str = "US", date: str | None = None) -> dict:
    """Fetch the latest Strategist regime signal from FundTeam API."""
    import requests

    url = f"{FUNDTEAM_BASE_URL}/api/strategist/{market}"
    if date:
        url += f"/{date}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[data] Strategist API failed: {e}")
        return {}


def load_local_prices(filepath: str | Path) -> pd.DataFrame:
    """Load prices from a local CSV/parquet file."""
    path = Path(filepath)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df
