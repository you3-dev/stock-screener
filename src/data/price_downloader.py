"""Price data downloader using yfinance.

Downloads daily OHLCV data for TSE stocks and stores as Parquet.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.config import load_config

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"
_PRICE_FILE = _CACHE_DIR / "prices.parquet"
_CHUNK_SIZE = 100
_CHUNK_SLEEP = 1.0
_FINAL_COLS = ["date", "ticker", "open", "high", "low", "close", "volume", "turnover"]


def _normalize_yf_result(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance download result (always MultiIndex) to long format."""
    # yfinance 1.x always returns MultiIndex: (Price, Ticker)
    df = raw.stack(level="Ticker", future_stack=True).reset_index()
    df = df.rename(columns={"Ticker": "ticker"})

    # Normalize column names to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Drop unnecessary columns
    for col in ["adj_close", "price"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Compute turnover = volume * close
    df["turnover"] = df["volume"] * df["close"]

    return df[_FINAL_COLS].dropna(subset=["close"])


def _download_chunks(
    tickers: list[str], start_str: str, end_str: str
) -> list[pd.DataFrame]:
    """Download price data in chunks to avoid rate limits."""
    frames = []
    total_chunks = (len(tickers) + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    for i in range(0, len(tickers), _CHUNK_SIZE):
        chunk = tickers[i : i + _CHUNK_SIZE]
        logger.info(
            "  Chunk %d/%d (%d tickers)", i // _CHUNK_SIZE + 1, total_chunks, len(chunk)
        )

        raw = yf.download(chunk, start=start_str, end=end_str, progress=False)

        if raw.empty:
            logger.warning("  No data returned for chunk")
            continue

        frames.append(_normalize_yf_result(raw))

        if i + _CHUNK_SIZE < len(tickers):
            time.sleep(_CHUNK_SLEEP)

    return frames


def download_prices(
    tickers: list[str],
    years: int | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data for given tickers via yfinance.

    Args:
        tickers: List of ticker symbols (e.g. ["7203.T", "6758.T"]).
        years: Lookback period in years. Defaults to config value.

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume, turnover
    """
    if years is None:
        cfg = load_config()
        years = cfg["backtest"]["lookback_years"]

    end = datetime.now()
    start = end - timedelta(days=years * 365)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    logger.info(
        "Downloading prices for %d tickers (%s to %s)", len(tickers), start_str, end_str
    )

    frames = _download_chunks(tickers, start_str, end_str)

    if not frames:
        return pd.DataFrame(columns=_FINAL_COLS)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info("Downloaded %d rows for %d tickers", len(result), result["ticker"].nunique())
    return result


def save_prices(df: pd.DataFrame) -> Path:
    """Save price DataFrame to Parquet cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PRICE_FILE, index=False)
    logger.info("Saved prices to %s", _PRICE_FILE)
    return _PRICE_FILE


def load_prices() -> pd.DataFrame | None:
    """Load cached price data. Returns None if no cache exists."""
    if not _PRICE_FILE.exists():
        return None
    return pd.read_parquet(_PRICE_FILE)


def update_prices(tickers: list[str], recent_days: int = 10) -> pd.DataFrame:
    """Update existing price cache with recent data.

    Downloads the last `recent_days` calendar days and merges
    with existing cached data.

    Args:
        tickers: List of ticker symbols.
        recent_days: Number of calendar days to re-download.

    Returns:
        Updated DataFrame.
    """
    existing = load_prices()

    end = datetime.now()
    start = end - timedelta(days=recent_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    logger.info("Updating prices: fetching %s to %s", start_str, end_str)

    frames = _download_chunks(tickers, start_str, end_str)

    if not frames:
        logger.warning("No recent data downloaded")
        return existing if existing is not None else pd.DataFrame()

    recent = pd.concat(frames, ignore_index=True)

    if existing is not None:
        # Remove overlapping dates then append new data
        cutoff = recent["date"].min()
        existing = existing[existing["date"] < cutoff]
        result = pd.concat([existing, recent], ignore_index=True)
    else:
        result = recent

    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    save_prices(result)
    return result
