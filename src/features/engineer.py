"""Feature engineering for stock screening.

Computes technical features from daily OHLCV price data.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data.config import load_config

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"
_FEATURES_FILE = _CACHE_DIR / "features.parquet"


def _turnover_ratio_5d(group: pd.DataFrame) -> pd.Series:
    """Turnover ratio vs 5-day average."""
    avg_5d = group["turnover"].rolling(window=5, min_periods=5).mean()
    return group["turnover"] / avg_5d


def _atr14_ratio(group: pd.DataFrame) -> pd.Series:
    """ATR(14) / close price ratio."""
    prev_close = group["close"].shift(1)
    tr = pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - prev_close).abs(),
            (group["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    return atr / group["close"]


def _high_20_break_flag(group: pd.DataFrame) -> pd.Series:
    """Whether today's high equals the 20-day rolling max."""
    rolling_max = group["high"].rolling(window=20, min_periods=20).max()
    return group["high"] >= rolling_max


def _recent_3day_return(group: pd.DataFrame) -> pd.Series:
    """3-day return: close / close(3 days ago) - 1."""
    return group["close"] / group["close"].shift(3) - 1


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute all screening features from price data.

    Args:
        prices: DataFrame with columns date, ticker, open, high, low, close, volume, turnover.
                Must be sorted by [ticker, date].

    Returns:
        DataFrame with original price columns plus:
        turnover_ratio_5d, atr14_ratio, high_20_break_flag, recent_3day_return, liquidity_flag
    """
    cfg = load_config()
    liq = cfg["liquidity"]

    logger.info("Computing features for %d rows", len(prices))

    df = prices.sort_values(["ticker", "date"]).copy()

    # Use transform-style via concat to avoid groupby.apply index issues
    parts = []
    for _ticker, group in df.groupby("ticker", sort=False):
        g = group.copy()
        g["turnover_ratio_5d"] = _turnover_ratio_5d(g)
        g["atr14_ratio"] = _atr14_ratio(g)
        g["high_20_break_flag"] = _high_20_break_flag(g)
        g["recent_3day_return"] = _recent_3day_return(g)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    df["liquidity_flag"] = (
        (df["volume"] >= liq["min_volume"])
        & (df["turnover"] >= liq["min_turnover"])
        & (df["close"] >= liq["price_min"])
        & (df["close"] <= liq["price_max"])
    )

    logger.info("Features computed. Rows with all features: %d", df.dropna().shape[0])
    return df


def save_features(df: pd.DataFrame) -> Path:
    """Save features DataFrame to Parquet cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_FEATURES_FILE, index=False)
    logger.info("Saved features to %s", _FEATURES_FILE)
    return _FEATURES_FILE


def load_features() -> pd.DataFrame | None:
    """Load cached features. Returns None if no cache exists."""
    if not _FEATURES_FILE.exists():
        return None
    return pd.read_parquet(_FEATURES_FILE)
