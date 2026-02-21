"""Backtest engine for stock screening.

Computes n-day forward open-to-open returns and drawdowns,
then produces weighted statistics using exponential decay.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import load_config

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"
_BACKTEST_FILE = _CACHE_DIR / "backtest_results.parquet"


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median.

    Args:
        values: Array of values.
        weights: Array of weights (must be positive).

    Returns:
        Weighted median value.
    """
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    sorted_idx = np.argsort(values)
    values = values[sorted_idx]
    weights = weights[sorted_idx]

    cumsum = np.cumsum(weights)
    half = cumsum[-1] / 2.0

    idx = np.searchsorted(cumsum, half)
    return float(values[min(idx, len(values) - 1)])


def _add_forward_returns(group: pd.DataFrame, max_n: int) -> pd.DataFrame:
    """Add forward return columns (return_1 .. return_N) to a single-ticker group.

    R(n) = Open(t+n) / Open(t) - 1
    """
    g = group.copy()
    for n in range(1, max_n + 1):
        g[f"return_{n}"] = g["open"].shift(-n) / g["open"] - 1
    return g


def _add_max_drawdown(group: pd.DataFrame, max_n: int) -> pd.DataFrame:
    """Add max drawdown columns (dd_1 .. dd_N) to a single-ticker group.

    DD(n) = min(Low[t], Low[t+1], ..., Low[t+n-1]) / Open(t) - 1
    """
    g = group.copy()
    lows = g["low"].values
    opens = g["open"].values
    length = len(g)

    for n in range(1, max_n + 1):
        dd_values = np.full(length, np.nan)
        for i in range(length - n + 1):
            min_low = np.min(lows[i : i + n])
            dd_values[i] = min_low / opens[i] - 1
        g[f"dd_{n}"] = dd_values
    return g


def _compute_weights(
    dates: pd.Series, reference_date: datetime, decay_lambda: float
) -> np.ndarray:
    """Compute exponential decay weights.

    weight = exp(-lambda * years_elapsed)
    """
    ref = pd.Timestamp(reference_date)
    days_elapsed = (ref - pd.to_datetime(dates)).dt.days
    years_elapsed = days_elapsed / 365.25
    return np.exp(-decay_lambda * years_elapsed.values)


def run_backtest(prices: pd.DataFrame) -> pd.DataFrame:
    """Run backtest on price data.

    For each ticker, computes forward returns and max drawdowns for n=1..max_hold_days,
    then produces weighted statistics using exponential decay.

    Args:
        prices: DataFrame with columns date, ticker, open, high, low, close, volume, turnover.

    Returns:
        DataFrame with columns:
        ticker, hold_days, weighted_median_return, weighted_win_rate,
        weighted_dd_median, sample_size
    """
    cfg = load_config()
    max_n = cfg["backtest"]["max_hold_days"]
    decay_lambda = cfg["backtest"]["decay_lambda"]
    reference_date = datetime.now()

    logger.info("Running backtest: max_hold_days=%d, decay_lambda=%.2f", max_n, decay_lambda)

    df = prices.sort_values(["ticker", "date"]).copy()

    # Compute forward returns and DD per ticker
    parts = []
    for _ticker, group in df.groupby("ticker", sort=False):
        g = _add_forward_returns(group, max_n)
        g = _add_max_drawdown(g, max_n)
        parts.append(g)
    enriched = pd.concat(parts, ignore_index=True)

    # Compute weights
    weights = _compute_weights(enriched["date"], reference_date, decay_lambda)

    # Aggregate statistics per ticker per hold_days
    results = []
    for ticker, tgroup in enriched.groupby("ticker", sort=False):
        tw = weights[tgroup.index]
        for n in range(1, max_n + 1):
            ret_col = f"return_{n}"
            dd_col = f"dd_{n}"

            valid_mask = tgroup[ret_col].notna() & tgroup[dd_col].notna()
            if valid_mask.sum() == 0:
                continue

            ret_vals = tgroup.loc[valid_mask, ret_col].values
            dd_vals = tgroup.loc[valid_mask, dd_col].values
            w = tw[valid_mask.values]

            w_median_ret = weighted_median(ret_vals, w)
            w_win_rate = float(np.average(ret_vals > 0, weights=w))
            w_dd_median = weighted_median(dd_vals, w)
            sample = int(valid_mask.sum())

            results.append(
                {
                    "ticker": ticker,
                    "hold_days": n,
                    "weighted_median_return": w_median_ret,
                    "weighted_win_rate": w_win_rate,
                    "weighted_dd_median": w_dd_median,
                    "sample_size": sample,
                }
            )

    result_df = pd.DataFrame(results)
    logger.info(
        "Backtest complete: %d ticker×hold_days combinations", len(result_df)
    )
    return result_df


def save_backtest_results(df: pd.DataFrame) -> Path:
    """Save backtest results to Parquet cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_BACKTEST_FILE, index=False)
    logger.info("Saved backtest results to %s", _BACKTEST_FILE)
    return _BACKTEST_FILE


def load_backtest_results() -> pd.DataFrame | None:
    """Load cached backtest results. Returns None if no cache exists."""
    if not _BACKTEST_FILE.exists():
        return None
    return pd.read_parquet(_BACKTEST_FILE)
