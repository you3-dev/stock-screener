"""Screening engine for stock entry candidates.

Combines features and backtest results to filter and rank
stocks that meet entry criteria.
"""

import logging

import pandas as pd

from src.data.config import load_config

logger = logging.getLogger(__name__)


def screen_candidates(
    features: pd.DataFrame,
    backtest_results: pd.DataFrame,
    target_date: str | None = None,
) -> pd.DataFrame:
    """Screen stocks and return ranked entry candidates.

    Args:
        features: DataFrame from compute_features() with columns including
            date, ticker, open, close, turnover_ratio_5d, atr14_ratio,
            high_20_break_flag, recent_3day_return, liquidity_flag.
        backtest_results: DataFrame from run_backtest() with columns
            ticker, hold_days, weighted_median_return, weighted_win_rate,
            weighted_dd_median, sample_size.
        target_date: Date string (YYYY-MM-DD) to screen. Defaults to latest date.

    Returns:
        DataFrame with columns: date, ticker, expected_return_1d, win_rate,
        recommended_hold_days, dd_median. Sorted by expected_return_1d descending.
    """
    cfg = load_config()
    ci = cfg["capital_inflow"]
    entry = cfg["entry"]
    exclusion = cfg["exclusion"]

    # --- 1. Target date extraction ---
    if target_date is None:
        target_date = features["date"].max()
    else:
        target_date = pd.Timestamp(target_date)

    df = features[features["date"] == target_date].copy()
    logger.info("Screening for date=%s: %d rows", target_date, len(df))

    if df.empty:
        return _empty_result()

    # --- 2. Liquidity filter ---
    df = df[df["liquidity_flag"]].copy()
    logger.info("After liquidity filter: %d rows", len(df))

    if df.empty:
        return _empty_result()

    # --- 3. Capital inflow conditions ---
    df = df[
        (df["turnover_ratio_5d"] >= ci["turnover_ratio_5d_min"])
        & (df["high_20_break_flag"])
        & (df["close"] > df["open"])  # bullish candle
        & (df["atr14_ratio"] >= ci["atr14_ratio_min"])
        & (df["recent_3day_return"] <= ci["recent_3day_return_max"])
    ].copy()
    logger.info("After capital inflow filter: %d rows", len(df))

    if df.empty:
        return _empty_result()

    # --- 4. Exclusion: limit-up approximation ---
    if exclusion["limit_up_next_day"]:
        df = df[((df["close"] - df["open"]) / df["open"]) < 0.15].copy()
        logger.info("After limit-up exclusion: %d rows", len(df))

    if df.empty:
        return _empty_result()

    # --- 5. Join backtest results (hold_days=1) for entry criteria ---
    bt_1d = backtest_results[backtest_results["hold_days"] == 1][
        ["ticker", "weighted_median_return", "weighted_win_rate", "weighted_dd_median"]
    ].copy()

    df = df.merge(bt_1d, on="ticker", how="inner")
    logger.info("After backtest join: %d rows", len(df))

    if df.empty:
        return _empty_result()

    df = df[
        (df["weighted_median_return"] >= entry["expected_return_1d_min"])
        & (df["weighted_win_rate"] >= entry["win_rate_min"])
        & (df["weighted_dd_median"] >= entry["dd_median_max"])
    ].copy()
    logger.info("After entry criteria filter: %d rows", len(df))

    if df.empty:
        return _empty_result()

    # --- 6. Recommended hold days (n with max expected return) ---
    best_hold = (
        backtest_results.loc[
            backtest_results.groupby("ticker")["weighted_median_return"].idxmax()
        ][["ticker", "hold_days"]]
        .rename(columns={"hold_days": "recommended_hold_days"})
    )
    df = df.merge(best_hold, on="ticker", how="left")

    # --- 7. Build output and rank ---
    result = (
        df[["ticker", "weighted_median_return", "weighted_win_rate",
            "recommended_hold_days", "weighted_dd_median"]]
        .rename(columns={
            "weighted_median_return": "expected_return_1d",
            "weighted_win_rate": "win_rate",
            "weighted_dd_median": "dd_median",
        })
        .copy()
    )
    result["date"] = target_date
    result = result[
        ["date", "ticker", "expected_return_1d", "win_rate",
         "recommended_hold_days", "dd_median"]
    ]
    result = result.sort_values("expected_return_1d", ascending=False).reset_index(drop=True)

    logger.info("Screening complete: %d candidates", len(result))
    return result


def _empty_result() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame(
        columns=["date", "ticker", "expected_return_1d", "win_rate",
                 "recommended_hold_days", "dd_median"]
    )
