"""Tests for Phase 2: feature engineering."""

import numpy as np
import pandas as pd

from src.features.engineer import compute_features


def _make_price_df(
    n_days: int = 25,
    ticker: str = "9999.T",
    base_close: float = 1000.0,
) -> pd.DataFrame:
    """Create a synthetic price DataFrame for testing."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    rng = np.random.default_rng(42)
    closes = base_close + np.cumsum(rng.normal(0, 10, n_days))
    highs = closes + rng.uniform(5, 20, n_days)
    lows = closes - rng.uniform(5, 20, n_days)
    opens = closes + rng.normal(0, 5, n_days)
    volumes = rng.integers(400_000, 600_000, n_days).astype(float)

    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    df["turnover"] = df["volume"] * df["close"]
    return df


class TestTurnoverRatio5d:
    def test_basic_calculation(self):
        df = _make_price_df(n_days=10, ticker="0001.T")
        # Set known turnover values
        df["turnover"] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        df["volume"] = 1.0
        df["close"] = df["turnover"]  # so turnover = volume * close

        result = compute_features(df)

        # Day index 4 (5th day): turnover=500, avg(100..500)=300 → ratio=500/300
        row4 = result[result["date"] == df["date"].iloc[4]].iloc[0]
        expected = 500 / ((100 + 200 + 300 + 400 + 500) / 5)
        assert abs(row4["turnover_ratio_5d"] - expected) < 0.001

    def test_first_4_days_are_nan(self):
        df = _make_price_df(n_days=10, ticker="0001.T")
        result = compute_features(df)
        # First 4 rows should be NaN (need 5 days for rolling)
        assert result["turnover_ratio_5d"].iloc[:4].isna().all()


class TestATR14Ratio:
    def test_basic_calculation(self):
        # Create 20 days of data with known values
        df = _make_price_df(n_days=20, ticker="0002.T")
        result = compute_features(df)

        # After 14 days, atr14_ratio should be a positive value
        valid = result["atr14_ratio"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_first_13_days_are_nan(self):
        df = _make_price_df(n_days=20, ticker="0002.T")
        result = compute_features(df)
        # shift(1) loses row 0, then rolling(14) needs 14 valid TR values
        # → first 13 rows (index 0-12) are NaN, index 13 is first valid
        assert result["atr14_ratio"].iloc[:13].isna().all()
        assert result["atr14_ratio"].iloc[13:].notna().all()


class TestHigh20BreakFlag:
    def test_new_high_detected(self):
        df = _make_price_df(n_days=25, ticker="0003.T")
        # Set all highs to 100, then last day to 200 → should flag
        df["high"] = 100.0
        df.iloc[-1, df.columns.get_loc("high")] = 200.0
        result = compute_features(df)
        assert bool(result["high_20_break_flag"].iloc[-1]) is True

    def test_no_break_when_below_max(self):
        df = _make_price_df(n_days=25, ticker="0003.T")
        # Set recent high within window to 999, last day to 100 → should not flag
        df["high"] = 100.0
        df.iloc[-5, df.columns.get_loc("high")] = 999.0
        result = compute_features(df)
        assert bool(result["high_20_break_flag"].iloc[-1]) is False


class TestRecent3dayReturn:
    def test_basic_calculation(self):
        df = _make_price_df(n_days=10, ticker="0004.T")
        df["close"] = [100, 101, 102, 103, 110, 108, 112, 115, 120, 125]
        result = compute_features(df)
        # Day 3: close=103, 3 days ago close=100 → return = 0.03
        row3 = result[result["date"] == df["date"].iloc[3]].iloc[0]
        assert abs(row3["recent_3day_return"] - 0.03) < 0.001

    def test_first_3_days_are_nan(self):
        df = _make_price_df(n_days=10, ticker="0004.T")
        result = compute_features(df)
        assert result["recent_3day_return"].iloc[:3].isna().all()


class TestLiquidityFlag:
    def test_all_conditions_met(self):
        df = _make_price_df(n_days=25, ticker="0005.T")
        df["close"] = 1000.0
        df["volume"] = 600_000.0
        df["turnover"] = df["volume"] * df["close"]
        result = compute_features(df)
        assert result["liquidity_flag"].all()

    def test_volume_too_low(self):
        df = _make_price_df(n_days=25, ticker="0005.T")
        df["close"] = 1000.0
        df["volume"] = 100.0  # way below 500K
        df["turnover"] = df["volume"] * df["close"]
        result = compute_features(df)
        assert not result["liquidity_flag"].any()

    def test_price_out_of_range(self):
        df = _make_price_df(n_days=25, ticker="0005.T")
        df["close"] = 10000.0  # above 6000
        df["volume"] = 600_000.0
        df["turnover"] = df["volume"] * df["close"]
        result = compute_features(df)
        assert not result["liquidity_flag"].any()


class TestComputeFeaturesSchema:
    def test_output_columns(self):
        df = _make_price_df(n_days=25)
        result = compute_features(df)
        expected = {
            "date", "ticker", "open", "high", "low", "close", "volume", "turnover",
            "turnover_ratio_5d", "atr14_ratio", "high_20_break_flag",
            "recent_3day_return", "liquidity_flag",
        }
        assert set(result.columns) == expected

    def test_multiple_tickers(self):
        df1 = _make_price_df(n_days=25, ticker="0001.T")
        df2 = _make_price_df(n_days=25, ticker="0002.T")
        combined = pd.concat([df1, df2], ignore_index=True)
        result = compute_features(combined)
        assert set(result["ticker"].unique()) == {"0001.T", "0002.T"}
        assert len(result) == 50
