"""Tests for Phase 3: backtest engine."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.backtest.engine import (
    _add_forward_returns,
    _add_max_drawdown,
    _compute_weights,
    run_backtest,
    weighted_median,
)


def _make_price_df(n_days: int = 30, ticker: str = "9999.T") -> pd.DataFrame:
    """Create a synthetic price DataFrame for backtest testing."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    rng = np.random.default_rng(42)
    base = 1000.0 + np.cumsum(rng.normal(0, 5, n_days))
    opens = base + rng.normal(0, 3, n_days)
    highs = np.maximum(opens, base) + rng.uniform(5, 15, n_days)
    lows = np.minimum(opens, base) - rng.uniform(5, 15, n_days)
    closes = base
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


class TestWeightedMedian:
    def test_equal_weights_matches_median(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        result = weighted_median(values, weights)
        assert result == 3.0

    def test_heavy_weight_on_high_value(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 100.0])
        result = weighted_median(values, weights)
        assert result == 3.0

    def test_heavy_weight_on_low_value(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([100.0, 1.0, 1.0])
        result = weighted_median(values, weights)
        assert result == 1.0

    def test_nan_values_ignored(self):
        values = np.array([1.0, np.nan, 3.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        result = weighted_median(values, weights)
        assert result == 3.0

    def test_empty_returns_nan(self):
        values = np.array([])
        weights = np.array([])
        result = weighted_median(values, weights)
        assert np.isnan(result)


class TestForwardReturns:
    def test_return_1(self):
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2025-01-01", periods=5),
                "ticker": "TEST.T",
                "open": [100.0, 102.0, 105.0, 103.0, 108.0],
                "high": [110.0] * 5,
                "low": [95.0] * 5,
                "close": [101.0] * 5,
                "volume": [1000.0] * 5,
                "turnover": [101000.0] * 5,
            }
        )
        result = _add_forward_returns(df, max_n=3)

        # R(1) for day 0: Open(1)/Open(0) - 1 = 102/100 - 1 = 0.02
        assert abs(result["return_1"].iloc[0] - 0.02) < 1e-10
        # R(2) for day 0: Open(2)/Open(0) - 1 = 105/100 - 1 = 0.05
        assert abs(result["return_2"].iloc[0] - 0.05) < 1e-10
        # R(1) for day 3: Open(4)/Open(3) - 1 = 108/103 - 1
        expected = 108.0 / 103.0 - 1
        assert abs(result["return_1"].iloc[3] - expected) < 1e-10

    def test_last_rows_are_nan(self):
        df = _make_price_df(n_days=10)
        result = _add_forward_returns(df, max_n=3)
        # Last 3 rows of return_3 should be NaN
        assert result["return_3"].iloc[-3:].isna().all()
        # Last 1 row of return_1 should be NaN
        assert np.isnan(result["return_1"].iloc[-1])


class TestMaxDrawdown:
    def test_basic_dd(self):
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2025-01-01", periods=5),
                "ticker": "TEST.T",
                "open": [100.0, 102.0, 105.0, 103.0, 108.0],
                "high": [110.0] * 5,
                "low": [95.0, 98.0, 90.0, 100.0, 105.0],
                "close": [101.0] * 5,
                "volume": [1000.0] * 5,
                "turnover": [101000.0] * 5,
            }
        )
        result = _add_max_drawdown(df, max_n=3)

        # DD(1) for day 0: min(low[0]) / open[0] - 1 = 95/100 - 1 = -0.05
        assert abs(result["dd_1"].iloc[0] - (-0.05)) < 1e-10
        # DD(3) for day 0: min(95, 98, 90) / 100 - 1 = -0.10
        assert abs(result["dd_3"].iloc[0] - (-0.10)) < 1e-10

    def test_dd_last_rows_nan(self):
        df = _make_price_df(n_days=10)
        result = _add_max_drawdown(df, max_n=5)
        # Last 4 rows of dd_5 should be NaN (need 5 consecutive days)
        assert result["dd_5"].iloc[-4:].isna().all()
        assert result["dd_5"].iloc[-5:].notna().iloc[0]


class TestDecayWeights:
    def test_today_weight_is_one(self):
        now = datetime.now()
        dates = pd.Series([pd.Timestamp(now)])
        weights = _compute_weights(dates, now, 0.3)
        assert abs(weights[0] - 1.0) < 0.01

    def test_one_year_ago(self):
        now = datetime(2025, 6, 1)
        dates = pd.Series([pd.Timestamp("2024-06-01")])
        weights = _compute_weights(dates, now, 0.3)
        expected = np.exp(-0.3 * 1.0)
        assert abs(weights[0] - expected) < 0.02

    def test_older_dates_get_lower_weight(self):
        now = datetime(2025, 6, 1)
        dates = pd.Series(
            [pd.Timestamp("2025-01-01"), pd.Timestamp("2023-01-01")]
        )
        weights = _compute_weights(dates, now, 0.3)
        assert weights[0] > weights[1]


class TestRunBacktest:
    def test_output_schema(self):
        df = _make_price_df(n_days=30, ticker="0001.T")
        result = run_backtest(df)
        expected_cols = {
            "ticker", "hold_days", "weighted_median_return",
            "weighted_win_rate", "weighted_dd_median", "sample_size",
        }
        assert set(result.columns) == expected_cols

    def test_hold_days_range(self):
        df = _make_price_df(n_days=30, ticker="0001.T")
        result = run_backtest(df)
        hold_days = sorted(result["hold_days"].unique())
        # With 30 days of data, should have results for n=1..10 at minimum
        assert 1 in hold_days
        assert len(hold_days) >= 10

    def test_multiple_tickers(self):
        df1 = _make_price_df(n_days=30, ticker="0001.T")
        df2 = _make_price_df(n_days=30, ticker="0002.T")
        combined = pd.concat([df1, df2], ignore_index=True)
        result = run_backtest(combined)
        assert set(result["ticker"].unique()) == {"0001.T", "0002.T"}

    def test_win_rate_between_0_and_1(self):
        df = _make_price_df(n_days=50, ticker="0001.T")
        result = run_backtest(df)
        assert (result["weighted_win_rate"] >= 0).all()
        assert (result["weighted_win_rate"] <= 1).all()

    def test_dd_is_negative_or_zero(self):
        df = _make_price_df(n_days=50, ticker="0001.T")
        result = run_backtest(df)
        assert (result["weighted_dd_median"] <= 0).all()

    def test_sample_size_positive(self):
        df = _make_price_df(n_days=30, ticker="0001.T")
        result = run_backtest(df)
        assert (result["sample_size"] > 0).all()
