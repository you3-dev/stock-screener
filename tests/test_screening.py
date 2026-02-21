"""Tests for the screening engine."""

import pandas as pd

from src.screening.screener import screen_candidates


def _make_features(n_tickers: int = 5, n_days: int = 1) -> pd.DataFrame:
    """Create synthetic features DataFrame with all conditions passing by default."""
    rows = []
    base_date = pd.Timestamp("2025-01-10")
    for d in range(n_days):
        date = base_date + pd.Timedelta(days=d)
        for i in range(n_tickers):
            rows.append(
                {
                    "date": date,
                    "ticker": f"{1000 + i}.T",
                    "open": 1000.0,
                    "high": 1050.0,
                    "low": 980.0,
                    "close": 1030.0,  # close > open => bullish
                    "volume": 600000,
                    "turnover": 600000000,
                    "turnover_ratio_5d": 2.0,
                    "atr14_ratio": 0.03,
                    "high_20_break_flag": True,
                    "recent_3day_return": 0.05,
                    "liquidity_flag": True,
                }
            )
    return pd.DataFrame(rows)


def _make_backtest(tickers: list[str], n_hold: int = 20) -> pd.DataFrame:
    """Create synthetic backtest results that pass entry criteria by default."""
    rows = []
    for ticker in tickers:
        for n in range(1, n_hold + 1):
            rows.append(
                {
                    "ticker": ticker,
                    "hold_days": n,
                    "weighted_median_return": 0.005 + n * 0.0001,
                    "weighted_win_rate": 0.60,
                    "weighted_dd_median": -0.01,
                    "sample_size": 100,
                }
            )
    return pd.DataFrame(rows)


class TestLiquidityFilter:
    def test_liquidity_false_excluded(self):
        features = _make_features(3)
        features.loc[0, "liquidity_flag"] = False
        tickers = features["ticker"].unique().tolist()
        bt = _make_backtest(tickers)

        result = screen_candidates(features, bt)
        assert features.loc[0, "ticker"] not in result["ticker"].values

    def test_all_illiquid_returns_empty(self):
        features = _make_features(3)
        features["liquidity_flag"] = False
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert len(result) == 0


class TestCapitalInflowConditions:
    def test_low_turnover_ratio_excluded(self):
        features = _make_features(2)
        features.loc[0, "turnover_ratio_5d"] = 1.0  # below 1.8
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert features.loc[0, "ticker"] not in result["ticker"].values

    def test_no_high_break_excluded(self):
        features = _make_features(2)
        features.loc[0, "high_20_break_flag"] = False
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert features.loc[0, "ticker"] not in result["ticker"].values

    def test_bearish_candle_excluded(self):
        features = _make_features(2)
        features.loc[0, "close"] = 990.0  # close < open => bearish
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert features.loc[0, "ticker"] not in result["ticker"].values

    def test_low_atr_excluded(self):
        features = _make_features(2)
        features.loc[0, "atr14_ratio"] = 0.01  # below 0.02
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert features.loc[0, "ticker"] not in result["ticker"].values

    def test_high_3day_return_excluded(self):
        features = _make_features(2)
        features.loc[0, "recent_3day_return"] = 0.15  # above 0.12
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert features.loc[0, "ticker"] not in result["ticker"].values


class TestExclusionConditions:
    def test_limit_up_excluded(self):
        features = _make_features(2)
        # Simulate limit-up: (close - open) / open >= 0.15
        features.loc[0, "open"] = 1000.0
        features.loc[0, "close"] = 1200.0  # 20% gain
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert features.loc[0, "ticker"] not in result["ticker"].values

    def test_below_limit_up_not_excluded(self):
        features = _make_features(2)
        features.loc[0, "open"] = 1000.0
        features.loc[0, "close"] = 1100.0  # 10% gain, below 15% threshold
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert features.loc[0, "ticker"] in result["ticker"].values


class TestEntryCriteria:
    def test_low_expected_return_excluded(self):
        features = _make_features(2)
        tickers = features["ticker"].unique().tolist()
        bt = _make_backtest(tickers)
        # Set hold_days=1 return below threshold for first ticker
        mask = (bt["ticker"] == tickers[0]) & (bt["hold_days"] == 1)
        bt.loc[mask, "weighted_median_return"] = 0.001  # below 0.0035

        result = screen_candidates(features, bt)
        assert tickers[0] not in result["ticker"].values

    def test_low_win_rate_excluded(self):
        features = _make_features(2)
        tickers = features["ticker"].unique().tolist()
        bt = _make_backtest(tickers)
        mask = (bt["ticker"] == tickers[0]) & (bt["hold_days"] == 1)
        bt.loc[mask, "weighted_win_rate"] = 0.50  # below 0.57

        result = screen_candidates(features, bt)
        assert tickers[0] not in result["ticker"].values

    def test_bad_dd_excluded(self):
        features = _make_features(2)
        tickers = features["ticker"].unique().tolist()
        bt = _make_backtest(tickers)
        mask = (bt["ticker"] == tickers[0]) & (bt["hold_days"] == 1)
        bt.loc[mask, "weighted_dd_median"] = -0.03  # below -0.018

        result = screen_candidates(features, bt)
        assert tickers[0] not in result["ticker"].values


class TestRecommendedHoldDays:
    def test_best_hold_days_selected(self):
        features = _make_features(1)
        ticker = features["ticker"].iloc[0]
        bt = _make_backtest([ticker])
        # Make hold_days=5 have the best return
        bt.loc[bt["hold_days"] == 5, "weighted_median_return"] = 0.10

        result = screen_candidates(features, bt)
        assert len(result) == 1
        assert result["recommended_hold_days"].iloc[0] == 5


class TestRankingOrder:
    def test_sorted_by_expected_return_desc(self):
        features = _make_features(3)
        tickers = features["ticker"].unique().tolist()
        bt = _make_backtest(tickers)
        # Set different 1d returns
        for i, ticker in enumerate(tickers):
            mask = (bt["ticker"] == ticker) & (bt["hold_days"] == 1)
            bt.loc[mask, "weighted_median_return"] = 0.004 + i * 0.002

        result = screen_candidates(features, bt)
        assert len(result) == 3
        returns = result["expected_return_1d"].tolist()
        assert returns == sorted(returns, reverse=True)


class TestOutputSchema:
    def test_output_columns(self):
        features = _make_features(2)
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        expected_cols = [
            "date", "ticker", "expected_return_1d", "win_rate",
            "recommended_hold_days", "dd_median",
        ]
        assert list(result.columns) == expected_cols

    def test_empty_result_has_correct_schema(self):
        features = _make_features(2)
        features["liquidity_flag"] = False
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        expected_cols = [
            "date", "ticker", "expected_return_1d", "win_rate",
            "recommended_hold_days", "dd_median",
        ]
        assert list(result.columns) == expected_cols
        assert len(result) == 0


class TestTargetDate:
    def test_specific_date_selected(self):
        features = _make_features(3, n_days=3)
        bt = _make_backtest(features["ticker"].unique().tolist())
        target = "2025-01-11"

        result = screen_candidates(features, bt, target_date=target)
        assert all(result["date"] == pd.Timestamp(target))

    def test_latest_date_by_default(self):
        features = _make_features(3, n_days=3)
        bt = _make_backtest(features["ticker"].unique().tolist())

        result = screen_candidates(features, bt)
        assert all(result["date"] == pd.Timestamp("2025-01-12"))
