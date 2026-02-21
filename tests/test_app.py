"""Tests for Phase 5: Streamlit UI helper functions."""

from datetime import date

import pandas as pd

from src.app.main import (
    compute_hold_days,
    compute_remaining_days,
    compute_total_score,
    format_display_table,
)


class TestComputeTotalScore:
    def test_single_row_returns_50(self):
        df = pd.DataFrame(
            {
                "expected_return_1d": [0.005],
                "win_rate": [0.60],
                "dd_median": [-0.015],
            }
        )
        score = compute_total_score(df)
        assert len(score) == 1
        assert score.iloc[0] == 50.0

    def test_higher_return_gets_higher_score(self):
        df = pd.DataFrame(
            {
                "expected_return_1d": [0.003, 0.010],
                "win_rate": [0.60, 0.60],
                "dd_median": [-0.015, -0.015],
            }
        )
        scores = compute_total_score(df)
        assert scores.iloc[1] > scores.iloc[0]

    def test_score_range_0_to_100(self):
        df = pd.DataFrame(
            {
                "expected_return_1d": [0.001, 0.005, 0.010],
                "win_rate": [0.50, 0.60, 0.70],
                "dd_median": [-0.03, -0.015, -0.005],
            }
        )
        scores = compute_total_score(df)
        assert (scores >= 0).all()
        assert (scores <= 100).all()

    def test_dd_closer_to_zero_is_better(self):
        df = pd.DataFrame(
            {
                "expected_return_1d": [0.005, 0.005],
                "win_rate": [0.60, 0.60],
                "dd_median": [-0.03, -0.005],
            }
        )
        scores = compute_total_score(df)
        assert scores.iloc[1] > scores.iloc[0]


class TestComputeHoldDays:
    def test_same_day_is_zero(self):
        today = date(2026, 2, 20)
        assert compute_hold_days(date(2026, 2, 20), today=today) == 0

    def test_one_business_day(self):
        assert compute_hold_days(date(2026, 2, 19), today=date(2026, 2, 20)) == 1

    def test_over_weekend(self):
        # Friday -> Monday = 1 business day
        assert compute_hold_days(date(2026, 2, 20), today=date(2026, 2, 23)) == 1

    def test_one_week(self):
        # Monday -> next Monday = 5 business days
        assert compute_hold_days(date(2026, 2, 16), today=date(2026, 2, 23)) == 5


class TestComputeRemainingDays:
    def test_full_remaining(self):
        assert compute_remaining_days(0, 20) == 20

    def test_partial_remaining(self):
        assert compute_remaining_days(15, 20) == 5

    def test_exceeded(self):
        assert compute_remaining_days(22, 20) == -2


class TestFormatDisplayTable:
    def test_join_company_name(self):
        candidates = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-02-20")],
                "ticker": ["7203.T"],
                "expected_return_1d": [0.005],
                "win_rate": [0.60],
                "recommended_hold_days": [5],
                "dd_median": [-0.015],
            }
        )
        ticker_master = pd.DataFrame(
            {
                "ticker": ["7203.T"],
                "company_name": ["トヨタ自動車"],
                "market": ["Prime"],
            }
        )
        result = format_display_table(candidates, ticker_master)
        assert "銘柄名" in result.columns
        assert result.iloc[0]["銘柄名"] == "トヨタ自動車"

    def test_missing_ticker_in_master(self):
        candidates = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-02-20")],
                "ticker": ["9999.T"],
                "expected_return_1d": [0.005],
                "win_rate": [0.60],
                "recommended_hold_days": [5],
                "dd_median": [-0.015],
            }
        )
        ticker_master = pd.DataFrame(
            {
                "ticker": ["7203.T"],
                "company_name": ["トヨタ自動車"],
                "market": ["Prime"],
            }
        )
        result = format_display_table(candidates, ticker_master)
        assert result.iloc[0]["銘柄名"] == "---"

    def test_none_ticker_master(self):
        candidates = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-02-20")],
                "ticker": ["7203.T"],
                "expected_return_1d": [0.005],
                "win_rate": [0.60],
                "recommended_hold_days": [5],
                "dd_median": [-0.015],
            }
        )
        result = format_display_table(candidates, None)
        assert result.iloc[0]["銘柄名"] == "---"

    def test_empty_candidates(self):
        candidates = pd.DataFrame(
            columns=[
                "date", "ticker", "expected_return_1d", "win_rate",
                "recommended_hold_days", "dd_median",
            ]
        )
        ticker_master = pd.DataFrame(columns=["ticker", "company_name", "market"])
        result = format_display_table(candidates, ticker_master)
        assert len(result) == 0

    def test_rank_column(self):
        candidates = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-02-20")] * 3,
                "ticker": ["1000.T", "2000.T", "3000.T"],
                "expected_return_1d": [0.008, 0.006, 0.004],
                "win_rate": [0.60, 0.58, 0.62],
                "recommended_hold_days": [3, 5, 7],
                "dd_median": [-0.01, -0.015, -0.02],
            }
        )
        result = format_display_table(candidates, None)
        assert list(result["ランク"]) == [1, 2, 3]

    def test_output_columns(self):
        candidates = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-02-20")],
                "ticker": ["7203.T"],
                "expected_return_1d": [0.005],
                "win_rate": [0.60],
                "recommended_hold_days": [5],
                "dd_median": [-0.015],
            }
        )
        result = format_display_table(candidates, None)
        expected_cols = [
            "ランク", "銘柄コード", "銘柄名", "1日期待値",
            "勝率", "推奨日数", "最大DD", "総合スコア",
        ]
        assert list(result.columns) == expected_cols
