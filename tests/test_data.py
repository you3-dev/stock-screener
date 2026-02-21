"""Tests for Phase 1: data acquisition modules."""

import pandas as pd
import pytest

from src.data.config import load_config


class TestConfig:
    def test_load_config_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_config_has_required_sections(self):
        cfg = load_config()
        for section in ["backtest", "liquidity", "capital_inflow", "entry", "portfolio"]:
            assert section in cfg, f"Missing config section: {section}"

    def test_config_backtest_values(self):
        cfg = load_config()
        assert cfg["backtest"]["lookback_years"] == 5
        assert cfg["backtest"]["decay_lambda"] == 0.3
        assert cfg["backtest"]["max_hold_days"] == 20

    def test_config_entry_thresholds(self):
        cfg = load_config()
        assert cfg["entry"]["expected_return_1d_min"] == 0.0035
        assert cfg["entry"]["win_rate_min"] == 0.57


@pytest.mark.slow
class TestTickerMaster:
    def test_fetch_prime_tickers(self):
        from src.data.ticker_master import fetch_prime_tickers

        df = fetch_prime_tickers()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"ticker", "company_name", "market"}
        # Prime stocks should be around 1500-1700
        assert len(df) > 1000
        assert len(df) < 2500

    def test_ticker_format(self):
        from src.data.ticker_master import fetch_prime_tickers

        df = fetch_prime_tickers()
        for ticker in df["ticker"]:
            assert ticker.endswith(".T"), f"Ticker {ticker} doesn't end with .T"
            code = ticker.replace(".T", "")
            assert code.isalnum(), f"Ticker code {code} is not alphanumeric"


@pytest.mark.slow
class TestPriceDownloader:
    def test_download_few_tickers(self):
        from src.data.price_downloader import download_prices

        tickers = ["7203.T", "6758.T", "9984.T"]
        df = download_prices(tickers, years=1)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "turnover"}
        assert set(df.columns) == expected_cols
        assert len(df) > 0
        # turnover should be volume * close
        sample = df.iloc[0]
        assert abs(sample["turnover"] - sample["volume"] * sample["close"]) < 0.01

    def test_download_single_ticker(self):
        from src.data.price_downloader import download_prices

        df = download_prices(["7203.T"], years=1)
        assert len(df) > 200  # ~250 trading days in a year
        assert df["ticker"].unique().tolist() == ["7203.T"]
