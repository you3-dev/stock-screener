"""Nightly batch pipeline: data update, features, backtest, screening.

Usage:
    uv run python scripts/run_pipeline.py
"""

import logging
import sys

from src.backtest.engine import run_backtest, save_backtest_results
from src.data.price_downloader import download_prices, save_prices
from src.data.ticker_master import load_ticker_master
from src.features.engineer import compute_features, save_features
from src.screening.screener import screen_candidates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full nightly pipeline."""
    logger.info("=== Nightly pipeline started ===")

    # 1. Ticker master
    logger.info("Step 1/6: Loading ticker master...")
    master = load_ticker_master(force_refresh=True)
    tickers = master["ticker"].tolist()
    logger.info("Loaded %d tickers", len(tickers))

    # 2. Price download
    logger.info("Step 2/6: Downloading prices...")
    prices = download_prices(tickers)
    save_prices(prices)
    logger.info("Prices: %d rows", len(prices))

    # 3. Feature engineering
    logger.info("Step 3/6: Computing features...")
    features = compute_features(prices)
    save_features(features)
    logger.info("Features: %d rows", len(features))

    # 4. Backtest (conditional on capital-inflow signal days)
    logger.info("Step 4/6: Running backtest...")
    backtest = run_backtest(prices, features)
    save_backtest_results(backtest)
    logger.info("Backtest: %d results", len(backtest))

    # 5. Screening
    logger.info("Step 5/6: Running screening...")
    candidates = screen_candidates(features, backtest)
    logger.info("Screening: %d candidates", len(candidates))

    # 6. Summary
    logger.info("Step 6/6: Summary")
    if candidates.empty:
        logger.info("No candidates found today.")
    else:
        logger.info("Top candidates:")
        for _, row in candidates.head(5).iterrows():
            logger.info(
                "  %s | return=%+.2f%% | win_rate=%.1f%% | hold=%dd | dd=%.2f%%",
                row["ticker"],
                row["expected_return_1d"] * 100,
                row["win_rate"] * 100,
                row["recommended_hold_days"],
                row["dd_median"] * 100,
            )

    logger.info("=== Nightly pipeline completed ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
