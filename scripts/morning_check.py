"""Pre-market gap check script (08:50 JST).

Compares previous close to latest price data and classifies
each screening candidate by gap threshold.

Usage:
    uv run python scripts/morning_check.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.backtest.engine import load_backtest_results
from src.data.config import load_config
from src.data.price_downloader import update_prices
from src.features.engineer import load_features
from src.screening.screener import screen_candidates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
_CHECK_FILE = _CACHE_DIR / "market_check.txt"


def main() -> None:
    """Run pre-market gap check."""
    logger.info("=== Morning gap check started ===")

    cfg = load_config()
    gap_cfg = cfg["premarket_gap"]
    ok_max = gap_cfg["ok_max"]
    caution_max = gap_cfg["caution_max"]

    # 1. Load previous screening candidates
    features = load_features()
    backtest = load_backtest_results()
    if features is None or backtest is None:
        logger.error("Missing cached data. Run nightly pipeline first.")
        sys.exit(1)

    candidates = screen_candidates(features, backtest)
    if candidates.empty:
        logger.info("No candidates to check.")
        _write_result("No candidates to check.", [])
        return

    candidate_tickers = candidates["ticker"].tolist()
    logger.info("Checking %d candidates", len(candidate_tickers))

    # 2. Update prices with recent data
    logger.info("Updating recent prices...")
    prices = update_prices(candidate_tickers, recent_days=3)

    if prices.empty:
        logger.warning("No price data available.")
        _write_result("No price data available.", [])
        return

    # 3. Get latest two closes for each candidate
    results = []
    for ticker in candidate_tickers:
        tp = prices[prices["ticker"] == ticker].sort_values("date")
        if len(tp) < 2:
            results.append({"ticker": ticker, "gap": None, "status": "データ不足"})
            continue

        prev_close = tp["close"].iloc[-2]
        latest_close = tp["close"].iloc[-1]
        gap = latest_close / prev_close - 1

        if gap <= ok_max:
            status = "OK"
        elif gap <= caution_max:
            status = "注意"
        else:
            status = "除外推奨"

        results.append({"ticker": ticker, "gap": gap, "status": status})

    # 4. Output results
    _write_result(
        f"Gap check: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        results,
    )

    # 5. Summary log
    for r in results:
        gap_str = f"{r['gap']:+.2%}" if r["gap"] is not None else "N/A"
        logger.info("  %s | gap=%s | %s", r["ticker"], gap_str, r["status"])

    ok_count = sum(1 for r in results if r["status"] == "OK")
    caution_count = sum(1 for r in results if r["status"] == "注意")
    exclude_count = sum(1 for r in results if r["status"] == "除外推奨")
    logger.info(
        "Summary: OK=%d, Caution=%d, Exclude=%d", ok_count, caution_count, exclude_count
    )

    logger.info("=== Morning gap check completed ===")


def _write_result(header: str, results: list[dict]) -> None:
    """Write gap check results to text file."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [header, "=" * 50]
    for r in results:
        gap_str = f"{r['gap']:+.2%}" if r.get("gap") is not None else "N/A"
        lines.append(f"{r['ticker']:>10s}  gap={gap_str:>8s}  {r['status']}")
    lines.append("")
    _CHECK_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results written to %s", _CHECK_FILE)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Morning check failed")
        sys.exit(1)
