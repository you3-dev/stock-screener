"""Nightly pipeline (layer-A-demoted edition, integration docs/06 §8).

Builds the caches the app needs: features (pattern watchlist), toxic-financing
events (layer B), and macro/regime (layer C).  It does NOT regenerate the old
per-ticker backtest cache — that ranking is invalid (docs/07), so the legacy
tabs show a notice instead.

Run ``scripts/fetch_prices.py`` first (it keeps adj_close and is resumable).

Usage:
    uv run python scripts/fetch_prices.py     # 1) prices (Prime+Standard+Growth)
    uv run python scripts/run_pipeline.py     # 2) features + events + macro
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.features.engineer import compute_features, save_features  # noqa: E402
from src.ingest import financing_events as fe  # noqa: E402
from src.overlay import regime  # noqa: E402
from src.screening.screener import screen_candidates_v2  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

_CACHE = _ROOT / "data_cache"
_FIN_DB = Path(r"D:\work\finance\data\db\screener.db")


def main() -> int:
    logger.info("=== pipeline (layer-A-demoted) started ===")
    prices_path = _CACHE / "prices.parquet"
    if not prices_path.exists():
        logger.error("no prices.parquet — run scripts/fetch_prices.py first")
        return 1
    prices = pd.read_parquet(prices_path)
    logger.info("prices: %d rows, %d tickers", len(prices), prices["ticker"].nunique())

    # 1. features (pattern watchlist)
    logger.info("Step 1/3: features")
    feats = compute_features(prices)
    save_features(feats)

    # 2. toxic-financing events (layer B)
    logger.info("Step 2/3: financing events")
    events = fe.load_events()
    if events.empty and _FIN_DB.exists():
        events = fe.import_from_sqlite(_FIN_DB)
    logger.info("events: %d (TierA=%d)", len(events),
                int((events["tier"] == "A").sum()) if len(events) else 0)

    # 3. macro / regime (layer C)
    logger.info("Step 3/3: macro/regime")
    macro = regime.fetch_macro(force=True)
    reg = regime.compute_regime(macro).dropna(subset=["risk_off"])
    last = reg.iloc[-1]
    logger.info("regime asof %s: risk_off=%s", pd.Timestamp(last["date"]).date(), bool(last["risk_off"]))

    cand = screen_candidates_v2(feats, events)
    flagged = int(cand["red_flag"].sum()) if "red_flag" in cand and not cand.empty else 0
    logger.info("today's watchlist: %d matches (%d red-flagged/excluded)", len(cand), flagged)
    logger.info("=== pipeline completed ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("pipeline failed")
        sys.exit(1)
