"""Pre-market check (08:50 JST) — layer-A-demoted edition (docs/12).

Morning value in the new model = the regime banner being current, plus a light
gap check on today's watchlist.  Does NOT mutate prices.parquet (the old
update_prices dropped adj_close); recent closes are fetched ad hoc.

Usage:
    uv run python scripts/morning_check.py
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

from src.data.config import load_config
from src.features.engineer import load_features
from src.ingest.financing_events import load_events
from src.overlay.integration import current_regime
from src.screening.screener import screen_candidates_v2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent / "data_cache"
_CHECK_FILE = _CACHE / "market_check.txt"


def main() -> None:
    logger.info("=== morning check started ===")
    try:
        from src.data.release_store import ensure_data_available
        ensure_data_available()
    except Exception:
        logger.warning("release fetch skipped", exc_info=True)

    cfg = load_config()
    ok_max = cfg["premarket_gap"]["ok_max"]
    caution_max = cfg["premarket_gap"]["caution_max"]

    # 1. regime (the key morning signal) — refreshes macro.parquet
    reg = current_regime()
    logger.info("regime asof %s: %s (size x%.1f)", reg["asof"], reg["label"], reg["size_factor"])

    # 2. today's watchlist (layer A demoted) + red flags (layer B)
    lines = [f"morning check {datetime.now():%Y-%m-%d %H:%M}", "=" * 50,
             f"地合い: {reg['label']} (asof {reg['asof']}) 推奨サイズ x{reg['size_factor']:.1f}"]
    feats = load_features()
    tickers: list[str] = []
    if feats is not None:
        cand = screen_candidates_v2(feats, load_events())
        watch = cand[~cand["red_flag"]] if "red_flag" in cand and not cand.empty else cand
        flagged = cand[cand["red_flag"]] if "red_flag" in cand and not cand.empty else cand.iloc[0:0]
        tickers = watch["ticker"].tolist()
        lines.append(f"ウォッチ {len(watch)}件 / 🚩回避 {len(flagged)}件")
    else:
        lines.append("features キャッシュ無し(夜間バッチ未実行)")

    # 3. light gap check (best-effort; no cache mutation)
    if tickers:
        lines.append("-" * 50)
        try:
            raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
            close = raw["Close"] if "Close" in raw else raw
            for t in tickers:
                s = close[t].dropna() if t in getattr(close, "columns", []) else close.dropna()
                if len(s) < 2:
                    lines.append(f"{t:>10s}  gap=     N/A  データ不足"); continue
                gap = float(s.iloc[-1] / s.iloc[-2] - 1)
                status = "OK" if gap <= ok_max else ("注意" if gap <= caution_max else "除外推奨")
                lines.append(f"{t:>10s}  gap={gap:+7.2%}  {status}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"(gap check skipped: {type(e).__name__})")

    _CACHE.mkdir(parents=True, exist_ok=True)
    _CHECK_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", _CHECK_FILE)
    logger.info("=== morning check completed ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("morning check failed")
        sys.exit(1)
