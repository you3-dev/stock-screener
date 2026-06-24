"""Resumable price fetch for the honest backtest (integration design docs/06).

Two fixes over ``src/data/price_downloader.py`` that the honest engine needs:

1. **Keeps ``adj_close``** (downloads with ``auto_adjust=False``) so returns can
   be split/dividend adjusted.  The product downloader drops it.
2. **Resumable** -- each chunk is written to ``data_cache/price_parts/`` and
   skipped on re-run.  The original holds every chunk in memory and saves once
   at the end, so a mid-run yfinance failure loses everything.  The finance
   project's hard rule: long batches must be resumable (the user powers the PC
   off mid-run).

Usage:
    uv run python scripts/fetch_prices.py                 # full Prime, 5y
    uv run python scripts/fetch_prices.py --limit 300     # quick sample
    uv run python scripts/fetch_prices.py --combine-only  # rebuild prices.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import yfinance as yf  # noqa: E402

from src.data.config import load_config  # noqa: E402
from src.data.ticker_master import load_ticker_master  # noqa: E402

logger = logging.getLogger(__name__)

_CACHE_DIR = _ROOT / "data_cache"
_PARTS_DIR = _CACHE_DIR / "price_parts"
_PRICE_FILE = _CACHE_DIR / "prices.parquet"
_FINAL_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "turnover"]


def _normalize(raw: pd.DataFrame, chunk: list[str]) -> pd.DataFrame:
    """yfinance result (auto_adjust=False) -> long format keeping adj_close."""
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.stack(level="Ticker", future_stack=True).reset_index()
    else:  # single-ticker fallback: columns are plain price fields
        df = raw.reset_index()
        df["Ticker"] = chunk[0]
    df = df.rename(columns={"Ticker": "ticker"})
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]  # auto_adjust already applied upstream
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    df["turnover"] = df["volume"] * df["close"]
    keep = [c for c in _FINAL_COLS if c in df.columns]
    return df[keep].dropna(subset=["close"])


def _part_path(i: int) -> Path:
    return _PARTS_DIR / f"part_{i:04d}.parquet"


def fetch(tickers: list[str], years: int, chunk_size: int, sleep: float) -> None:
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=years * 365)
    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    _PARTS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = sorted(set(tickers))  # deterministic chunking for stable resume
    total = (len(tickers) + chunk_size - 1) // chunk_size
    logger.info("Fetching %d tickers in %d chunks (%s..%s)", len(tickers), total, start_str, end_str)

    for ci, i in enumerate(range(0, len(tickers), chunk_size)):
        part = _part_path(i)
        if part.exists():
            logger.info("  chunk %d/%d: skip (exists)", ci + 1, total)
            continue
        chunk = tickers[i : i + chunk_size]
        for attempt in range(3):
            try:
                raw = yf.download(
                    chunk, start=start_str, end=end_str,
                    auto_adjust=False, progress=False, threads=True,
                )
                if raw is None or raw.empty:
                    logger.warning("  chunk %d/%d: empty (attempt %d)", ci + 1, total, attempt + 1)
                    time.sleep(sleep * (attempt + 1))
                    continue
                df = _normalize(raw, chunk)
                if df.empty:
                    logger.warning("  chunk %d/%d: no usable rows", ci + 1, total)
                    break
                df.to_parquet(part, index=False)
                logger.info("  chunk %d/%d: saved %d rows, %d tickers",
                            ci + 1, total, len(df), df["ticker"].nunique())
                break
            except Exception as e:  # noqa: BLE001 - keep going on any yfinance error
                logger.warning("  chunk %d/%d failed (attempt %d): %s: %s",
                               ci + 1, total, attempt + 1, type(e).__name__, e)
                time.sleep(sleep * (attempt + 1))
        time.sleep(sleep)


def combine() -> Path | None:
    parts = sorted(_PARTS_DIR.glob("part_*.parquet")) if _PARTS_DIR.exists() else []
    if not parts:
        logger.warning("no part files to combine")
        return None
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"]).reset_index(drop=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PRICE_FILE, index=False)
    logger.info("combined %d parts -> %s (%d rows, %d tickers)",
                len(parts), _PRICE_FILE, len(df), df["ticker"].nunique())
    return _PRICE_FILE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Resumable Prime price fetch (keeps adj_close)")
    p.add_argument("--years", type=int, default=None, help="lookback years (default: config)")
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None, help="only first N tickers (quick sample)")
    p.add_argument("--combine-only", action="store_true", help="just rebuild prices.parquet")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    years = args.years or load_config()["backtest"]["lookback_years"]

    if not args.combine_only:
        master = load_ticker_master()
        tickers = master["ticker"].tolist()
        if args.limit:
            tickers = sorted(set(tickers))[: args.limit]
        fetch(tickers, years, args.chunk_size, args.sleep)
    combine()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
