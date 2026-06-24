"""Verify the avoidance overlay (layer B, docs/06 §5), BY MARKET SEGMENT.

After the universe expansion (docs/09) the Tier-A toxic-financing events that
were 88% outside Prime should now map into Standard/Growth, where finance found
the real underperformance.  We measure Tier A in Small caps and the overlay's
drawdown effect on a small-cap long basket.

Usage:
    uv run python scripts/run_avoidance_overlay.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtest import honest  # noqa: E402
from src.data.ticker_master import load_ticker_master  # noqa: E402
from src.ingest import financing_events as fe  # noqa: E402
from src.overlay import avoidance  # noqa: E402

logger = logging.getLogger(__name__)
_CACHE = _ROOT / "data_cache"
_DEFAULT_FIN_DB = Path(r"D:\work\finance\data\db\screener.db")
_SMALL = ["Standard", "Growth"]


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.2f}%"


def _block(vals: np.ndarray) -> str:
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        return "  (none)"
    return (f"n={len(v):4d}  median={_pct(np.median(v))}  mean={_pct(np.mean(v))}  "
            f"loss={(v<0).mean():.0%}  deep(<-10%)={(v<-0.10).mean():.0%}")


def event_study(prices, events, mindex, label) -> None:
    rr = honest.compute_event_returns(prices, events, mindex, horizons=(5, 20, 60))
    studied = rr["ticker"].nunique() if not rr.empty else 0
    print(f"\n=== {label}: {len(events)} events ({events['ticker'].nunique()} tickers), studied {studied} ===")
    if rr.empty:
        print("  (no price-covered events)"); return
    for n in (5, 20, 60):
        print(f"  +{n:>2}d neut: {_block(rr[rr['hold_days']==n]['adj_return'].to_numpy())}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--finance-db", type=Path, default=_DEFAULT_FIN_DB)
    p.add_argument("--window", type=int, default=avoidance.DEFAULT_WINDOW)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    events = fe.load_events()
    if events.empty:
        if not args.finance_db.exists():
            logger.error("no events and finance DB not found at %s", args.finance_db); return 1
        events = fe.import_from_sqlite(args.finance_db)

    prices = pd.read_parquet(_CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    master = load_ticker_master()
    mkt = dict(zip(master["ticker"], master["market"]))
    universe = set(prices["ticker"].unique())
    mindex = honest.build_universe_index(prices)

    # Tier A coverage by market segment
    a_all = avoidance.tier_a_events(events)
    a_all = a_all.copy()
    a_all["market"] = a_all["ticker"].map(mkt).fillna("(not listed/ETF/foreign)")
    print(f"\nTier A (deduped): {len(a_all)} events, {a_all['ticker'].nunique()} tickers")
    print("  by market:", a_all["market"].value_counts().to_dict())

    a_small = a_all[a_all["market"].isin(_SMALL) & a_all["ticker"].isin(universe)]
    a_prime = a_all[(a_all["market"] == "Prime") & a_all["ticker"].isin(universe)]
    b_small = avoidance.dedup_events(events[events["tier"] == "B"])
    b_small = b_small[b_small["ticker"].map(mkt).isin(_SMALL) & b_small["ticker"].isin(universe)]

    event_study(prices, a_small, mindex, "Tier A in Small (Std+Growth)")
    event_study(prices, a_prime, mindex, "Tier A in Prime (contrast)")
    event_study(prices, b_small, mindex, "Tier B in Small (contrast)")

    # overlay effect on the honest small-cap signal basket
    hr_path = _CACHE / "honest_returns.parquet"
    if hr_path.exists():
        hr = pd.read_parquet(hr_path)
        if "market" in hr.columns:
            hr = hr[hr["market"].isin(_SMALL)]
        n_pick = 18 if 18 in hr["hold_days"].unique() else int(hr["hold_days"].min())
        basket = hr[hr["hold_days"] == n_pick][["ticker", "entry_date", "adj_return", "dd"]].copy()
        basket = basket.rename(columns={"entry_date": "date"})
        ann = avoidance.annotate(basket, events, window_days=args.window)
        n_flag = int(ann["red_flag"].sum())
        print(f"\n=== overlay effect on Small honest basket (n={n_pick}, {len(ann)} signals) ===")
        print(f"  flagged by Tier A overlay (window {args.window}d): {n_flag} ({n_flag/max(len(ann),1):.2%})")
        if n_flag:
            kept = ann[~ann["red_flag"]]
            print(f"  all     : median adj {_pct(ann['adj_return'].median())}  "
                  f"dd_median {_pct(ann['dd'].median())}  worst {_pct(ann['adj_return'].min())}  "
                  f"deep(<-20%) {(ann['adj_return']<-0.20).mean():.1%}")
            print(f"  excluded: median adj {_pct(ann[ann['red_flag']]['adj_return'].median())}  "
                  f"worst {_pct(ann[ann['red_flag']]['adj_return'].min())}  "
                  f"deep(<-20%) {(ann[ann['red_flag']]['adj_return']<-0.20).mean():.1%}")
            print(f"  kept    : median adj {_pct(kept['adj_return'].median())}  "
                  f"dd_median {_pct(kept['dd'].median())}  worst {_pct(kept['adj_return'].min())}  "
                  f"deep(<-20%) {(kept['adj_return']<-0.20).mean():.1%}")

    print("\n(see docs/09 for the verdict)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
