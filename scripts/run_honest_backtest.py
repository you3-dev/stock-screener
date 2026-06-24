"""Run the honest backtest and print a GO/NO-GO verdict, BY MARKET SEGMENT.

Layer A re-validation (docs/06 §4, docs/07).  After the universe expansion
(docs/09) we compare Prime vs Standard vs Growth vs Small(=Std+Growth) to see
whether the honest technical edge that was ~0 in Prime appears in small caps.

Usage:
    uv run python scripts/run_honest_backtest.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtest import honest  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.ticker_master import load_ticker_master  # noqa: E402
from src.features.engineer import compute_features  # noqa: E402

logger = logging.getLogger(__name__)
_CACHE = _ROOT / "data_cache"
_PRICES = _CACHE / "prices.parquet"


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.2f}%"


def seg_verdict(ret: pd.DataFrame, cfg: dict, cost: float, label: str) -> dict:
    """Print one compact verdict line for a segment; return a summary dict."""
    if ret.empty:
        print(f"  {label:9s} | no signals")
        return {"label": label, "go": False}
    h = honest.summarize_by_horizon(ret, cost)
    valid = h[h["n_obs"] >= 30]
    best = valid.loc[valid["net_median"].idxmax()] if not valid.empty else h.iloc[0]
    bn = int(best["hold_days"])
    n1 = h[h["hold_days"] == 1]
    n1net = n1.iloc[0]["net_median"] if not n1.empty else float("nan")
    by_year = honest.summarize_by_period(ret, cost, bn)
    years_pos = float((by_year["net_median"] > 0).mean()) if len(by_year) else 0.0
    oos = honest.oos_split(ret, cost, bn)
    hold_pos = bool(oos) and oos["holdout"]["net_median"] > 0
    go = (best["net_median"] > 0 and best["net_win_rate"] >= 0.50
          and years_pos >= 0.6 and hold_pos)
    print(f"  {label:9s} | sig={int(n1.iloc[0]['n_obs']) if not n1.empty else 0:5d} | "
          f"n=1 net {_pct(n1net):>8} | best n={bn:2d} net {_pct(best['net_median']):>8} "
          f"win {best['net_win_rate']*100:4.0f}% | yrs+ {years_pos*100:3.0f}% | "
          f"holdout {'+' if hold_pos else '-'} | {'GO' if go else 'no'}")
    return {"label": label, "go": go, "best_n": bn,
            "best_net": float(best["net_median"]), "years_pos": years_pos}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if not _PRICES.exists():
        logger.error("price cache not found: %s (run scripts/fetch_prices.py)", _PRICES)
        return 1

    cfg = load_config()
    cost = honest.round_trip_cost_frac(cfg)
    max_n = cfg["backtest"]["max_hold_days"]

    prices = pd.read_parquet(_PRICES)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    if "adj_close" not in prices.columns:
        logger.error("prices.parquet has no adj_close -- re-fetch"); return 1

    master = load_ticker_master()
    mkt = dict(zip(master["ticker"], master["market"])) if "market" in master.columns else {}
    by_mkt = {m: prices[prices['ticker'].isin([t for t,mm in mkt.items() if mm==m])]['ticker'].nunique()
              for m in ["Prime", "Standard", "Growth"]}
    print(f"\nuniverse: {prices['ticker'].nunique()} tickers {by_mkt}, "
          f"{prices['date'].min().date()}..{prices['date'].max().date()}, {len(prices):,} rows")

    feats = compute_features(prices)
    mindex = honest.build_universe_index(prices)
    honest_ret = honest.compute_signal_returns(feats, mindex, max_n=max_n, entry_shift=1, cfg=cfg)
    naive_ret = honest.compute_signal_returns(feats, mindex, max_n=max_n, entry_shift=0,
                                              gap_exclude=None, cfg=cfg)
    honest_ret["market"] = honest_ret["ticker"].map(mkt)
    naive_ret["market"] = naive_ret["ticker"].map(mkt)
    print(f"signals (honest): {honest_ret[honest_ret['hold_days']==1].shape[0]:,}\n")

    print(f"=== per-segment verdict (cost {cost*100:.2f}% round-trip) ===")
    small = honest_ret[honest_ret["market"].isin(["Standard", "Growth"])]
    seg_verdict(honest_ret, cfg, cost, "ALL")
    for m in ["Prime", "Standard", "Growth"]:
        seg_verdict(honest_ret[honest_ret["market"] == m], cfg, cost, m)
    seg_verdict(small, cfg, cost, "Small")

    # detailed naive-vs-honest contrast for the Small segment (the focus)
    print("\n=== Small (Std+Growth): naive (look-ahead) vs honest by horizon ===")
    nv_small = naive_ret[naive_ret["market"].isin(["Standard", "Growth"])]
    hs = honest.summarize_by_horizon(small, cost)
    ns = honest.summarize_by_horizon(nv_small, 0.0)
    print(f"{'n':>3} {'n_obs':>6} | {'naive raw':>10} | {'honest adj':>11} {'net':>9} {'net_win':>8} {'dd':>8}")
    for n in [1, 2, 3, 5, 10, 20]:
        h = hs[hs["hold_days"] == n]
        nv = ns[ns["hold_days"] == n]
        if h.empty:
            continue
        h = h.iloc[0]
        nvr = _pct(nv.iloc[0]["raw_median"]) if not nv.empty else "n/a"
        print(f"{n:>3} {int(h['n_obs']):>6} | {nvr:>10} | {_pct(h['adj_median']):>11} "
              f"{_pct(h['net_median']):>9} {h['net_win_rate']*100:>7.1f}% {_pct(h['dd_median']):>8}")

    if not small.empty:
        valid = hs[hs["n_obs"] >= 30]
        bn = int(valid.loc[valid["net_median"].idxmax()]["hold_days"]) if not valid.empty else 1
        print(f"\n=== Small: net median by year (n={bn}) ===")
        for _, r in honest.summarize_by_period(small, cost, bn).iterrows():
            print(f"  {int(r['year'])}: {_pct(r['net_median'])} win={r['win_rate']*100:.0f}% n={int(r['n'])}")

    _CACHE.mkdir(parents=True, exist_ok=True)
    honest_ret.to_parquet(_CACHE / "honest_returns.parquet", index=False)
    print("\nsaved honest_returns.parquet (with market column)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
