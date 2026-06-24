"""Verify the regime/risk overlay (layer C, docs/06 §6).

Does gating out new exposure during (high-correlation) risk-off shrink an
arbitrary long basket's max drawdown by MORE than it shrinks its return?
The basket = the cross-sectional MEDIAN daily return of the universe (a proxy
for an equal-weight long), for All and for Small (Std+Growth) separately.

Verdict: overlay improves return / |maxDD| AND lowers maxDD -> GO.
If it only lowers both proportionally, it is just beta reduction -> reject.

Usage:
    uv run python scripts/run_regime_overlay.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtest.honest import DAILY_CLIP  # noqa: E402
from src.data.ticker_master import load_ticker_master  # noqa: E402
from src.overlay import regime  # noqa: E402

logger = logging.getLogger(__name__)
_CACHE = _ROOT / "data_cache"
_SMALL = ["Standard", "Growth"]


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.1f}%"


def basket_returns(prices: pd.DataFrame, agg: str = "mean") -> pd.Series:
    """Daily cross-sectional return (>=5 names), indexed by date.

    ``mean`` = an equal-weight daily-rebalanced long (what a diversified long
    actually earns); ``median`` = robust path (understates due to skew).
    """
    df = prices[["date", "ticker", "adj_close"]].sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker", sort=False)["adj_close"].pct_change().clip(-DAILY_CLIP, DAILY_CLIP)
    daily = df.dropna(subset=["ret"]).groupby("date")["ret"].agg([agg, "count"])
    return daily[daily["count"] >= 5][agg]


def metrics(ret: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    max_dd = float((eq / eq.cummax() - 1).min())
    total = float(eq.iloc[-1] - 1)
    ratio = total / abs(max_dd) if max_dd < 0 else float("nan")
    return {"total": total, "max_dd": max_dd, "ratio": ratio}


def evaluate(basket: pd.Series, regime_df: pd.DataFrame, gate_col: str, label: str) -> None:
    gate = regime.gate_for_dates(regime_df, basket.index, gate_col)
    gate = gate.reindex(basket.index).fillna(False).astype(bool)
    base = metrics(basket)
    overlay_ret = basket.where(~gate.values, 0.0)
    ov = metrics(overlay_ret)
    in_mkt = float((~gate).mean())
    better = ov["max_dd"] > base["max_dd"] and (
        pd.notna(ov["ratio"]) and pd.notna(base["ratio"]) and ov["ratio"] > base["ratio"]
    )
    print(f"  {label:24s} | base: ret {_pct(base['total']):>7} DD {_pct(base['max_dd']):>7} "
          f"r/DD {base['ratio']:5.2f} | overlay: ret {_pct(ov['total']):>7} DD {_pct(ov['max_dd']):>7} "
          f"r/DD {ov['ratio']:5.2f} | in-mkt {in_mkt*100:3.0f}% | {'GO' if better else 'no'}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    prices = pd.read_parquet(_CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    master = load_ticker_master()
    mkt = dict(zip(master["ticker"], master["market"]))

    macro = regime.fetch_macro()
    print(f"\nmacro: {macro['date'].min().date()}..{macro['date'].max().date()} ({len(macro)} rows)")

    all_basket = basket_returns(prices, "mean")
    small_basket = basket_returns(prices[prices["ticker"].map(mkt).isin(_SMALL)], "mean")

    # regime definitions to test (avoid cherry-picking: report several)
    variants = [
        ("corr>=0.5, GSPC<50dMA", dict(corr_threshold=0.5), "gate_hi_corr_riskoff"),
        ("corr>=0.4, GSPC<50dMA", dict(corr_threshold=0.4), "gate_hi_corr_riskoff"),
        ("risk-off only (GSPC<50dMA)", dict(), "gate_riskoff"),
    ]
    for basket, bname in [(all_basket, "ALL universe"), (small_basket, "Small (Std+Growth)")]:
        print(f"\n=== basket: {bname} ({len(basket)} days) ===")
        for vlabel, kwargs, gate_col in variants:
            reg = regime.compute_regime(macro, **kwargs)
            evaluate(basket, reg, gate_col, vlabel)

    print("\nnote: overlay = fully in cash on gated days (strong form). "
          "GO requires DD shrinks AND return/|DD| improves (not mere beta reduction).")
    print("(see docs/10 for the verdict)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
