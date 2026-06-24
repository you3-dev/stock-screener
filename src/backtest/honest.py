"""Honest backtest engine (integration design docs/06 §4).

Supersedes the look-ahead / raw-return / per-ticker-in-sample logic of
``engine.py`` with the validated discipline ported (as *method*, not code)
from the sibling ``finance`` project:

- **Entry at the NEXT trading day's open.**  The original engine measured
  ``Open(t)/Open(t)`` while the signal needs day-t's *close* (bullish candle,
  20-day-high break, today's turnover) -- unknowable at day-t's open.  Live
  trading enters the next open, so we measure from ``Open(t+1)``.
- **Open-to-open returns**, matching the product's "exit at a later open".
- **Split/dividend adjusted** via ``adj_close`` (requires the price cache to
  keep ``adj_close``; see ``scripts/fetch_prices.py``).
- **Market/size neutralized** against the universe's daily cross-sectional
  *median* return -- not large-cap TOPIX.  The finance project showed that a
  single large-cap benchmark makes size-beta look like signal.
- **Pooled across all tickers** -- no per-ticker in-sample selection (the
  screener ranked each ticker by its *own* past conditional return on the
  same 5-year sample, which is overfitting).
- **Round-trip transaction costs** deducted (see ``config.costs``).
- Reported **by period** and on an **out-of-sample holdout** so that
  period-dependence is visible (finance design §6.4).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.config import load_config

logger = logging.getLogger(__name__)

DAILY_CLIP = 0.50  # clip daily returns at ±50% (unadjusted-split guard)
SPLIT_LOW, SPLIT_HIGH = 0.5, 2.0  # within-window split-glitch bounds

_DEFAULT_COSTS = {"commission_bps": 0.0, "spread_bps": 7.5, "slippage_bps": 7.5}


def round_trip_cost_frac(cfg: dict | None = None) -> float:
    """Round-trip cost as a fraction (both legs).

    Each component is per-side in basis points; a round trip pays both legs.
    Defaults are deliberately conservative for a market-on-open entry on a
    stock that broke out yesterday (adverse selection at the open).
    """
    costs = (cfg or load_config()).get("costs", _DEFAULT_COSTS)
    per_side = (
        costs.get("commission_bps", _DEFAULT_COSTS["commission_bps"])
        + costs.get("spread_bps", _DEFAULT_COSTS["spread_bps"])
        + costs.get("slippage_bps", _DEFAULT_COSTS["slippage_bps"])
    )
    return 2.0 * per_side / 1e4


def build_universe_index(prices: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional MEDIAN return, compounded to a level series.

    Returns a Series indexed by date (first day = 1.0).  A day needs >=5
    names to contribute a median; otherwise the level carries forward.
    Ported from ``finance/src/backtest/market.py``.
    """
    df = prices[["date", "ticker", "adj_close"]].sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker", sort=False)["adj_close"].pct_change()
    df = df.dropna(subset=["ret"])
    df["ret"] = df["ret"].clip(-DAILY_CLIP, DAILY_CLIP)

    daily = df.groupby("date")["ret"].agg(["median", "count"]).sort_index()
    level = 1.0
    levels: dict = {}
    for d, row in daily.iterrows():
        if row["count"] >= 5:
            level *= 1.0 + row["median"]
        levels[d] = level
    return pd.Series(levels, name="mindex")


def signal_mask(features: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """Capital-inflow signal mask -- identical conditions to the screener."""
    cfg = cfg or load_config()
    ci = cfg["capital_inflow"]
    excl = cfg["exclusion"]
    mask = (
        features["liquidity_flag"]
        & (features["turnover_ratio_5d"] >= ci["turnover_ratio_5d_min"])
        & features["high_20_break_flag"]
        & (features["close"] > features["open"])  # bullish candle
        & (features["atr14_ratio"] >= ci["atr14_ratio_min"])
        & (features["recent_3day_return"] <= ci["recent_3day_return_max"])
    )
    if excl.get("limit_up_next_day", False):
        mask = mask & (((features["close"] - features["open"]) / features["open"]) < 0.15)
    return mask.fillna(False)


def compute_signal_returns(
    features: pd.DataFrame,
    mindex: pd.Series,
    max_n: int = 20,
    entry_shift: int = 1,
    gap_exclude: float | None = 0.05,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Long DataFrame of per-(ticker, signal, hold_days) returns.

    Args:
        features: output of ``features.engineer.compute_features`` -- must
            include ``adj_close`` alongside raw OHLCV.
        mindex: universe median-return level series (``build_universe_index``).
        max_n: maximum holding period in trading days.
        entry_shift: 0 reproduces the original (look-ahead) same-day entry;
            1 is the honest next-open entry.  Used to contrast the two.
        gap_exclude: skip entries gapping up >= this fraction vs the prior
            close (mirrors the live 08:50 pre-market check; look-ahead safe
            because the gap is known at the open we buy).  None disables it.

    Returns:
        columns: ticker, entry_date, hold_days, raw_return, adj_return, dd
    """
    cfg = cfg or load_config()
    feat = features.sort_values(["ticker", "date"]).copy()
    factor = feat["adj_close"] / feat["close"]
    feat["adj_open"] = feat["open"] * factor
    feat["adj_low"] = feat["low"] * factor
    feat["_signal"] = signal_mask(feat, cfg)

    mi = mindex.to_dict()
    records: list[tuple] = []

    for ticker, g in feat.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        adj_open = g["adj_open"].to_numpy(dtype=float)
        adj_close = g["adj_close"].to_numpy(dtype=float)
        adj_low = g["adj_low"].to_numpy(dtype=float)
        dates = list(g["date"])
        nrows = len(g)
        for i in np.flatnonzero(g["_signal"].to_numpy()):
            entry_i = i + entry_shift
            if entry_i >= nrows:
                continue
            entry_open = adj_open[entry_i]
            if not np.isfinite(entry_open) or entry_open <= 0:
                continue
            # gap exclusion at entry (look-ahead safe)
            if gap_exclude is not None and entry_shift >= 1:
                prev_close = adj_close[i]
                if np.isfinite(prev_close) and prev_close > 0:
                    if entry_open / prev_close - 1.0 >= gap_exclude:
                        continue
            # split-glitch guard over the whole window
            last_i = min(entry_i + max_n, nrows - 1)
            win = adj_close[entry_i : last_i + 1]
            bad = False
            for a, b in zip(win, win[1:]):
                if a > 0 and np.isfinite(a) and np.isfinite(b):
                    r = b / a
                    if r < SPLIT_LOW or r > SPLIT_HIGH:
                        bad = True
                        break
            if bad:
                continue

            entry_date = dates[entry_i]
            mi_e = mi.get(entry_date)
            running_low = adj_low[entry_i]
            for n in range(1, max_n + 1):
                exit_i = entry_i + n
                if exit_i >= nrows:
                    break
                exit_open = adj_open[exit_i]
                if np.isfinite(adj_low[exit_i]):
                    running_low = min(running_low, adj_low[exit_i])
                if not np.isfinite(exit_open) or exit_open <= 0:
                    continue
                raw = exit_open / entry_open - 1.0
                mi_x = mi.get(dates[exit_i])
                adj = raw - (mi_x / mi_e - 1.0) if (mi_e and mi_x) else np.nan
                dd = running_low / entry_open - 1.0
                records.append((ticker, entry_date, n, raw, adj, dd))

    return pd.DataFrame.from_records(
        records, columns=["ticker", "entry_date", "hold_days", "raw_return", "adj_return", "dd"]
    )


def _adjust(prices: pd.DataFrame) -> pd.DataFrame:
    """Add split-adjusted open/low columns from adj_close (returns a copy)."""
    df = prices.sort_values(["ticker", "date"]).copy()
    factor = df["adj_close"] / df["close"]
    df["adj_open"] = df["open"] * factor
    df["adj_low"] = df["low"] * factor
    return df


def compute_event_returns(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    mindex: pd.Series,
    horizons=(5, 20, 60),
) -> pd.DataFrame:
    """Forward returns for arbitrary events (layer B avoidance study).

    Mirrors finance/src/backtest/returns.py: entry at the first trading day's
    open AFTER the disclosure, exit at adj_close at +n, market-neutralized by
    the universe median index, split-glitch guarded.

    Args:
        events: DataFrame with columns ``ticker`` and ``disclosed_at``.
    Returns:
        long DataFrame: ticker, disclosed_at, entry_date, hold_days,
        raw_return, adj_return, dd
    """
    adj = _adjust(prices)
    by_ticker = {tkr: g.reset_index(drop=True) for tkr, g in adj.groupby("ticker", sort=False)}
    mi = mindex.to_dict()
    max_n = max(horizons)
    records: list[tuple] = []

    for ev in events.itertuples(index=False):
        g = by_ticker.get(ev.ticker)
        if g is None:
            continue
        disc = str(ev.disclosed_at)[:10]
        dates = g["date"]
        # first trading day strictly after the disclosure date
        after = dates[dates > pd.Timestamp(disc)]
        if after.empty:
            continue
        entry_i = after.index[0]
        adj_open = g["adj_open"].to_numpy(dtype=float)
        adj_close = g["adj_close"].to_numpy(dtype=float)
        adj_low = g["adj_low"].to_numpy(dtype=float)
        dlist = list(g["date"])
        n_rows = len(g)
        entry_open = adj_open[entry_i]
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
        # split-glitch guard
        last_i = min(entry_i + max_n, n_rows - 1)
        win = adj_close[entry_i : last_i + 1]
        if any(
            (a > 0 and np.isfinite(a) and np.isfinite(b) and not (SPLIT_LOW <= b / a <= SPLIT_HIGH))
            for a, b in zip(win, win[1:])
        ):
            continue
        entry_date = dlist[entry_i]
        mi_e = mi.get(entry_date)
        running_low = adj_low[entry_i]
        for n in horizons:
            exit_i = entry_i + n
            if exit_i >= n_rows:
                continue
            running_low = min(running_low, np.nanmin(adj_low[entry_i : exit_i + 1]))
            exit_close = adj_close[exit_i]
            if not np.isfinite(exit_close) or exit_close <= 0:
                continue
            raw = exit_close / entry_open - 1.0
            mi_x = mi.get(dlist[exit_i])
            adj_r = raw - (mi_x / mi_e - 1.0) if (mi_e and mi_x) else np.nan
            dd = running_low / entry_open - 1.0
            records.append((ev.ticker, ev.disclosed_at, entry_date, n, raw, adj_r, dd))

    return pd.DataFrame.from_records(
        records,
        columns=["ticker", "disclosed_at", "entry_date", "hold_days", "raw_return", "adj_return", "dd"],
    )


def summarize_by_horizon(returns: pd.DataFrame, cost_frac: float) -> pd.DataFrame:
    """Per-horizon pooled stats (raw, neutralized, and net-of-cost)."""
    out = []
    for n, g in returns.groupby("hold_days"):
        adj = g["adj_return"].dropna()
        net = adj - cost_frac
        out.append(
            {
                "hold_days": int(n),
                "n_obs": int(len(g)),
                "raw_median": float(g["raw_return"].median()),
                "adj_median": float(adj.median()) if len(adj) else np.nan,
                "adj_mean": float(adj.mean()) if len(adj) else np.nan,
                "net_median": float(net.median()) if len(net) else np.nan,
                "net_win_rate": float((net > 0).mean()) if len(net) else np.nan,
                "dd_median": float(g["dd"].median()),
            }
        )
    return pd.DataFrame(out).sort_values("hold_days").reset_index(drop=True)


def summarize_by_period(returns: pd.DataFrame, cost_frac: float, hold_days: int) -> pd.DataFrame:
    """Net-of-cost neutralized median by calendar year for one horizon."""
    g = returns[returns["hold_days"] == hold_days].copy()
    g = g.dropna(subset=["adj_return"])
    g["year"] = pd.to_datetime(g["entry_date"]).dt.year
    g["net"] = g["adj_return"] - cost_frac
    out = g.groupby("year")["net"].agg(
        net_median="median", win_rate=lambda s: (s > 0).mean(), n="count"
    )
    return out.reset_index()


def oos_split(
    returns: pd.DataFrame, cost_frac: float, hold_days: int, holdout_days: int = 365
) -> dict:
    """In-sample vs out-of-sample (last ``holdout_days``) net median."""
    g = returns[returns["hold_days"] == hold_days].dropna(subset=["adj_return"]).copy()
    if g.empty:
        return {}
    g["entry_date"] = pd.to_datetime(g["entry_date"])
    cutoff = g["entry_date"].max() - pd.Timedelta(days=holdout_days)
    g["net"] = g["adj_return"] - cost_frac
    ins, oos = g[g["entry_date"] <= cutoff], g[g["entry_date"] > cutoff]
    summ = lambda s: {  # noqa: E731
        "net_median": float(s.median()) if len(s) else np.nan,
        "win_rate": float((s > 0).mean()) if len(s) else np.nan,
        "n": int(len(s)),
    }
    return {"cutoff": str(cutoff.date()), "in_sample": summ(ins["net"]), "holdout": summ(oos["net"])}
