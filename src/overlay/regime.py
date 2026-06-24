"""Regime / risk overlay (integration design docs/06 §6, layer C).

NOT a direction signal (that would violate finance rule §1 "don't use macro as
a selection signal").  Pure risk management: when JP and US move together AND
are in a downtrend (systemic risk-off, diversification fails), suppress new
swing entries / cut size.  The MVP question: does that shrink an arbitrary long
basket's max drawdown by MORE than it shrinks its return?

Look-ahead safe: the regime for day t is decided from data through t-1
(``.shift(1)``), so it is actionable at t's open.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent.parent / "data_cache"
_MACRO_FILE = _CACHE / "macro.parquet"
_TICKERS = ["^N225", "^GSPC"]


def fetch_macro(years: int = 5, force: bool = False) -> pd.DataFrame:
    """Daily close for ^N225 and ^GSPC (cached). Columns: date, n225, gspc."""
    if _MACRO_FILE.exists() and not force:
        return pd.read_parquet(_MACRO_FILE)
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=years * 365)
    raw = yf.download(_TICKERS, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    close = raw["Close"].reset_index()
    close.columns = [str(c).lower().replace(" ", "_") for c in close.columns]
    close = close.rename(columns={"^n225": "n225", "^gspc": "gspc"})
    close["date"] = pd.to_datetime(close["date"]).dt.normalize()
    _CACHE.mkdir(parents=True, exist_ok=True)
    close[["date", "n225", "gspc"]].to_parquet(_MACRO_FILE, index=False)
    logger.info("macro saved: %d rows %s..%s", len(close), close["date"].min().date(),
                close["date"].max().date())
    return close[["date", "n225", "gspc"]]


def compute_regime(macro: pd.DataFrame, corr_window: int = 60, ma_window: int = 50,
                   corr_threshold: float = 0.5) -> pd.DataFrame:
    """Per-date regime flags (look-ahead safe via shift(1)).

    Returns columns: date, corr, risk_off, high_corr, and gates
    ``gate_riskoff`` / ``gate_hi_corr_riskoff`` (True = suppress new entries).
    """
    m = macro.sort_values("date").copy()
    # N225 and GSPC trade on different calendars -> scattered NaN.  Clean each
    # series on its OWN valid dates (else a 50d MA / rolling corr spanning any
    # holiday NaN becomes NaN and the gate never fires).
    g = m[["date", "gspc"]].dropna().copy()
    g["r_g"] = g["gspc"].pct_change()
    g["gspc_ma"] = g["gspc"].rolling(ma_window).mean()
    g["risk_off"] = g["gspc"] < g["gspc_ma"]

    n = m[["date", "n225"]].dropna().copy()
    n["r_n"] = n["n225"].pct_change()

    # correlation on the two markets' common dates
    j = pd.merge(n[["date", "r_n"]], g[["date", "r_g"]], on="date", how="inner").sort_values("date")
    j["corr"] = j["r_n"].rolling(corr_window).corr(j["r_g"])

    # backbone = GSPC trading dates (risk-off basis); bring corr via ffill
    out = g[["date", "risk_off"]].merge(j[["date", "corr"]], on="date", how="left").sort_values("date")
    out["corr"] = out["corr"].ffill()
    out["high_corr"] = out["corr"] >= corr_threshold
    # decide from info through t-1 -> actionable at t's open
    out["gate_riskoff"] = out["risk_off"].shift(1).fillna(False)
    out["gate_hi_corr_riskoff"] = (out["high_corr"] & out["risk_off"]).shift(1).fillna(False)
    return out[["date", "corr", "risk_off", "high_corr", "gate_riskoff", "gate_hi_corr_riskoff"]]


def gate_for_dates(regime: pd.DataFrame, dates: pd.Series, gate_col: str) -> pd.Series:
    """Map a regime gate onto arbitrary (JP) trading dates via forward-fill.

    Returns a boolean Series aligned to ``dates`` (True = suppress new entry).
    """
    g = regime[["date", gate_col]].sort_values("date").set_index("date")[gate_col]
    target = pd.to_datetime(pd.Series(sorted(set(dates)))).dt.normalize()
    mapped = g.reindex(g.index.union(target)).ffill().reindex(target).fillna(False)
    return mapped
