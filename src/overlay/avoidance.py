"""Avoidance overlay (integration design docs/06 §5, layer B).

Hard-excludes / red-flags tickers that recently announced toxic financing
(moving-strike warrants / MSCB = Tier A).  This is the one signal that held
up in the finance project (検証10): not a prediction, an *exclusion* that
trims the left tail.  "Don't lose" is itself value (CLAUDE.md positioning).
"""
from __future__ import annotations

import pandas as pd

from src.ingest.financing_events import load_events

DEDUP_DAYS = 90  # collapse a ticker's repeated announcements in one episode
DEFAULT_WINDOW = 60  # avoid for this many calendar days after the announcement


def dedup_events(events: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest announcement per ticker within each 90-day episode."""
    if events.empty:
        return events
    ev = events.copy()
    ev["disclosed_date"] = pd.to_datetime(ev["disclosed_at"]).dt.normalize()
    ev = ev.sort_values(["ticker", "disclosed_date"])
    keep, last = [], {}
    for row in ev.itertuples(index=False):
        prev = last.get(row.ticker)
        if prev is not None and (row.disclosed_date - prev).days < DEDUP_DAYS:
            continue
        last[row.ticker] = row.disclosed_date
        keep.append(row)
    return pd.DataFrame(keep)


def tier_a_events(events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Deduped Tier A (moving-strike) events with ticker + disclosed_date."""
    ev = events if events is not None else load_events()
    ev = ev[ev["tier"] == "A"]
    return dedup_events(ev)


def annotate(
    candidates: pd.DataFrame,
    events: pd.DataFrame | None = None,
    window_days: int = DEFAULT_WINDOW,
    date_col: str = "date",
    tier: str = "A",
) -> pd.DataFrame:
    """Add ``red_flag`` / ``flag_subtype`` / ``flag_date`` to candidate rows.

    A row is flagged if its ticker announced a Tier-``tier`` financing event
    on or before the row's date and within ``window_days`` calendar days.
    """
    out = candidates.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    ev = (events if events is not None else load_events())
    ev = ev[ev["tier"] == tier]
    ev = dedup_events(ev)
    out["red_flag"] = False
    out["flag_subtype"] = None
    out["flag_date"] = pd.NaT
    if ev.empty or out.empty:
        return out

    ev = ev.sort_values("disclosed_date").copy()
    ev["disclosed_date"] = ev["disclosed_date"].astype("datetime64[ns]")
    out[date_col] = out[date_col].astype("datetime64[ns]")
    left = out.sort_values(date_col).reset_index()  # keep original index
    merged = pd.merge_asof(
        left, ev[["ticker", "disclosed_date", "subtype"]].rename(columns={"subtype": "_sub"}),
        left_on=date_col, right_on="disclosed_date", by="ticker", direction="backward",
    )
    within = (merged[date_col] - merged["disclosed_date"]).dt.days
    flagged = within.between(0, window_days)
    merged["red_flag"] = flagged.fillna(False)
    res = merged.set_index("index")
    out["red_flag"] = res["red_flag"].reindex(out.index).fillna(False)
    out["flag_subtype"] = res["_sub"].where(res["red_flag"]).reindex(out.index)
    out["flag_date"] = res["disclosed_date"].where(res["red_flag"]).reindex(out.index)
    return out


def exclusion_mask(
    candidates: pd.DataFrame, events: pd.DataFrame | None = None,
    window_days: int = DEFAULT_WINDOW, date_col: str = "date", tier: str = "A",
) -> pd.Series:
    """Boolean Series: True = exclude (recently flagged). Aligned to candidates.index."""
    return annotate(candidates, events, window_days, date_col, tier)["red_flag"]
