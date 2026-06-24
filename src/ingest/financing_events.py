"""Toxic-financing event ingest (integration design docs/06 §5, layer B).

The one signal that survived in the sibling ``finance`` project (検証10):
companies that announce **moving-strike warrants / MSCB** (Tier A) materially
underperform afterwards.  We don't predict -- we *exclude* them (avoidance).

Events are classified from the TDnet disclosure **title only** (the PDF body
is not needed and vanishes ~30 days after disclosure; titles are retrievable
~3 years back).  Source = yanoshin TDnet WEB-API (free, no key).

Two ways to populate ``data_cache/financing_events.parquet``:
- ``import_from_sqlite()`` -- seed from finance's already-ingested 2.5y table
  (fast, and kind to the free API -- no redundant re-fetch).
- ``ingest_forward()`` -- resumable daily yanoshin pull for ongoing operation.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _ROOT / "data_cache"
_EVENTS_FILE = _CACHE / "financing_events.parquet"
_DONE_FILE = _CACHE / "financing" / "done_days.txt"

BASE_URL = "https://webapi.yanoshin.jp/webapi/tdnet/list"
MIN_SLEEP = 3.0  # free individual-run API: >=3s between requests (finance rule 4)
DEFAULT_LIMIT = 3000

# Follow-up / correction / cancellation titles are NOT the announcement event.
_EXCLUDE = (
    "払込完了", "発行完了", "割当完了", "行使状況", "行使価額の修正に関するお知らせ",
    "行使価額の決定", "取得状況", "途中経過", "終了", "完了", "訂正", "中止", "停止",
    "自己株式の処分", "自己株式の取得",
)

EVENT_COLS = ["code", "ticker", "disclosed_at", "tier", "subtype", "title", "company"]


def classify_financing(title: str | None) -> str | None:
    """Classify a disclosure title into a toxic-financing subtype, else None.

    Prefix ``A_`` = moving-strike (most toxic) / ``B_`` = ordinary private
    placement.  Ported verbatim from finance/src/backtest/landmine.py.
    """
    t = title or ""
    if any(x in t for x in _EXCLUDE):
        return None
    if not ("発行" in t or "割当" in t or "募集" in t):
        return None
    moving = ("行使価額修正" in t or "行使価額の修正条項" in t
              or "行使価額修正条項" in t or "ＭＳＣＢ" in t or "MSCB" in t)
    if "新株予約権" in t and moving:
        return "A_行使価額修正新株予約権"
    if "転換社債型新株予約権付社債" in t and ("修正" in t or "第三者割当" in t):
        return "A_MSCB社債"
    if "第三者割当" in t:
        if "新株予約権" in t:
            return "B_第三者割当新株予約権"
        if "新株式" in t or "募集株式" in t or "増資" in t:
            return "B_第三者割当増資"
    return None


def code_to_ticker(code: str | int | None) -> str | None:
    """yanoshin/TDnet 5-digit code -> yfinance ticker (e.g. 76030 -> 7603.T)."""
    if code is None:
        return None
    s = str(code).strip()
    if len(s) >= 5:
        s = s[:4]
    return f"{s}.T" if s else None


def _events_to_rows(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=EVENT_COLS)
    return df


def import_from_sqlite(db_path: str | Path) -> pd.DataFrame:
    """Seed events from finance's ``financing_events`` table (one-time)."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT code, disclosed_at, subtype, title, company FROM financing_events"
        ).fetchall()
    finally:
        conn.close()
    records = []
    for code, disclosed_at, subtype, title, company in rows:
        records.append({
            "code": str(code), "ticker": code_to_ticker(code),
            "disclosed_at": disclosed_at, "tier": (subtype or "?")[0],
            "subtype": subtype, "title": title, "company": company or "",
        })
    df = _events_to_rows(records)
    _save(df)
    logger.info("imported %d events from %s (TierA=%d)",
                len(df), db_path, int((df["tier"] == "A").sum()))
    return df


def _save(df: pd.DataFrame) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["ticker", "disclosed_at"]).drop_duplicates(
        ["code", "disclosed_at", "subtype"]).reset_index(drop=True)
    df.to_parquet(_EVENTS_FILE, index=False)
    return _EVENTS_FILE


def load_events() -> pd.DataFrame:
    """Load events parquet (empty frame if none)."""
    if not _EVENTS_FILE.exists():
        return _events_to_rows([])
    return pd.read_parquet(_EVENTS_FILE)


# --- forward ingest via yanoshin (for live operation) ---

def _fetch_day(ymd: str, limit: int = DEFAULT_LIMIT) -> dict:
    url = f"{BASE_URL}/{ymd}-{ymd}.json?limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "stock-screener/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _done_days() -> set[str]:
    if not _DONE_FILE.exists():
        return set()
    return set(_DONE_FILE.read_text(encoding="utf-8").split())


def _mark_done(ymd: str) -> None:
    _DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _DONE_FILE.open("a", encoding="utf-8") as f:
        f.write(ymd + "\n")


def ingest_forward(date_from: date, date_to: date, sleep: float = MIN_SLEEP) -> int:
    """Resumable daily yanoshin pull; appends classified events. Returns new count."""
    done = _done_days()
    existing = load_events()
    records = existing.to_dict("records")
    added = 0
    d = date_from
    while d <= date_to:
        ymd = d.strftime("%Y%m%d")
        if ymd in done:
            d += timedelta(days=1)
            continue
        try:
            payload = _fetch_day(ymd)
        except Exception as e:  # noqa: BLE001 - transient API errors: retry next run
            logger.warning("%s fetch failed: %s: %s", ymd, type(e).__name__, e)
            time.sleep(sleep * 2)
            d += timedelta(days=1)
            continue
        for item in payload.get("items", []):
            t = item.get("Tdnet", item)
            sub = classify_financing(t.get("title"))
            if sub and t.get("company_code"):
                records.append({
                    "code": str(t["company_code"]),
                    "ticker": code_to_ticker(t["company_code"]),
                    "disclosed_at": t.get("pubdate"), "tier": sub[0],
                    "subtype": sub, "title": t.get("title") or "",
                    "company": t.get("company_name") or "",
                })
                added += 1
        _save(_events_to_rows(records))
        _mark_done(ymd)
        time.sleep(sleep)
        d += timedelta(days=1)
    logger.info("forward ingest: +%d events", added)
    return added
