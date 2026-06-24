"""TSE Prime stock master list management.

Fetches the listed companies file from JPX and extracts Prime market tickers.
"""

import logging
import urllib.request
from pathlib import Path

import pandas as pd
import xlrd

logger = logging.getLogger(__name__)

_JPX_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"
_CACHE_FILE = _CACHE_DIR / "ticker_master.parquet"


# JPX市場区分(部分一致)→ 正規化ラベル
_MARKET_MAP = {"プライム": "Prime", "スタンダード": "Standard", "グロース": "Growth"}
DEFAULT_MARKETS = ("Prime", "Standard", "Growth")


def fetch_listed_tickers(markets: tuple[str, ...] = DEFAULT_MARKETS) -> pd.DataFrame:
    """Download JPX listed companies XLS and return equity tickers.

    Args:
        markets: normalized market labels to include (Prime/Standard/Growth).
            ETF/REIT/foreign/PRO are always excluded (内国株式の3市場のみ)。

    Returns:
        DataFrame with columns: ticker, company_name, market
    """
    logger.info("Downloading JPX listed companies from %s", _JPX_URL)
    data = urllib.request.urlopen(_JPX_URL).read()
    wb = xlrd.open_workbook(file_contents=data, encoding_override="cp932")
    ws = wb.sheet_by_index(0)

    rows = []
    for r in range(1, ws.nrows):
        raw_market = ws.cell_value(r, 3)
        norm = next((v for k, v in _MARKET_MAP.items() if k in raw_market), None)
        if norm is None or norm not in markets:
            continue
        code = ws.cell_value(r, 1)
        if isinstance(code, float):  # code may be float (e.g. 1301.0) or string
            code = str(int(code))
        rows.append({"ticker": f"{code}.T", "company_name": ws.cell_value(r, 2),
                     "market": norm})

    df = pd.DataFrame(rows)
    by_mkt = df["market"].value_counts().to_dict()
    logger.info("Found %d tickers %s", len(df), by_mkt)
    return df


# backward-compatible alias (Prime only)
def fetch_prime_tickers() -> pd.DataFrame:
    return fetch_listed_tickers(markets=("Prime",))


def load_ticker_master(force_refresh: bool = False) -> pd.DataFrame:
    """Load ticker master from cache, or fetch from JPX if not cached.

    Args:
        force_refresh: If True, always re-download from JPX.

    Returns:
        DataFrame with columns: ticker, company_name, market
    """
    if not force_refresh and _CACHE_FILE.exists():
        logger.info("Loading ticker master from cache: %s", _CACHE_FILE)
        return pd.read_parquet(_CACHE_FILE)

    df = fetch_listed_tickers()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE_FILE, index=False)
    logger.info("Saved ticker master to %s", _CACHE_FILE)
    return df
