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


def fetch_prime_tickers() -> pd.DataFrame:
    """Download JPX listed companies XLS and return Prime market tickers.

    Returns:
        DataFrame with columns: ticker, company_name, market
    """
    logger.info("Downloading JPX listed companies from %s", _JPX_URL)
    data = urllib.request.urlopen(_JPX_URL).read()
    wb = xlrd.open_workbook(file_contents=data, encoding_override="cp932")
    ws = wb.sheet_by_index(0)

    rows = []
    for r in range(1, ws.nrows):
        market = ws.cell_value(r, 3)
        if "プライム" not in market:
            continue
        code = ws.cell_value(r, 1)
        # Code may be float (e.g. 1301.0) or string
        if isinstance(code, float):
            code = str(int(code))
        ticker = f"{code}.T"
        company_name = ws.cell_value(r, 2)
        rows.append({"ticker": ticker, "company_name": company_name, "market": "Prime"})

    df = pd.DataFrame(rows)
    logger.info("Found %d Prime tickers", len(df))
    return df


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

    df = fetch_prime_tickers()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE_FILE, index=False)
    logger.info("Saved ticker master to %s", _CACHE_FILE)
    return df
