"""Forward-ingest toxic-financing events from yanoshin (nightly Action, docs/12).

Appends the last N days of classified financing disclosures to
``data_cache/financing_events.parquet`` (layer B).  Idempotent: re-saving
dedups, so overlapping windows across nightly runs are safe.  Free API, no key,
3s sleep between requests (finance rule 4).

Usage:
    uv run python scripts/ingest_events.py --days-back 14
    uv run python scripts/ingest_events.py --days-back 90   # one-time bootstrap
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.ingest import financing_events as fe  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="yanoshin forward ingest of toxic-financing events")
    p.add_argument("--days-back", type=int, default=14, help="lookback window in calendar days")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    end = date.today()
    start = end - timedelta(days=args.days_back)
    added = fe.ingest_forward(start, end)
    total = len(fe.load_events())
    print(f"ingested {added} new events ({start}..{end}); total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
