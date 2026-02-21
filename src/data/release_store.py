"""GitHub Releases data store.

Downloads data cache files from GitHub Releases so the Streamlit app
(and morning_check) can run without Git LFS.
"""

import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from src.data.config import load_config

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"

_DATA_FILES = [
    "prices.parquet",
    "features.parquet",
    "backtest_results.parquet",
    "ticker_master.parquet",
]


class _StripAuthRedirectHandler(HTTPRedirectHandler):
    """Strip Authorization header on redirect.

    GitHub API returns 302 to a signed S3/Azure URL for asset downloads.
    Sending the Authorization header to the storage backend causes errors.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            new_req.remove_header("Authorization")
        return new_req


def _get_repo() -> str:
    """Resolve GitHub repository (owner/name)."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    cfg = load_config()
    return cfg.get("release_store", {}).get("repo", "")


def _get_tag() -> str:
    cfg = load_config()
    return cfg.get("release_store", {}).get("tag", "data-latest")


def _get_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN")


def _fetch_asset_map(repo: str, tag: str, token: str | None) -> dict[str, str]:
    """Return {filename: api_url} for assets in the given release."""
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urlopen(req) as resp:
        data = json.loads(resp.read())

    return {asset["name"]: asset["url"] for asset in data.get("assets", [])}


def _download_asset(api_url: str, target: Path, token: str | None) -> None:
    """Download a single release asset via the GitHub API."""
    opener = build_opener(_StripAuthRedirectHandler)
    req = Request(api_url)
    req.add_header("Accept", "application/octet-stream")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with opener.open(req) as resp:
        target.write_bytes(resp.read())


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def download_release_data(target_dir: Path | None = None) -> bool:
    """Download all data parquet files from the ``data-latest`` release.

    Returns ``True`` when every required file was downloaded.
    """
    if target_dir is None:
        target_dir = _CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    repo = _get_repo()
    if not repo:
        logger.warning(
            "GitHub repository not configured. "
            "Set GITHUB_REPOSITORY or release_store.repo in settings.yaml"
        )
        return False

    tag = _get_tag()
    token = _get_token()

    logger.info("Downloading data from release %s (tag=%s) ...", repo, tag)

    try:
        assets = _fetch_asset_map(repo, tag, token)
    except HTTPError as e:
        if e.code == 404:
            logger.warning("Release '%s' not found. Run the nightly pipeline first.", tag)
        else:
            logger.warning("Failed to fetch release info: %s", e)
        return False
    except URLError as e:
        logger.warning("Network error: %s", e)
        return False

    all_ok = True
    for filename in _DATA_FILES:
        if filename not in assets:
            logger.warning("Asset '%s' not found in release", filename)
            all_ok = False
            continue

        target = target_dir / filename
        try:
            _download_asset(assets[filename], target, token)
            size_mb = target.stat().st_size / 1024 / 1024
            logger.info("  %s (%.1f MB)", filename, size_mb)
        except Exception:
            logger.warning("Failed to download %s", filename, exc_info=True)
            all_ok = False

    return all_ok


def ensure_data_available() -> bool:
    """Make sure all data parquet files exist locally.

    If any file is missing, attempt to download from the GitHub Release.
    Returns ``True`` when all required files are present after this call.
    """
    missing = [f for f in _DATA_FILES if not (_CACHE_DIR / f).exists()]

    if not missing:
        return True

    logger.info("Missing data files: %s — downloading from GitHub Release ...", missing)
    download_release_data()

    # Re-check
    still_missing = [f for f in _DATA_FILES if not (_CACHE_DIR / f).exists()]
    if still_missing:
        logger.error("Still missing after download: %s", still_missing)
        return False
    return True
