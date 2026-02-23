"""GitHub Actions workflow trigger via REST API."""

import json
import logging
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.data.config import load_config

logger = logging.getLogger(__name__)


def _get_repo() -> str:
    """Get repository name from config."""
    cfg = load_config()
    return cfg.get("release_store", {}).get("repo", "")


def trigger_workflow(workflow_file: str, token: str, ref: str = "master") -> dict[str, object]:
    """Trigger a GitHub Actions workflow via workflow_dispatch.

    Returns dict with ``success`` (bool) and ``message`` (str).
    """
    repo = _get_repo()
    if not repo:
        return {"success": False, "message": "リポジトリが設定されていません。"}

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    data = json.dumps({"ref": ref}).encode()

    req = Request(url, data=data, method="POST")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req) as resp:
            resp.read()
        return {"success": True, "message": "ワークフローをトリガーしました。"}
    except HTTPError as e:
        if e.code == 404:
            msg = f"ワークフロー '{workflow_file}' が見つかりません。"
        elif e.code == 422:
            msg = "ワークフローが無効です（ref ブランチが存在しない等）。"
        else:
            msg = f"API エラー: {e.code} {e.reason}"
        return {"success": False, "message": msg}


def fetch_workflow_runs(workflow_file: str, token: str, per_page: int = 5) -> list[dict]:
    """Fetch recent workflow runs for a given workflow file."""
    repo = _get_repo()
    if not repo:
        return []

    url = (
        f"https://api.github.com/repos/{repo}"
        f"/actions/workflows/{workflow_file}/runs?per_page={per_page}"
    )

    req = Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")

    try:
        with urlopen(req) as resp:
            data = json.loads(resp.read())

        runs = []
        for run in data.get("workflow_runs", []):
            runs.append(
                {
                    "id": run["id"],
                    "status": run["status"],
                    "conclusion": run.get("conclusion"),
                    "created_at": run["created_at"],
                    "html_url": run["html_url"],
                }
            )
        return runs
    except Exception:
        logger.warning("Failed to fetch workflow runs", exc_info=True)
        return []


def format_run_status(run: dict) -> tuple[str, str]:
    """Return (icon, label) for a workflow run status."""
    status = run.get("status", "")
    conclusion = run.get("conclusion")

    if status == "completed":
        icons = {"success": "✅", "failure": "❌", "cancelled": "⚪"}
        labels = {"success": "成功", "failure": "失敗", "cancelled": "キャンセル"}
        icon = icons.get(conclusion, "❓")
        label = labels.get(conclusion, conclusion or "不明")
    elif status == "in_progress":
        icon = "🔄"
        label = "実行中"
    elif status == "queued":
        icon = "⏳"
        label = "待機中"
    else:
        icon = "❓"
        label = status

    return icon, label
