"""Tests for pipeline trigger module."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

from src.app.pipeline_trigger import (
    fetch_workflow_runs,
    format_run_status,
    trigger_workflow,
)


class TestTriggerWorkflow:
    @patch("src.app.pipeline_trigger._get_repo", return_value="owner/repo")
    @patch("src.app.pipeline_trigger.urlopen")
    def test_success(self, mock_urlopen, _mock_repo):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = trigger_workflow("nightly.yml", "fake-token")
        assert result["success"] is True
        assert "トリガー" in result["message"]

    @patch("src.app.pipeline_trigger._get_repo", return_value="owner/repo")
    @patch("src.app.pipeline_trigger.urlopen")
    def test_sends_correct_request(self, mock_urlopen, _mock_repo):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        trigger_workflow("nightly.yml", "fake-token", ref="develop")

        req = mock_urlopen.call_args[0][0]
        assert "owner/repo" in req.full_url
        assert "nightly.yml" in req.full_url
        assert req.get_header("Authorization") == "Bearer fake-token"
        body = json.loads(req.data)
        assert body["ref"] == "develop"

    @patch("src.app.pipeline_trigger._get_repo", return_value="")
    def test_no_repo(self, _mock_repo):
        result = trigger_workflow("nightly.yml", "fake-token")
        assert result["success"] is False
        assert "リポジトリ" in result["message"]

    @patch("src.app.pipeline_trigger._get_repo", return_value="owner/repo")
    @patch("src.app.pipeline_trigger.urlopen")
    def test_404_error(self, mock_urlopen, _mock_repo):
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="", code=404, msg="Not Found", hdrs={}, fp=BytesIO()
        )
        result = trigger_workflow("nightly.yml", "fake-token")
        assert result["success"] is False
        assert "見つかりません" in result["message"]

    @patch("src.app.pipeline_trigger._get_repo", return_value="owner/repo")
    @patch("src.app.pipeline_trigger.urlopen")
    def test_422_error(self, mock_urlopen, _mock_repo):
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="", code=422, msg="Unprocessable", hdrs={}, fp=BytesIO()
        )
        result = trigger_workflow("nightly.yml", "fake-token")
        assert result["success"] is False
        assert "無効" in result["message"]

    @patch("src.app.pipeline_trigger._get_repo", return_value="owner/repo")
    @patch("src.app.pipeline_trigger.urlopen")
    def test_other_http_error(self, mock_urlopen, _mock_repo):
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="", code=500, msg="Server Error", hdrs={}, fp=BytesIO()
        )
        result = trigger_workflow("nightly.yml", "fake-token")
        assert result["success"] is False
        assert "500" in result["message"]


class TestFetchWorkflowRuns:
    @patch("src.app.pipeline_trigger._get_repo", return_value="owner/repo")
    @patch("src.app.pipeline_trigger.urlopen")
    def test_success(self, mock_urlopen, _mock_repo):
        runs_data = {
            "workflow_runs": [
                {
                    "id": 123,
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": "2026-02-22T12:00:00Z",
                    "html_url": "https://github.com/owner/repo/actions/runs/123",
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(runs_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        runs = fetch_workflow_runs("nightly.yml", "fake-token")
        assert len(runs) == 1
        assert runs[0]["id"] == 123
        assert runs[0]["status"] == "completed"
        assert runs[0]["conclusion"] == "success"

    @patch("src.app.pipeline_trigger._get_repo", return_value="")
    def test_no_repo(self, _mock_repo):
        runs = fetch_workflow_runs("nightly.yml", "fake-token")
        assert runs == []

    @patch("src.app.pipeline_trigger._get_repo", return_value="owner/repo")
    @patch("src.app.pipeline_trigger.urlopen")
    def test_network_error_returns_empty(self, mock_urlopen, _mock_repo):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")
        runs = fetch_workflow_runs("nightly.yml", "fake-token")
        assert runs == []

    @patch("src.app.pipeline_trigger._get_repo", return_value="owner/repo")
    @patch("src.app.pipeline_trigger.urlopen")
    def test_empty_runs(self, mock_urlopen, _mock_repo):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"workflow_runs": []}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        runs = fetch_workflow_runs("nightly.yml", "fake-token")
        assert runs == []


class TestFormatRunStatus:
    def test_completed_success(self):
        icon, label = format_run_status({"status": "completed", "conclusion": "success"})
        assert icon == "✅"
        assert label == "成功"

    def test_completed_failure(self):
        icon, label = format_run_status({"status": "completed", "conclusion": "failure"})
        assert icon == "❌"
        assert label == "失敗"

    def test_completed_cancelled(self):
        icon, label = format_run_status({"status": "completed", "conclusion": "cancelled"})
        assert icon == "⚪"
        assert label == "キャンセル"

    def test_in_progress(self):
        icon, label = format_run_status({"status": "in_progress", "conclusion": None})
        assert icon == "🔄"
        assert label == "実行中"

    def test_queued(self):
        icon, label = format_run_status({"status": "queued", "conclusion": None})
        assert icon == "⏳"
        assert label == "待機中"

    def test_unknown_status(self):
        icon, label = format_run_status({"status": "waiting", "conclusion": None})
        assert icon == "❓"
        assert label == "waiting"

    def test_empty_dict(self):
        icon, label = format_run_status({})
        assert icon == "❓"
