"""Tests for pile.prefetch — scrum data prefetch and query intent detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pile.prefetch import (
    _safe_call,
    detect_query_intent,
    detect_scrum_type,
    prefetch_query_data,
    prefetch_scrum_data,
)


# ---------------------------------------------------------------------------
# detect_scrum_type
# ---------------------------------------------------------------------------


class TestDetectScrumType:
    @pytest.mark.parametrize("query,expected", [
        ("standup report", "standup"),
        ("daily meeting notes", "standup"),
        ("stand-up summary", "standup"),
    ])
    def test_standup(self, query, expected):
        assert detect_scrum_type(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("sprint review for this week", "sprint_review"),
        ("sprint progress update", "sprint_review"),
        ("sprint summary", "sprint_review"),
    ])
    def test_sprint_review(self, query, expected):
        assert detect_scrum_type(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("velocity chart", "velocity"),
        ("team velocity", "velocity"),
    ])
    def test_velocity(self, query, expected):
        assert detect_scrum_type(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("workload distribution", "workload"),
        ("who is working on what", "workload"),
        ("overload check", "workload"),
    ])
    def test_workload(self, query, expected):
        assert detect_scrum_type(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("blockers list", "blockers"),
        ("blocked items", "blockers"),
        ("any problems?", "blockers"),
    ])
    def test_blockers(self, query, expected):
        assert detect_scrum_type(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("retro notes", "retro"),
        ("retrospective prep", "retro"),
        ("meeting prep", "retro"),
    ])
    def test_retro(self, query, expected):
        assert detect_scrum_type(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("cycle time analysis", "cycle_time"),
        ("lead time for issues", "cycle_time"),
    ])
    def test_cycle_time(self, query, expected):
        assert detect_scrum_type(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("data quality check", "data_quality"),
    ])
    def test_data_quality(self, query, expected):
        assert detect_scrum_type(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("stakeholder update", "stakeholder"),
        ("summary for management", "stakeholder"),
    ])
    def test_stakeholder(self, query, expected):
        assert detect_scrum_type(query) == expected

    def test_unknown_returns_general(self):
        assert detect_scrum_type("something random") == "general"

    def test_empty_query(self):
        assert detect_scrum_type("") == "general"

    def test_whitespace_only(self):
        assert detect_scrum_type("   ") == "general"


# ---------------------------------------------------------------------------
# detect_query_intent
# ---------------------------------------------------------------------------


class TestDetectQueryIntent:
    @pytest.mark.parametrize("query,expected", [
        ("show in progress issues", "in_progress"),
        ("to do list", "to_do"),
        ("xong roi", "done_recent"),
        ("testing items", "testing"),
        ("code review queue", "code_review"),
    ])
    def test_status_intents(self, query, expected):
        assert detect_query_intent(query) == expected

    def test_issue_key_detection(self):
        assert detect_query_intent("What is TETRA-123?") == "issue:TETRA-123"
        assert detect_query_intent("Tell me about PROJ-42") == "issue:PROJ-42"

    def test_no_match_returns_none(self):
        assert detect_query_intent("something random") is None

    def test_empty_query(self):
        assert detect_query_intent("") is None


# ---------------------------------------------------------------------------
# _safe_call
# ---------------------------------------------------------------------------


class TestSafeCall:
    def test_success(self):
        fn = MagicMock(return_value="result data")
        assert _safe_call(fn, key="val") == "result data"
        fn.assert_called_once_with(key="val")

    def test_exception_returns_empty(self):
        fn = MagicMock(side_effect=Exception("boom"), __name__="mock_fn")
        assert _safe_call(fn) == ""


# ---------------------------------------------------------------------------
# prefetch_scrum_data
# ---------------------------------------------------------------------------


class TestPrefetchScrumData:
    """Tests for prefetch_scrum_data with all Jira calls mocked."""

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_standup_fetches_sprint_issues_and_recent(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="sprint data") as mock_sc:
            result = prefetch_scrum_data("standup report", board_id=1)
            assert "sprint data" in result
            assert mock_sc.call_count >= 2

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_velocity_fetches_board_and_closed_sprints(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="velocity data") as mock_sc:
            result = prefetch_scrum_data("velocity chart", board_id=1)
            assert "velocity data" in result
            assert mock_sc.call_count >= 3

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_workload_fetches_sprint_issues(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="workload data") as mock_sc:
            result = prefetch_scrum_data("workload analysis", board_id=1)
            assert "workload data" in result

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_blockers_fetches_blocked_issues(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="blocker data") as mock_sc:
            result = prefetch_scrum_data("blockers list", board_id=1)
            assert "blocker data" in result
            assert mock_sc.call_count >= 2

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_retro_fetches_board_sprint_and_done(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="retro data") as mock_sc:
            result = prefetch_scrum_data("retro notes", board_id=1)
            assert "retro data" in result
            assert mock_sc.call_count >= 3

    @patch("pile.prefetch._get_done_issue_keys", return_value=["TETRA-1", "TETRA-2"])
    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_cycle_time_fetches_changelogs(self, mock_pk, mock_sid, mock_done):
        with patch("pile.prefetch._safe_call", return_value="cycle data") as mock_sc:
            result = prefetch_scrum_data("cycle time analysis", board_id=1)
            assert "cycle data" in result
            # 1 for sprint issues + 2 for changelogs
            assert mock_sc.call_count >= 3

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_data_quality_fetches_sprint_issues(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="quality data") as mock_sc:
            result = prefetch_scrum_data("data quality check", board_id=1)
            assert "quality data" in result

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_sprint_review_fetches_board_and_issues(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="review data") as mock_sc:
            result = prefetch_scrum_data("sprint review meeting", board_id=1)
            assert "review data" in result
            assert mock_sc.call_count >= 2

    @patch("pile.prefetch._get_active_sprint_id", return_value=None)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_no_active_sprint_still_works(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="") as mock_sc:
            result = prefetch_scrum_data("standup report", board_id=1)
            # Should still return something (fallback message)
            assert result

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_all_errors_returns_fallback_message(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value=""):
            result = prefetch_scrum_data("standup report", board_id=1)
            assert "No data could be fetched" in result

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_truncation_at_3000_chars(self, mock_pk, mock_sid):
        long_data = "x" * 4000
        with patch("pile.prefetch._safe_call", return_value=long_data):
            result = prefetch_scrum_data("sprint review", board_id=1)
            assert len(result) < 4000
            assert "[... truncated]" in result

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_error_parts_filtered_out(self, mock_pk, mock_sid):
        call_count = 0

        def alternating_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return "Error: something broke"
            return "good data"

        with patch("pile.prefetch._safe_call", side_effect=alternating_call):
            result = prefetch_scrum_data("sprint review", board_id=1)
            assert "Error" not in result

    @patch("pile.prefetch._get_active_sprint_id", return_value=42)
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_general_type_fetches_board_and_issues(self, mock_pk, mock_sid):
        with patch("pile.prefetch._safe_call", return_value="general data") as mock_sc:
            result = prefetch_scrum_data("how is the team doing?", board_id=1)
            assert "general data" in result
            assert mock_sc.call_count >= 2


# ---------------------------------------------------------------------------
# prefetch_query_data
# ---------------------------------------------------------------------------


class TestPrefetchQueryData:
    @patch("pile.prefetch._safe_call")
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_in_progress_intent(self, mock_pk, mock_sc):
        mock_sc.return_value = "issue list"
        result = prefetch_query_data("show in progress items")
        assert result == "issue list"

    @patch("pile.prefetch._safe_call")
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_issue_key_intent(self, mock_pk, mock_sc):
        mock_sc.return_value = "issue details"
        result = prefetch_query_data("Tell me about TETRA-123")
        assert result == "issue details"

    def test_no_intent_returns_none(self):
        result = prefetch_query_data("something random with no patterns")
        assert result is None

    @patch("pile.prefetch._safe_call")
    @patch("pile.prefetch._project_key", return_value="TETRA")
    def test_code_review_intent(self, mock_pk, mock_sc):
        mock_sc.return_value = "review issues"
        result = prefetch_query_data("code review queue")
        assert result == "review issues"


# ---------------------------------------------------------------------------
# _get_active_sprint_id (via Jira API mock)
# ---------------------------------------------------------------------------


class TestGetActiveSprintId:
    @patch("pile.tools.jira_tools._jira_client")
    def test_returns_sprint_id(self, mock_client_fn):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"values": [{"id": 99}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_fn.return_value.get.return_value = mock_resp

        from pile.prefetch import _get_active_sprint_id
        assert _get_active_sprint_id(board_id=1) == 99

    @patch("pile.tools.jira_tools._jira_client", side_effect=Exception("connection error"))
    def test_returns_none_on_error(self, mock_client_fn):
        from pile.prefetch import _get_active_sprint_id
        assert _get_active_sprint_id(board_id=1) is None

    @patch("pile.tools.jira_tools._jira_client")
    def test_returns_none_when_no_sprints(self, mock_client_fn):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"values": []}
        mock_resp.raise_for_status = MagicMock()
        mock_client_fn.return_value.get.return_value = mock_resp

        from pile.prefetch import _get_active_sprint_id
        assert _get_active_sprint_id(board_id=1) is None


# ---------------------------------------------------------------------------
# _get_done_issue_keys
# ---------------------------------------------------------------------------


class TestGetDoneIssueKeys:
    def test_no_sprint_returns_empty(self):
        from pile.prefetch import _get_done_issue_keys
        assert _get_done_issue_keys(None) == []

    @patch("pile.tools.jira_tools._jira_client")
    def test_returns_keys(self, mock_client_fn):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "issues": [{"key": "TETRA-1"}, {"key": "TETRA-2"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client_fn.return_value.get.return_value = mock_resp

        from pile.prefetch import _get_done_issue_keys
        assert _get_done_issue_keys(42) == ["TETRA-1", "TETRA-2"]

    @patch("pile.tools.jira_tools._jira_client", side_effect=Exception("fail"))
    def test_returns_empty_on_error(self, mock_client_fn):
        from pile.prefetch import _get_done_issue_keys
        assert _get_done_issue_keys(42) == []


# ---------------------------------------------------------------------------
# _project_key
# ---------------------------------------------------------------------------


class TestProjectKey:
    @patch("pile.config.settings")
    def test_returns_configured_key(self, mock_settings):
        mock_settings.jira_project_key = "MYPROJ"
        from pile.prefetch import _project_key
        assert _project_key() == "MYPROJ"
