"""Tests for Jira tools with mocked HTTP."""

from unittest.mock import patch

import httpx
import pytest

from pile.tools.jira_tools import jira_get_board, jira_get_issue, jira_search


def _mock_response(json_data, status_code=200):
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://test.atlassian.net"),
    )


@pytest.fixture(autouse=True)
def _reset_jira_client():
    """Reset the shared Jira client between tests."""
    import pile.tools.jira_tools as mod
    old = mod._client
    mod._client = None
    yield
    if mod._client and not mod._client.is_closed:
        mod._client.close()
    mod._client = old


class TestJiraSearch:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_issues_found(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"issues": []})
        result = jira_search("project=TEST")
        assert result == "No issues found."

    @patch("pile.tools.jira_tools._jira_client")
    def test_issues_returned(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "TEST-1",
                    "fields": {
                        "summary": "Fix bug",
                        "status": {"name": "Open"},
                        "assignee": {"displayName": "Alice"},
                        "priority": {"name": "High"},
                        "issuetype": {"name": "Bug"},
                    },
                }
            ]
        })
        result = jira_search("project=TEST")
        assert "TEST-1" in result
        assert "Fix bug" in result
        assert "Alice" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_unassigned_issue(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "TEST-2",
                    "fields": {
                        "summary": "No owner",
                        "status": {"name": "Open"},
                        "assignee": None,
                        "priority": None,
                        "issuetype": {"name": "Task"},
                    },
                }
            ]
        })
        result = jira_search("project=TEST")
        assert "Unassigned" in result


class TestJiraGetIssue:
    @patch("pile.tools.jira_tools._jira_client")
    def test_full_issue_details(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "key": "TEST-1",
            "fields": {
                "summary": "Test issue",
                "issuetype": {"name": "Story"},
                "status": {"name": "In Progress"},
                "priority": {"name": "Medium"},
                "assignee": {"displayName": "Bob"},
                "reporter": {"displayName": "Alice"},
                "created": "2026-04-01T10:00:00.000+0000",
                "updated": "2026-04-02T10:00:00.000+0000",
                "description": None,
                "issuelinks": [],
            },
        })
        result = jira_get_issue("TEST-1")
        assert "TEST-1" in result
        assert "In Progress" in result
        assert "Bob" in result


class TestJiraGetBoard:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_boards(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"values": []})
        result = jira_get_board("TEST")
        assert "No boards found" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_boards_returned(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "values": [{"name": "Test Board", "id": 1, "type": "scrum"}]
        })
        result = jira_get_board("TEST")
        assert "Test Board" in result


class TestJiraErrorHandling:
    @patch("pile.tools.jira_tools._jira_client")
    def test_connection_error(self, mock_client):
        mock_client.return_value.get.side_effect = httpx.ConnectError("fail")
        result = jira_search("project=TEST")
        assert "Error" in result
        assert "Cannot connect" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_timeout_error(self, mock_client):
        mock_client.return_value.get.side_effect = httpx.TimeoutException("timeout")
        result = jira_search("project=TEST")
        assert "timed out" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_auth_error(self, mock_client):
        resp = _mock_response({}, status_code=401)
        mock_client.return_value.get.side_effect = httpx.HTTPStatusError(
            "401", request=resp.request, response=resp,
        )
        result = jira_search("project=TEST")
        assert "authentication failed" in result.lower()
