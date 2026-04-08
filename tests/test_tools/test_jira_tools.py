"""Tests for Jira tools with mocked HTTP."""

from unittest.mock import patch, MagicMock

import httpx
import pytest

from pile.tools.jira_tools import (
    jira_add_comment,
    jira_create_issue,
    jira_create_sprint,
    jira_curl_command,
    jira_get_backlog,
    jira_get_board,
    jira_get_board_config,
    jira_get_changelog,
    jira_get_epic_issues,
    jira_get_epics,
    jira_get_issue,
    jira_get_sprint,
    jira_get_sprint_issues,
    jira_link_issues,
    jira_list_boards,
    jira_move_to_backlog,
    jira_move_to_sprint,
    jira_search,
    jira_transition_issue,
    jira_update_issue,
)


def _mock_response(json_data, status_code=200):
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://test.atlassian.net"),
    )


def _mock_error_response(status_code):
    """Create a mock response that raises HTTPStatusError."""
    resp = _mock_response({}, status_code=status_code)
    return httpx.HTTPStatusError(
        str(status_code), request=resp.request, response=resp,
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


# ---------------------------------------------------------------------------
# jira_curl_command
# ---------------------------------------------------------------------------

class TestJiraCurlCommand:
    def test_known_action(self):
        result = jira_curl_command("list_boards")
        assert "rest/agile/1.0/board" in result

    def test_unknown_action(self):
        result = jira_curl_command("nonexistent")
        assert "Unknown action" in result
        assert "list_boards" in result

    def test_all_actions_generate(self):
        from pile.tools.jira_tools import _CURL_COMMANDS
        for action in _CURL_COMMANDS:
            result = jira_curl_command(action)
            assert "curl" in result


# ---------------------------------------------------------------------------
# jira_search
# ---------------------------------------------------------------------------

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

    @patch("pile.tools.jira_tools._jira_client")
    def test_subtask_filter_appended(self, mock_client):
        """When include_subtasks=False and no issuetype in JQL, filter is appended."""
        mock_client.return_value.get.return_value = _mock_response({"issues": []})
        jira_search("project=TEST", include_subtasks=False)
        call_kwargs = mock_client.return_value.get.call_args
        jql_sent = call_kwargs.kwargs.get("params", {}).get("jql", "") if call_kwargs.kwargs.get("params") else call_kwargs[1]["params"]["jql"]
        assert "subtaskIssueTypes" in jql_sent

    @patch("pile.tools.jira_tools._jira_client")
    def test_subtask_filter_not_appended_when_issuetype_present(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"issues": []})
        jira_search("project=TEST AND issuetype=Bug", include_subtasks=False)
        call_kwargs = mock_client.return_value.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert "subtaskIssueTypes" not in params.get("jql", "")

    @patch("pile.tools.jira_tools._jira_client")
    def test_include_subtasks(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"issues": []})
        jira_search("project=TEST", include_subtasks=True)
        call_kwargs = mock_client.return_value.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert "subtaskIssueTypes" not in params.get("jql", "")


# ---------------------------------------------------------------------------
# jira_get_issue
# ---------------------------------------------------------------------------

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

    @patch("pile.tools.jira_tools._jira_client")
    def test_issue_with_links(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "key": "TEST-5",
            "fields": {
                "summary": "Linked issue",
                "issuetype": {"name": "Task"},
                "status": {"name": "Open"},
                "priority": None,
                "assignee": None,
                "reporter": None,
                "created": "2026-04-01T10:00:00.000+0000",
                "updated": "2026-04-02T10:00:00.000+0000",
                "description": None,
                "issuelinks": [
                    {
                        "type": {"outward": "blocks"},
                        "outwardIssue": {"key": "TEST-6"},
                    },
                    {
                        "type": {"inward": "is blocked by"},
                        "inwardIssue": {"key": "TEST-7"},
                    },
                ],
            },
        })
        result = jira_get_issue("TEST-5")
        assert "blocks" in result
        assert "TEST-6" in result
        assert "is blocked by" in result
        assert "TEST-7" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_issue_with_story_points(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "key": "TEST-8",
            "fields": {
                "summary": "SP issue",
                "issuetype": {"name": "Story"},
                "status": {"name": "Open"},
                "priority": {"name": "Low"},
                "assignee": {"displayName": "Carol"},
                "reporter": {"displayName": "Dave"},
                "created": "2026-04-01T10:00:00.000+0000",
                "updated": "2026-04-02T10:00:00.000+0000",
                "description": None,
                "issuelinks": [],
                "story_points": 5,
                "customfield_10016": 5,
            },
        })
        result = jira_get_issue("TEST-8")
        assert "5" in result


# ---------------------------------------------------------------------------
# jira_get_sprint
# ---------------------------------------------------------------------------

class TestJiraGetSprint:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_sprints(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"values": []})
        result = jira_get_sprint(1, state="active")
        assert "No active sprints" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_sprints_returned(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "values": [
                {
                    "name": "Sprint 1",
                    "id": 10,
                    "state": "active",
                    "startDate": "2026-04-01T00:00:00.000Z",
                    "endDate": "2026-04-14T00:00:00.000Z",
                    "goal": "Finish MVP",
                }
            ]
        })
        result = jira_get_sprint(1)
        assert "Sprint 1" in result
        assert "Finish MVP" in result
        assert "2026-04-01" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_sprint_without_dates(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "values": [
                {
                    "name": "Sprint 2",
                    "id": 11,
                    "state": "future",
                    "startDate": None,
                    "endDate": None,
                }
            ]
        })
        result = jira_get_sprint(1, state="future")
        assert "Sprint 2" in result
        assert "N/A" in result


# ---------------------------------------------------------------------------
# jira_get_sprint_issues
# ---------------------------------------------------------------------------

class TestJiraGetSprintIssues:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_issues(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"issues": []})
        result = jira_get_sprint_issues(10)
        assert "No issues" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_issues_grouped_by_status(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "T-1",
                    "fields": {
                        "summary": "Task A",
                        "status": {"name": "To Do"},
                        "issuetype": {"name": "Task", "subtask": False},
                        "assignee": {"displayName": "Alice"},
                        "customfield_10016": 3,
                    },
                },
                {
                    "key": "T-2",
                    "fields": {
                        "summary": "Task B",
                        "status": {"name": "Done"},
                        "issuetype": {"name": "Task", "subtask": False},
                        "assignee": None,
                        "customfield_10016": None,
                    },
                },
            ]
        })
        result = jira_get_sprint_issues(10)
        assert "T-1" in result
        assert "T-2" in result
        assert "To Do" in result
        assert "Done" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_subtasks_excluded_by_default(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "T-1",
                    "fields": {
                        "summary": "Parent",
                        "status": {"name": "Open"},
                        "issuetype": {"name": "Task", "subtask": False},
                        "assignee": None,
                    },
                },
                {
                    "key": "T-1-SUB",
                    "fields": {
                        "summary": "Subtask",
                        "status": {"name": "Open"},
                        "issuetype": {"name": "Sub-task", "subtask": True},
                        "assignee": None,
                    },
                },
            ]
        })
        result = jira_get_sprint_issues(10, include_subtasks=False)
        assert "T-1" in result
        assert "T-1-SUB" not in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_subtasks_included(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "T-1-SUB",
                    "fields": {
                        "summary": "Subtask",
                        "status": {"name": "Open"},
                        "issuetype": {"name": "Sub-task", "subtask": True},
                        "assignee": None,
                    },
                },
            ]
        })
        result = jira_get_sprint_issues(10, include_subtasks=True)
        assert "T-1-SUB" in result


# ---------------------------------------------------------------------------
# jira_list_boards
# ---------------------------------------------------------------------------

class TestJiraListBoards:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_boards(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"values": []})
        result = jira_list_boards()
        assert "No boards found" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_boards_listed(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "values": [
                {"name": "Board A", "id": 1, "type": "scrum", "location": {"projectKey": "PROJ"}},
                {"name": "Board B", "id": 2, "type": "kanban", "location": {"projectKey": "PROJ"}},
            ]
        })
        result = jira_list_boards(project_key="PROJ")
        assert "Board A" in result
        assert "Board B" in result
        assert "2 boards found" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_boards_without_project_filter(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "values": [{"name": "Global", "id": 3, "type": "scrum", "location": {}}]
        })
        result = jira_list_boards()
        assert "Global" in result


# ---------------------------------------------------------------------------
# jira_get_board
# ---------------------------------------------------------------------------

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

    @patch("pile.tools.jira_tools._jira_client")
    def test_board_with_active_sprint_and_issues(self, mock_client):
        """Full path: board found -> active sprint -> sprint issues."""
        client = mock_client.return_value
        client.get.side_effect = [
            _mock_response({"values": [{"name": "Board X", "id": 1, "type": "scrum"}]}),
            _mock_response({
                "values": [{
                    "name": "Sprint 5",
                    "id": 50,
                    "state": "active",
                    "startDate": "2026-04-01T00:00:00.000Z",
                    "endDate": "2026-04-14T00:00:00.000Z",
                    "goal": "Ship it",
                }]
            }),
            _mock_response({
                "issues": [
                    {
                        "key": "T-1",
                        "fields": {
                            "summary": "A",
                            "status": {"name": "To Do"},
                            "issuetype": {"name": "Task", "subtask": False},
                        },
                    },
                    {
                        "key": "T-2",
                        "fields": {
                            "summary": "B",
                            "status": {"name": "Done"},
                            "issuetype": {"name": "Task", "subtask": False},
                        },
                    },
                ]
            }),
        ]
        result = jira_get_board("TEST")
        assert "Board X" in result
        assert "Sprint 5" in result
        assert "Ship it" in result
        assert "Work items: 2" in result
        assert "To Do: 1" in result
        assert "Done: 1" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_board_no_active_sprint(self, mock_client):
        client = mock_client.return_value
        client.get.side_effect = [
            _mock_response({"values": [{"name": "Board Y", "id": 2, "type": "kanban"}]}),
            _mock_response({"values": []}),
        ]
        result = jira_get_board("TEST")
        assert "Board Y" in result
        assert "No active sprint" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_board_sprint_fetch_exception(self, mock_client):
        """Sprint fetch fails gracefully."""
        client = mock_client.return_value
        client.get.side_effect = [
            _mock_response({"values": [{"name": "Board Z", "id": 3, "type": "scrum"}]}),
            Exception("network error"),
        ]
        result = jira_get_board("TEST")
        assert "Board Z" in result


# ---------------------------------------------------------------------------
# jira_get_backlog
# ---------------------------------------------------------------------------

class TestJiraGetBacklog:
    @patch("pile.tools.jira_tools._jira_client")
    def test_empty_backlog(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"issues": []})
        result = jira_get_backlog(1)
        assert "Backlog is empty" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_backlog_with_items(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "T-10",
                    "fields": {
                        "summary": "Backlog item",
                        "status": {"name": "Open"},
                        "issuetype": {"name": "Task", "subtask": False},
                        "assignee": {"displayName": "Eve"},
                        "priority": {"name": "Medium"},
                        "customfield_10016": 2,
                    },
                }
            ]
        })
        result = jira_get_backlog(1)
        assert "T-10" in result
        assert "Backlog item" in result
        assert "Eve" in result
        assert "[2 SP]" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_backlog_excludes_subtasks(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "T-11",
                    "fields": {
                        "summary": "Sub",
                        "status": {"name": "Open"},
                        "issuetype": {"name": "Sub-task", "subtask": True},
                        "assignee": None,
                        "priority": None,
                    },
                }
            ]
        })
        result = jira_get_backlog(1, include_subtasks=False)
        assert "Backlog is empty" in result


# ---------------------------------------------------------------------------
# jira_get_epics
# ---------------------------------------------------------------------------

class TestJiraGetEpics:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_epics(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"values": []})
        result = jira_get_epics(1)
        assert "No epics found" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_epics_listed(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "values": [
                {"key": "E-1", "name": "Epic One", "done": False},
                {"key": "E-2", "name": "Epic Two", "done": True},
            ]
        })
        result = jira_get_epics(1)
        assert "E-1" in result
        assert "Epic One" in result
        assert "In Progress" in result
        assert "E-2" in result
        assert "Done" in result


# ---------------------------------------------------------------------------
# jira_get_epic_issues
# ---------------------------------------------------------------------------

class TestJiraGetEpicIssues:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_issues(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"issues": []})
        result = jira_get_epic_issues("E-1")
        assert "No issues in epic" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_issues_grouped(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "T-20",
                    "fields": {
                        "summary": "Epic task",
                        "status": {"name": "In Progress"},
                        "issuetype": {"name": "Task", "subtask": False},
                        "assignee": {"displayName": "Frank"},
                        "customfield_10016": 8,
                    },
                }
            ]
        })
        result = jira_get_epic_issues("E-1")
        assert "T-20" in result
        assert "Epic task" in result
        assert "Frank" in result
        assert "[8 SP]" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_subtasks_excluded(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "issues": [
                {
                    "key": "T-21",
                    "fields": {
                        "summary": "Subtask of epic",
                        "status": {"name": "Open"},
                        "issuetype": {"name": "Sub-task", "subtask": True},
                        "assignee": None,
                    },
                }
            ]
        })
        result = jira_get_epic_issues("E-1", include_subtasks=False)
        assert "No issues in epic" in result


# ---------------------------------------------------------------------------
# jira_get_board_config
# ---------------------------------------------------------------------------

class TestJiraGetBoardConfig:
    @patch("pile.tools.jira_tools._jira_client")
    def test_full_config(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "columnConfig": {
                "columns": [
                    {"name": "To Do", "statuses": [{"self": "https://x/status/1"}], "min": None, "max": 5},
                    {"name": "Done", "statuses": [{"id": "2"}]},
                ]
            },
            "estimation": {
                "field": {"displayName": "Story Points", "fieldId": "customfield_10016"},
            },
            "filter": {"name": "Sprint Filter", "id": "12345"},
            "subQuery": {"query": "ORDER BY rank"},
        })
        result = jira_get_board_config(1)
        assert "To Do" in result
        assert "Done" in result
        assert "Story Points" in result
        assert "Sprint Filter" in result
        assert "ORDER BY rank" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_minimal_config(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({})
        result = jira_get_board_config(1)
        assert "Board Configuration" in result


# ---------------------------------------------------------------------------
# jira_get_changelog
# ---------------------------------------------------------------------------

class TestJiraGetChangelog:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_entries(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({"values": []})
        result = jira_get_changelog("T-1")
        assert "No changelog entries" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_entries_returned(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "values": [
                {
                    "author": {"displayName": "Grace"},
                    "created": "2026-04-01T10:30:00.000Z",
                    "items": [
                        {"field": "status", "fromString": "Open", "toString": "In Progress"},
                    ],
                }
            ]
        })
        result = jira_get_changelog("T-1")
        assert "Grace" in result
        assert "status" in result
        assert "Open" in result
        assert "In Progress" in result
        assert "→" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_entry_with_none_values(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "values": [
                {
                    "author": None,
                    "created": "2026-04-01T10:30:00.000Z",
                    "items": [
                        {"field": "assignee", "fromString": None, "toString": "Hank"},
                    ],
                }
            ]
        })
        result = jira_get_changelog("T-1")
        assert "Hank" in result
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# jira_create_issue
# ---------------------------------------------------------------------------

class TestJiraCreateIssue:
    @patch("pile.tools.jira_tools._jira_client")
    def test_create_minimal(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({"key": "T-100"})
        result = jira_create_issue(summary="New task")
        assert "T-100" in result
        assert "created" in result.lower()

    @patch("pile.tools.jira_tools._jira_client")
    def test_create_with_all_fields(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({"key": "T-101"})
        result = jira_create_issue(
            summary="Full task",
            issue_type="Bug",
            description="Some desc",
            assignee_id="abc123",
            priority="High",
        )
        assert "T-101" in result
        payload = mock_client.return_value.post.call_args.kwargs.get("json") or mock_client.return_value.post.call_args[1]["json"]
        assert payload["fields"]["issuetype"]["name"] == "Bug"
        assert payload["fields"]["assignee"]["accountId"] == "abc123"
        assert payload["fields"]["priority"]["name"] == "High"
        assert "description" in payload["fields"]


# ---------------------------------------------------------------------------
# jira_transition_issue
# ---------------------------------------------------------------------------

class TestJiraTransitionIssue:
    @patch("pile.tools.jira_tools._jira_client")
    def test_successful_transition(self, mock_client):
        client = mock_client.return_value
        client.get.return_value = _mock_response({
            "transitions": [
                {"id": "31", "name": "In Progress"},
                {"id": "41", "name": "Done"},
            ]
        })
        client.post.return_value = _mock_response({}, status_code=204)
        result = jira_transition_issue("T-1", "Done")
        assert "transitioned" in result.lower()
        assert "Done" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_transition_not_found(self, mock_client):
        mock_client.return_value.get.return_value = _mock_response({
            "transitions": [
                {"id": "31", "name": "In Progress"},
            ]
        })
        result = jira_transition_issue("T-1", "Nonexistent")
        assert "not found" in result.lower()
        assert "In Progress" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_transition_case_insensitive(self, mock_client):
        client = mock_client.return_value
        client.get.return_value = _mock_response({
            "transitions": [{"id": "31", "name": "In Progress"}]
        })
        client.post.return_value = _mock_response({}, status_code=204)
        result = jira_transition_issue("T-1", "in progress")
        assert "transitioned" in result.lower()


# ---------------------------------------------------------------------------
# jira_add_comment
# ---------------------------------------------------------------------------

class TestJiraAddComment:
    @patch("pile.tools.jira_tools._jira_client")
    def test_add_comment(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({"id": "1001"})
        result = jira_add_comment("T-1", "Hello!")
        assert "Comment added" in result
        assert "T-1" in result


# ---------------------------------------------------------------------------
# jira_update_issue
# ---------------------------------------------------------------------------

class TestJiraUpdateIssue:
    @patch("pile.tools.jira_tools._jira_client")
    def test_no_fields(self, mock_client):
        result = jira_update_issue("T-1")
        assert "No fields to update" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_update_summary(self, mock_client):
        mock_client.return_value.put.return_value = _mock_response({}, status_code=204)
        result = jira_update_issue("T-1", summary="New title")
        assert "updated" in result.lower()
        assert "summary" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_update_multiple_fields(self, mock_client):
        mock_client.return_value.put.return_value = _mock_response({}, status_code=204)
        result = jira_update_issue(
            "T-1",
            summary="Title",
            priority="High",
            story_points=5.0,
            labels="frontend,backend",
        )
        assert "summary" in result
        assert "priority" in result
        payload = mock_client.return_value.put.call_args.kwargs.get("json") or mock_client.return_value.put.call_args[1]["json"]
        assert payload["fields"]["customfield_10016"] == 5.0
        assert payload["fields"]["labels"] == ["frontend", "backend"]

    @patch("pile.tools.jira_tools._jira_client")
    def test_unassign(self, mock_client):
        mock_client.return_value.put.return_value = _mock_response({}, status_code=204)
        result = jira_update_issue("T-1", assignee_id="none")
        assert "assignee" in result
        payload = mock_client.return_value.put.call_args.kwargs.get("json") or mock_client.return_value.put.call_args[1]["json"]
        assert payload["fields"]["assignee"] is None

    @patch("pile.tools.jira_tools._jira_client")
    def test_assign_user(self, mock_client):
        mock_client.return_value.put.return_value = _mock_response({}, status_code=204)
        result = jira_update_issue("T-1", assignee_id="user123")
        payload = mock_client.return_value.put.call_args.kwargs.get("json") or mock_client.return_value.put.call_args[1]["json"]
        assert payload["fields"]["assignee"]["accountId"] == "user123"

    @patch("pile.tools.jira_tools._jira_client")
    def test_update_description(self, mock_client):
        mock_client.return_value.put.return_value = _mock_response({}, status_code=204)
        result = jira_update_issue("T-1", description="New desc")
        assert "description" in result


# ---------------------------------------------------------------------------
# jira_move_to_sprint
# ---------------------------------------------------------------------------

class TestJiraMoveToSprint:
    @patch("pile.tools.jira_tools._jira_client")
    def test_move_issues(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({}, status_code=204)
        result = jira_move_to_sprint(50, "T-1,T-2,T-3")
        assert "Moved 3 issues" in result
        assert "sprint 50" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_empty_keys(self, mock_client):
        result = jira_move_to_sprint(50, "")
        assert "No issue keys" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_whitespace_keys(self, mock_client):
        result = jira_move_to_sprint(50, " , , ")
        assert "No issue keys" in result


# ---------------------------------------------------------------------------
# jira_move_to_backlog
# ---------------------------------------------------------------------------

class TestJiraMoveToBacklog:
    @patch("pile.tools.jira_tools._jira_client")
    def test_move_to_backlog(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({}, status_code=204)
        result = jira_move_to_backlog("T-1,T-2")
        assert "Moved 2 issues to backlog" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_empty_keys(self, mock_client):
        result = jira_move_to_backlog("")
        assert "No issue keys" in result


# ---------------------------------------------------------------------------
# jira_create_sprint
# ---------------------------------------------------------------------------

class TestJiraCreateSprint:
    @patch("pile.tools.jira_tools._jira_client")
    def test_create_minimal(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({
            "name": "Sprint 10",
            "id": 100,
            "state": "future",
        })
        result = jira_create_sprint(1, "Sprint 10")
        assert "Sprint 10" in result
        assert "100" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_create_with_all_fields(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({
            "name": "Sprint 11",
            "id": 101,
            "state": "future",
        })
        result = jira_create_sprint(
            1, "Sprint 11",
            goal="Deliver feature X",
            start_date="2026-04-07",
            end_date="2026-04-21",
        )
        assert "Sprint 11" in result
        payload = mock_client.return_value.post.call_args.kwargs.get("json") or mock_client.return_value.post.call_args[1]["json"]
        assert payload["goal"] == "Deliver feature X"
        assert payload["startDate"] == "2026-04-07"
        assert payload["endDate"] == "2026-04-21"


# ---------------------------------------------------------------------------
# jira_link_issues
# ---------------------------------------------------------------------------

class TestJiraLinkIssues:
    @patch("pile.tools.jira_tools._jira_client")
    def test_link_issues(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({}, status_code=201)
        result = jira_link_issues("T-1", "T-2", link_type="Blocks")
        assert "Linked" in result
        assert "T-1" in result
        assert "T-2" in result
        assert "Blocks" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_link_default_type(self, mock_client):
        mock_client.return_value.post.return_value = _mock_response({}, status_code=201)
        result = jira_link_issues("T-1", "T-2")
        assert "Blocks" in result


# ---------------------------------------------------------------------------
# Error handling (_safe_jira_call decorator)
# ---------------------------------------------------------------------------

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
        mock_client.return_value.get.side_effect = _mock_error_response(401)
        result = jira_search("project=TEST")
        assert "authentication failed" in result.lower()

    @patch("pile.tools.jira_tools._jira_client")
    def test_forbidden_error(self, mock_client):
        mock_client.return_value.get.side_effect = _mock_error_response(403)
        result = jira_search("project=TEST")
        assert "forbidden" in result.lower()

    @patch("pile.tools.jira_tools._jira_client")
    def test_not_found_error(self, mock_client):
        mock_client.return_value.get.side_effect = _mock_error_response(404)
        result = jira_search("project=TEST")
        assert "404" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_rate_limit_error(self, mock_client):
        mock_client.return_value.get.side_effect = _mock_error_response(429)
        result = jira_search("project=TEST")
        assert "rate limit" in result.lower()

    @patch("pile.tools.jira_tools._jira_client")
    def test_generic_http_error(self, mock_client):
        mock_client.return_value.get.side_effect = _mock_error_response(500)
        result = jira_search("project=TEST")
        assert "500" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_unexpected_error(self, mock_client):
        mock_client.return_value.get.side_effect = RuntimeError("boom")
        result = jira_search("project=TEST")
        assert "Error" in result
        assert "boom" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_connection_error_on_post(self, mock_client):
        mock_client.return_value.post.side_effect = httpx.ConnectError("fail")
        result = jira_add_comment("T-1", "test")
        assert "Cannot connect" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_timeout_on_create_issue(self, mock_client):
        mock_client.return_value.post.side_effect = httpx.TimeoutException("timeout")
        result = jira_create_issue(summary="Test")
        assert "timed out" in result

    @patch("pile.tools.jira_tools._jira_client")
    def test_auth_error_on_update(self, mock_client):
        mock_client.return_value.put.side_effect = _mock_error_response(401)
        result = jira_update_issue("T-1", summary="New")
        assert "authentication failed" in result.lower()
