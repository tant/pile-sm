"""Jira REST API tools for the Jira Agent."""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from pydantic import Field

from agent_framework import tool
from pile.config import settings
from pile.tools.utils import extract_text, make_adf

logger = logging.getLogger("pile.tools.jira")

_client: httpx.Client | None = None


def _jira_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            base_url=settings.jira_base_url,
            auth=(settings.jira_email, settings.jira_api_token),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=30.0,
        )
    return _client


def _safe_jira_call(func):
    """Decorator to handle common Jira API errors gracefully."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except httpx.ConnectError:
            return f"Error: Cannot connect to Jira at {settings.jira_base_url}."
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                return "Error: Jira authentication failed. Check credentials."
            if status == 403:
                return "Error: Jira access forbidden."
            if status == 404:
                return "Error: Not found (404)."
            if status == 429:
                return "Error: Jira rate limit exceeded. Try again later."
            return f"Error: Jira API returned {status}."
        except httpx.TimeoutException:
            return "Error: Jira request timed out."
        except Exception as e:
            logger.exception("Unexpected Jira error")
            return f"Error: {e}"
    return wrapper


@_safe_jira_call
def jira_search(
    jql: Annotated[str, Field(description="JQL query string, e.g. 'project=TETRA AND sprint in openSprints()'")],
    max_results: Annotated[int, Field(description="Maximum number of results to return")] = 10,
) -> str:
    """Search Jira issues using JQL. Returns a summary list of matching issues."""
    client = _jira_client()
    resp = client.get(
        "/rest/api/3/search/jql",
        params={
            "jql": jql,
            "maxResults": max_results,
            "fields": "summary,status,assignee,priority,issuetype",
        },
    )
    resp.raise_for_status()
    issues = resp.json().get("issues", [])
    if not issues:
        return "No issues found."
    lines = []
    for issue in issues:
        key = issue["key"]
        fields = issue["fields"]
        summary = fields["summary"]
        status = fields["status"]["name"]
        assignee = (fields.get("assignee") or {}).get("displayName", "Unassigned")
        priority = (fields.get("priority") or {}).get("name", "None")
        lines.append(f"- **{key}** [{status}] {summary} (assignee: {assignee}, priority: {priority})")
    return "\n".join(lines)


@_safe_jira_call
def jira_get_issue(
    issue_key: Annotated[str, Field(description="Jira issue key, e.g. TETRA-123")],
) -> str:
    """Get detailed information about a specific Jira issue including links and comments."""
    client = _jira_client()
    resp = client.get(f"/rest/api/3/issue/{issue_key}")
    resp.raise_for_status()
    data = resp.json()
    fields = data["fields"]
    lines = [
        f"**{data['key']}: {fields['summary']}**",
        f"- Type: {fields['issuetype']['name']}",
        f"- Status: {fields['status']['name']}",
        f"- Priority: {(fields.get('priority') or {}).get('name', 'None')}",
        f"- Assignee: {(fields.get('assignee') or {}).get('displayName', 'Unassigned')}",
        f"- Reporter: {(fields.get('reporter') or {}).get('displayName', 'Unknown')}",
        f"- Created: {fields['created'][:10]}",
        f"- Updated: {fields['updated'][:10]}",
        f"- Story Points: {fields.get('story_points') or fields.get('customfield_10016') or 'N/A'}",
        f"- Description: {extract_text(fields.get('description')) or 'No description'}",
    ]
    issue_links = fields.get("issuelinks", [])
    if issue_links:
        lines.append("- Links:")
        for link in issue_links:
            if "outwardIssue" in link:
                lines.append(f"  - {link['type']['outward']} {link['outwardIssue']['key']}")
            if "inwardIssue" in link:
                lines.append(f"  - {link['type']['inward']} {link['inwardIssue']['key']}")
    return "\n".join(lines)


@_safe_jira_call
def jira_get_sprint(
    board_id: Annotated[int, Field(description="Jira board ID")],
    state: Annotated[str, Field(description="Sprint state: active, future, or closed")] = "active",
) -> str:
    """Get sprint information for a Jira board."""
    client = _jira_client()
    resp = client.get(
        f"/rest/agile/1.0/board/{board_id}/sprint",
        params={"state": state, "maxResults": 5},
    )
    resp.raise_for_status()
    sprints = resp.json().get("values", [])
    if not sprints:
        return f"No {state} sprints found for board {board_id}."
    lines = []
    for s in sprints:
        start = s.get("startDate", "N/A")[:10] if s.get("startDate") else "N/A"
        end = s.get("endDate", "N/A")[:10] if s.get("endDate") else "N/A"
        goal = s.get("goal", "")
        lines.append(
            f"- **{s['name']}** (id: {s['id']}, state: {s['state']})\n"
            f"  Start: {start} | End: {end}"
        )
        if goal:
            lines.append(f"  Goal: {goal}")
    return "\n".join(lines)


@_safe_jira_call
def jira_get_sprint_issues(
    sprint_id: Annotated[int, Field(description="Sprint ID")],
    max_results: Annotated[int, Field(description="Maximum number of results")] = 50,
) -> str:
    """Get all issues in a specific sprint, grouped by status."""
    client = _jira_client()
    resp = client.get(
        f"/rest/agile/1.0/sprint/{sprint_id}/issue",
        params={"maxResults": max_results},
    )
    resp.raise_for_status()
    issues = resp.json().get("issues", [])
    if not issues:
        return "No issues in this sprint."
    by_status: dict[str, list] = {}
    for issue in issues:
        status = issue["fields"]["status"]["name"]
        by_status.setdefault(status, []).append(issue)
    lines = [f"**Sprint {sprint_id}** — {len(issues)} issues total"]
    for status, items in sorted(by_status.items()):
        lines.append(f"\n### {status} ({len(items)})")
        for i in items:
            assignee = (i["fields"].get("assignee") or {}).get("displayName", "Unassigned")
            sp = i["fields"].get("story_points") or i["fields"].get("customfield_10016") or ""
            sp_str = f" [{sp} SP]" if sp else ""
            lines.append(f"- {i['key']}: {i['fields']['summary']}{sp_str} ({assignee})")
    return "\n".join(lines)


@_safe_jira_call
def jira_get_board(
    project_key: Annotated[str, Field(description="Project key, e.g. TETRA")] = "TETRA",
) -> str:
    """Get Jira board information for a project."""
    client = _jira_client()
    resp = client.get(
        "/rest/agile/1.0/board",
        params={"projectKeyOrId": project_key, "maxResults": 5},
    )
    resp.raise_for_status()
    boards = resp.json().get("values", [])
    if not boards:
        return f"No boards found for project {project_key}."
    lines = []
    for b in boards:
        lines.append(f"- **{b['name']}** (id: {b['id']}, type: {b['type']})")
    return "\n".join(lines)


# --- Write tools (require approval) ---


@tool(approval_mode="always_require")
@_safe_jira_call
def jira_create_issue(
    summary: Annotated[str, Field(description="Issue title")],
    issue_type: Annotated[str, Field(description="Type: Task, Bug, Story, Epic")] = "Task",
    description: Annotated[str | None, Field(description="Issue description")] = None,
    assignee_id: Annotated[str | None, Field(description="Assignee account ID")] = None,
    priority: Annotated[str | None, Field(description="Priority: Highest, High, Medium, Low, Lowest")] = None,
) -> str:
    """Create a new Jira issue. Requires user approval before execution."""
    payload: dict = {
        "fields": {
            "project": {"key": settings.jira_project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
    }
    if description:
        payload["fields"]["description"] = make_adf(description)
    if assignee_id:
        payload["fields"]["assignee"] = {"accountId": assignee_id}
    if priority:
        payload["fields"]["priority"] = {"name": priority}
    client = _jira_client()
    resp = client.post("/rest/api/3/issue", json=payload)
    resp.raise_for_status()
    key = resp.json()["key"]
    return f"Issue created: {key}"


@tool(approval_mode="always_require")
@_safe_jira_call
def jira_transition_issue(
    issue_key: Annotated[str, Field(description="Issue key, e.g. TETRA-123")],
    transition_name: Annotated[str, Field(description="Target status name, e.g. In Progress, Done")],
) -> str:
    """Transition a Jira issue to a new status. Requires user approval."""
    client = _jira_client()
    resp = client.get(f"/rest/api/3/issue/{issue_key}/transitions")
    resp.raise_for_status()
    transitions = resp.json()["transitions"]
    match = next((t for t in transitions if t["name"].lower() == transition_name.lower()), None)
    if not match:
        available = ", ".join(t["name"] for t in transitions)
        return f"Transition '{transition_name}' not found. Available: {available}"
    resp = client.post(
        f"/rest/api/3/issue/{issue_key}/transitions",
        json={"transition": {"id": match["id"]}},
    )
    resp.raise_for_status()
    return f"{issue_key} transitioned to '{transition_name}'."


@tool(approval_mode="always_require")
@_safe_jira_call
def jira_add_comment(
    issue_key: Annotated[str, Field(description="Issue key, e.g. TETRA-123")],
    comment: Annotated[str, Field(description="Comment text")],
) -> str:
    """Add a comment to a Jira issue. Requires user approval."""
    client = _jira_client()
    resp = client.post(
        f"/rest/api/3/issue/{issue_key}/comment",
        json={"body": make_adf(comment)},
    )
    resp.raise_for_status()
    return f"Comment added to {issue_key}."
