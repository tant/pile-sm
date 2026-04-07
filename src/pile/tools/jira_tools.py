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


_CURL_COMMANDS = {
    "list_boards": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/agile/1.0/board"',
    "get_board": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/agile/1.0/board/{{BOARD_ID}}"',
    "active_sprint": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/agile/1.0/board/{{BOARD_ID}}/sprint?state=active"',
    "sprint_issues": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/agile/1.0/sprint/{{SPRINT_ID}}/issue"',
    "search": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/api/3/search/jql?jql={{JQL}}&maxResults=10"',
    "get_issue": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/api/3/issue/{{ISSUE_KEY}}"',
    "backlog": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/agile/1.0/board/{{BOARD_ID}}/backlog"',
    "epics": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/agile/1.0/board/{{BOARD_ID}}/epic"',
    "board_config": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/agile/1.0/board/{{BOARD_ID}}/configuration"',
    "changelog": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/api/3/issue/{{ISSUE_KEY}}/changelog"',
    "create_issue": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" -H "Content-Type: application/json" -X POST "{base}/rest/api/3/issue" -d \'{{JSON_BODY}}\'',
    "transition": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" -H "Content-Type: application/json" -X POST "{base}/rest/api/3/issue/{{ISSUE_KEY}}/transitions" -d \'{{JSON_BODY}}\'',
    "myself": 'curl -s -u "$JIRA_EMAIL:$JIRA_API_TOKEN" -H "Accept: application/json" "{base}/rest/api/3/myself"',
}


def jira_curl_command(
    action: Annotated[str, Field(description="Action: list_boards, get_board, active_sprint, sprint_issues, search, get_issue, backlog, epics, board_config, changelog, create_issue, transition, myself")],
) -> str:
    """Generate a curl command for a Jira API action. User can copy and run it manually."""
    template = _CURL_COMMANDS.get(action)
    if not template:
        available = ", ".join(_CURL_COMMANDS.keys())
        return f"Unknown action '{action}'. Available: {available}"
    return template.format(base=settings.jira_base_url)


@_safe_jira_call
def jira_search(
    jql: Annotated[str, Field(description="JQL query string, e.g. 'project=TETRA AND sprint in openSprints()'")],
    max_results: Annotated[int, Field(description="Maximum number of results to return")] = 10,
    include_subtasks: Annotated[bool, Field(description="Include sub-tasks in results (default: false, only work items)")] = False,
) -> str:
    """Search Jira issues using JQL. By default excludes sub-tasks — set include_subtasks=true to include them."""
    if not include_subtasks and "issuetype" not in jql.lower():
        jql = f"({jql}) AND issuetype not in subtaskIssueTypes()"
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
    include_subtasks: Annotated[bool, Field(description="Include sub-tasks (default: false, only work items)")] = False,
) -> str:
    """Get work items in a sprint, grouped by status. Excludes sub-tasks by default."""
    client = _jira_client()
    resp = client.get(
        f"/rest/agile/1.0/sprint/{sprint_id}/issue",
        params={"maxResults": max_results},
    )
    resp.raise_for_status()
    issues = resp.json().get("issues", [])
    if not include_subtasks:
        issues = [i for i in issues if not i["fields"]["issuetype"].get("subtask", False)]
    if not issues:
        return "No issues in this sprint."
    by_status: dict[str, list] = {}
    for issue in issues:
        status = issue["fields"]["status"]["name"]
        by_status.setdefault(status, []).append(issue)
    lines = [f"**Sprint {sprint_id}** — {len(issues)} work items"]
    for status, items in sorted(by_status.items()):
        lines.append(f"\n### {status} ({len(items)})")
        for i in items:
            assignee = (i["fields"].get("assignee") or {}).get("displayName", "Unassigned")
            sp = i["fields"].get("story_points") or i["fields"].get("customfield_10016") or ""
            sp_str = f" [{sp} SP]" if sp else ""
            lines.append(f"- {i['key']}: {i['fields']['summary']}{sp_str} ({assignee})")
    return "\n".join(lines)


@_safe_jira_call
def jira_list_boards(
    project_key: Annotated[str | None, Field(description="Project key to filter (optional, lists all boards if not given)")] = None,
) -> str:
    """List all Jira boards, optionally filtered by project."""
    client = _jira_client()
    params: dict = {"maxResults": 50}
    if project_key:
        params["projectKeyOrId"] = project_key
    resp = client.get("/rest/agile/1.0/board", params=params)
    resp.raise_for_status()
    boards = resp.json().get("values", [])
    if not boards:
        return "No boards found."
    lines = [f"**{len(boards)} boards found:**"]
    for b in boards:
        project = b.get("location", {}).get("projectKey", "")
        lines.append(f"- **{b['name']}** (id: {b['id']}, type: {b['type']}, project: {project})")
    return "\n".join(lines)


@_safe_jira_call
def jira_get_board(
    project_key: Annotated[str | None, Field(description="Project key (uses configured default if not given)")] = None,
) -> str:
    """Get board detail with active sprint summary. Returns board, sprint, and issue counts in one call."""
    client = _jira_client()
    key = project_key or settings.jira_project_key

    # Get board
    resp = client.get("/rest/agile/1.0/board", params={"projectKeyOrId": key, "maxResults": 5})
    resp.raise_for_status()
    boards = resp.json().get("values", [])
    if not boards:
        return f"No boards found for project {key}."

    board = boards[0]
    lines = [f"**Board: {board['name']}** (id: {board['id']}, type: {board['type']})"]

    # Auto-fetch active sprint
    try:
        resp = client.get(f"/rest/agile/1.0/board/{board['id']}/sprint", params={"state": "active", "maxResults": 1})
        resp.raise_for_status()
        sprints = resp.json().get("values", [])
        if sprints:
            s = sprints[0]
            start = s.get("startDate", "N/A")[:10] if s.get("startDate") else "N/A"
            end = s.get("endDate", "N/A")[:10] if s.get("endDate") else "N/A"
            lines.append(f"\n**Active Sprint: {s['name']}** (id: {s['id']})")
            lines.append(f"  Start: {start} | End: {end}")
            if s.get("goal"):
                lines.append(f"  Goal: {s['goal']}")

            # Auto-fetch sprint issue counts (work items only, exclude sub-tasks)
            try:
                resp = client.get(f"/rest/agile/1.0/sprint/{s['id']}/issue", params={"maxResults": 100})
                resp.raise_for_status()
                issues = [i for i in resp.json().get("issues", []) if not i["fields"]["issuetype"].get("subtask", False)]
                by_status: dict[str, int] = {}
                for issue in issues:
                    status = issue["fields"]["status"]["name"]
                    by_status[status] = by_status.get(status, 0) + 1
                lines.append(f"  Work items: {len(issues)} (excluding sub-tasks)")
                for status, count in sorted(by_status.items()):
                    lines.append(f"  - {status}: {count}")
            except Exception:
                pass
        else:
            lines.append("\nNo active sprint.")
    except Exception:
        pass

    return "\n".join(lines)


# --- Backlog, Epic, Board Config, Changelog ---


@_safe_jira_call
def jira_get_backlog(
    board_id: Annotated[int, Field(description="Jira board ID")],
    max_results: Annotated[int, Field(description="Maximum number of results")] = 30,
    include_subtasks: Annotated[bool, Field(description="Include sub-tasks (default: false)")] = False,
) -> str:
    """Get backlog issues (not in any active/future sprint) for a board."""
    client = _jira_client()
    resp = client.get(
        f"/rest/agile/1.0/board/{board_id}/backlog",
        params={"maxResults": max_results, "fields": "summary,status,assignee,priority,issuetype,customfield_10016"},
    )
    resp.raise_for_status()
    issues = resp.json().get("issues", [])
    if not include_subtasks:
        issues = [i for i in issues if not i["fields"]["issuetype"].get("subtask", False)]
    if not issues:
        return f"Backlog is empty for board {board_id}."
    lines = [f"**Backlog** — {len(issues)} work items"]
    for i in issues:
        f = i["fields"]
        assignee = (f.get("assignee") or {}).get("displayName", "Unassigned")
        sp = f.get("story_points") or f.get("customfield_10016") or ""
        sp_str = f" [{sp} SP]" if sp else ""
        priority = (f.get("priority") or {}).get("name", "")
        lines.append(f"- {i['key']}: {f['summary']}{sp_str} ({assignee}, {priority})")
    return "\n".join(lines)


@_safe_jira_call
def jira_get_epics(
    board_id: Annotated[int, Field(description="Jira board ID")],
    done: Annotated[bool, Field(description="Include completed epics (default: false)")] = False,
) -> str:
    """List epics on a board."""
    client = _jira_client()
    resp = client.get(
        f"/rest/agile/1.0/board/{board_id}/epic",
        params={"maxResults": 50, "done": str(done).lower()},
    )
    resp.raise_for_status()
    epics = resp.json().get("values", [])
    if not epics:
        return f"No epics found on board {board_id}."
    lines = [f"**{len(epics)} epics:**"]
    for e in epics:
        status = "Done" if e.get("done") else "In Progress"
        lines.append(f"- **{e['key']}**: {e.get('name') or e.get('summary', 'N/A')} [{status}]")
    return "\n".join(lines)


@_safe_jira_call
def jira_get_epic_issues(
    epic_key: Annotated[str, Field(description="Epic issue key, e.g. PROJ-10")],
    max_results: Annotated[int, Field(description="Maximum number of results")] = 30,
    include_subtasks: Annotated[bool, Field(description="Include sub-tasks (default: false)")] = False,
) -> str:
    """Get all issues belonging to an epic."""
    client = _jira_client()
    resp = client.get(
        f"/rest/agile/1.0/epic/{epic_key}/issue",
        params={"maxResults": max_results, "fields": "summary,status,assignee,issuetype,customfield_10016"},
    )
    resp.raise_for_status()
    issues = resp.json().get("issues", [])
    if not include_subtasks:
        issues = [i for i in issues if not i["fields"]["issuetype"].get("subtask", False)]
    if not issues:
        return f"No issues in epic {epic_key}."
    by_status: dict[str, list] = {}
    for issue in issues:
        status = issue["fields"]["status"]["name"]
        by_status.setdefault(status, []).append(issue)
    lines = [f"**Epic {epic_key}** — {len(issues)} work items"]
    for status, items in sorted(by_status.items()):
        lines.append(f"\n### {status} ({len(items)})")
        for i in items:
            assignee = (i["fields"].get("assignee") or {}).get("displayName", "Unassigned")
            sp = i["fields"].get("story_points") or i["fields"].get("customfield_10016") or ""
            sp_str = f" [{sp} SP]" if sp else ""
            lines.append(f"- {i['key']}: {i['fields']['summary']}{sp_str} ({assignee})")
    return "\n".join(lines)


@_safe_jira_call
def jira_get_board_config(
    board_id: Annotated[int, Field(description="Jira board ID")],
) -> str:
    """Get board configuration: columns (workflow mapping), estimation field, filter."""
    client = _jira_client()
    resp = client.get(f"/rest/agile/1.0/board/{board_id}/configuration")
    resp.raise_for_status()
    config = resp.json()
    lines = [f"**Board Configuration (id: {board_id})**"]

    # Columns
    cols = config.get("columnConfig", {}).get("columns", [])
    if cols:
        lines.append("\n**Columns:**")
        for col in cols:
            statuses = ", ".join(s.get("self", "").split("/")[-1] if "self" in s else str(s.get("id", "")) for s in col.get("statuses", []))
            constraint = ""
            if col.get("min") is not None or col.get("max") is not None:
                constraint = f" (min: {col.get('min', '-')}, max: {col.get('max', '-')})"
            lines.append(f"- {col['name']}{constraint}")

    # Estimation
    est = config.get("estimation", {})
    if est:
        field = est.get("field", {})
        lines.append(f"\n**Estimation:** {field.get('displayName', 'N/A')} (field: {field.get('fieldId', 'N/A')})")

    # Filter
    filt = config.get("filter", {})
    if filt:
        lines.append(f"\n**Filter:** {filt.get('name', 'N/A')} (id: {filt.get('id', 'N/A')})")
        if config.get("subQuery", {}).get("query"):
            lines.append(f"**Sub-query:** {config['subQuery']['query']}")

    return "\n".join(lines)


@_safe_jira_call
def jira_get_changelog(
    issue_key: Annotated[str, Field(description="Issue key, e.g. PROJ-123")],
    max_results: Annotated[int, Field(description="Maximum changelog entries")] = 20,
) -> str:
    """Get issue change history (status transitions, field changes). Useful for cycle time analysis."""
    client = _jira_client()
    resp = client.get(
        f"/rest/api/3/issue/{issue_key}/changelog",
        params={"maxResults": max_results},
    )
    resp.raise_for_status()
    entries = resp.json().get("values", [])
    if not entries:
        return f"No changelog entries for {issue_key}."
    lines = [f"**Changelog for {issue_key}** ({len(entries)} entries)"]
    for entry in entries:
        author = (entry.get("author") or {}).get("displayName", "Unknown")
        created = entry.get("created", "")[:16].replace("T", " ")
        for item in entry.get("items", []):
            field = item.get("field", "")
            from_val = item.get("fromString", "") or ""
            to_val = item.get("toString", "") or ""
            lines.append(f"- {created} | {author} | {field}: {from_val} → {to_val}")
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


@tool(approval_mode="always_require")
@_safe_jira_call
def jira_update_issue(
    issue_key: Annotated[str, Field(description="Issue key, e.g. PROJ-123")],
    summary: Annotated[str | None, Field(description="New title")] = None,
    description: Annotated[str | None, Field(description="New description")] = None,
    assignee_id: Annotated[str | None, Field(description="Assignee account ID (use 'none' to unassign)")] = None,
    priority: Annotated[str | None, Field(description="Priority: Highest, High, Medium, Low, Lowest")] = None,
    story_points: Annotated[float | None, Field(description="Story points value")] = None,
    labels: Annotated[str | None, Field(description="Comma-separated labels to SET (replaces existing)")] = None,
) -> str:
    """Update fields on a Jira issue. Only provided fields are changed. Requires user approval."""
    fields: dict = {}
    if summary:
        fields["summary"] = summary
    if description:
        fields["description"] = make_adf(description)
    if assignee_id is not None:
        fields["assignee"] = None if assignee_id.lower() == "none" else {"accountId": assignee_id}
    if priority:
        fields["priority"] = {"name": priority}
    if story_points is not None:
        fields["customfield_10016"] = story_points
    if labels is not None:
        fields["labels"] = [l.strip() for l in labels.split(",") if l.strip()]
    if not fields:
        return "No fields to update."
    client = _jira_client()
    resp = client.put(f"/rest/api/3/issue/{issue_key}", json={"fields": fields})
    resp.raise_for_status()
    updated = ", ".join(fields.keys())
    return f"{issue_key} updated: {updated}."


@tool(approval_mode="always_require")
@_safe_jira_call
def jira_move_to_sprint(
    sprint_id: Annotated[int, Field(description="Target sprint ID")],
    issue_keys: Annotated[str, Field(description="Comma-separated issue keys, e.g. 'PROJ-1,PROJ-2'")],
) -> str:
    """Move issues to a sprint. Requires user approval."""
    keys = [k.strip() for k in issue_keys.split(",") if k.strip()]
    if not keys:
        return "No issue keys provided."
    client = _jira_client()
    resp = client.post(
        f"/rest/agile/1.0/sprint/{sprint_id}/issue",
        json={"issues": keys},
    )
    resp.raise_for_status()
    return f"Moved {len(keys)} issues to sprint {sprint_id}: {', '.join(keys)}"


@tool(approval_mode="always_require")
@_safe_jira_call
def jira_move_to_backlog(
    issue_keys: Annotated[str, Field(description="Comma-separated issue keys, e.g. 'PROJ-1,PROJ-2'")],
) -> str:
    """Move issues back to backlog (remove from sprint). Requires user approval."""
    keys = [k.strip() for k in issue_keys.split(",") if k.strip()]
    if not keys:
        return "No issue keys provided."
    client = _jira_client()
    resp = client.post(
        "/rest/agile/1.0/backlog/issue",
        json={"issues": keys},
    )
    resp.raise_for_status()
    return f"Moved {len(keys)} issues to backlog: {', '.join(keys)}"


@tool(approval_mode="always_require")
@_safe_jira_call
def jira_create_sprint(
    board_id: Annotated[int, Field(description="Board ID to create sprint on")],
    name: Annotated[str, Field(description="Sprint name")],
    goal: Annotated[str | None, Field(description="Sprint goal")] = None,
    start_date: Annotated[str | None, Field(description="Start date (ISO format: 2026-04-07)")] = None,
    end_date: Annotated[str | None, Field(description="End date (ISO format: 2026-04-21)")] = None,
) -> str:
    """Create a new sprint on a board. Requires user approval."""
    payload: dict = {"name": name, "originBoardId": board_id}
    if goal:
        payload["goal"] = goal
    if start_date:
        payload["startDate"] = start_date
    if end_date:
        payload["endDate"] = end_date
    client = _jira_client()
    resp = client.post("/rest/agile/1.0/sprint", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return f"Sprint created: {data['name']} (id: {data['id']}, state: {data['state']})"


@tool(approval_mode="always_require")
@_safe_jira_call
def jira_link_issues(
    inward_issue: Annotated[str, Field(description="Inward issue key (e.g. the blocked issue)")],
    outward_issue: Annotated[str, Field(description="Outward issue key (e.g. the blocking issue)")],
    link_type: Annotated[str, Field(description="Link type: Blocks, Relates, Duplicate, Cloners")] = "Blocks",
) -> str:
    """Create a link between two issues (e.g. 'PROJ-1 blocks PROJ-2'). Requires user approval."""
    client = _jira_client()
    resp = client.post(
        "/rest/api/3/issueLink",
        json={
            "type": {"name": link_type},
            "inwardIssue": {"key": inward_issue},
            "outwardIssue": {"key": outward_issue},
        },
    )
    resp.raise_for_status()
    return f"Linked: {inward_issue} ←[{link_type}]→ {outward_issue}"
