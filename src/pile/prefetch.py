"""Data prefetch for Scrum Agent — fetch Jira data before LLM call.

Instead of letting the model decide which tools to call (causing loops),
we deterministically fetch data based on query type, then inject it into
the prompt. Model only needs to analyze, not hunt for data.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("pile.prefetch")


# Query type detection — maps scrum query patterns to prefetch strategies
_SCRUM_PATTERNS: list[tuple[str, list[str]]] = [
    ("standup", [
        r"standup\b", r"stand-up\b", r"daily\b",
    ]),
    ("sprint_review", [
        r"sprint\s+review", r"sprint.*ti[eế]n\s+đ[oộ]",
        r"sprint.*progress", r"sprint.*summary",
        r"sprint.*cho\s+(s[eế]p|meeting|team)",
        r"t[oổ]ng\s+h[oợ]p.*sprint", r"t[oó]m\s+t[aắ]t.*sprint",
    ]),
    ("velocity", [
        r"velocity\b", r"t[oố]c\s+đ[oộ]",
    ]),
    ("workload", [
        r"workload\b", r"qu[aá]\s+t[aả]i", r"overload",
        r"r[aả]nh\b", r"ai\s+đang",
        r"who\s+is\s+working",
    ]),
    ("blockers", [
        r"blocke[rd]", r"b[iị]\s+block", r"\bblock\b",
        r"v[aấ]n\s+đ[eề]", r"problem",
        r"c[aầ]n\s+ch[uú]\s+[yý]",
    ]),
    ("retro", [
        r"retro", r"retrospective",
        r"meeting\s+prep",
    ]),
    ("cycle_time", [
        r"cycle\s+time", r"lead\s+time", r"chu\s+k[yỳ]",
        r"m[aấ]t\s+bao\s+l[aâ]u",
    ]),
    ("data_quality", [
        r"data\s+quality", r"thi[eế]u\s+th[oô]ng\s+tin",
    ]),
    ("stakeholder", [
        r"stakeholder", r"cho\s+s[eế]p",
        r"b[aá]o\s+c[aá]o", r"summary\b",
    ]),
]


def detect_scrum_type(query: str) -> str:
    """Detect which type of scrum query this is."""
    q = query.lower().strip()
    for qtype, patterns in _SCRUM_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return qtype
    return "general"


def prefetch_scrum_data(query: str, board_id: int) -> str:
    """Fetch Jira data needed for a scrum query. Returns formatted data string."""
    from pile.tools.jira_tools import (
        jira_get_board,
        jira_get_changelog,
        jira_get_sprint,
        jira_get_sprint_issues,
        jira_search,
    )

    qtype = detect_scrum_type(query)
    logger.info("Prefetch: '%s' → type=%s", query[:40], qtype)

    parts: list[str] = []

    # Almost all scrum queries need sprint issues
    sprint_id = _get_active_sprint_id(board_id)

    if qtype == "standup":
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))
        parts.append(_safe_call(
            jira_search,
            jql=f"project = {_project_key()} AND updated >= -1d ORDER BY updated DESC",
            max_results=20,
        ))

    elif qtype == "velocity":
        parts.append(_safe_call(jira_get_board))
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))
        # Get closed sprints for comparison
        parts.append(_safe_call(jira_get_sprint, board_id=board_id, state="closed"))

    elif qtype in ("sprint_review", "stakeholder", "general"):
        parts.append(_safe_call(jira_get_board))
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))

    elif qtype == "workload":
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))

    elif qtype == "blockers":
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))
        parts.append(_safe_call(
            jira_search,
            jql=f"project = {_project_key()} AND status = Blocked ORDER BY priority DESC",
            max_results=20,
        ))

    elif qtype == "retro":
        parts.append(_safe_call(jira_get_board))
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))
        parts.append(_safe_call(
            jira_search,
            jql=f"project = {_project_key()} AND status changed TO Done AFTER -14d ORDER BY updated DESC",
            max_results=20,
        ))

    elif qtype == "cycle_time":
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))
        # Fetch changelog for recently Done issues to calculate time per status
        done_keys = _get_done_issue_keys(sprint_id)
        for key in done_keys[:5]:  # limit to 5 to avoid too much data
            parts.append(_safe_call(jira_get_changelog, issue_key=key))

    elif qtype == "data_quality":
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))

    else:
        # Fallback: board + sprint issues
        parts.append(_safe_call(jira_get_board))
        if sprint_id:
            parts.append(_safe_call(jira_get_sprint_issues, sprint_id=sprint_id))

    data = "\n\n".join(p for p in parts if p and not p.startswith("Error"))
    if not data:
        data = "No data could be fetched. Respond based on your knowledge."

    logger.info("Prefetch: %d chars of data fetched", len(data))
    return data


# --- Prefetch for Jira query intents ---

_QUERY_PATTERNS: dict[str, str] = {
    "in_progress": "project = {project} AND status = 'In Progress' ORDER BY priority DESC",
    "to_do": "project = {project} AND status = 'To Do' ORDER BY priority DESC",
    "done_recent": "project = {project} AND status = Done AND updated >= -7d ORDER BY updated DESC",
    "testing": "project = {project} AND status = 'Testing' ORDER BY priority DESC",
    "code_review": "project = {project} AND status = 'Code Review' ORDER BY priority DESC",
    "my_issues": "project = {project} AND assignee = currentUser() ORDER BY priority DESC",
}

_QUERY_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("in_progress", [r"in\s+progress", r"đang\s+l[aà]m", r"đang\s+th[uự]c\s+hi[eệ]n"]),
    ("to_do", [r"to\s+do\b", r"ch[uư]a\s+l[aà]m", r"c[aầ]n\s+l[aà]m"]),
    ("done_recent", [r"\bdone\b.*tu[aầ]n", r"ho[aà]n\s+th[aà]nh", r"xong\b"]),
    ("testing", [r"testing\b", r"đang\s+test", r"ki[eể]m\s+th[uử]"]),
    ("code_review", [r"code\s+review", r"review\b"]),
]


def detect_query_intent(query: str) -> str | None:
    """Detect if query matches a known JQL pattern. Returns intent key or None."""
    q = query.lower().strip()
    for intent, patterns in _QUERY_INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return intent
    # Issue key detection
    match = re.search(r"([A-Z]+-\d+)", query)
    if match:
        return f"issue:{match.group(1)}"
    return None


def prefetch_query_data(query: str) -> str | None:
    """Prefetch Jira data for common query patterns. Returns data or None."""
    from pile.tools.jira_tools import jira_search, jira_get_issue

    intent = detect_query_intent(query)
    if not intent:
        return None

    logger.info("Query prefetch: '%s' → intent=%s", query[:40], intent)

    if intent.startswith("issue:"):
        issue_key = intent.split(":", 1)[1]
        return _safe_call(jira_get_issue, issue_key=issue_key)

    jql_template = _QUERY_PATTERNS.get(intent)
    if jql_template:
        jql = jql_template.format(project=_project_key())
        return _safe_call(jira_search, jql=jql, max_results=15)

    return None


def _get_active_sprint_id(board_id: int) -> int | None:
    """Get the active sprint ID for a board via Jira API directly."""
    try:
        from pile.tools.jira_tools import _jira_client

        client = _jira_client()
        resp = client.get(
            f"/rest/agile/1.0/board/{board_id}/sprint",
            params={"state": "active", "maxResults": 1},
        )
        resp.raise_for_status()
        sprints = resp.json().get("values", [])
        if sprints:
            return sprints[0]["id"]
    except Exception as e:
        logger.warning("Failed to get active sprint ID: %s", e)
    return None


def _get_done_issue_keys(sprint_id: int | None) -> list[str]:
    """Extract issue keys with status Done from a sprint."""
    if not sprint_id:
        return []
    try:
        from pile.tools.jira_tools import _jira_client
        client = _jira_client()
        resp = client.get(
            f"/rest/agile/1.0/sprint/{sprint_id}/issue",
            params={"maxResults": 50, "fields": "status", "jql": "status = Done"},
        )
        resp.raise_for_status()
        issues = resp.json().get("issues", [])
        return [i["key"] for i in issues]
    except Exception:
        return []


def _project_key() -> str:
    from pile.config import settings
    return settings.jira_project_key


def _safe_call(func, **kwargs) -> str:
    """Call a Jira tool function safely, return result or empty string."""
    try:
        return func(**kwargs)
    except Exception as e:
        logger.warning("Prefetch %s failed: %s", func.__name__, e)
        return ""
