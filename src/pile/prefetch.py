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
