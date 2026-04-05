"""Deterministic router — keyword-based agent selection for small models.

Replaces HandoffBuilder to avoid tool bloat. Routes 70%+ queries by keyword,
falls back to LLM classification only for ambiguous cases.
"""

from __future__ import annotations

import re

# Keyword patterns → agent name mapping (checked in order)
_ROUTES: list[tuple[str, list[str]]] = [
    ("memory", [
        r"nh[oớ]\s+gi[uú]p", r"remember\b", r"qu[eê]n\b", r"forget\b",
        r"knowledge\b", r"t[aà]i\s+li[eệ]u",
        r"load\s+(file|document|pdf)", r"ingest\b",
        r"memory\b",
    ]),
    ("browser", [
        r"m[oở]\s+trang", r"open\s+(url|page|web)",
        r"login\s+v[aà]o", r"đ[aă]ng\s+nh[aậ]p",
        r"screenshot\b", r"scrape\b", r"browse\b",
        r"vnexpress|github\.com|gitlab\.com|atlassian",
        r"https?://",
    ]),
    ("jira_query", [
        r"cho\s+t[oô]i\s+l[eệ]nh", r"curl\b", r"l[eệ]nh\s+l[aấ]y",
        r"changelog\b", r"l[iị]ch\s+s[uử]",
    ]),
    ("jira_write", [
        r"t[aạ]o\s+(issue|bug|task|story|epic|ticket)",
        r"create\s+(issue|bug|task|story|epic|ticket)",
        r"assign\b", r"chuy[eể]n\s+tr[aạ]ng\s+th[aá]i",
        r"transition\b", r"comment\b", r"link\s+issue",
        r"s[uử]a\s+(issue|bug|task)", r"update\s+issue",
        r"c[aậ]p\s+nh[aậ]t\s+(issue|bug|task)",
    ]),
    ("board", [
        r"\bboard\b", r"li[eệ]t\s+k[eê]\s+.*board",
        r"board\s+config", r"c[aấ]u\s+h[iì]nh\s+board",
    ]),
    ("sprint", [
        r"\bsprint\b", r"t[aạ]o\s+sprint",
        r"chuy[eể]n.*v[aà]o\s+sprint", r"move.*sprint",
        r"v[eề]\s+backlog", r"move.*backlog",
    ]),
    ("epic", [
        r"\bepic\b", r"\bbacklog\b",
    ]),
    ("git", [
        r"\bgit\b", r"\bcommit\b", r"\bbranch\b",
        r"\bdiff\b", r"\bblame\b", r"code\s+change",
    ]),
    ("scrum", [
        r"standup\b", r"velocity\b", r"workload\b",
        r"burndown\b", r"cycle\s+time", r"retro",
        r"sprint\s+review", r"sprint\s+planning",
        r"blocke[rd]", r"qu[aá]\s+t[aả]i",
        r"data\s+quality", r"thi[eế]u\s+th[oô]ng\s+tin",
        r"t[oổ]ng\s+h[oợ]p", r"b[aá]o\s+c[aá]o",
        r"stakeholder", r"meeting\s+prep",
    ]),
    ("jira_query", [
        r"t[iì]m\b", r"search\b", r"jql\b",
        r"issue\b", r"ticket\b",
        r"tr[aạ]ng\s+th[aá]i", r"status\b",
    ]),
]


def route_query(query: str) -> str | None:
    """Route a user query to an agent name by keyword matching.

    Returns agent key (e.g. "board", "jira_query") or None if ambiguous.
    """
    q = query.lower().strip()
    for agent_key, patterns in _ROUTES:
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return agent_key
    return None
