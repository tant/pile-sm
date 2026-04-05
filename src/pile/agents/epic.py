"""Epic Agent — epic and backlog management."""

from __future__ import annotations

from pile.tools.jira_tools import jira_get_backlog, jira_get_epic_issues, jira_get_epics

EPIC_INSTRUCTIONS = """\
You are an epic and backlog specialist for project {project_key}.

You help users view epics and backlog items.

Capabilities:
- List epics on a board (jira_get_epics)
- View issues in an epic grouped by status (jira_get_epic_issues)
- View backlog issues not in any sprint (jira_get_backlog)

Examples:
- "Các epic trên board 1?" → jira_get_epics(board_id=1)
- "Issues trong epic PROJ-10?" → jira_get_epic_issues(epic_key="PROJ-10")
- "Backlog có gì?" → jira_get_backlog(board_id=1)

Rules:
- Respond in the same language as the user (Vietnamese or English).
"""


def create_epic_agent(client, middleware=None):
    """Create the Epic Agent."""
    from pile.config import settings

    return client.as_agent(
        name="EpicAgent",
        description="Epic and backlog specialist: list epics, epic issues, backlog",
        instructions=EPIC_INSTRUCTIONS.format(project_key=settings.jira_project_key),
        tools=[jira_get_epics, jira_get_epic_issues, jira_get_backlog],
        middleware=middleware,
    )
