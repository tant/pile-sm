"""Sprint Agent — sprint viewing, creation, and issue movement."""

from __future__ import annotations

from pile.tools.jira_tools import (
    jira_create_sprint,
    jira_get_sprint,
    jira_get_sprint_issues,
    jira_move_to_backlog,
    jira_move_to_sprint,
)

SPRINT_INSTRUCTIONS = """\
You are a sprint management specialist for project {project_key}.

You help users view sprints and move issues between sprints and backlog.

Capabilities:
- View sprints for a board: active, future, closed (jira_get_sprint)
- View all issues in a sprint grouped by status (jira_get_sprint_issues)
- Create a new sprint (jira_create_sprint) — requires approval
- Move issues into a sprint (jira_move_to_sprint) — requires approval
- Move issues back to backlog (jira_move_to_backlog) — requires approval

Examples:
- "Sprint hiện tại có gì?" → jira_get_sprint_issues
- "Các sprint trên board 1?" → jira_get_sprint(board_id=1, state="active")
- "Chuyển PROJ-1, PROJ-2 vào sprint 15" → jira_move_to_sprint
- "Đưa PROJ-3 về backlog" → jira_move_to_backlog
- "Tạo sprint mới" → jira_create_sprint

Rules:
- All write operations require user approval.
- Respond in the same language as the user (Vietnamese or English).
"""


def create_sprint_agent(client, middleware=None):
    """Create the Sprint Agent."""
    from pile.config import settings

    return client.as_agent(
        name="SprintAgent",
        description="Sprint specialist: view sprints, move issues, create sprints",
        instructions=SPRINT_INSTRUCTIONS.format(project_key=settings.jira_project_key),
        tools=[jira_get_sprint, jira_get_sprint_issues, jira_create_sprint, jira_move_to_sprint, jira_move_to_backlog],
        middleware=middleware,
    )
