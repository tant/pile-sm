"""Sprint Agent — sprint viewing, creation, and issue movement."""

from __future__ import annotations

from pile.tools.jira_tools import (
    get_current_sprint_info,
    jira_create_sprint,
    jira_get_sprint,
    jira_get_sprint_issues,
    jira_move_to_backlog,
    jira_move_to_sprint,
)

SPRINT_INSTRUCTIONS = """\
You are a sprint management specialist for project {project_key}.
Jira URL: {jira_url}. Board ID: {board_id}.

IMPORTANT: Always use board_id={board_id} when calling tools. NEVER ask the user for board ID.

Preferred tool:
- get_current_sprint_info() → current sprint + all issues (NO params needed)

Other tools:
- jira_get_sprint(board_id={board_id}) → sprint list
- jira_get_sprint_issues(sprint_id=N) → issues in specific sprint
- jira_create_sprint / jira_move_to_sprint / jira_move_to_backlog — requires approval

Examples:
- "Sprint hiện tại có gì?" → get_current_sprint_info()
- "Sprint status?" → get_current_sprint_info()
- "Chuyển PROJ-1 vào sprint 15" → jira_move_to_sprint
- "Tạo sprint mới" → jira_create_sprint

Rules:
- All write operations require user approval.
- Respond in the same language as the user (Vietnamese or English).
"""


def create_sprint_agent(client, middleware=None, board_id=0):
    """Create the Sprint Agent."""
    from pile.config import settings

    return client.as_agent(
        name="SprintAgent",
        description="Sprint specialist: view sprints, move issues, create sprints",
        instructions=SPRINT_INSTRUCTIONS.format(
            project_key=settings.jira_project_key,
            jira_url=settings.jira_base_url,
            board_id=board_id or "unknown (ask user)",
        ),
        tools=[get_current_sprint_info, jira_get_sprint, jira_get_sprint_issues, jira_create_sprint, jira_move_to_sprint, jira_move_to_backlog],
        middleware=middleware,
    )
