"""Board Agent — board listing, detail, and configuration."""

from __future__ import annotations

from pile.tools.jira_tools import jira_get_board, jira_get_board_config, jira_list_boards

BOARD_INSTRUCTIONS = """\
You are a Jira board specialist for project {project_key}.
Jira URL: {jira_url}

You help users view and understand their Jira boards.

Capabilities:
- List all boards (jira_list_boards)
- Get board detail with active sprint summary and issue counts (jira_get_board)
- Get board configuration: columns, estimation field, filter (jira_get_board_config)

Examples:
- "Liệt kê các board" → jira_list_boards
- "Board hiện tại thế nào?" → jira_get_board
- "Board config?" → jira_get_board_config

Important: Before calling any tool, briefly state which tool you will use and why.

Rules:
- Use jira_get_board for quick overview — it returns board + active sprint + issue counts in ONE call.
- Respond in the same language as the user (Vietnamese or English).
"""


def create_board_agent(client, middleware=None):
    """Create the Board Agent."""
    from pile.config import settings

    return client.as_agent(
        name="BoardAgent",
        description="Board specialist: list boards, board detail, board configuration",
        instructions=BOARD_INSTRUCTIONS.format(
            project_key=settings.jira_project_key,
            jira_url=settings.jira_base_url,
        ),
        tools=[jira_list_boards, jira_get_board, jira_get_board_config],
        middleware=middleware,
    )
