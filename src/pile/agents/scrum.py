"""Scrum Agent — Scrum Master assistant with prefetched Jira data + optional deep-dive tools."""

from __future__ import annotations

from pile.config import settings

SCRUM_INSTRUCTIONS = """\
You are a Scrum Master for project {project_key} ({jira_url}).

When the user's message contains Jira data, analyze that data and answer the question.
Include specific numbers, issue keys (TETRA-xxx), and assignee names.
Do NOT say you cannot access the system — the data is provided to you.

Respond in the same language as the user (Vietnamese or English).
{git_note}
{memory_note}
"""


def create_scrum_agent(client, middleware=None, **kwargs):
    """Create the Scrum Agent."""
    git_note = "Git is not configured — skip git-related analysis."
    memory_note = ""

    git_tools = []
    if settings.git_repo_list:
        from pile.tools.git_tools import git_diff, git_log
        git_tools = [git_log, git_diff]
        git_note = (
            "You also have git_log and git_diff for commit history.\n"
            f"Repos: {', '.join(r.path for r in settings.git_repo_list)}"
        )

    memory_tools = []
    if settings.memory_enabled:
        from pile.tools.memory_tools import memory_search
        memory_tools = [memory_search]
        memory_note = "You have memory_search for past decisions and knowledge base."

    from pile.tools.jira_tools import (
        jira_get_board,
        jira_get_changelog,
        jira_get_issue,
        jira_get_sprint_issues,
        jira_search,
        get_current_sprint_info,
        search_project_issues,
    )
    tools = [
        get_current_sprint_info, search_project_issues,
        jira_search, jira_get_issue, jira_get_board,
        jira_get_sprint_issues, jira_get_changelog,
    ] + git_tools + memory_tools
    instructions = SCRUM_INSTRUCTIONS.format(
        project_key=settings.jira_project_key,
        jira_url=settings.jira_base_url,
        git_note=git_note,
        memory_note=memory_note,
    )

    return client.as_agent(
        name="ScrumAgent",
        description="Scrum Master: standup, planning, retro, coaching, reports, data quality, timeline tracking",
        instructions=instructions,
        tools=tools,
        middleware=middleware,
    )
