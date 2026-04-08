"""Scrum Agent — Scrum Master assistant with prefetched Jira data + optional deep-dive tools."""

from __future__ import annotations

from pile.config import settings

SCRUM_INSTRUCTIONS = """\
You are an experienced Scrum Master for project {project_key} ({jira_url}). Analyze the data below and respond with actionable insights.

{prefetch_data}

Rules:
- Analyze the data above. Do NOT say you need more data — work with what you have.
- Include specific numbers, percentages, and issue keys in your analysis.
- Respond in the same language as the user (Vietnamese or English).
{git_note}
{memory_note}
"""

SCRUM_INSTRUCTIONS_NO_DATA = """\
You are an experienced Scrum Master assistant for project {project_key} ({jira_url}).

Use the available tools to gather data, then analyze and present insights.

Rules:
- Call 1-2 tools MAX, then analyze the data you received. Do NOT call the same tool twice.
- Provide actionable insights with specific data points.
- Respond in the same language as the user (Vietnamese or English).
{git_note}
{memory_note}
"""


def create_scrum_agent(client, middleware=None, prefetch_data: str = ""):
    """Create the Scrum Agent.

    If prefetch_data is provided, the agent receives data in its instructions
    and only has deep-dive tools (changelog, search). This prevents tool loops.
    If no prefetch_data, falls back to full tool set.
    """
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

    if prefetch_data:
        # Prefetch mode: data in prompt, no Jira tools needed
        tools = git_tools + memory_tools
        instructions = SCRUM_INSTRUCTIONS.format(
            project_key=settings.jira_project_key,
            jira_url=settings.jira_base_url,
            prefetch_data=prefetch_data,
            git_note=git_note,
            memory_note=memory_note,
        )
    else:
        # Fallback: full tools (when prefetch is not possible)
        from pile.tools.jira_tools import (
            jira_get_board,
            jira_get_changelog,
            jira_get_issue,
            jira_get_sprint_issues,
            jira_search,
        )
        tools = [
            jira_search, jira_get_issue, jira_get_board,
            jira_get_sprint_issues, jira_get_changelog,
        ] + git_tools + memory_tools
        instructions = SCRUM_INSTRUCTIONS_NO_DATA.format(
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
