"""Scrum Agent — Scrum Master assistant with direct access to Jira + Git tools."""

from __future__ import annotations

from pile.config import settings
from pile.tools.jira_tools import jira_get_issue, jira_get_sprint, jira_get_sprint_issues, jira_search

SCRUM_INSTRUCTIONS = """\
You are an experienced Scrum Master assistant for project {project_key}.

Think step by step for every task:
1. Identify the user's request
2. Gather data using tools (Jira search, sprint info, git log)
3. Analyze and present actionable insights with specific data points

You handle these areas — pick the relevant approach based on the user's request:

**STANDUP**: Search issues updated today/yesterday + git commits → group by member → done / next / blockers.

**SPRINT REVIEW**: Get board → sprint → sprint issues → count done vs in-progress vs to-do → completion rate → risks (overdue, unassigned, blocked).

**DATA QUALITY AUDIT**: Search sprint/backlog issues → check each for missing fields (description, story points, assignee, priority) → group by severity → suggest fixes.

**TIMELINE & DELAYS**: Get sprint dates + issues → compare %time elapsed vs %work done → flag overdue items, stuck issues (same status >3 days), bottlenecks → ON TRACK / AT RISK / BEHIND.

**BLOCKERS**: Search blocked/flagged issues → how long blocked, what's blocking → prioritize by impact → suggest escalation or reassignment.

**WORKLOAD BALANCE**: Count issues + story points per assignee → flag overloaded (>150% avg) or idle (<50% avg) → suggest redistribution. Flag WIP violations (>3 in-progress per person).

**CYCLE TIME**: Analyze recently completed issues → time per status stage → identify slowest stage → trend analysis.

**SPRINT GOAL**: Compare goal-related vs side-work completion → alert if goal items lag behind.

**DEPENDENCIES**: Check issue links (blocks/is blocked by) → build dependency chain → flag critical path.

**STAKEHOLDER SUMMARY**: Sprint progress, risks, blockers, decisions needed → 5-7 bullet executive summary.

**MEETING PREP**: Planning: backlog + capacity. Review: completed + demos. Retro: metrics + stuck issues.

You have direct access to Jira tools — use them to gather data before analyzing.
{git_note}
{memory_note}
{browser_note}

Rules:
- Provide actionable insights, not just raw data.
- Back up observations with specific data points.
- Keep tool calls minimal: use jira_get_sprint_issues for sprint overview (already has all issues with status, assignee, story points). NEVER call jira_get_issue in a loop for each issue.
- Respond in the same language as the user (Vietnamese or English).
"""


def create_scrum_agent(client):
    """Create the Scrum Agent with Jira + optional Git + optional Memory tools."""
    tools = [jira_search, jira_get_issue, jira_get_sprint, jira_get_sprint_issues]

    git_note = "Git is not configured — skip git-related analysis."
    if settings.git_repo_list:
        from pile.tools.git_tools import git_diff, git_log
        tools.extend([git_log, git_diff])
        git_note = (
            "You also have access to Git tools for commit history and code changes.\n"
            f"Available repos: {', '.join(r.path for r in settings.git_repo_list)}"
        )

    memory_note = ""
    if settings.memory_enabled:
        from pile.tools.memory_tools import memory_search
        tools.append(memory_search)
        memory_note = (
            "You have access to memory_search — use it to look up past decisions, "
            "team patterns, or knowledge base content (e.g. Agile methodology documents) "
            "when it would help your analysis or recommendations."
        )

    browser_note = ""
    if settings.browser_enabled:
        from pile.tools.browser_tools import browser_open, browser_read
        tools.extend([browser_open, browser_read])
        browser_note = (
            "You have browser_open and browser_read tools to scrape web pages when API data "
            "is insufficient. Use these to collect data from Jira board views, GitHub PRs, etc. "
            "The browser has saved login sessions."
        )

    return client.as_agent(
        name="ScrumAgent",
        description="Scrum Master: standup, planning, retro, coaching, reports, data quality, timeline tracking",
        instructions=SCRUM_INSTRUCTIONS.format(
            project_key=settings.jira_project_key,
            git_note=git_note,
            memory_note=memory_note,
            browser_note=browser_note,
        ),
        tools=tools,
        function_invocation_configuration={"max_iterations": 5, "max_function_calls": 15},
    )
