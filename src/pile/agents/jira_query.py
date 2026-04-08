"""Jira Query Agent — search, view issues, changelog, curl commands."""

from __future__ import annotations

from pile.tools.jira_tools import (
    jira_curl_command, jira_get_changelog, jira_get_issue, jira_search,
    search_project_issues, get_current_sprint_info,
)

JIRA_QUERY_INSTRUCTIONS = """\
You are a Jira query specialist for project {project_key} at {jira_url}.

IMPORTANT: Always use project={project_key}. NEVER ask the user for project name.

Preferred tools (simple, no JQL needed):
- search_project_issues(status="In Progress") → issues by status
- search_project_issues(assignee="Tan Tran") → issues by assignee
- get_current_sprint_info() → current sprint + all issues
- jira_get_issue(issue_key="TETRA-42") → one specific issue

Advanced tools (only when needed):
- jira_search(jql="...") → custom JQL query
- jira_get_changelog(issue_key="...") → issue change history

Examples:
- "Issue đang In Progress?" → search_project_issues(status="In Progress")
- "Tân đang làm gì?" → search_project_issues(assignee="Tan Tran")
- "Sprint hiện tại?" → get_current_sprint_info()
- "TETRA-42 thế nào?" → jira_get_issue(issue_key="TETRA-42")

Rules:
- Call each tool ONCE. Do NOT repeat the same call.
- Respond in the same language as the user (Vietnamese or English).
"""


def create_jira_query_agent(client, middleware=None):
    """Create the Jira Query Agent (read-only + curl)."""
    from pile.config import settings

    return client.as_agent(
        name="JiraQueryAgent",
        description="Jira query: search issues, view issue details, changelog, curl commands",
        instructions=JIRA_QUERY_INSTRUCTIONS.format(
            project_key=settings.jira_project_key,
            jira_url=settings.jira_base_url,
        ),
        tools=[search_project_issues, get_current_sprint_info, jira_get_issue, jira_search, jira_get_changelog, jira_curl_command],
        middleware=middleware,
    )
