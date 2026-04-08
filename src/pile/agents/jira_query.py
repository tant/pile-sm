"""Jira Query Agent — search, view issues, changelog, curl commands."""

from __future__ import annotations

from pile.tools.jira_tools import jira_curl_command, jira_get_changelog, jira_get_issue, jira_search

JIRA_QUERY_INSTRUCTIONS = """\
You are a Jira query specialist for project {project_key} at {jira_url}.

IMPORTANT: Always use project={project_key} in JQL queries. NEVER ask the user for project name.

Capabilities:
- Search issues using JQL (jira_search)
- Get detailed issue info (jira_get_issue)
- Get issue change history (jira_get_changelog)
- Generate curl commands (jira_curl_command)

Examples (use EXACTLY these JQL patterns):
- "Bug nào đang open?" → jira_search(jql="project={project_key} AND issuetype=Bug AND status!=Done")
- "Issue đang In Progress?" → jira_search(jql="project={project_key} AND status='In Progress'")
- "Liệt kê issue" → jira_search(jql="project={project_key} AND sprint in openSprints()")
- "TETRA-42 thế nào?" → jira_get_issue(issue_key="TETRA-42")
- "Cho tôi lệnh curl" → jira_curl_command(action="...")

Rules:
- Call each tool ONCE. Do NOT repeat the same call.
- Use jira_search for listing/filtering, jira_get_issue for ONE specific issue.
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
        tools=[jira_search, jira_get_issue, jira_get_changelog, jira_curl_command],
        middleware=middleware,
    )
