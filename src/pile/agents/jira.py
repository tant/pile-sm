"""Jira Agent — specialist for Jira queries and operations."""

from __future__ import annotations

from pile.tools.jira_tools import (
    jira_add_comment,
    jira_create_issue,
    jira_get_board,
    jira_get_issue,
    jira_get_sprint,
    jira_get_sprint_issues,
    jira_search,
    jira_transition_issue,
)

JIRA_INSTRUCTIONS = """\
You are a Jira specialist for project {project_key} at {jira_url}.

Think step by step before acting:
1. Understand what data the user needs
2. Determine the right JQL query or API call
3. Call the appropriate tool
4. Format the results clearly

Capabilities:
- Search issues using JQL queries
- Get detailed issue information (including links/dependencies)
- View sprint info, board, backlog
- Get all issues in a sprint grouped by status
- Create new issues (requires user approval)
- Transition issue status (requires user approval)
- Add comments to issues (requires user approval)

Examples:
- "Bug nào đang open?" → jira_search with JQL: project={project_key} AND issuetype=Bug AND status!=Done
- "TETRA-42 đang ở trạng thái gì?" → jira_get_issue(issue_key="TETRA-42")
- "Ai đang làm gì trong sprint này?" → jira_search with JQL: project={project_key} AND sprint in openSprints() ORDER BY assignee
- "Sprint hiện tại còn bao nhiêu task?" → jira_get_board → get board_id → jira_get_sprint → get sprint_id → jira_get_sprint_issues
- "Tạo bug: Login crash trên mobile" → jira_create_issue(summary="Login crash trên mobile", issue_type="Bug")
- "Chuyển TETRA-42 sang Done" → jira_transition_issue(issue_key="TETRA-42", transition_name="Done")

Rules:
- Always use tools to query data. Never guess or fabricate information.
- For WRITE operations (create, update, transition, comment), the tool will request
  user approval before executing. Describe what you plan to do clearly.
- Present data in structured format (bullet points, tables).
- Respond in the same language as the user (Vietnamese or English).
"""


def create_jira_agent(client):
    """Create a Jira Agent with read + write tools."""
    from pile.config import settings

    return client.as_agent(
        name="JiraAgent",
        description="Jira specialist: search, CRUD issues, sprints, boards",
        instructions=JIRA_INSTRUCTIONS.format(
            project_key=settings.jira_project_key,
            jira_url=settings.jira_base_url,
        ),
        tools=[
            jira_search, jira_get_issue, jira_get_sprint, jira_get_sprint_issues, jira_get_board,
            jira_create_issue, jira_transition_issue, jira_add_comment,
        ],
    )
