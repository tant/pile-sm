"""Git Agent — specialist for Git repository queries."""

from __future__ import annotations

from pile.config import settings
from pile.tools.git_tools import git_blame, git_branch_list, git_diff, git_log, git_show

GIT_INSTRUCTIONS = """\
You are a Git specialist. You help users understand their code repository history and changes.

Think step by step:
1. Identify what the user wants to know about the repository
2. Determine which tool and parameters to use
3. Call the tool and format results clearly

Available repositories:
{repos}

Capabilities:
- View commit history with filters (author, date, branch)
- Compare branches and commits (diff)
- List and describe branches
- Show commit details
- Link commits to Jira issues via commit message patterns (e.g. TETRA-123)
- Show who last modified lines of a file (blame)

Rules:
- Always use tools to query data. Never guess.
- When showing diffs, summarize the changes rather than dumping raw output.
- Respond in the same language as the user (Vietnamese or English).
"""


def create_git_agent(client):
    """Create a Git Agent. Returns None if no git repos are configured."""
    repos = settings.git_repo_list
    if not repos:
        return None

    repos_str = "\n".join(
        f"- {r.path}" + (" (private, credentials configured)" if r.has_credentials else "")
        for r in repos
    )

    return client.as_agent(
        name="GitAgent",
        description="Git specialist: commits, branches, diffs, blame",
        instructions=GIT_INSTRUCTIONS.format(repos=repos_str),
        tools=[git_log, git_diff, git_branch_list, git_show, git_blame],

    )
