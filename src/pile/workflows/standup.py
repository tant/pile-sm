"""Sequential workflow — daily standup report pipeline."""

from __future__ import annotations

from pile.agents.git import create_git_agent
from pile.agents.jira_query import create_jira_query_agent
from pile.agents.scrum import create_scrum_agent
from pile.client import create_client


def create_workflow():
    """Build a Sequential workflow for standup reports.

    Pipeline: JiraQuery gathers data -> Git gathers commits (if available) -> Scrum synthesizes.
    """
    from agent_framework.orchestrations import SequentialBuilder

    client = create_client()

    jira_query = create_jira_query_agent(client)
    scrum = create_scrum_agent(client)
    git = create_git_agent(client)  # None if no repos configured

    participants = [jira_query]
    if git:
        participants.append(git)
    participants.append(scrum)

    return SequentialBuilder(participants=participants).build()
