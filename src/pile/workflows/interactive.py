"""Interactive Handoff workflow — the primary Q&A interaction mode."""

from __future__ import annotations

from pile.agents.git import create_git_agent
from pile.agents.jira import create_jira_agent
from pile.agents.scrum import create_scrum_agent
from pile.agents.triage import create_triage_agent
from pile.client import create_client
from pile.middleware import ToolCallTracker


def create_workflow():
    """Build the Handoff workflow with all available agents.

    Returns (workflow, tracker) — tracker can be used to inspect tool calls after each run.
    """
    from agent_framework.orchestrations import HandoffBuilder

    client = create_client()
    tracker = ToolCallTracker()

    triage = create_triage_agent(client, middleware=[tracker])
    jira = create_jira_agent(client, middleware=[tracker])
    scrum = create_scrum_agent(client, middleware=[tracker])
    git = create_git_agent(client, middleware=[tracker])  # None if no repos configured

    participants = [triage, jira, scrum]
    if git:
        participants.append(git)

    builder = (
        HandoffBuilder(
            name="pile_sm",
            participants=participants,
        )
        .with_start_agent(triage)
        .add_handoff(triage, [a for a in [jira, git, scrum] if a])
        .add_handoff(scrum, [a for a in [jira, git, triage] if a])
        .add_handoff(jira, [a for a in [triage, scrum] if a])
    )

    if git:
        builder = builder.add_handoff(git, [triage, scrum])

    return builder.build(), tracker
