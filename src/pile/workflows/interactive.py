"""Interactive Handoff workflow — the primary Q&A interaction mode."""

from __future__ import annotations

from pile.agents.board import create_board_agent
from pile.agents.epic import create_epic_agent
from pile.agents.git import create_git_agent
from pile.agents.jira_query import create_jira_query_agent
from pile.agents.jira_write import create_jira_write_agent
from pile.agents.scrum import create_scrum_agent
from pile.agents.sprint import create_sprint_agent
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
    jira_query = create_jira_query_agent(client, middleware=[tracker])
    jira_write = create_jira_write_agent(client, middleware=[tracker])
    board = create_board_agent(client, middleware=[tracker])
    sprint = create_sprint_agent(client, middleware=[tracker])
    epic = create_epic_agent(client, middleware=[tracker])
    scrum = create_scrum_agent(client, middleware=[tracker])
    git = create_git_agent(client, middleware=[tracker])  # None if no repos configured

    participants = [triage, jira_query, jira_write, board, sprint, epic, scrum]
    if git:
        participants.append(git)

    # Triage routes to all agents
    triage_targets = [a for a in [jira_query, jira_write, board, sprint, epic, git, scrum] if a]

    builder = (
        HandoffBuilder(
            name="pile_sm",
            participants=participants,
        )
        .with_start_agent(triage)
        .add_handoff(triage, triage_targets)
        # Jira agents can escalate to Scrum or go back to Triage
        .add_handoff(jira_query, [triage, scrum])
        .add_handoff(jira_write, [triage, scrum])
        # Board/Sprint/Epic can go to each other or back to Triage
        .add_handoff(board, [triage, sprint, scrum])
        .add_handoff(sprint, [triage, board, scrum])
        .add_handoff(epic, [triage, sprint, scrum])
        # Scrum can query data from Jira agents
        .add_handoff(scrum, [a for a in [triage, jira_query, board, sprint, epic, git] if a])
    )

    if git:
        builder = builder.add_handoff(git, [triage, scrum])

    return builder.build(), tracker
