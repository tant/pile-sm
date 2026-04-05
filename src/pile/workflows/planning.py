"""GroupChat workflow — sprint planning discussion."""

from __future__ import annotations

from pile.agents.epic import create_epic_agent
from pile.agents.git import create_git_agent
from pile.agents.jira_query import create_jira_query_agent
from pile.agents.scrum import create_scrum_agent
from pile.agents.sprint import create_sprint_agent
from pile.client import create_client


def create_workflow():
    """Build a GroupChat workflow for sprint planning.

    JiraQuery presents issue data, Sprint shows sprint status, Epic shows backlog/epics,
    Git (if available) shows code context, Scrum synthesizes and moderates.
    Round-robin speaker selection. Terminates after 8+ assistant messages.
    """
    from agent_framework.orchestrations import GroupChatBuilder, GroupChatState

    client = create_client()

    jira_query = create_jira_query_agent(client)
    sprint = create_sprint_agent(client)
    epic = create_epic_agent(client)
    scrum = create_scrum_agent(client)
    git = create_git_agent(client)  # None if no repos configured

    participants = [jira_query, sprint, epic]
    if git:
        participants.append(git)
    participants.append(scrum)

    def round_robin(state: GroupChatState) -> str:
        names = list(state.participants.keys())
        return names[state.current_round % len(names)]

    return GroupChatBuilder(
        participants=participants,
        selection_func=round_robin,
        termination_condition=lambda msgs: sum(1 for m in msgs if m.role == "assistant") >= 8,
    ).build()
