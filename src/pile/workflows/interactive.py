"""Interactive workflow — deterministic router + focused agent execution.

Replaces HandoffBuilder (which added 7 transfer tools to Triage, overwhelming
the 9B model) with keyword-based routing. Each agent runs independently with
only its own 3-5 tools visible.
"""

from __future__ import annotations

import logging

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
from pile.router import route_query

logger = logging.getLogger("pile.workflow")


def create_workflow():
    """Build the routed workflow with all available agents.

    Returns (workflow, tracker).
    """
    client = create_client()
    tracker = ToolCallTracker()

    agents = {
        "triage": create_triage_agent(client, middleware=[tracker]),
        "jira_query": create_jira_query_agent(client, middleware=[tracker]),
        "jira_write": create_jira_write_agent(client, middleware=[tracker]),
        "board": create_board_agent(client, middleware=[tracker]),
        "sprint": create_sprint_agent(client, middleware=[tracker]),
        "epic": create_epic_agent(client, middleware=[tracker]),
        "scrum": create_scrum_agent(client, middleware=[tracker]),
    }

    git = create_git_agent(client, middleware=[tracker])
    if git:
        agents["git"] = git

    workflow = RoutedWorkflow(agents, tracker, client)
    return workflow, tracker


class RoutedWorkflow:
    """Simple workflow: deterministic route → run single agent → return.

    No HandoffBuilder, no transfer tools. Each agent only sees its own tools.
    For ambiguous queries, falls back to Triage agent (memory/browser ops).
    """

    def __init__(self, agents: dict, tracker: ToolCallTracker, client):
        self.agents = agents
        self.tracker = tracker
        self.client = client
        self._is_running = False

    def _reset_running_flag(self):
        self._is_running = False

    def run(self, message=None, *, stream=False, responses=None,
            include_status_events=False, **kwargs):
        """Run the workflow — route query to agent, execute, return stream."""
        if responses is not None:
            # Handle pending responses (approval, follow-up)
            return self._run_with_responses(responses, stream=stream)
        return self._run_query(message, stream=stream)

    async def _run_query(self, message: str, stream: bool = False):
        """Route and execute a user query."""
        from agent_framework._workflows._events import WorkflowEvent

        self._is_running = True
        try:
            # Deterministic routing
            agent_key = route_query(message)
            if agent_key and agent_key in self.agents:
                agent = self.agents[agent_key]
                logger.info("Route: '%s' → %s (keyword match)", message[:50], agent.name)
            else:
                # Fallback to triage for ambiguous queries
                agent = self.agents["triage"]
                logger.info("Route: '%s' → %s (fallback)", message[:50], agent.name)

            # Emit executor_invoked
            yield WorkflowEvent.executor_invoked(agent.name)

            # Run the agent
            if stream:
                result_stream = agent.run(message, stream=True)
                async for update in result_stream:
                    if update.text:
                        yield WorkflowEvent.output(agent.name, update)
                response = await result_stream.get_final_response()
            else:
                response = await agent.run(message)
                if response.text:
                    yield WorkflowEvent.output(agent.name, response)

            # Emit executor_completed
            yield WorkflowEvent.executor_completed(agent.name)

        except Exception as e:
            logger.exception("Workflow error: %s", e)
            yield WorkflowEvent.executor_failed(agent.name if 'agent' in dir() else "unknown", str(e))
        finally:
            self._is_running = False

    async def _run_with_responses(self, responses: dict, stream: bool = False):
        """Handle approval/follow-up responses — not used in simple routing."""
        from agent_framework._workflows._events import WorkflowEvent

        # For now, yield nothing — approval handling stays in chainlit_app
        return
        yield  # Make this an async generator
