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
from pile.cache import get_cached, set_cached
from pile.router import smart_route

logger = logging.getLogger("pile.workflow")


def _detect_board_id():
    """Auto-detect default board ID from Jira on startup."""
    from pile.config import settings
    if settings.default_board_id:
        return settings.default_board_id
    if not settings.jira_project_key:
        return 0
    try:
        import httpx
        resp = httpx.get(
            f"{settings.jira_base_url}/rest/agile/1.0/board",
            params={"projectKeyOrId": settings.jira_project_key, "maxResults": 1},
            auth=(settings.jira_email, settings.jira_api_token),
            headers={"Accept": "application/json"},
            timeout=15.0,
        )
        resp.raise_for_status()
        boards = resp.json().get("values", [])
        if boards:
            board_id = boards[0]["id"]
            settings.default_board_id = board_id
            logger.info("Auto-detected board ID: %d (%s)", board_id, boards[0]["name"])
            return board_id
    except Exception as e:
        logger.warning("Failed to auto-detect board ID: %s", e)
    return 0


def create_workflow():
    """Build the routed workflow with all available agents.

    Returns (workflow, tracker).
    """
    client = create_client()
    tracker = ToolCallTracker()
    board_id = _detect_board_id()

    agents = {
        "triage": create_triage_agent(client, middleware=[tracker]),
        "jira_query": create_jira_query_agent(client, middleware=[tracker]),
        "jira_write": create_jira_write_agent(client, middleware=[tracker]),
        "board": create_board_agent(client, middleware=[tracker]),
        "sprint": create_sprint_agent(client, middleware=[tracker], board_id=board_id),
        "epic": create_epic_agent(client, middleware=[tracker], board_id=board_id),
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
    Each agent has its own session for conversation history.
    """

    def __init__(self, agents: dict, tracker: ToolCallTracker, client):
        self.agents = agents
        self.tracker = tracker
        self.client = client
        self._is_running = False
        self._sessions: dict[str, "AgentSession"] = {}

    def _get_session(self, agent_key: str):
        """Get or create a session for an agent (keeps conversation history)."""
        from agent_framework import AgentSession
        if agent_key not in self._sessions:
            self._sessions[agent_key] = AgentSession()
        return self._sessions[agent_key]

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
        from agent_framework._types import AgentResponseUpdate

        self._is_running = True
        try:
            # Check cache first (read-only queries)
            agent_key = smart_route(message)
            cached = get_cached(message)
            if cached and agent_key not in ("jira_write", "memory", "browser"):
                cached_text, cached_agent = cached
                logger.info("Cache hit: '%s' → %s", message[:50], cached_agent)
                yield WorkflowEvent.executor_invoked(f"{cached_agent} (cached)")
                yield WorkflowEvent.output(cached_agent, AgentResponseUpdate(text=cached_text))
                yield WorkflowEvent.executor_completed(cached_agent)
                return

            # Route to agent
            agent = self.agents.get(agent_key, self.agents["triage"])
            session = self._get_session(agent_key or "triage")
            logger.info("Route: '%s' → %s", message[:50], agent.name)

            # Emit executor_invoked
            yield WorkflowEvent.executor_invoked(agent.name)

            # Run the agent with session (keeps conversation history)
            full_text = ""
            if stream:
                result_stream = agent.run(message, stream=True, session=session)
                async for update in result_stream:
                    if update.text:
                        full_text += update.text
                        yield WorkflowEvent.output(agent.name, update)
                response = await result_stream.get_final_response()
            else:
                response = await agent.run(message, session=session)
                if response.text:
                    full_text = response.text
                    yield WorkflowEvent.output(agent.name, response)

            # Cache read-only responses
            if full_text and agent_key not in ("jira_write", "memory", "browser"):
                set_cached(message, full_text, agent.name)

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
