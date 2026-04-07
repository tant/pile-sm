"""Interactive workflow — deterministic router + focused agent execution.

Replaces HandoffBuilder (which added 7 transfer tools to Triage, overwhelming
the 9B model) with keyword-based routing. Each agent runs independently with
only its own 3-5 tools visible.

Recovery: if the first agent produces a poor result (empty, no tool calls,
all errors), the workflow re-routes to a fallback agent automatically.
"""

from __future__ import annotations

import asyncio
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
from pile.prefetch import prefetch_scrum_data
from pile.context import recall, learn

logger = logging.getLogger("pile.workflow")

# --- Recovery: fallback chains and failure detection ---

# When agent A fails, try the first available fallback in order.
_FALLBACK_CHAINS: dict[str, list[str]] = {
    "triage": ["jira_query", "scrum", "sprint"],
    "board": ["sprint", "jira_query"],
    "sprint": ["scrum", "jira_query"],
    "epic": ["sprint", "jira_query"],
    "scrum": ["jira_query", "sprint"],
    "jira_query": ["scrum", "sprint"],
    "git": ["jira_query"],
}

# Never retry write operations — too risky.
_NO_RETRY = {"jira_write"}


def _is_error_result(result: str | None) -> bool:
    """Check if a tool call result looks like an error."""
    if not result:
        return False
    r = result.lower()
    return any(s in r for s in ("error", "404", "400", "401", "403", "timeout", "not found", "failed"))


def _detect_failure(
    full_text: str, tool_calls: list, agent_key: str,
    has_prefetch: bool = False,
) -> bool:
    """Detect if an agent run produced a poor result that warrants re-routing."""
    text = (full_text or "").strip()

    # Empty or trivially short response — something went wrong.
    if len(text) < 20:
        return True

    # Agent has tools but made zero calls — it didn't know what to do.
    # Exceptions: triage (greetings), scrum with prefetch (data in prompt, no tools needed).
    if not tool_calls and agent_key != "triage" and not has_prefetch:
        return True

    # All tool calls returned errors — wrong agent for this query.
    if tool_calls and all(_is_error_result(c.result) for c in tool_calls):
        return True

    return False


def _get_fallback(agent_key: str, available_agents: set[str]) -> str | None:
    """Get the first available fallback agent for a failed agent."""
    chain = _FALLBACK_CHAINS.get(agent_key, [])
    for candidate in chain:
        if candidate in available_agents and candidate != agent_key:
            return candidate
    return None


# --- Workflow setup ---


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

    workflow = RoutedWorkflow(agents, tracker, client, board_id=board_id)
    return workflow, tracker


class RoutedWorkflow:
    """Deterministic route → run agent → recovery if needed → return.

    Each agent only sees its own tools. If an agent produces a poor result
    (empty, no tool calls, all errors), the workflow re-routes to a fallback
    agent from _FALLBACK_CHAINS. Max 1 retry to keep things fast.
    """

    def __init__(self, agents: dict, tracker: ToolCallTracker, client, board_id: int = 0):
        self.agents = agents
        self.tracker = tracker
        self.client = client
        self.board_id = board_id
        self._is_running = False
        self._sessions: dict = {}
        self.last_agent_key: str = ""

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
            return self._run_with_responses(responses, stream=stream)
        return self._run_query(message, stream=stream)

    async def _execute_agent(self, agent_key: str, message: str):
        """Run a single agent, yielding streaming updates.

        Yields AgentResponseUpdate objects as they arrive.
        Caller accumulates full_text for recovery detection.
        """
        agent = self.agents.get(agent_key, self.agents["triage"])
        session = self._get_session(agent_key)

        # Drain any leftover tool calls from previous runs.
        self.tracker.drain()

        result_stream = agent.run(message, stream=True, session=session)
        async for update in result_stream:
            yield update
        await result_stream.get_final_response()

    async def _stream_agent_events(self, agent_key: str, message: str, tool_events: asyncio.Queue):
        """Stream an agent run, yielding WorkflowEvents for text and tool calls.

        After iteration completes, read self._last_full_text and
        self._last_tool_calls for recovery detection.
        """
        from agent_framework._workflows._events import WorkflowEvent

        agent = self.agents.get(agent_key, self.agents["triage"])
        agent_name = agent.name
        self.last_agent_key = agent_key.replace("_scrum_prefetch", "scrum")
        logger.info("Route: '%s' -> %s", message[:50], agent_name)
        yield WorkflowEvent.executor_invoked(agent_name)

        self._last_full_text = ""

        async for update in self._execute_agent(agent_key, message):
            # Drain tool events that arrived during this iteration
            while not tool_events.empty():
                try:
                    evt = tool_events.get_nowait()
                    if evt[0] == "tool_start":
                        yield WorkflowEvent.emit(agent_name, {"type": "tool_start", "name": evt[1], "args": evt[2]})
                    elif evt[0] == "tool_end":
                        yield WorkflowEvent.emit(agent_name, {"type": "tool_end", "record": evt[1]})
                except asyncio.QueueEmpty:
                    break

            if update.text:
                self._last_full_text += update.text
                yield WorkflowEvent.output(agent_name, update)

        # Drain remaining tool events after stream ends
        while not tool_events.empty():
            try:
                evt = tool_events.get_nowait()
                if evt[0] == "tool_start":
                    yield WorkflowEvent.emit(agent_name, {"type": "tool_start", "name": evt[1], "args": evt[2]})
                elif evt[0] == "tool_end":
                    yield WorkflowEvent.emit(agent_name, {"type": "tool_end", "record": evt[1]})
            except asyncio.QueueEmpty:
                break

        self._last_tool_calls = self.tracker.drain()

    async def _run_query(self, message: str, stream: bool = False):
        """Route and execute a user query, with recovery on failure."""
        from agent_framework._workflows._events import WorkflowEvent
        from agent_framework._types import AgentResponseUpdate, Content

        self._is_running = True
        agent_name = "unknown"
        try:
            # Check cache first (read-only queries)
            agent_key = smart_route(message)
            cached = get_cached(message)
            if cached and agent_key not in ("jira_write", "memory", "browser"):
                cached_text, cached_agent = cached
                logger.info("Cache hit: '%s' -> %s", message[:50], cached_agent)
                yield WorkflowEvent.executor_invoked(f"{cached_agent} (cached)")
                yield WorkflowEvent.output(cached_agent, AgentResponseUpdate(text=cached_text))
                yield WorkflowEvent.executor_completed(cached_agent)
                return

            # --- Prefetch data for scrum queries ---
            has_prefetch = False
            if agent_key == "scrum" and self.board_id:
                data = prefetch_scrum_data(message, self.board_id)
                if data:
                    has_prefetch = True
                    self.agents["_scrum_prefetch"] = create_scrum_agent(
                        self.client,
                        middleware=[self.tracker],
                        prefetch_data=data,
                    )
                    agent_key = "_scrum_prefetch"
                    self._sessions.pop("_scrum_prefetch", None)

            # --- Auto-recall: inject memory context ---
            enriched_message = message
            if agent_key not in ("triage", "memory"):
                memory_context = recall(message)
                if memory_context:
                    enriched_message = f"{message}\n\n{memory_context}"

            # --- Wire tool event callbacks ---
            tool_events: asyncio.Queue = asyncio.Queue()

            async def _on_tool_start(name, args):
                await tool_events.put(("tool_start", name, args))

            async def _on_tool_end(record):
                await tool_events.put(("tool_end", record))

            self.tracker.on_tool_start = _on_tool_start
            self.tracker.on_tool_end = _on_tool_end

            # --- Stream first agent ---
            async for event in self._stream_agent_events(agent_key, enriched_message, tool_events):
                yield event
            full_text = self._last_full_text
            tool_calls = self._last_tool_calls

            # --- Recovery check ---
            is_failure = (
                agent_key not in _NO_RETRY
                and _detect_failure(full_text, tool_calls, agent_key, has_prefetch)
            )

            if is_failure:
                fallback_key = _get_fallback(agent_key, set(self.agents.keys()))
                if fallback_key:
                    agent = self.agents.get(agent_key, self.agents["triage"])
                    logger.info(
                        "Recovery: %s failed (text=%d, tools=%d), retrying -> %s",
                        agent_key, len(full_text), len(tool_calls), fallback_key,
                    )
                    yield WorkflowEvent.executor_completed(agent.name)

                    # --- Fallback attempt ---
                    async for event in self._stream_agent_events(fallback_key, enriched_message, tool_events):
                        yield event
                    full_text = self._last_full_text
                    tool_calls = self._last_tool_calls

                    if full_text and len(full_text.strip()) > 20:
                        original_key = agent_key.replace("_scrum_prefetch", "scrum")
                        learn(
                            message,
                            f"Query '{message}' failed on {original_key}, "
                            f"succeeded on {fallback_key}.",
                        )

                    agent_key = fallback_key
                    is_failure = False

            # Cache read-only responses
            agent = self.agents.get(agent_key, self.agents["triage"])
            agent_name = agent.name
            if full_text and agent_key not in ("jira_write", "memory", "browser"):
                if not is_failure:
                    set_cached(message, full_text, agent_name)

            yield WorkflowEvent.executor_completed(agent_name)

        except Exception as e:
            logger.exception("Workflow error: %s", e)
            yield WorkflowEvent.executor_failed(agent_name, str(e))
        finally:
            self._is_running = False
            self.tracker.on_tool_start = None
            self.tracker.on_tool_end = None

    async def _run_with_responses(self, responses: dict, stream: bool = False):
        """Handle approval/follow-up responses — not used in simple routing."""
        # For now, yield nothing — approval handling stays in chainlit_app
        return
        yield  # Make this an async generator
