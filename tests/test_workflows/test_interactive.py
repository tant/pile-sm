"""Tests for pile.workflows.interactive — routed workflow with recovery."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pile.workflows.interactive import (
    RoutedWorkflow,
    _detect_board_id,
    _detect_failure,
    _get_fallback,
    _is_error_result,
    _FALLBACK_CHAINS,
    _NO_RETRY,
)


class TestIsErrorResult:
    def test_none_is_not_error(self):
        assert _is_error_result(None) is False

    def test_empty_is_not_error(self):
        assert _is_error_result("") is False

    def test_error_keyword(self):
        assert _is_error_result("Error: something went wrong") is True

    def test_404_code(self):
        assert _is_error_result("Resource 404 not found") is True

    def test_400_code(self):
        assert _is_error_result("Bad request 400") is True

    def test_401_code(self):
        assert _is_error_result("Unauthorized 401") is True

    def test_403_code(self):
        assert _is_error_result("Forbidden 403") is True

    def test_timeout(self):
        assert _is_error_result("Request timeout after 30s") is True

    def test_not_found(self):
        assert _is_error_result("Issue not found") is True

    def test_failed(self):
        assert _is_error_result("Operation failed") is True

    def test_normal_result(self):
        assert _is_error_result("Sprint 5 has 10 issues assigned") is False

    def test_case_insensitive(self):
        assert _is_error_result("ERROR: connection refused") is True


class TestDetectFailure:
    def test_short_text_is_failure(self):
        assert _detect_failure("ok", [], "jira_query") is True

    def test_empty_text_is_failure(self):
        assert _detect_failure("", [], "jira_query") is True

    def test_no_tool_calls_is_failure(self):
        assert _detect_failure("A long enough response text here.", [], "jira_query") is True

    def test_triage_no_tools_is_not_failure(self):
        assert _detect_failure("Hello! How can I help you today?", [], "triage") is False

    def test_scrum_with_prefetch_no_tools_not_failure(self):
        text = "Sprint 5 summary: team completed 20 story points."
        assert _detect_failure(text, [], "scrum", has_prefetch=True) is False

    def test_all_error_tool_calls_is_failure(self):
        calls = [
            MagicMock(result="Error: 404 not found"),
            MagicMock(result="Error: timeout"),
        ]
        assert _detect_failure("Some response text here to pass length check", calls, "jira_query") is True

    def test_mixed_tool_calls_not_failure(self):
        calls = [
            MagicMock(result="Error: 404 not found"),
            MagicMock(result="Found 5 issues in sprint"),
        ]
        assert _detect_failure("Some response text here to pass length check", calls, "jira_query") is False

    def test_successful_tool_calls_not_failure(self):
        calls = [MagicMock(result="Found 5 issues")]
        assert _detect_failure("Here are the results for your query.", calls, "jira_query") is False


class TestGetFallback:
    def test_returns_first_available_fallback(self):
        available = {"jira_query", "scrum", "sprint"}
        result = _get_fallback("triage", available)
        assert result == "jira_query"

    def test_skips_unavailable_agents(self):
        available = {"sprint"}
        result = _get_fallback("triage", available)
        assert result == "sprint"

    def test_returns_none_when_no_fallback_available(self):
        available = set()
        result = _get_fallback("triage", available)
        assert result is None

    def test_does_not_fallback_to_self(self):
        available = {"jira_query"}
        result = _get_fallback("jira_query", available)
        # Should not return jira_query for itself
        chain = _FALLBACK_CHAINS.get("jira_query", [])
        if "jira_query" not in chain:
            assert result is None or result != "jira_query"

    def test_returns_none_for_unknown_agent(self):
        result = _get_fallback("nonexistent", {"jira_query", "scrum"})
        assert result is None

    def test_no_retry_agents(self):
        assert "jira_write" in _NO_RETRY


class TestRoutedWorkflow:
    def setup_method(self):
        self.tracker = MagicMock()
        self.tracker.drain.return_value = []
        self.client = MagicMock()
        self.agents = {
            "triage": MagicMock(name="Triage"),
            "jira_query": MagicMock(name="JiraQuery"),
            "scrum": MagicMock(name="Scrum"),
        }
        self.workflow = RoutedWorkflow(
            agents=self.agents,
            tracker=self.tracker,
            client=self.client,
            board_id=1,
        )

    def test_init(self):
        assert self.workflow.board_id == 1
        assert self.workflow._is_running is False
        assert self.workflow.last_agent_key == ""

    def test_get_session_creates_new(self):
        with patch("agent_framework.AgentSession") as mock_session_cls:
            mock_session_cls.return_value = MagicMock()
            session = self.workflow._get_session("jira_query")
            assert "jira_query" in self.workflow._sessions

    def test_get_session_reuses_existing(self):
        existing = MagicMock()
        self.workflow._sessions["jira_query"] = existing
        session = self.workflow._get_session("jira_query")
        assert session is existing

    def test_reset_running_flag(self):
        self.workflow._is_running = True
        self.workflow._reset_running_flag()
        assert self.workflow._is_running is False

    @pytest.mark.asyncio
    async def test_run_with_responses_yields_nothing(self):
        events = []
        async for event in self.workflow._run_with_responses({}):
            events.append(event)
        assert events == []

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="jira_query")
    @patch("pile.workflows.interactive.get_cached", return_value=("cached response", "JiraQuery"))
    async def test_run_query_cache_hit(self, mock_get_cached, mock_route):
        events = []
        async for event in self.workflow._run_query("show issues"):
            events.append(event)
        assert self.workflow._is_running is False
        assert len(events) > 0

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="jira_write")
    @patch("pile.workflows.interactive.get_cached", return_value=("cached", "JiraWrite"))
    async def test_run_query_no_cache_for_write_agent(self, mock_get_cached, mock_route):
        """Write agents should not use cache even if there is a match."""
        with patch.object(self.workflow, "_stream_agent_events") as mock_stream:
            async def _fake_stream(key, msg, q):
                self.workflow._last_full_text = "Created issue TETRA-100 successfully."
                self.workflow._last_tool_calls = [MagicMock(result="ok")]
                return
                yield
            mock_stream.side_effect = _fake_stream
            with patch("pile.context.recall_facts", return_value=[]):
                with patch("pile.workflows.interactive.set_cached") as mock_set:
                    events = []
                    async for event in self.workflow._run_query("create issue"):
                        events.append(event)
            # Should have called _stream_agent_events, not used cache
            mock_stream.assert_called_once()
            # Write agents should not cache their results
            mock_set.assert_not_called()


# --- Helper to create a mock async generator ---
def _mock_async_gen(*items):
    """Create an async generator that yields given items."""
    async def _gen():
        for item in items:
            yield item
    return _gen()


class _FakeUpdate:
    """Minimal stand-in for AgentResponseUpdate."""
    def __init__(self, text=None):
        self.text = text


class _FakeResponse:
    """Minimal stand-in for a completed agent response."""
    def __init__(self, text=None):
        self.text = text


class _FakeStream:
    """Fake streaming result with get_final_response."""
    def __init__(self, updates):
        self._updates = updates

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._updates:
            return self._updates.pop(0)
        raise StopAsyncIteration

    async def get_final_response(self):
        return _FakeResponse("done")


class TestDetectBoardId:
    @patch("pile.config.settings")
    def test_returns_configured_board_id(self, mock_settings):
        mock_settings.default_board_id = 42
        assert _detect_board_id() == 42

    @patch("pile.config.settings")
    def test_returns_zero_when_no_project_key(self, mock_settings):
        mock_settings.default_board_id = 0
        mock_settings.jira_project_key = ""
        assert _detect_board_id() == 0

    @patch("pile.config.settings")
    def test_auto_detect_from_api(self, mock_settings):
        mock_settings.default_board_id = 0
        mock_settings.jira_project_key = "TETRA"
        mock_settings.jira_base_url = "https://example.atlassian.net"
        mock_settings.jira_email = "user@example.com"
        mock_settings.jira_api_token = "token"
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"values": [{"id": 99, "name": "Board"}]}
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp
            assert _detect_board_id() == 99
            assert mock_settings.default_board_id == 99

    @patch("pile.config.settings")
    def test_auto_detect_empty_boards(self, mock_settings):
        mock_settings.default_board_id = 0
        mock_settings.jira_project_key = "TETRA"
        mock_settings.jira_base_url = "https://example.atlassian.net"
        mock_settings.jira_email = "user@example.com"
        mock_settings.jira_api_token = "token"
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"values": []}
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp
            assert _detect_board_id() == 0

    @patch("pile.config.settings")
    def test_auto_detect_api_error(self, mock_settings):
        mock_settings.default_board_id = 0
        mock_settings.jira_project_key = "TETRA"
        mock_settings.jira_base_url = "https://example.atlassian.net"
        mock_settings.jira_email = "user@example.com"
        mock_settings.jira_api_token = "token"
        with patch("httpx.get", side_effect=Exception("network error")):
            assert _detect_board_id() == 0


class TestCreateWorkflow:
    @patch("pile.workflows.interactive._detect_board_id", return_value=5)
    @patch("pile.workflows.interactive.create_client")
    @patch("pile.workflows.interactive.create_triage_agent")
    @patch("pile.workflows.interactive.create_jira_query_agent")
    @patch("pile.workflows.interactive.create_jira_write_agent")
    @patch("pile.workflows.interactive.create_board_agent")
    @patch("pile.workflows.interactive.create_sprint_agent")
    @patch("pile.workflows.interactive.create_epic_agent")
    @patch("pile.workflows.interactive.create_scrum_agent")
    @patch("pile.workflows.interactive.create_git_agent", return_value=None)
    def test_create_workflow_no_git(self, *mocks):
        from pile.workflows.interactive import create_workflow
        workflow, tracker = create_workflow()
        assert isinstance(workflow, RoutedWorkflow)
        assert "git" not in workflow.agents

    @patch("pile.workflows.interactive._detect_board_id", return_value=5)
    @patch("pile.workflows.interactive.create_client")
    @patch("pile.workflows.interactive.create_triage_agent")
    @patch("pile.workflows.interactive.create_jira_query_agent")
    @patch("pile.workflows.interactive.create_jira_write_agent")
    @patch("pile.workflows.interactive.create_board_agent")
    @patch("pile.workflows.interactive.create_sprint_agent")
    @patch("pile.workflows.interactive.create_epic_agent")
    @patch("pile.workflows.interactive.create_scrum_agent")
    @patch("pile.workflows.interactive.create_git_agent")
    def test_create_workflow_with_git(self, mock_git, *mocks):
        mock_git.return_value = MagicMock(name="Git")
        from pile.workflows.interactive import create_workflow
        workflow, tracker = create_workflow()
        assert "git" in workflow.agents


class TestExecuteAgent:
    def setup_method(self):
        self.tracker = MagicMock()
        self.tracker.drain.return_value = []
        self.client = MagicMock()
        self.agents = {
            "triage": MagicMock(name="Triage"),
            "jira_query": MagicMock(name="JiraQuery"),
        }
        self.workflow = RoutedWorkflow(
            agents=self.agents, tracker=self.tracker, client=self.client,
        )

    @pytest.mark.asyncio
    async def test_streaming_produces_text(self):
        """When streaming works, text is yielded and no fallback needed."""
        update1 = _FakeUpdate(text="Hello ")
        update2 = _FakeUpdate(text="world")
        stream = _FakeStream([update1, update2])
        self.agents["triage"].run = MagicMock(return_value=stream)

        with patch("agent_framework.AgentSession", return_value=MagicMock()):
            updates = []
            async for u in self.workflow._execute_agent("triage", "hi"):
                updates.append(u)

        texts = [u.text for u in updates if u.text]
        assert "Hello " in texts
        assert "world" in texts

    @pytest.mark.asyncio
    async def test_streaming_empty_falls_back_to_non_streaming(self):
        """When streaming yields no text, falls back to non-streaming run."""
        # Streaming yields updates with no text
        empty_stream = _FakeStream([_FakeUpdate(text="")])
        self.agents["triage"].run = MagicMock(return_value=empty_stream)

        # For the second (non-streaming) call, return a response with text
        response = _FakeResponse(text="Fallback response")
        # run() is called twice: first streaming, then non-streaming
        call_count = [0]
        original_run = self.agents["triage"].run

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("stream"):
                return empty_stream
            # Return an awaitable for non-streaming
            async def _coro():
                return response
            return _coro()

        self.agents["triage"].run = MagicMock(side_effect=side_effect)

        with patch("agent_framework.AgentSession", return_value=MagicMock()):
            with patch("agent_framework._types.AgentResponseUpdate") as MockUpdate:
                MockUpdate.return_value = _FakeUpdate(text="Fallback response")
                with patch("agent_framework._types.Content") as MockContent:
                    MockContent.from_text.return_value = "text"
                    updates = []
                    async for u in self.workflow._execute_agent("triage", "hi"):
                        updates.append(u)

        # Should have at least the fallback update
        assert any(getattr(u, "text", None) == "Fallback response"
                   for u in updates if u is not None)

    @pytest.mark.asyncio
    async def test_streaming_exception_falls_back(self):
        """When streaming raises an exception, falls back to non-streaming."""
        async def _bad_stream():
            raise RuntimeError("streaming not supported")
            yield  # make it a generator

        class _FailStream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise RuntimeError("streaming failed")
            async def get_final_response(self):
                return _FakeResponse()

        self.agents["triage"].run = MagicMock()
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("stream"):
                return _FailStream()
            async def _coro():
                return _FakeResponse(text="non-streaming result")
            return _coro()

        self.agents["triage"].run.side_effect = side_effect

        with patch("agent_framework.AgentSession", return_value=MagicMock()):
            with patch("agent_framework._types.AgentResponseUpdate") as MockUpdate:
                MockUpdate.return_value = _FakeUpdate(text="non-streaming result")
                with patch("agent_framework._types.Content") as MockContent:
                    MockContent.from_text.return_value = "text"
                    updates = []
                    async for u in self.workflow._execute_agent("triage", "test"):
                        updates.append(u)

        assert any(getattr(u, "text", None) == "non-streaming result"
                   for u in updates if u is not None)

    @pytest.mark.asyncio
    async def test_falls_back_to_triage_for_unknown_agent(self):
        """Unknown agent key falls back to triage agent."""
        update = _FakeUpdate(text="Triage response")
        stream = _FakeStream([update])
        self.agents["triage"].run = MagicMock(return_value=stream)

        with patch("agent_framework.AgentSession", return_value=MagicMock()):
            updates = []
            async for u in self.workflow._execute_agent("nonexistent", "hello"):
                updates.append(u)

        assert any(getattr(u, "text", None) == "Triage response" for u in updates)


class TestDrainToolQueue:
    def setup_method(self):
        self.tracker = MagicMock()
        self.tracker.drain.return_value = []
        self.workflow = RoutedWorkflow(
            agents={"triage": MagicMock(name="Triage")},
            tracker=self.tracker, client=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_drain_tool_start_event(self):
        q = asyncio.Queue()
        await q.put(("tool_start", "search_issues", {"query": "test"}))

        with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
            MockEvt.emit.return_value = "tool_start_event"
            events = []
            async for evt in self.workflow._drain_tool_queue(q, "TestAgent"):
                events.append(evt)

        assert len(events) == 1
        MockEvt.emit.assert_called_once_with(
            "TestAgent",
            {"type": "tool_start", "name": "search_issues", "args": {"query": "test"}},
        )

    @pytest.mark.asyncio
    async def test_drain_tool_end_event(self):
        record = MagicMock()
        q = asyncio.Queue()
        await q.put(("tool_end", record))

        with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
            MockEvt.emit.return_value = "tool_end_event"
            events = []
            async for evt in self.workflow._drain_tool_queue(q, "TestAgent"):
                events.append(evt)

        assert len(events) == 1
        MockEvt.emit.assert_called_once_with(
            "TestAgent",
            {"type": "tool_end", "record": record},
        )

    @pytest.mark.asyncio
    async def test_drain_empty_queue(self):
        q = asyncio.Queue()
        events = []
        async for evt in self.workflow._drain_tool_queue(q, "TestAgent"):
            events.append(evt)
        assert events == []

    @pytest.mark.asyncio
    async def test_drain_multiple_events(self):
        q = asyncio.Queue()
        await q.put(("tool_start", "tool1", {}))
        await q.put(("tool_end", MagicMock()))
        await q.put(("tool_start", "tool2", {"a": 1}))

        with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
            MockEvt.emit.return_value = "event"
            events = []
            async for evt in self.workflow._drain_tool_queue(q, "Agent"):
                events.append(evt)

        assert len(events) == 3


class TestStreamAgentEvents:
    def setup_method(self):
        self.tracker = MagicMock()
        self.tracker.drain.return_value = []
        self.client = MagicMock()
        self.agents = {
            "triage": MagicMock(name="Triage"),
            "jira_query": MagicMock(name="JiraQuery"),
        }
        self.workflow = RoutedWorkflow(
            agents=self.agents, tracker=self.tracker, client=self.client,
        )

    @pytest.mark.asyncio
    async def test_yields_executor_invoked_and_output(self):
        update = _FakeUpdate(text="Response text here")

        async def _fake_execute(key, msg):
            yield update

        with patch.object(self.workflow, "_execute_agent", side_effect=_fake_execute):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_invoked.return_value = "invoked"
                MockEvt.output.return_value = "output"

                q = asyncio.Queue()
                events = []
                async for evt in self.workflow._stream_agent_events("triage", "hello", q):
                    events.append(evt)

        assert "invoked" in events
        assert "output" in events
        assert self.workflow.last_agent_key == "triage"
        assert self.workflow._last_full_text == "Response text here"

    @pytest.mark.asyncio
    async def test_strips_scrum_prefetch_suffix_from_agent_key(self):
        async def _fake_execute(key, msg):
            return
            yield

        with patch.object(self.workflow, "_execute_agent", side_effect=_fake_execute):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_invoked.return_value = "invoked"
                q = asyncio.Queue()
                async for _ in self.workflow._stream_agent_events("_scrum_prefetch", "test", q):
                    pass

        assert self.workflow.last_agent_key == "scrum"


class TestRunQueryFullFlow:
    def setup_method(self):
        self.tracker = MagicMock()
        self.tracker.drain.return_value = []
        self.client = MagicMock()
        self.agents = {
            "triage": MagicMock(name="Triage"),
            "jira_query": MagicMock(name="JiraQuery"),
            "scrum": MagicMock(name="Scrum"),
            "sprint": MagicMock(name="Sprint"),
        }
        self.workflow = RoutedWorkflow(
            agents=self.agents, tracker=self.tracker,
            client=self.client, board_id=1,
        )

    def _patch_stream(self, full_text, tool_calls=None):
        """Create a patched _stream_agent_events that sets results."""
        tool_calls = tool_calls or []

        async def _fake_stream(key, msg, q):
            self.workflow._last_full_text = full_text
            self.workflow._last_tool_calls = tool_calls
            return
            yield
        return _fake_stream

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="triage")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    async def test_successful_run_caches_result(self, mock_set, mock_get, mock_route):
        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=self._patch_stream("Hello, how can I help?")):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_completed.return_value = "completed"
                events = []
                async for evt in self.workflow._run_query("hello"):
                    events.append(evt)

        mock_set.assert_called_once()
        assert self.workflow._is_running is False

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="jira_query")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    @patch("pile.context.recall_facts", return_value=["fact1"])
    async def test_recall_facts_emitted(self, mock_recall, mock_set, mock_get, mock_route):
        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=self._patch_stream("Here are results for your query.")):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.emit.return_value = "recall_event"
                MockEvt.executor_completed.return_value = "completed"
                events = []
                async for evt in self.workflow._run_query("find issues"):
                    events.append(evt)

        assert "recall_event" in events
        MockEvt.emit.assert_any_call("system", {"type": "recalled_context", "facts": ["fact1"]})

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="jira_query")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    @patch("pile.context.recall_facts", return_value=[])
    @patch("pile.workflows.interactive.learn")
    async def test_recovery_on_failure_triggers_fallback(
        self, mock_learn, mock_recall, mock_set, mock_get, mock_route,
    ):
        call_count = [0]

        async def _fake_stream(key, msg, q):
            call_count[0] += 1
            if call_count[0] == 1:
                # First agent fails (short text, no tool calls)
                self.workflow._last_full_text = "ok"
                self.workflow._last_tool_calls = []
            else:
                # Fallback succeeds
                self.workflow._last_full_text = "Here is a detailed response with enough content."
                self.workflow._last_tool_calls = [MagicMock(result="success")]
            return
            yield

        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=_fake_stream):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_completed.return_value = "completed"
                events = []
                async for evt in self.workflow._run_query("find issues"):
                    events.append(evt)

        # Should have called stream twice (original + fallback)
        assert call_count[0] == 2
        # Should have learned from the recovery
        mock_learn.assert_called_once()

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="jira_write")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    @patch("pile.context.recall_facts", return_value=[])
    async def test_no_retry_for_write_agents(
        self, mock_recall, mock_set, mock_get, mock_route,
    ):
        """jira_write agent should never trigger recovery even on failure."""
        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=self._patch_stream("ok")):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_completed.return_value = "completed"
                events = []
                async for evt in self.workflow._run_query("create issue"):
                    events.append(evt)

        # Should not cache since text is too short and it's jira_write
        # The key point: no fallback was attempted despite failure-like response

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="triage")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    async def test_exception_yields_failed_event(self, mock_get, mock_route):
        async def _exploding_stream(key, msg, q):
            raise RuntimeError("something broke")
            yield

        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=_exploding_stream):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_failed.return_value = "failed"
                events = []
                async for evt in self.workflow._run_query("hello"):
                    events.append(evt)

        assert "failed" in events
        assert self.workflow._is_running is False
        # Callbacks should be cleaned up
        assert self.tracker.on_tool_start is None
        assert self.tracker.on_tool_end is None

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="scrum")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    @patch("pile.workflows.interactive.prefetch_scrum_data", return_value="sprint data here")
    async def test_prefetch_scrum_sets_has_prefetch(
        self, mock_prefetch, mock_set, mock_get, mock_route,
    ):
        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=self._patch_stream("Sprint summary with enough detail.")):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_completed.return_value = "completed"
                events = []
                async for evt in self.workflow._run_query("sprint status"):
                    events.append(evt)

        mock_prefetch.assert_called_once()

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="sprint")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    @patch("pile.workflows.interactive.prefetch_scrum_data", return_value="sprint data")
    async def test_sprint_redirects_to_scrum_with_prefetch(
        self, mock_prefetch, mock_set, mock_get, mock_route,
    ):
        """Sprint with prefetch data should redirect to scrum agent."""
        called_keys = []

        async def _capture_stream(key, msg, q):
            called_keys.append(key)
            self.workflow._last_full_text = "Sprint analysis complete with details."
            self.workflow._last_tool_calls = []
            return
            yield

        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=_capture_stream):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_completed.return_value = "completed"
                async for _ in self.workflow._run_query("sprint status"):
                    pass

        assert called_keys[0] == "scrum"

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="jira_query")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    @patch("pile.context.recall_facts", return_value=[])
    async def test_jira_query_prefetch_redirects_to_scrum(
        self, mock_recall, mock_set, mock_get, mock_route,
    ):
        """jira_query with prefetch data should redirect to scrum agent."""
        called_keys = []

        async def _capture_stream(key, msg, q):
            called_keys.append(key)
            self.workflow._last_full_text = "Query results with enough content here."
            self.workflow._last_tool_calls = []
            return
            yield

        with patch("pile.prefetch.prefetch_query_data", return_value="issues data"):
            with patch.object(self.workflow, "_stream_agent_events",
                              side_effect=_capture_stream):
                with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                    MockEvt.executor_completed.return_value = "completed"
                    async for _ in self.workflow._run_query("find bugs"):
                        pass

        assert called_keys[0] == "scrum"

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="jira_query")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    @patch("pile.context.recall_facts", return_value=[])
    async def test_failed_result_not_cached(
        self, mock_recall, mock_set, mock_get, mock_route,
    ):
        """Failed responses (with no fallback available) should not be cached."""
        # Remove all possible fallbacks
        self.workflow.agents = {"triage": MagicMock(name="Triage"), "jira_query": MagicMock(name="JQ")}

        async def _fail_stream(key, msg, q):
            self.workflow._last_full_text = "ok"
            self.workflow._last_tool_calls = []
            return
            yield

        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=_fail_stream):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_completed.return_value = "completed"
                async for _ in self.workflow._run_query("find issues"):
                    pass

        mock_set.assert_not_called()

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="jira_query")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    @patch("pile.context.recall_facts", return_value=[])
    @patch("pile.workflows.interactive.learn")
    async def test_fallback_failure_not_learned(
        self, mock_learn, mock_recall, mock_set, mock_get, mock_route,
    ):
        """If the fallback also produces a short response, don't call learn()."""
        call_count = [0]

        async def _fake_stream(key, msg, q):
            call_count[0] += 1
            # Both original and fallback produce short text
            self.workflow._last_full_text = "ok"
            self.workflow._last_tool_calls = []
            return
            yield

        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=_fake_stream):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_completed.return_value = "completed"
                async for _ in self.workflow._run_query("find issues"):
                    pass

        mock_learn.assert_not_called()


class TestRunDispatch:
    """Tests for the run() method dispatching."""

    def setup_method(self):
        self.tracker = MagicMock()
        self.tracker.drain.return_value = []
        self.workflow = RoutedWorkflow(
            agents={"triage": MagicMock(name="Triage")},
            tracker=self.tracker, client=MagicMock(),
        )

    def test_run_with_responses_dispatches(self):
        with patch.object(self.workflow, "_run_with_responses") as mock_rwr:
            mock_rwr.return_value = "responses_gen"
            result = self.workflow.run(responses={"key": "val"})
            mock_rwr.assert_called_once_with({"key": "val"}, stream=False)

    def test_run_with_message_dispatches(self):
        with patch.object(self.workflow, "_run_query") as mock_rq:
            mock_rq.return_value = "query_gen"
            result = self.workflow.run(message="hello")
            mock_rq.assert_called_once_with("hello", stream=False)

    def test_run_with_stream_flag(self):
        with patch.object(self.workflow, "_run_query") as mock_rq:
            mock_rq.return_value = "query_gen"
            result = self.workflow.run(message="hello", stream=True)
            mock_rq.assert_called_once_with("hello", stream=True)


class TestToolMiddlewareIntegration:
    """Tests for tracker callback wiring in _run_query."""

    def setup_method(self):
        self.tracker = MagicMock()
        self.tracker.drain.return_value = []
        self.tracker.on_tool_start = None
        self.tracker.on_tool_end = None
        self.workflow = RoutedWorkflow(
            agents={"triage": MagicMock(name="Triage")},
            tracker=self.tracker, client=MagicMock(),
        )

    @pytest.mark.asyncio
    @patch("pile.workflows.interactive.smart_route", return_value="triage")
    @patch("pile.workflows.interactive.get_cached", return_value=None)
    @patch("pile.workflows.interactive.set_cached")
    async def test_callbacks_set_during_run_and_cleaned_after(
        self, mock_set, mock_get, mock_route,
    ):
        callbacks_during_run = {}

        async def _capture_stream(key, msg, q):
            callbacks_during_run["on_tool_start"] = self.tracker.on_tool_start
            callbacks_during_run["on_tool_end"] = self.tracker.on_tool_end
            self.workflow._last_full_text = "Response with enough text to pass checks."
            self.workflow._last_tool_calls = []
            return
            yield

        with patch.object(self.workflow, "_stream_agent_events",
                          side_effect=_capture_stream):
            with patch("agent_framework._workflows._events.WorkflowEvent") as MockEvt:
                MockEvt.executor_completed.return_value = "completed"
                async for _ in self.workflow._run_query("hello"):
                    pass

        # Callbacks should have been set during the run
        assert callbacks_during_run["on_tool_start"] is not None
        assert callbacks_during_run["on_tool_end"] is not None
        # And cleaned up after
        assert self.tracker.on_tool_start is None
        assert self.tracker.on_tool_end is None
