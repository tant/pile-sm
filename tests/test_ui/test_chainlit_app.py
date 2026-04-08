"""Tests for pile.ui.chainlit_app — handlers, helpers, and session management."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: replace the chainlit module with a mock before importing the app
# ---------------------------------------------------------------------------

@pytest.fixture()
def cl_mock():
    """Provide a MagicMock standing in for the `chainlit` package.

    This avoids chainlit's custom __getattr__ which breaks unittest.mock.patch.
    """
    mock = MagicMock()
    mock.Message.return_value.send = AsyncMock()
    mock.Step.return_value.send = AsyncMock()
    mock.Step.return_value.update = AsyncMock()
    mock.Step.return_value.remove = AsyncMock()
    mock.Step.return_value.id = "step-id"
    mock.Step.return_value.output = ""
    mock.Step.return_value.name = ""
    mock.Plotly.return_value.send = AsyncMock()
    mock.Starter = lambda **kw: SimpleNamespace(**kw)

    # Decorators should be transparent pass-throughs
    mock.set_starters = lambda f: f
    mock.on_chat_start = lambda f: f
    mock.on_message = lambda f: f
    mock.on_stop = lambda f: f
    mock.on_chat_end = lambda f: f

    mock.user_session = MagicMock()
    return mock


@pytest.fixture()
def app(cl_mock):
    """Import (or re-import) chainlit_app with the mocked chainlit module."""
    original = sys.modules.get("chainlit")
    sys.modules["chainlit"] = cl_mock

    # Force re-import so module-level `import chainlit as cl` picks up the mock
    mod_name = "pile.ui.chainlit_app"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)

    yield mod

    # Restore
    if original is not None:
        sys.modules["chainlit"] = original
    else:
        sys.modules.pop("chainlit", None)
    sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# Async iterator helper
# ---------------------------------------------------------------------------

class AsyncIterFromList:
    """Wraps a list as an async iterator for mocking workflow.run()."""

    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _ev(etype, data=None, executor_id=None):
    """Create a mock workflow event."""
    ev = SimpleNamespace(type=etype, data=data)
    if executor_id is not None:
        ev.executor_id = executor_id
    return ev


# ---------------------------------------------------------------------------
# summarize_args (pure function — import directly, no mock needed)
# ---------------------------------------------------------------------------

from pile.ui.chainlit_app import summarize_args


class TestSummarizeArgs:
    def test_empty(self):
        assert summarize_args({}) == ""

    def test_single_key(self):
        assert summarize_args({"key": "val"}) == "key=val"

    def test_truncation(self):
        result = summarize_args({"k": "x" * 50})
        assert result == "k=" + "x" * 30 + "..."

    def test_max_keys_default(self):
        args = {chr(i): str(i) for i in range(97, 102)}  # a-e
        result = summarize_args(args)
        assert "+2 more" in result

    def test_custom_max_keys(self):
        result = summarize_args({"a": "1", "b": "2", "c": "3"}, max_keys=2)
        assert "+1 more" in result

    def test_custom_max_val_len(self):
        result = summarize_args({"k": "abcde"}, max_val_len=3)
        assert result == "k=abc..."

    def test_exact_length_no_ellipsis(self):
        result = summarize_args({"k": "abc"}, max_val_len=3)
        assert result == "k=abc"

    def test_non_string_value(self):
        result = summarize_args({"n": 42})
        assert result == "n=42"


# ---------------------------------------------------------------------------
# AGENT_CONFIG
# ---------------------------------------------------------------------------

from pile.ui.chainlit_app import AGENT_CONFIG


class TestAgentConfig:
    def test_known_agents(self):
        assert "TriageAgent" in AGENT_CONFIG
        assert AGENT_CONFIG["TriageAgent"]["label"] == "Routing"

    def test_all_entries_have_type_and_label(self):
        for name, cfg in AGENT_CONFIG.items():
            assert "type" in cfg, f"{name} missing type"
            assert "label" in cfg, f"{name} missing label"


# ---------------------------------------------------------------------------
# set_starters
# ---------------------------------------------------------------------------

class TestSetStarters:
    @pytest.mark.asyncio
    async def test_returns_six_starters(self, app, cl_mock):
        result = await app.set_starters()
        assert len(result) == 6

    @pytest.mark.asyncio
    async def test_starter_labels(self, app, cl_mock):
        result = await app.set_starters()
        labels = {s.label for s in result}
        assert "Sprint Status" in labels
        assert "Blockers" in labels
        assert "Backlog" in labels


# ---------------------------------------------------------------------------
# on_chat_start
# ---------------------------------------------------------------------------

class TestOnChatStart:
    @pytest.mark.asyncio
    async def test_successful_init(self, app, cl_mock):
        with patch.object(app, "setup_app_logger"), \
             patch.object(app, "setup_inference_logger"), \
             patch.object(app, "ensure_models"), \
             patch.object(app, "create_workflow") as cw_mock, \
             patch("pile.health.run_health_checks", return_value=[]):
            mock_wf, mock_tracker = MagicMock(), MagicMock()
            cw_mock.return_value = (mock_wf, mock_tracker)

            await app.on_chat_start()

            cl_mock.user_session.set.assert_any_call("workflow", mock_wf)
            cl_mock.user_session.set.assert_any_call("tracker", mock_tracker)

    @pytest.mark.asyncio
    async def test_health_check_warnings_sent(self, app, cl_mock):
        with patch.object(app, "setup_app_logger"), \
             patch.object(app, "setup_inference_logger"), \
             patch.object(app, "ensure_models"), \
             patch.object(app, "create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("pile.health.run_health_checks", return_value=["Jira unreachable"]):
            await app.on_chat_start()

            # A warning message was sent
            assert cl_mock.Message.return_value.send.called

    @pytest.mark.asyncio
    async def test_workflow_creation_failure(self, app, cl_mock):
        with patch.object(app, "setup_app_logger"), \
             patch.object(app, "setup_inference_logger"), \
             patch.object(app, "ensure_models"), \
             patch.object(app, "create_workflow", side_effect=RuntimeError("boom")), \
             patch("pile.health.run_health_checks", return_value=[]):
            await app.on_chat_start()

            # Error message sent
            assert cl_mock.Message.return_value.send.called
            # workflow not stored
            set_calls = [c for c in cl_mock.user_session.set.call_args_list
                         if c[0][0] == "workflow"]
            assert len(set_calls) == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_facts_runs(self, app, cl_mock):
        with patch.object(app, "setup_app_logger"), \
             patch.object(app, "setup_inference_logger"), \
             patch.object(app, "ensure_models"), \
             patch.object(app, "create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("pile.health.run_health_checks", return_value=[]), \
             patch("pile.memory.store.cleanup_expired_facts", return_value=3) as cleanup:
            await app.on_chat_start()
            cleanup.assert_called_once_with(max_age_days=7)

    @pytest.mark.asyncio
    async def test_cleanup_expired_facts_exception_ignored(self, app, cl_mock):
        with patch.object(app, "setup_app_logger"), \
             patch.object(app, "setup_inference_logger"), \
             patch.object(app, "ensure_models"), \
             patch.object(app, "create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("pile.health.run_health_checks", return_value=[]), \
             patch("pile.memory.store.cleanup_expired_facts", side_effect=Exception("db err")):
            # Should not raise
            await app.on_chat_start()


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------

class TestOnMessage:
    @pytest.mark.asyncio
    async def test_no_workflow_sends_error(self, app, cl_mock):
        cl_mock.user_session.get.return_value = None

        msg = MagicMock()
        msg.content = "hello"
        msg.elements = []

        await app.on_message(msg)

        cl_mock.Message.assert_called_with(content="Workflow not initialized. Please refresh.")

    @pytest.mark.asyncio
    async def test_plain_message_runs_workflow(self, app, cl_mock):
        mock_wf = MagicMock()
        cl_mock.user_session.get.return_value = mock_wf

        msg = MagicMock()
        msg.content = "hello"
        msg.elements = []

        with patch.object(app, "_handle_file_uploads", new_callable=AsyncMock, return_value=[]), \
             patch.object(app, "_run_workflow", new_callable=AsyncMock) as run_mock:
            await app.on_message(msg)
            run_mock.assert_called_once_with(mock_wf, user_input="hello")

    @pytest.mark.asyncio
    async def test_empty_message_no_run(self, app, cl_mock):
        mock_wf = MagicMock()
        cl_mock.user_session.get.return_value = mock_wf

        msg = MagicMock()
        msg.content = ""
        msg.elements = []

        with patch.object(app, "_handle_file_uploads", new_callable=AsyncMock, return_value=[]), \
             patch.object(app, "_run_workflow", new_callable=AsyncMock) as run_mock:
            await app.on_message(msg)
            run_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_only_no_text(self, app, cl_mock):
        mock_wf = MagicMock()
        cl_mock.user_session.get.return_value = mock_wf

        msg = MagicMock()
        msg.content = ""
        msg.elements = [MagicMock()]

        with patch.object(app, "_handle_file_uploads", new_callable=AsyncMock, return_value=["report.pdf"]), \
             patch.object(app, "_run_workflow", new_callable=AsyncMock) as run_mock:
            await app.on_message(msg)
            call_input = run_mock.call_args[1]["user_input"]
            assert "report.pdf" in call_input

    @pytest.mark.asyncio
    async def test_upload_with_text(self, app, cl_mock):
        mock_wf = MagicMock()
        cl_mock.user_session.get.return_value = mock_wf

        msg = MagicMock()
        msg.content = "summarize this"
        msg.elements = [MagicMock()]

        with patch.object(app, "_handle_file_uploads", new_callable=AsyncMock, return_value=["doc.pdf"]), \
             patch.object(app, "_run_workflow", new_callable=AsyncMock) as run_mock:
            await app.on_message(msg)
            call_input = run_mock.call_args[1]["user_input"]
            assert "doc.pdf" in call_input
            assert "summarize this" in call_input


# ---------------------------------------------------------------------------
# _handle_file_uploads
# ---------------------------------------------------------------------------

class TestHandleFileUploads:
    @pytest.mark.asyncio
    async def test_no_elements(self, app, cl_mock):
        msg = MagicMock()
        msg.elements = []
        result = await app._handle_file_uploads(msg)
        assert result == []

    @pytest.mark.asyncio
    async def test_memory_disabled(self, app, cl_mock):
        elem = MagicMock()
        elem.path = "/tmp/test.pdf"
        msg = MagicMock()
        msg.elements = [elem]

        settings_mock = MagicMock()
        settings_mock.memory_enabled = False
        with patch("pile.config.settings", settings_mock):
            result = await app._handle_file_uploads(msg)
        assert result == []

    @pytest.mark.asyncio
    async def test_successful_ingest(self, app, cl_mock):
        elem = MagicMock()
        elem.path = "/tmp/test.pdf"
        elem.name = "test.pdf"
        msg = MagicMock()
        msg.elements = [elem]

        settings_mock = MagicMock()
        settings_mock.memory_enabled = True
        ingest_result = {"doc_name": "test.pdf", "pages": 5, "chunks": 12}

        with patch("pile.config.settings", settings_mock), \
             patch("pile.memory.ingest.ingest_file", return_value=ingest_result):
            result = await app._handle_file_uploads(msg)
        assert result == ["test.pdf"]

    @pytest.mark.asyncio
    async def test_ingest_failure(self, app, cl_mock):
        elem = MagicMock()
        elem.path = "/tmp/bad.pdf"
        elem.name = "bad.pdf"
        msg = MagicMock()
        msg.elements = [elem]

        settings_mock = MagicMock()
        settings_mock.memory_enabled = True
        with patch("pile.config.settings", settings_mock), \
             patch("pile.memory.ingest.ingest_file", side_effect=ValueError("corrupt")):
            result = await app._handle_file_uploads(msg)
        assert result == []

    @pytest.mark.asyncio
    async def test_element_without_path_skipped(self, app, cl_mock):
        elem = MagicMock()
        elem.path = None
        msg = MagicMock()
        msg.elements = [elem]

        settings_mock = MagicMock()
        settings_mock.memory_enabled = True
        with patch("pile.config.settings", settings_mock):
            result = await app._handle_file_uploads(msg)
        assert result == []


# ---------------------------------------------------------------------------
# _send_charts_if_any
# ---------------------------------------------------------------------------

class TestSendChartsIfAny:
    @pytest.mark.asyncio
    async def test_short_content_skipped(self, app, cl_mock):
        msg = MagicMock()
        msg.content = "short"
        await app._send_charts_if_any(msg)  # no crash

    @pytest.mark.asyncio
    async def test_no_content_skipped(self, app, cl_mock):
        msg = MagicMock()
        msg.content = None
        await app._send_charts_if_any(msg)

    @pytest.mark.asyncio
    async def test_chart_detected_and_sent(self, app, cl_mock):
        chart_data = MagicMock()
        chart_data.title = "Test Chart"
        fig_mock = MagicMock()

        msg = MagicMock()
        msg.content = "x" * 30
        msg.id = "msg-1"

        with patch("pile.ui.charts.detect_charts", return_value=[chart_data]), \
             patch("pile.ui.charts.build_chart", return_value=fig_mock):
            await app._send_charts_if_any(msg)

        cl_mock.Plotly.assert_called_once_with(name="Test Chart", figure=fig_mock, size="large")
        cl_mock.Plotly.return_value.send.assert_called_once_with(for_id="msg-1")

    @pytest.mark.asyncio
    async def test_chart_exception_suppressed(self, app, cl_mock):
        msg = MagicMock()
        msg.content = "x" * 30
        with patch("pile.ui.charts.detect_charts", side_effect=RuntimeError("boom")):
            await app._send_charts_if_any(msg)  # no raise


# ---------------------------------------------------------------------------
# _run_workflow_once — event streaming
# ---------------------------------------------------------------------------

class TestRunWorkflowOnce:
    @pytest.mark.asyncio
    async def test_simple_output(self, app, cl_mock):
        output_data = SimpleNamespace(text="Hello world")
        events = [_ev("output", data=output_data)]

        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList(events)

        with patch.object(app, "_send_charts_if_any", new_callable=AsyncMock):
            await app._run_workflow_once(workflow, user_input="hi")

        # Final message sent with the output text
        cl_mock.Message.assert_called()

    @pytest.mark.asyncio
    async def test_executor_invoked_creates_step(self, app, cl_mock):
        events = [
            _ev("executor_invoked", executor_id="TriageAgent"),
            _ev("executor_completed"),
        ]
        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList(events)

        await app._run_workflow_once(workflow, user_input="hi")

        # At least root step + agent step
        assert cl_mock.Step.call_count >= 2

    @pytest.mark.asyncio
    async def test_tool_start_and_end(self, app, cl_mock):
        record = SimpleNamespace(name="search_issues", duration_ms=1500, result="found 3")
        events = [
            _ev("executor_invoked", executor_id="JiraQueryAgent"),
            _ev("data", data={"type": "tool_start", "name": "search_issues", "args": {"q": "test"}}),
            _ev("data", data={"type": "tool_end", "record": record}),
            _ev("executor_completed"),
        ]
        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList(events)

        await app._run_workflow_once(workflow, user_input="hi")

    @pytest.mark.asyncio
    async def test_recalled_context(self, app, cl_mock):
        events = [
            _ev("data", data={"type": "recalled_context", "facts": ["fact1", "fact2"]}),
        ]
        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList(events)

        await app._run_workflow_once(workflow, user_input="hi")

    @pytest.mark.asyncio
    async def test_executor_failed(self, app, cl_mock):
        events = [
            _ev("executor_invoked", executor_id="JiraQueryAgent"),
            _ev("executor_failed", data="something broke"),
        ]
        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList(events)

        await app._run_workflow_once(workflow, user_input="hi")

    @pytest.mark.asyncio
    async def test_cancelled_error(self, app, cl_mock):
        async def cancel_iter(*a, **kw):
            raise asyncio.CancelledError()
            yield  # noqa: unreachable — makes this an async generator

        workflow = MagicMock()
        workflow.run.return_value = cancel_iter()

        result = await app._run_workflow_once(workflow, user_input="hi")

        assert result == []
        workflow._reset_running_flag.assert_called_once()

    @pytest.mark.asyncio
    async def test_general_exception(self, app, cl_mock):
        async def error_iter(*a, **kw):
            raise RuntimeError("workflow exploded")
            yield  # noqa

        workflow = MagicMock()
        workflow.run.return_value = error_iter()

        with pytest.raises(RuntimeError, match="workflow exploded"):
            await app._run_workflow_once(workflow, user_input="hi")

        workflow._reset_running_flag.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_final_text_no_message(self, app, cl_mock):
        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList([])

        await app._run_workflow_once(workflow, user_input="hi")

        # Message should not be called when there is no output
        cl_mock.Message.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_mode(self, app, cl_mock):
        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList([])

        await app._run_workflow_once(workflow, responses={"key": "val"})

        workflow.run.assert_called_once_with(
            responses={"key": "val"}, stream=True, include_status_events=True,
        )

    @pytest.mark.asyncio
    async def test_unknown_agent_uses_id_as_label(self, app, cl_mock):
        events = [
            _ev("executor_invoked", executor_id="UnknownAgent"),
            _ev("executor_completed"),
        ]
        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList(events)

        await app._run_workflow_once(workflow, user_input="hi")

    @pytest.mark.asyncio
    async def test_output_with_list_data_ignored(self, app, cl_mock):
        events = [_ev("output", data=["item1", "item2"])]
        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList(events)

        await app._run_workflow_once(workflow, user_input="hi")

        cl_mock.Message.assert_not_called()

    @pytest.mark.asyncio
    async def test_bg_summarize_triggered_for_long_output(self, app, cl_mock):
        long_text = "A" * 60
        output_data = SimpleNamespace(text=long_text)
        events = [_ev("output", data=output_data)]

        workflow = MagicMock()
        workflow.run.return_value = AsyncIterFromList(events)

        with patch.object(app, "_send_charts_if_any", new_callable=AsyncMock), \
             patch("pile.context.summarize_turn") as summarize_mock:
            await app._run_workflow_once(workflow, user_input="hi")
            # Give background task a chance to run
            await asyncio.sleep(0.05)

        summarize_mock.assert_called_once()


# ---------------------------------------------------------------------------
# on_stop
# ---------------------------------------------------------------------------

class TestOnStop:
    @pytest.mark.asyncio
    async def test_resets_running_flag(self, app, cl_mock):
        mock_wf = MagicMock()
        cl_mock.user_session.get.return_value = mock_wf

        await app.on_stop()

        mock_wf._reset_running_flag.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_workflow(self, app, cl_mock):
        cl_mock.user_session.get.return_value = None
        await app.on_stop()  # no raise


# ---------------------------------------------------------------------------
# on_chat_end
# ---------------------------------------------------------------------------

class TestOnChatEnd:
    @pytest.mark.asyncio
    async def test_clears_workflow(self, app, cl_mock):
        await app.on_chat_end()
        cl_mock.user_session.set.assert_called_once_with("workflow", None)


# ---------------------------------------------------------------------------
# _cleanup (atexit handler)
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_calls_unload_all(self):
        with patch("pile.models.manager.unload_all") as unload:
            from pile.ui.chainlit_app import _cleanup
            _cleanup()
            unload.assert_called_once()

    def test_exception_suppressed(self):
        with patch("pile.models.manager.unload_all", side_effect=RuntimeError("fail")):
            from pile.ui.chainlit_app import _cleanup
            _cleanup()  # no raise

    def test_is_callable(self):
        from pile.ui.chainlit_app import _cleanup
        assert callable(_cleanup)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_prints_hint(self, capsys):
        from pile.ui.chainlit_app import main
        main()
        captured = capsys.readouterr()
        assert "chainlit run" in captured.out


# ---------------------------------------------------------------------------
# _run_workflow (thin wrapper)
# ---------------------------------------------------------------------------

class TestRunWorkflow:
    @pytest.mark.asyncio
    async def test_delegates_to_run_workflow_once(self, app, cl_mock):
        with patch.object(app, "_run_workflow_once", new_callable=AsyncMock) as once_mock:
            mock_wf = MagicMock()
            await app._run_workflow(mock_wf, user_input="hi")
            once_mock.assert_called_once_with(mock_wf, user_input="hi")
