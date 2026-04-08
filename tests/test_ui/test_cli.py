"""Tests for pile.ui.cli — interactive terminal chat entry point."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def _async_iter(items):
    for item in items:
        yield item


async def _async_iter_raise(exc_type):
    raise exc_type()
    yield  # noqa: unreachable


@contextmanager
def _patch_run_setup(health_errors=None):
    """Patch all _run setup dependencies (loggers, models, health checks)."""
    if health_errors is None:
        health_errors = []
    with patch("pile.models.logging.setup_app_logger"), \
         patch("pile.models.logging.setup_inference_logger"), \
         patch("pile.models.manager.ensure_models"), \
         patch("pile.health.run_health_checks", return_value=health_errors):
        yield


# ---------------------------------------------------------------------------
# _create_slash_workflow
# ---------------------------------------------------------------------------

class TestCreateSlashWorkflow:
    def _call(self, command):
        from pile.ui.cli import _create_slash_workflow
        return _create_slash_workflow(command)

    @patch("pile.workflows.standup.create_workflow")
    def test_standup_returns_workflow_and_prompt(self, mock_create):
        mock_wf = MagicMock()
        mock_create.return_value = mock_wf
        result = self._call("/standup")
        assert result is not None
        wf, prompt = result
        assert wf is mock_wf
        assert "standup" in prompt.lower()

    @patch("pile.workflows.planning.create_workflow")
    def test_planning_returns_workflow_and_prompt(self, mock_create):
        mock_wf = MagicMock()
        mock_create.return_value = mock_wf
        result = self._call("/planning")
        assert result is not None
        wf, prompt = result
        assert wf is mock_wf
        assert "planning" in prompt.lower()

    def test_unknown_command_returns_none(self):
        assert self._call("/unknown") is None
        assert self._call("hello") is None
        assert self._call("") is None


# ---------------------------------------------------------------------------
# _stream_workflow
# ---------------------------------------------------------------------------

class TestStreamWorkflow:
    @pytest.mark.asyncio
    async def test_streams_text_output(self, capsys):
        from pile.ui.cli import _stream_workflow

        event = MagicMock()
        event.type = "output"
        event.data.text = "Hello world"

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter([event]))

        pending = []
        await _stream_workflow(mock_wf, "test", pending)

        captured = capsys.readouterr()
        assert "Hello world" in captured.out
        assert pending == []

    @pytest.mark.asyncio
    async def test_collects_request_info(self):
        from pile.ui.cli import _stream_workflow

        event = MagicMock()
        event.type = "request_info"
        event.data = "some request"

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter([event]))

        pending = []
        await _stream_workflow(mock_wf, "test", pending)
        assert len(pending) == 1
        assert pending[0] is event

    @pytest.mark.asyncio
    async def test_streams_list_output(self, capsys):
        from pile.ui.cli import _stream_workflow

        msg = MagicMock()
        msg.text = "message text"
        msg.role = "assistant"
        msg.author_name = "Bot"

        event = MagicMock()
        event.type = "output"
        event.data = [msg]

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter([event]))

        pending = []
        await _stream_workflow(mock_wf, "test", pending)

        captured = capsys.readouterr()
        assert "Bot" in captured.out
        assert "message text" in captured.out

    @pytest.mark.asyncio
    async def test_list_output_uses_role_when_no_author_name(self, capsys):
        from pile.ui.cli import _stream_workflow

        msg = MagicMock()
        msg.text = "some text"
        msg.role = "assistant"
        msg.author_name = None

        event = MagicMock()
        event.type = "output"
        event.data = [msg]

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter([event]))

        pending = []
        await _stream_workflow(mock_wf, "test", pending)

        captured = capsys.readouterr()
        assert "assistant" in captured.out


# ---------------------------------------------------------------------------
# _stream_responses
# ---------------------------------------------------------------------------

class TestStreamResponses:
    @pytest.mark.asyncio
    async def test_streams_text_output(self, capsys):
        from pile.ui.cli import _stream_responses

        event = MagicMock()
        event.type = "output"
        event.data.text = "response text"

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter([event]))

        pending = []
        await _stream_responses(mock_wf, {"id": "resp"}, pending)

        captured = capsys.readouterr()
        assert "response text" in captured.out

    @pytest.mark.asyncio
    async def test_collects_request_info(self):
        from pile.ui.cli import _stream_responses

        event = MagicMock()
        event.type = "request_info"

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter([event]))

        pending = []
        await _stream_responses(mock_wf, {}, pending)
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_ignores_output_without_text(self, capsys):
        from pile.ui.cli import _stream_responses

        event = MagicMock()
        event.type = "output"
        event.data.text = ""

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter([event]))

        pending = []
        await _stream_responses(mock_wf, {}, pending)

        captured = capsys.readouterr()
        assert captured.out.strip() == ""


# ---------------------------------------------------------------------------
# _handle_pending_requests — handoff agent
# ---------------------------------------------------------------------------

class TestHandlePendingRequestsHandoff:
    def _make_handoff_request(self):
        from agent_framework.orchestrations import HandoffAgentUserRequest

        msg1 = MagicMock()
        msg1.text = "Question?"
        msg1.author_name = "Agent"

        # Create a subclass so isinstance() passes, but use MagicMock for flexibility
        class FakeHandoff(HandoffAgentUserRequest):
            def __init__(self):
                pass  # skip dataclass init

        handoff = FakeHandoff()
        handoff.agent_response = MagicMock()
        handoff.agent_response.messages = [msg1]
        handoff.create_response = MagicMock(return_value={"response": "ok"})

        req = MagicMock()
        req.request_id = "req-1"
        req.data = handoff

        return req

    @pytest.mark.asyncio
    async def test_handoff_follow_up(self, capsys):
        from pile.ui.cli import _handle_pending_requests

        req = self._make_handoff_request()
        mock_wf = MagicMock()

        with patch("builtins.input", return_value="follow up"), \
             patch("pile.ui.cli._stream_responses", new_callable=AsyncMock) as mock_stream:
            result = await _handle_pending_requests(mock_wf, [req])

        assert result is True
        mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_handoff_eof_returns_false(self, capsys):
        from pile.ui.cli import _handle_pending_requests

        req = self._make_handoff_request()
        mock_wf = MagicMock()

        with patch("builtins.input", side_effect=EOFError):
            result = await _handle_pending_requests(mock_wf, [req])

        assert result is False

    @pytest.mark.asyncio
    async def test_handoff_quit_returns_false(self, capsys):
        from pile.ui.cli import _handle_pending_requests

        req = self._make_handoff_request()
        mock_wf = MagicMock()

        with patch("builtins.input", return_value="quit"):
            result = await _handle_pending_requests(mock_wf, [req])

        assert result is False

    @pytest.mark.asyncio
    async def test_handoff_exit_returns_false(self, capsys):
        from pile.ui.cli import _handle_pending_requests

        req = self._make_handoff_request()
        mock_wf = MagicMock()

        with patch("builtins.input", return_value="exit"):
            result = await _handle_pending_requests(mock_wf, [req])

        assert result is False


# ---------------------------------------------------------------------------
# _handle_pending_requests — function approval
# ---------------------------------------------------------------------------

class TestHandlePendingRequestsApproval:
    def _make_approval_request(self):
        func_call = MagicMock()
        func_call.name = "search_jira"
        func_call.parse_arguments.return_value = {"query": "test"}

        approval_data = MagicMock()
        approval_data.type = "function_approval_request"
        approval_data.function_call = func_call
        # Ensure isinstance(data, HandoffAgentUserRequest) is False
        del approval_data.__class__

        req = MagicMock()
        req.request_id = "req-2"
        req.data = approval_data

        return req

    @pytest.mark.asyncio
    async def test_approval_yes(self, capsys):
        from pile.ui.cli import _handle_pending_requests

        req = self._make_approval_request()
        mock_wf = MagicMock()

        with patch("builtins.input", return_value="y"), \
             patch("pile.ui.cli._stream_responses", new_callable=AsyncMock) as mock_stream:
            result = await _handle_pending_requests(mock_wf, [req])

        assert result is True
        mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_approval_no(self, capsys):
        from pile.ui.cli import _handle_pending_requests

        req = self._make_approval_request()
        mock_wf = MagicMock()

        with patch("builtins.input", return_value="n"), \
             patch("pile.ui.cli._stream_responses", new_callable=AsyncMock):
            result = await _handle_pending_requests(mock_wf, [req])

        assert result is True
        req.data.to_function_approval_response.assert_called_once_with(approved=False)

    @pytest.mark.asyncio
    async def test_approval_keyboard_interrupt_returns_false(self, capsys):
        from pile.ui.cli import _handle_pending_requests

        req = self._make_approval_request()
        mock_wf = MagicMock()

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = await _handle_pending_requests(mock_wf, [req])

        assert result is False

    @pytest.mark.asyncio
    async def test_approval_eof_returns_false(self, capsys):
        from pile.ui.cli import _handle_pending_requests

        req = self._make_approval_request()
        mock_wf = MagicMock()

        with patch("builtins.input", side_effect=EOFError):
            result = await _handle_pending_requests(mock_wf, [req])

        assert result is False


# ---------------------------------------------------------------------------
# _handle_pending_requests — empty list
# ---------------------------------------------------------------------------

class TestHandlePendingRequestsEmpty:
    @pytest.mark.asyncio
    async def test_no_requests_returns_true(self):
        from pile.ui.cli import _handle_pending_requests

        result = await _handle_pending_requests(MagicMock(), [])
        assert result is True


# ---------------------------------------------------------------------------
# _run — REPL loop
# ---------------------------------------------------------------------------

class TestRun:
    @pytest.mark.asyncio
    async def test_quit_exits_loop(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", side_effect=["quit"]):
            await _run()

        captured = capsys.readouterr()
        assert "Bye!" in captured.out

    @pytest.mark.asyncio
    async def test_exit_command(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", side_effect=["exit"]):
            await _run()

        captured = capsys.readouterr()
        assert "Bye!" in captured.out

    @pytest.mark.asyncio
    async def test_q_command(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", side_effect=["q"]):
            await _run()

        captured = capsys.readouterr()
        assert "Bye!" in captured.out

    @pytest.mark.asyncio
    async def test_eof_exits_loop(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", side_effect=EOFError):
            await _run()

        captured = capsys.readouterr()
        assert "Bye!" in captured.out

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_exits_loop(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", side_effect=KeyboardInterrupt):
            await _run()

        captured = capsys.readouterr()
        assert "Bye!" in captured.out

    @pytest.mark.asyncio
    async def test_empty_input_continues(self, capsys):
        from pile.ui.cli import _run

        mock_input = MagicMock(side_effect=["", "quit"])
        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", mock_input):
            await _run()

        assert mock_input.call_count == 2

    @pytest.mark.asyncio
    async def test_ready_message_printed(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", side_effect=["quit"]):
            await _run()

        captured = capsys.readouterr()
        assert "Pile SM ready" in captured.out
        assert "/standup" in captured.out
        assert "/planning" in captured.out


# ---------------------------------------------------------------------------
# _run — health check warnings
# ---------------------------------------------------------------------------

class TestRunHealthChecks:
    @pytest.mark.asyncio
    async def test_prints_health_warnings(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(health_errors=["models err", "jira err"]), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", side_effect=["quit"]):
            await _run()

        captured = capsys.readouterr()
        assert "Health check warnings" in captured.err
        assert "models err" in captured.err
        assert "jira err" in captured.err

    @pytest.mark.asyncio
    async def test_no_warnings_when_healthy(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(health_errors=[]), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("builtins.input", side_effect=["quit"]):
            await _run()

        captured = capsys.readouterr()
        assert "Health check warnings" not in captured.err


# ---------------------------------------------------------------------------
# _run — workflow init failure
# ---------------------------------------------------------------------------

class TestRunWorkflowInitFailure:
    @pytest.mark.asyncio
    async def test_init_failure_prints_error(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", side_effect=RuntimeError("init failed")):
            await _run()

        captured = capsys.readouterr()
        assert "Failed to initialize" in captured.err
        assert "init failed" in captured.err


# ---------------------------------------------------------------------------
# _run — user message routing
# ---------------------------------------------------------------------------

class TestRunMessageRouting:
    @pytest.mark.asyncio
    async def test_routes_message_to_workflow(self, capsys):
        from pile.ui.cli import _run

        event = MagicMock()
        event.type = "output"
        event.data.text = "response"

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter([event]))

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(mock_wf, MagicMock())), \
             patch("builtins.input", side_effect=["hello world", "quit"]):
            await _run()

        mock_wf.run.assert_called_once_with("hello world", stream=True)
        captured = capsys.readouterr()
        assert "response" in captured.out

    @pytest.mark.asyncio
    async def test_slash_command_dispatched(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("pile.ui.cli._create_slash_workflow") as mock_slash, \
             patch("pile.ui.cli._stream_workflow", new_callable=AsyncMock) as mock_stream, \
             patch("pile.ui.cli._handle_pending_requests", new_callable=AsyncMock, return_value=True) as mock_handle, \
             patch("builtins.input", side_effect=["/standup", "quit"]):

            slash_wf = MagicMock()
            mock_slash.return_value = (slash_wf, "standup prompt")

            await _run()

        mock_stream.assert_called_once()
        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_slash_command_error_printed(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("pile.ui.cli._create_slash_workflow", return_value=(MagicMock(), "prompt")), \
             patch("pile.ui.cli._stream_workflow", new_callable=AsyncMock, side_effect=RuntimeError("workflow error")), \
             patch("builtins.input", side_effect=["/standup", "quit"]):
            await _run()

        captured = capsys.readouterr()
        assert "Workflow error" in captured.err

    @pytest.mark.asyncio
    async def test_slash_handle_returns_false_exits(self, capsys):
        from pile.ui.cli import _run

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(MagicMock(), MagicMock())), \
             patch("pile.ui.cli._create_slash_workflow", return_value=(MagicMock(), "prompt")), \
             patch("pile.ui.cli._stream_workflow", new_callable=AsyncMock), \
             patch("pile.ui.cli._handle_pending_requests", new_callable=AsyncMock, return_value=False), \
             patch("builtins.input", side_effect=["/standup", "should not reach"]):
            await _run()

        # If _handle_pending_requests returns False, _run returns without asking for more input


# ---------------------------------------------------------------------------
# _run — keyboard interrupt during streaming
# ---------------------------------------------------------------------------

class TestRunStreamingInterrupt:
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_during_stream(self, capsys):
        from pile.ui.cli import _run

        mock_wf = MagicMock()
        mock_wf.run = MagicMock(return_value=_async_iter_raise(KeyboardInterrupt))

        with _patch_run_setup(), \
             patch("pile.ui.cli.create_workflow", return_value=(mock_wf, MagicMock())), \
             patch("builtins.input", side_effect=["hello", "quit"]):
            await _run()

        captured = capsys.readouterr()
        assert "[Stopped]" in captured.out
        mock_wf._reset_running_flag.assert_called_once()


# ---------------------------------------------------------------------------
# main — entry point
# ---------------------------------------------------------------------------

class TestMain:
    @patch("pile.models.manager.unload_all")
    @patch("pile.ui.cli.asyncio.run")
    def test_main_calls_asyncio_run(self, mock_run, mock_unload):
        from pile.ui.cli import main
        main()
        mock_run.assert_called_once()
        mock_unload.assert_called_once()

    @patch("pile.models.manager.unload_all")
    @patch("pile.ui.cli.asyncio.run", side_effect=KeyboardInterrupt)
    def test_main_keyboard_interrupt(self, mock_run, mock_unload, capsys):
        from pile.ui.cli import main
        main()
        captured = capsys.readouterr()
        assert "Bye!" in captured.out
        mock_unload.assert_called_once()

    @patch("pile.models.manager.unload_all")
    @patch("pile.ui.cli.asyncio.run")
    def test_main_always_unloads_on_error(self, mock_run, mock_unload):
        from pile.ui.cli import main

        mock_run.side_effect = RuntimeError("unexpected")
        with pytest.raises(RuntimeError):
            main()

        mock_unload.assert_called_once()

    @patch("pile.models.manager.unload_all", side_effect=RuntimeError("unload failed"))
    @patch("pile.ui.cli.asyncio.run")
    def test_main_unload_failure_propagates(self, mock_run, mock_unload):
        from pile.ui.cli import main

        with pytest.raises(RuntimeError, match="unload failed"):
            main()
