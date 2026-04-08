"""Tests for Triage Agent creation and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestCreateTriageAgent:
    """Test create_triage_agent under various config scenarios."""

    @patch("pile.agents.triage.settings")
    def test_no_memory_no_browser(self, mock_settings):
        mock_settings.memory_enabled = False
        mock_settings.browser_enabled = False

        client = MagicMock()

        from pile.agents.triage import TRIAGE_INSTRUCTIONS_NO_MEMORY, create_triage_agent

        create_triage_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        assert call_kwargs["name"] == "TriageAgent"
        assert call_kwargs["tools"] == []
        assert call_kwargs["instructions"] == TRIAGE_INSTRUCTIONS_NO_MEMORY

    @patch("pile.agents.triage.settings")
    def test_memory_enabled(self, mock_settings):
        mock_settings.memory_enabled = True
        mock_settings.browser_enabled = False

        client = MagicMock()
        mock_mem_mod = MagicMock()

        from pile.agents.triage import TRIAGE_INSTRUCTIONS, create_triage_agent

        with patch.dict("sys.modules", {"pile.tools.memory_tools": mock_mem_mod}):
            create_triage_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        # 6 memory tools
        assert len(call_kwargs["tools"]) == 6
        assert call_kwargs["instructions"] == TRIAGE_INSTRUCTIONS

    @patch("pile.agents.triage.settings")
    def test_browser_enabled_no_memory(self, mock_settings):
        mock_settings.memory_enabled = False
        mock_settings.browser_enabled = True

        client = MagicMock()
        mock_browser_mod = MagicMock()

        from pile.agents.triage import TRIAGE_INSTRUCTIONS_NO_MEMORY, create_triage_agent

        with patch.dict("sys.modules", {"pile.tools.browser_tools": mock_browser_mod}):
            create_triage_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        # 6 browser tools
        assert len(call_kwargs["tools"]) == 6
        assert call_kwargs["instructions"] == TRIAGE_INSTRUCTIONS_NO_MEMORY

    @patch("pile.agents.triage.settings")
    def test_both_memory_and_browser_enabled(self, mock_settings):
        mock_settings.memory_enabled = True
        mock_settings.browser_enabled = True

        client = MagicMock()
        mock_mem_mod = MagicMock()
        mock_browser_mod = MagicMock()

        from pile.agents.triage import TRIAGE_INSTRUCTIONS, create_triage_agent

        with patch.dict("sys.modules", {
            "pile.tools.memory_tools": mock_mem_mod,
            "pile.tools.browser_tools": mock_browser_mod,
        }):
            create_triage_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        # 6 memory + 6 browser = 12
        assert len(call_kwargs["tools"]) == 12
        assert call_kwargs["instructions"] == TRIAGE_INSTRUCTIONS

    @patch("pile.agents.triage.settings")
    def test_middleware_passed_through(self, mock_settings):
        mock_settings.memory_enabled = False
        mock_settings.browser_enabled = False

        client = MagicMock()
        middleware = MagicMock()

        from pile.agents.triage import create_triage_agent

        create_triage_agent(client, middleware=middleware)

        call_kwargs = client.as_agent.call_args.kwargs
        assert call_kwargs["middleware"] is middleware

    @patch("pile.agents.triage.settings")
    def test_agent_description(self, mock_settings):
        mock_settings.memory_enabled = False
        mock_settings.browser_enabled = False

        client = MagicMock()

        from pile.agents.triage import create_triage_agent

        create_triage_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        assert "Routes" in call_kwargs["description"]
        assert "memory" in call_kwargs["description"].lower() or "browser" in call_kwargs["description"].lower()


class TestTriageInstructions:
    """Test instruction templates."""

    def test_triage_instructions_content(self):
        from pile.agents.triage import TRIAGE_INSTRUCTIONS

        assert "memory" in TRIAGE_INSTRUCTIONS.lower()
        assert "browser" in TRIAGE_INSTRUCTIONS.lower()
        assert "Jira" in TRIAGE_INSTRUCTIONS

    def test_triage_no_memory_instructions_content(self):
        from pile.agents.triage import TRIAGE_INSTRUCTIONS_NO_MEMORY

        assert "Jira" in TRIAGE_INSTRUCTIONS_NO_MEMORY
