"""Tests for Scrum Agent creation and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pile.config import GitRepo


class TestCreateScrumAgent:
    """Test create_scrum_agent under various config scenarios."""

    @patch("pile.agents.scrum.settings")
    def test_no_git_no_memory(self, mock_settings):
        mock_settings.git_repo_list = []
        mock_settings.memory_enabled = False
        mock_settings.jira_project_key = "TEST"
        mock_settings.jira_base_url = "https://test.atlassian.net"

        client = MagicMock()
        client.as_agent.return_value = MagicMock(name="ScrumAgent")

        mock_jira_mod = MagicMock()
        from pile.agents.scrum import create_scrum_agent

        with patch.dict("sys.modules", {"pile.tools.jira_tools": mock_jira_mod}):
            agent = create_scrum_agent(client)

        client.as_agent.assert_called_once()
        call_kwargs = client.as_agent.call_args.kwargs
        assert call_kwargs["name"] == "ScrumAgent"
        # 7 jira tools (get_current_sprint_info, search_project_issues, jira_search,
        # jira_get_issue, jira_get_board, jira_get_sprint_issues, jira_get_changelog)
        assert len(call_kwargs["tools"]) == 7
        assert "TEST" in call_kwargs["instructions"]
        assert "Git is not configured" in call_kwargs["instructions"]

    @patch("pile.agents.scrum.settings")
    def test_with_git(self, mock_settings):
        mock_settings.git_repo_list = [GitRepo(path="/repo/a")]
        mock_settings.memory_enabled = False
        mock_settings.jira_project_key = "PROJ"
        mock_settings.jira_base_url = "https://proj.atlassian.net"

        client = MagicMock()

        mock_jira_mod = MagicMock()
        mock_git_mod = MagicMock()
        from pile.agents.scrum import create_scrum_agent

        with patch.dict("sys.modules", {
            "pile.tools.jira_tools": mock_jira_mod,
            "pile.tools.git_tools": mock_git_mod,
        }):
            agent = create_scrum_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        # 7 jira + 2 git = 9
        assert len(call_kwargs["tools"]) == 9
        assert "git_log" in call_kwargs["instructions"]
        assert "/repo/a" in call_kwargs["instructions"]

    @patch("pile.agents.scrum.settings")
    def test_with_memory(self, mock_settings):
        mock_settings.git_repo_list = []
        mock_settings.memory_enabled = True
        mock_settings.jira_project_key = "MEM"
        mock_settings.jira_base_url = "https://mem.atlassian.net"

        client = MagicMock()
        mock_jira_mod = MagicMock()
        mock_mem_mod = MagicMock()

        from pile.agents.scrum import create_scrum_agent

        with patch.dict("sys.modules", {
            "pile.tools.jira_tools": mock_jira_mod,
            "pile.tools.memory_tools": mock_mem_mod,
        }):
            agent = create_scrum_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        # 7 jira + 1 memory = 8
        assert len(call_kwargs["tools"]) == 8
        assert "memory_search" in call_kwargs["instructions"]

    @patch("pile.agents.scrum.settings")
    def test_with_git_and_memory(self, mock_settings):
        mock_settings.git_repo_list = [GitRepo(path="/repo/x")]
        mock_settings.memory_enabled = True
        mock_settings.jira_project_key = "ALL"
        mock_settings.jira_base_url = "https://all.atlassian.net"

        client = MagicMock()

        mock_jira_mod = MagicMock()
        mock_git_mod = MagicMock()
        mock_mem_mod = MagicMock()

        from pile.agents.scrum import create_scrum_agent

        with patch.dict("sys.modules", {
            "pile.tools.jira_tools": mock_jira_mod,
            "pile.tools.git_tools": mock_git_mod,
            "pile.tools.memory_tools": mock_mem_mod,
        }):
            agent = create_scrum_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        # 7 jira + 2 git + 1 memory = 10
        assert len(call_kwargs["tools"]) == 10
        assert "/repo/x" in call_kwargs["instructions"]
        assert "memory_search" in call_kwargs["instructions"]

    @patch("pile.agents.scrum.settings")
    def test_middleware_passed_through(self, mock_settings):
        mock_settings.git_repo_list = []
        mock_settings.memory_enabled = False
        mock_settings.jira_project_key = "MW"
        mock_settings.jira_base_url = "https://mw.atlassian.net"

        client = MagicMock()
        middleware = MagicMock()
        mock_jira_mod = MagicMock()

        from pile.agents.scrum import create_scrum_agent

        with patch.dict("sys.modules", {"pile.tools.jira_tools": mock_jira_mod}):
            create_scrum_agent(client, middleware=middleware)

        call_kwargs = client.as_agent.call_args.kwargs
        assert call_kwargs["middleware"] is middleware

    @patch("pile.agents.scrum.settings")
    def test_description_is_set(self, mock_settings):
        mock_settings.git_repo_list = []
        mock_settings.memory_enabled = False
        mock_settings.jira_project_key = "D"
        mock_settings.jira_base_url = "https://d.atlassian.net"

        client = MagicMock()
        mock_jira_mod = MagicMock()

        from pile.agents.scrum import create_scrum_agent

        with patch.dict("sys.modules", {"pile.tools.jira_tools": mock_jira_mod}):
            create_scrum_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        assert "Scrum Master" in call_kwargs["description"]

    @patch("pile.agents.scrum.settings")
    def test_jira_url_in_instructions(self, mock_settings):
        mock_settings.git_repo_list = []
        mock_settings.memory_enabled = False
        mock_settings.jira_project_key = "URL"
        mock_settings.jira_base_url = "https://url.atlassian.net"

        client = MagicMock()
        mock_jira_mod = MagicMock()

        from pile.agents.scrum import create_scrum_agent

        with patch.dict("sys.modules", {"pile.tools.jira_tools": mock_jira_mod}):
            create_scrum_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        assert "https://url.atlassian.net" in call_kwargs["instructions"]

    @patch("pile.agents.scrum.settings")
    def test_kwargs_accepted(self, mock_settings):
        """Verify create_scrum_agent accepts **kwargs without error."""
        mock_settings.git_repo_list = []
        mock_settings.memory_enabled = False
        mock_settings.jira_project_key = "KW"
        mock_settings.jira_base_url = "https://kw.atlassian.net"

        client = MagicMock()
        mock_jira_mod = MagicMock()

        from pile.agents.scrum import create_scrum_agent

        with patch.dict("sys.modules", {"pile.tools.jira_tools": mock_jira_mod}):
            create_scrum_agent(client, extra_param="ignored")

        client.as_agent.assert_called_once()


class TestScrumInstructions:
    """Test that instruction templates contain expected content."""

    def test_scrum_instructions_placeholders(self):
        from pile.agents.scrum import SCRUM_INSTRUCTIONS

        assert "{project_key}" in SCRUM_INSTRUCTIONS
        assert "{jira_url}" in SCRUM_INSTRUCTIONS
        assert "{git_note}" in SCRUM_INSTRUCTIONS
        assert "{memory_note}" in SCRUM_INSTRUCTIONS

    def test_scrum_instructions_content(self):
        from pile.agents.scrum import SCRUM_INSTRUCTIONS

        assert "Scrum Master" in SCRUM_INSTRUCTIONS
