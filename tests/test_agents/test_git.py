"""Tests for Git Agent creation and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pile.config import GitRepo


class TestCreateGitAgent:
    """Test create_git_agent under various config scenarios."""

    @patch("pile.agents.git.settings")
    def test_no_repos_returns_none(self, mock_settings):
        mock_settings.git_repo_list = []

        client = MagicMock()

        from pile.agents.git import create_git_agent

        result = create_git_agent(client)

        assert result is None
        client.as_agent.assert_not_called()

    @patch("pile.agents.git.settings")
    def test_single_public_repo(self, mock_settings):
        mock_settings.git_repo_list = [GitRepo(path="/home/user/project")]

        client = MagicMock()
        client.as_agent.return_value = MagicMock(name="GitAgent")

        from pile.agents.git import create_git_agent

        agent = create_git_agent(client)

        assert agent is not None
        call_kwargs = client.as_agent.call_args.kwargs
        assert call_kwargs["name"] == "GitAgent"
        assert "/home/user/project" in call_kwargs["instructions"]
        assert "private" not in call_kwargs["instructions"]
        assert len(call_kwargs["tools"]) == 5

    @patch("pile.agents.git.settings")
    def test_repo_with_credentials(self, mock_settings):
        repo = GitRepo(path="/private/repo", token="ghp_secret123")
        mock_settings.git_repo_list = [repo]

        client = MagicMock()

        from pile.agents.git import create_git_agent

        create_git_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        assert "private, credentials configured" in call_kwargs["instructions"]
        assert "/private/repo" in call_kwargs["instructions"]

    @patch("pile.agents.git.settings")
    def test_multiple_repos(self, mock_settings):
        mock_settings.git_repo_list = [
            GitRepo(path="/repo/a"),
            GitRepo(path="/repo/b", token="tok"),
            GitRepo(path="/repo/c"),
        ]

        client = MagicMock()

        from pile.agents.git import create_git_agent

        create_git_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        instructions = call_kwargs["instructions"]
        assert "/repo/a" in instructions
        assert "/repo/b" in instructions
        assert "/repo/c" in instructions

    @patch("pile.agents.git.settings")
    def test_middleware_passed_through(self, mock_settings):
        mock_settings.git_repo_list = [GitRepo(path="/repo")]

        client = MagicMock()
        middleware = MagicMock()

        from pile.agents.git import create_git_agent

        create_git_agent(client, middleware=middleware)

        call_kwargs = client.as_agent.call_args.kwargs
        assert call_kwargs["middleware"] is middleware

    @patch("pile.agents.git.settings")
    def test_tools_registered(self, mock_settings):
        mock_settings.git_repo_list = [GitRepo(path="/repo")]

        client = MagicMock()

        from pile.agents.git import create_git_agent, git_blame, git_branch_list, git_diff, git_log, git_show

        create_git_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        tools = call_kwargs["tools"]
        assert git_log in tools
        assert git_diff in tools
        assert git_branch_list in tools
        assert git_show in tools
        assert git_blame in tools

    @patch("pile.agents.git.settings")
    def test_description_content(self, mock_settings):
        mock_settings.git_repo_list = [GitRepo(path="/repo")]

        client = MagicMock()

        from pile.agents.git import create_git_agent

        create_git_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        assert "Git" in call_kwargs["description"]

    @patch("pile.agents.git.settings")
    def test_repo_without_credentials_no_private_label(self, mock_settings):
        repo = GitRepo(path="/public/repo", username="user")
        # username alone is not enough for has_credentials
        mock_settings.git_repo_list = [repo]

        client = MagicMock()

        from pile.agents.git import create_git_agent

        create_git_agent(client)

        call_kwargs = client.as_agent.call_args.kwargs
        assert "private" not in call_kwargs["instructions"]


class TestGitInstructions:
    """Test the instruction template."""

    def test_git_instructions_has_repos_placeholder(self):
        from pile.agents.git import GIT_INSTRUCTIONS

        assert "{repos}" in GIT_INSTRUCTIONS

    def test_git_instructions_content(self):
        from pile.agents.git import GIT_INSTRUCTIONS

        assert "Git specialist" in GIT_INSTRUCTIONS
        assert "commit" in GIT_INSTRUCTIONS.lower()
