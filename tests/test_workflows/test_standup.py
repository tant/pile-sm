"""Tests for pile.workflows.standup — Sequential standup report workflow."""

from unittest.mock import MagicMock, patch


class TestCreateWorkflow:
    @patch("pile.workflows.standup.create_git_agent", return_value=None)
    @patch("pile.workflows.standup.create_scrum_agent")
    @patch("pile.workflows.standup.create_jira_query_agent")
    @patch("pile.workflows.standup.create_client")
    def test_creates_workflow_without_git(
        self, mock_client, mock_jira, mock_scrum, mock_git
    ):
        mock_client.return_value = MagicMock()
        mock_jira.return_value = MagicMock(name="JiraQuery")
        mock_scrum.return_value = MagicMock(name="Scrum")

        with patch("agent_framework.orchestrations.SequentialBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()

            from pile.workflows.standup import create_workflow
            result = create_workflow()

            mock_builder_cls.assert_called_once()
            call_kwargs = mock_builder_cls.call_args
            participants = call_kwargs.kwargs.get("participants")

            # Without git: [jira_query, scrum]
            assert len(participants) == 2
            mock_builder.build.assert_called_once()

    @patch("pile.workflows.standup.create_git_agent")
    @patch("pile.workflows.standup.create_scrum_agent")
    @patch("pile.workflows.standup.create_jira_query_agent")
    @patch("pile.workflows.standup.create_client")
    def test_creates_workflow_with_git(
        self, mock_client, mock_jira, mock_scrum, mock_git
    ):
        mock_client.return_value = MagicMock()
        mock_jira.return_value = MagicMock(name="JiraQuery")
        mock_scrum.return_value = MagicMock(name="Scrum")
        mock_git.return_value = MagicMock(name="Git")

        with patch("agent_framework.orchestrations.SequentialBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()

            from pile.workflows.standup import create_workflow
            result = create_workflow()

            call_kwargs = mock_builder_cls.call_args
            participants = call_kwargs.kwargs.get("participants")

            # With git: [jira_query, git, scrum]
            assert len(participants) == 3

    @patch("pile.workflows.standup.create_git_agent", return_value=None)
    @patch("pile.workflows.standup.create_scrum_agent")
    @patch("pile.workflows.standup.create_jira_query_agent")
    @patch("pile.workflows.standup.create_client")
    def test_participant_order(
        self, mock_client, mock_jira, mock_scrum, mock_git
    ):
        mock_client.return_value = MagicMock()
        jira = MagicMock(name="JiraQuery")
        scrum = MagicMock(name="Scrum")
        mock_jira.return_value = jira
        mock_scrum.return_value = scrum

        with patch("agent_framework.orchestrations.SequentialBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()

            from pile.workflows.standup import create_workflow
            create_workflow()

            call_kwargs = mock_builder_cls.call_args
            participants = call_kwargs.kwargs.get("participants")

            # Jira first, scrum last
            assert participants[0] is jira
            assert participants[-1] is scrum

    @patch("pile.workflows.standup.create_git_agent")
    @patch("pile.workflows.standup.create_scrum_agent")
    @patch("pile.workflows.standup.create_jira_query_agent")
    @patch("pile.workflows.standup.create_client")
    def test_git_inserted_in_middle(
        self, mock_client, mock_jira, mock_scrum, mock_git
    ):
        mock_client.return_value = MagicMock()
        jira = MagicMock(name="JiraQuery")
        scrum = MagicMock(name="Scrum")
        git = MagicMock(name="Git")
        mock_jira.return_value = jira
        mock_scrum.return_value = scrum
        mock_git.return_value = git

        with patch("agent_framework.orchestrations.SequentialBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()

            from pile.workflows.standup import create_workflow
            create_workflow()

            call_kwargs = mock_builder_cls.call_args
            participants = call_kwargs.kwargs.get("participants")

            # Order: jira, git, scrum
            assert participants[0] is jira
            assert participants[1] is git
            assert participants[2] is scrum
