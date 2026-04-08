"""Tests for pile.workflows.planning — GroupChat sprint planning workflow."""

from unittest.mock import MagicMock, patch


class TestCreateWorkflow:
    @patch("pile.workflows.planning.create_git_agent", return_value=None)
    @patch("pile.workflows.planning.create_scrum_agent")
    @patch("pile.workflows.planning.create_epic_agent")
    @patch("pile.workflows.planning.create_sprint_agent")
    @patch("pile.workflows.planning.create_jira_query_agent")
    @patch("pile.workflows.planning.create_client")
    def test_creates_workflow_without_git(
        self, mock_client, mock_jira, mock_sprint, mock_epic, mock_scrum, mock_git
    ):
        mock_client.return_value = MagicMock()
        mock_jira.return_value = MagicMock(name="JiraQuery")
        mock_sprint.return_value = MagicMock(name="Sprint")
        mock_epic.return_value = MagicMock(name="Epic")
        mock_scrum.return_value = MagicMock(name="Scrum")

        with patch("agent_framework.orchestrations.GroupChatBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()

            from pile.workflows.planning import create_workflow
            result = create_workflow()

            mock_builder_cls.assert_called_once()
            call_kwargs = mock_builder_cls.call_args
            participants = call_kwargs.kwargs.get("participants")

            # Without git, participants = [jira, sprint, epic, scrum]
            assert len(participants) == 4
            mock_builder.build.assert_called_once()

    @patch("pile.workflows.planning.create_git_agent")
    @patch("pile.workflows.planning.create_scrum_agent")
    @patch("pile.workflows.planning.create_epic_agent")
    @patch("pile.workflows.planning.create_sprint_agent")
    @patch("pile.workflows.planning.create_jira_query_agent")
    @patch("pile.workflows.planning.create_client")
    def test_creates_workflow_with_git(
        self, mock_client, mock_jira, mock_sprint, mock_epic, mock_scrum, mock_git
    ):
        mock_client.return_value = MagicMock()
        mock_jira.return_value = MagicMock(name="JiraQuery")
        mock_sprint.return_value = MagicMock(name="Sprint")
        mock_epic.return_value = MagicMock(name="Epic")
        mock_scrum.return_value = MagicMock(name="Scrum")
        mock_git.return_value = MagicMock(name="Git")

        with patch("agent_framework.orchestrations.GroupChatBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()

            from pile.workflows.planning import create_workflow
            result = create_workflow()

            call_kwargs = mock_builder_cls.call_args
            participants = call_kwargs.kwargs.get("participants")

            # With git, participants = [jira, sprint, epic, git, scrum]
            assert len(participants) == 5

    @patch("pile.workflows.planning.create_git_agent", return_value=None)
    @patch("pile.workflows.planning.create_scrum_agent")
    @patch("pile.workflows.planning.create_epic_agent")
    @patch("pile.workflows.planning.create_sprint_agent")
    @patch("pile.workflows.planning.create_jira_query_agent")
    @patch("pile.workflows.planning.create_client")
    def test_round_robin_selection(
        self, mock_client, mock_jira, mock_sprint, mock_epic, mock_scrum, mock_git
    ):
        mock_client.return_value = MagicMock()
        mock_jira.return_value = MagicMock(name="JiraQuery")
        mock_sprint.return_value = MagicMock(name="Sprint")
        mock_epic.return_value = MagicMock(name="Epic")
        mock_scrum.return_value = MagicMock(name="Scrum")

        with patch("agent_framework.orchestrations.GroupChatBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()

            from pile.workflows.planning import create_workflow
            create_workflow()

            call_kwargs = mock_builder_cls.call_args
            selection_func = call_kwargs.kwargs.get("selection_func")

            state = MagicMock()
            state.participants = {"A": 1, "B": 2, "C": 3}
            state.current_round = 0
            assert selection_func(state) == "A"
            state.current_round = 1
            assert selection_func(state) == "B"
            state.current_round = 2
            assert selection_func(state) == "C"
            state.current_round = 3
            assert selection_func(state) == "A"  # wraps around

    @patch("pile.workflows.planning.create_git_agent", return_value=None)
    @patch("pile.workflows.planning.create_scrum_agent")
    @patch("pile.workflows.planning.create_epic_agent")
    @patch("pile.workflows.planning.create_sprint_agent")
    @patch("pile.workflows.planning.create_jira_query_agent")
    @patch("pile.workflows.planning.create_client")
    def test_termination_condition(
        self, mock_client, mock_jira, mock_sprint, mock_epic, mock_scrum, mock_git
    ):
        mock_client.return_value = MagicMock()
        mock_jira.return_value = MagicMock(name="JiraQuery")
        mock_sprint.return_value = MagicMock(name="Sprint")
        mock_epic.return_value = MagicMock(name="Epic")
        mock_scrum.return_value = MagicMock(name="Scrum")

        with patch("agent_framework.orchestrations.GroupChatBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder_cls.return_value = mock_builder
            mock_builder.build.return_value = MagicMock()

            from pile.workflows.planning import create_workflow
            create_workflow()

            call_kwargs = mock_builder_cls.call_args
            term_cond = call_kwargs.kwargs.get("termination_condition")

            # Under 8 assistant messages = not terminated
            msgs = [MagicMock(role="assistant") for _ in range(7)]
            msgs.append(MagicMock(role="user"))
            assert term_cond(msgs) is False

            # 8+ assistant messages = terminated
            msgs = [MagicMock(role="assistant") for _ in range(8)]
            assert term_cond(msgs) is True
