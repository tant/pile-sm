"""Tests for pile.router — keyword, LLM classifier, and embedding routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pile.router import (
    _VALID_AGENTS,
    _cosine_similarity,
    route_query,
    route_query_with_embedding,
    route_query_with_llm,
    smart_route,
)


# ---------------------------------------------------------------------------
# Phase 1: keyword matching (route_query)
# ---------------------------------------------------------------------------


class TestRouteQueryKeywords:
    """route_query returns the correct agent for keyword-matched queries."""

    @pytest.mark.parametrize("query,expected", [
        ("hello", "triage"),
        ("Hi there", "triage"),
        ("hey!", "triage"),
        ("good morning", "triage"),
        ("xin chao", "triage"),
        ("chao ban", "triage"),
    ])
    def test_greeting_routes_to_triage(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("remember this", "memory"),
        ("forget about that", "memory"),
        ("load file abc.pdf", "memory"),
        ("ingest documents", "memory"),
        ("knowledge base search", "memory"),
    ])
    def test_memory_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("open url https://example.com", "browser"),
        ("screenshot the page", "browser"),
        ("scrape that site", "browser"),
        ("browse the page", "browser"),
        ("https://vnexpress.net", "browser"),
        ("check github.com/org/repo", "browser"),
    ])
    def test_browser_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("curl the API", "jira_query"),
        ("changelog for TETRA-100", "jira_query"),
    ])
    def test_jira_query_curl_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("show in progress issues", "jira_query"),
        ("code review tickets", "jira_query"),
        ("to do items", "jira_query"),
    ])
    def test_jira_status_filter_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("create issue for login bug", "jira_write"),
        ("assign to Bob", "jira_write"),
        ("transition this ticket", "jira_write"),
        ("comment on issue", "jira_write"),
        ("update issue fields", "jira_write"),
    ])
    def test_jira_write_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("show the board", "board"),
        ("board config", "board"),
    ])
    def test_board_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("standup report", "scrum"),
        ("velocity chart", "scrum"),
        ("workload analysis", "scrum"),
        ("burndown chart", "scrum"),
        ("blockers list", "scrum"),
        ("summary of sprint", "scrum"),
        ("retro notes", "scrum"),
    ])
    def test_scrum_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("create sprint", "sprint"),
        ("move to backlog", "sprint"),
        ("list sprints", "sprint"),
        ("sprint nao dang active", "sprint"),
    ])
    def test_sprint_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("show the epic", "epic"),
        ("backlog items", "epic"),
    ])
    def test_epic_keywords(self, query, expected):
        assert route_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("git log", "git"),
        ("show commit history", "git"),
        ("branch list", "git"),
        ("diff between main and dev", "git"),
        ("blame this file", "git"),
    ])
    def test_git_keywords(self, query, expected):
        assert route_query(query) == expected

    def test_issue_key_routes_to_jira_query(self):
        assert route_query("TETRA-123") == "jira_query"
        assert route_query("What is PROJ-42?") == "jira_query"

    @pytest.mark.parametrize("query,expected", [
        ("search for bugs", "jira_query"),
        ("ticket status", "jira_query"),
    ])
    def test_jira_query_broad_keywords(self, query, expected):
        assert route_query(query) == expected

    def test_no_match_returns_none(self):
        assert route_query("") is None
        assert route_query("something totally unrelated xyz") is None

    def test_empty_query(self):
        assert route_query("") is None

    def test_whitespace_only(self):
        assert route_query("   ") is None

    def test_case_insensitivity(self):
        assert route_query("HELLO") == "triage"
        assert route_query("GIT log") == "git"

    def test_first_match_wins(self):
        # "epic TETRA-893" should match epic before jira_query issue-key pattern
        assert route_query("epic TETRA-893") == "epic"


# ---------------------------------------------------------------------------
# Phase 2: LLM classifier (route_query_with_llm)
# ---------------------------------------------------------------------------


class TestRouteQueryWithLLM:
    """route_query_with_llm delegates to call_router_model and parses result."""

    @patch("pile.client.call_router_model")
    def test_valid_agent_returned(self, mock_call):
        mock_call.return_value = "scrum"
        assert route_query_with_llm("team overview") == "scrum"
        mock_call.assert_called_once()

    @patch("pile.client.call_router_model")
    def test_strips_punctuation(self, mock_call):
        mock_call.return_value = "jira_query."
        assert route_query_with_llm("who has bugs?") == "jira_query"

    @patch("pile.client.call_router_model")
    def test_takes_first_word(self, mock_call):
        mock_call.return_value = "git some extra text"
        assert route_query_with_llm("show commits") == "git"

    @patch("pile.client.call_router_model")
    def test_invalid_agent_falls_back_to_triage(self, mock_call):
        mock_call.return_value = "unknown_agent"
        assert route_query_with_llm("do something weird") == "triage"

    @patch("pile.client.call_router_model")
    def test_empty_response_falls_back_to_triage(self, mock_call):
        mock_call.return_value = ""
        assert route_query_with_llm("anything") == "triage"

    @patch("pile.client.call_router_model")
    def test_none_response_falls_back_to_triage(self, mock_call):
        mock_call.return_value = None
        assert route_query_with_llm("anything") == "triage"


# ---------------------------------------------------------------------------
# Phase 3: Embedding similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0, 0], [1, 2]) == 0.0
        assert _cosine_similarity([1, 2], [0, 0]) == 0.0


class TestRouteQueryWithEmbedding:
    @patch("pile.router._embed_texts")
    @patch("pile.router._get_embeddings")
    def test_returns_best_match(self, mock_get_emb, mock_embed):
        mock_get_emb.return_value = {
            "git": [1.0, 0.0],
            "scrum": [0.0, 1.0],
        }
        mock_embed.return_value = [[0.9, 0.1]]
        assert route_query_with_embedding("show commits") == "git"

    @patch("pile.router._get_embeddings")
    def test_no_embeddings_returns_triage(self, mock_get_emb):
        mock_get_emb.return_value = {}
        assert route_query_with_embedding("anything") == "triage"

    @patch("pile.router._embed_texts", side_effect=Exception("model error"))
    @patch("pile.router._get_embeddings", return_value={"git": [1.0, 0.0]})
    def test_exception_returns_triage(self, mock_get_emb, mock_embed):
        assert route_query_with_embedding("anything") == "triage"


# ---------------------------------------------------------------------------
# smart_route — orchestrates phases
# ---------------------------------------------------------------------------


class TestSmartRoute:
    def test_keyword_match_used_first(self):
        assert smart_route("git log") == "git"

    @patch("pile.client.call_router_model", return_value="scrum")
    def test_falls_through_to_llm(self, mock_call):
        result = smart_route("how is the team doing this week?")
        assert result == "scrum"
        mock_call.assert_called_once()

    @patch("pile.client.call_router_model", return_value="triage")
    def test_llm_returns_triage_for_ambiguous(self, mock_call):
        result = smart_route("something very vague")
        assert result == "triage"


# ---------------------------------------------------------------------------
# Valid agents constant
# ---------------------------------------------------------------------------


class TestValidAgents:
    def test_all_expected_agents_present(self):
        expected = {
            "jira_query", "jira_write", "board", "sprint", "epic",
            "git", "scrum", "memory", "browser", "triage",
        }
        assert _VALID_AGENTS == expected
