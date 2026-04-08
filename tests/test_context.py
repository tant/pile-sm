"""Tests for session memory summarization and recall."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


def test_summarize_turn_extracts_facts():
    """summarize_turn calls router model and stores non-duplicate facts."""
    stored = []

    def mock_add(content, memory_type, source):
        stored.append(content)
        return f"mem_{len(stored)}"

    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router, \
         patch("pile.memory.store.search_memories", return_value=[]), \
         patch("pile.memory.store.add_memory", side_effect=mock_add):
        mock_settings.memory_enabled = True
        mock_router.return_value = "- Sprint 5: 68 tasks\n- Done: 11"

        from pile.context import summarize_turn
        summarize_turn("Sprint thế nào?", "Sprint 5 có 68 tasks, 11 done.")

    assert len(stored) == 2
    assert "Sprint 5: 68 tasks" in stored[0]
    assert "Done: 11" in stored[1]


def test_summarize_turn_skips_duplicates():
    """summarize_turn skips facts that already exist in memory."""
    stored = []

    def mock_search(query, n_results=1):
        if "Sprint 5" in query:
            return [{"distance": 0.1, "content": "Sprint 5: 68 tasks"}]
        return []

    def mock_add(content, memory_type, source):
        stored.append(content)
        return "mem_1"

    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router, \
         patch("pile.memory.store.search_memories", side_effect=mock_search), \
         patch("pile.memory.store.add_memory", side_effect=mock_add):
        mock_settings.memory_enabled = True
        mock_router.return_value = "- Sprint 5: 68 tasks\n- Blocker: TETRA-101"

        from pile.context import summarize_turn
        summarize_turn("Sprint thế nào?", "Sprint 5 có 68 tasks. Blocker TETRA-101.")

    assert len(stored) == 1
    assert "TETRA-101" in stored[0]


def test_summarize_turn_skips_none_response():
    """summarize_turn returns early when router returns NONE."""
    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router, \
         patch("pile.memory.store.add_memory") as mock_add:
        mock_settings.memory_enabled = True
        mock_router.return_value = "NONE"

        from pile.context import summarize_turn
        summarize_turn("hello", "Hi there! How can I help you today? I'm here to assist.")

    mock_add.assert_not_called()


def test_summarize_turn_skips_short_response():
    """summarize_turn skips when agent response is too short."""
    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router:
        mock_settings.memory_enabled = True

        from pile.context import summarize_turn
        summarize_turn("hi", "Hello!")

    mock_router.assert_not_called()


def test_summarize_turn_skips_when_memory_disabled():
    """summarize_turn does nothing when memory is disabled."""
    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router:
        mock_settings.memory_enabled = False

        from pile.context import summarize_turn
        summarize_turn("Sprint?", "Sprint 5 has 68 tasks and is running from April 6 to April 12.")

    mock_router.assert_not_called()


def test_recall_facts_returns_list():
    """recall_facts returns a list of fact strings."""
    with patch("pile.context.settings") as mock_settings, \
         patch("pile.memory.store.search_memories") as mock_search:
        mock_settings.memory_enabled = True
        mock_search.return_value = [
            {"content": "Sprint 5: 68 tasks", "distance": 0.5, "metadata": {"type": "session_fact"}},
            {"content": "Tân: PR-PO Epic 4", "distance": 0.6, "metadata": {"type": "session_fact"}},
            {"content": "irrelevant", "distance": 0.9, "metadata": {"type": "session_fact"}},
        ]

        from pile.context import recall_facts
        facts = recall_facts("sprint status")

    assert len(facts) == 2
    assert "Sprint 5: 68 tasks" in facts
    assert "Tân: PR-PO Epic 4" in facts
