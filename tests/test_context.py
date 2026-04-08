"""Tests for context module: recall, learn, compress, summarize, and parse."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from pile.context import _parse_facts


# ---- _parse_facts ----

def test_parse_facts_bullet_points():
    """_parse_facts extracts bullet-pointed facts."""
    text = "- Sprint 5 has 68 tasks\n- Done: 11 tasks\n* Blocker: TETRA-101"
    facts = _parse_facts(text)
    assert len(facts) == 3
    assert "Sprint 5 has 68 tasks" in facts[0]
    assert "Done: 11 tasks" in facts[1]
    assert "Blocker: TETRA-101" in facts[2]


def test_parse_facts_skips_short_lines():
    """_parse_facts ignores lines with 5 or fewer characters."""
    text = "- ok\n- This is a real fact\n- hi"
    facts = _parse_facts(text)
    assert len(facts) == 1
    assert "This is a real fact" in facts[0]


def test_parse_facts_skips_none():
    """_parse_facts ignores lines that say NONE."""
    text = "NONE\n- NONE\n- A valid fact here"
    facts = _parse_facts(text)
    assert len(facts) == 1
    assert "A valid fact here" in facts[0]


def test_parse_facts_empty_input():
    """_parse_facts returns empty list for empty input."""
    assert _parse_facts("") == []
    assert _parse_facts("   ") == []


def test_parse_facts_no_bullets():
    """_parse_facts handles lines without bullet markers."""
    text = "Sprint 5 has 68 tasks\nDone: 11 tasks"
    facts = _parse_facts(text)
    assert len(facts) == 2


# ---- recall ----

def test_recall_returns_formatted_hint():
    """recall returns formatted hint string when memories found."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories") as mock_search:
        mock_settings.memory_enabled = True
        mock_search.return_value = [
            {"content": "Sprint 5: 68 tasks", "distance": 0.4},
            {"content": "Deadline is Friday", "distance": 0.6},
        ]

        from pile.context import recall
        result = recall("sprint status")

    assert "Relevant context from memory:" in result
    assert "Sprint 5: 68 tasks" in result
    assert "Deadline is Friday" in result


def test_recall_filters_by_distance():
    """recall excludes memories with distance >= RECALL_MAX_DISTANCE."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories") as mock_search:
        mock_settings.memory_enabled = True
        mock_search.return_value = [
            {"content": "relevant", "distance": 0.5},
            {"content": "irrelevant", "distance": 0.9},
        ]

        from pile.context import recall
        result = recall("test query")

    assert "relevant" in result
    assert "irrelevant" not in result


def test_recall_returns_empty_when_no_results():
    """recall returns empty string when search returns nothing."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories", return_value=[]):
        mock_settings.memory_enabled = True

        from pile.context import recall
        result = recall("something")

    assert result == ""


def test_recall_returns_empty_when_memory_disabled():
    """recall returns empty string when memory is disabled."""
    with patch("pile.config.settings") as mock_settings:
        mock_settings.memory_enabled = False

        from pile.context import recall
        result = recall("test")

    assert result == ""


def test_recall_returns_empty_on_exception():
    """recall returns empty string when an exception occurs."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories", side_effect=RuntimeError("db error")):
        mock_settings.memory_enabled = True

        from pile.context import recall
        result = recall("test")

    assert result == ""


def test_recall_skips_empty_content():
    """recall skips memories with empty content."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories") as mock_search:
        mock_settings.memory_enabled = True
        mock_search.return_value = [
            {"content": "", "distance": 0.3},
            {"content": None, "distance": 0.3},
            {"content": "valid fact", "distance": 0.3},
        ]

        from pile.context import recall
        result = recall("test")

    assert "valid fact" in result
    assert result.count("- ") == 1


def test_recall_returns_empty_when_all_filtered():
    """recall returns empty when all results are above distance threshold."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories") as mock_search:
        mock_settings.memory_enabled = True
        mock_search.return_value = [
            {"content": "too far", "distance": 1.5},
        ]

        from pile.context import recall
        result = recall("test")

    assert result == ""


# ---- learn ----

def test_learn_saves_compressed_lesson():
    """learn compresses lesson and saves to memory."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories", return_value=[]), \
         patch("pile.memory.store.add_memory") as mock_add, \
         patch("pile.context._compress", return_value="Compressed fact about testing"):
        mock_settings.memory_enabled = True

        from pile.context import learn
        learn("test query", "A long lesson about how testing works in the project")

    mock_add.assert_called_once_with("Compressed fact about testing", memory_type="auto_learn", source="system")


def test_learn_skips_when_memory_disabled():
    """learn does nothing when memory is disabled."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.add_memory") as mock_add:
        mock_settings.memory_enabled = False

        from pile.context import learn
        learn("query", "lesson text")

    mock_add.assert_not_called()


def test_learn_skips_duplicate():
    """learn skips saving when similar memory already exists."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories") as mock_search, \
         patch("pile.memory.store.add_memory") as mock_add:
        mock_settings.memory_enabled = True
        mock_search.return_value = [{"distance": 0.1, "content": "existing fact"}]

        from pile.context import learn
        learn("query", "existing fact about testing")

    mock_add.assert_not_called()


def test_learn_skips_short_compressed():
    """learn skips saving when compressed text is too short."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories", return_value=[]), \
         patch("pile.memory.store.add_memory") as mock_add, \
         patch("pile.context._compress", return_value="ok"):
        mock_settings.memory_enabled = True

        from pile.context import learn
        learn("query", "lesson")

    mock_add.assert_not_called()


def test_learn_skips_empty_compressed():
    """learn skips saving when compressed text is empty."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories", return_value=[]), \
         patch("pile.memory.store.add_memory") as mock_add, \
         patch("pile.context._compress", return_value=""):
        mock_settings.memory_enabled = True

        from pile.context import learn
        learn("query", "lesson")

    mock_add.assert_not_called()


def test_learn_returns_on_exception():
    """learn handles exceptions gracefully."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories", side_effect=RuntimeError("db error")):
        mock_settings.memory_enabled = True

        from pile.context import learn
        learn("query", "lesson")  # should not raise


# ---- _compress ----

def test_compress_uses_router_model():
    """_compress returns router model output when available."""
    with patch("pile.client.call_router_model", return_value="Compressed statement"):
        from pile.context import _compress
        result = _compress("A long explanation about testing")

    assert result == "Compressed statement"


def test_compress_truncates_when_router_returns_none():
    """_compress truncates text when router model returns None."""
    with patch("pile.client.call_router_model", return_value=None):
        from pile.context import _compress
        result = _compress("Short text")

    assert result == "Short text"


def test_compress_truncates_long_text_at_word_boundary():
    """_compress truncates at word boundary when text exceeds 200 chars."""
    long_text = "word " * 60  # 300 chars
    with patch("pile.client.call_router_model", return_value=None):
        from pile.context import _compress
        result = _compress(long_text)

    assert len(result) <= 200
    assert not result.endswith(" ")


def test_compress_returns_empty_string_from_router():
    """_compress falls back to truncation when router returns empty string."""
    with patch("pile.client.call_router_model", return_value=""):
        from pile.context import _compress
        result = _compress("Some lesson text here")

    # empty string is falsy, so it should fall back to truncation
    assert result == "Some lesson text here"


# ---- summarize_turn ----

def test_summarize_turn_extracts_facts():
    """summarize_turn calls router model and stores non-duplicate facts."""
    stored = []

    def mock_add(content, memory_type, source):
        stored.append(content)
        return f"mem_{len(stored)}"

    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model") as mock_router, \
         patch("pile.memory.store.search_memories", return_value=[]), \
         patch("pile.memory.store.add_memory", side_effect=mock_add):
        mock_settings.memory_enabled = True
        mock_router.return_value = "- Sprint 5: 68 tasks\n- Done: 11"

        from pile.context import summarize_turn
        summarize_turn("Sprint thế nào?", "Sprint 5 có 68 tasks: 35 To Do, 15 Testing, 1 In Progress, 11 Done.")

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

    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model") as mock_router, \
         patch("pile.memory.store.search_memories", side_effect=mock_search), \
         patch("pile.memory.store.add_memory", side_effect=mock_add):
        mock_settings.memory_enabled = True
        mock_router.return_value = "- Sprint 5: 68 tasks\n- Blocker: TETRA-101"

        from pile.context import summarize_turn
        summarize_turn("Sprint thế nào?", "Sprint 5 có 68 tasks: 35 To Do, 15 Testing. Blocker TETRA-101 chưa resolve.")

    assert len(stored) == 1
    assert "TETRA-101" in stored[0]


def test_summarize_turn_skips_none_response():
    """summarize_turn returns early when router returns NONE."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model") as mock_router, \
         patch("pile.memory.store.add_memory") as mock_add:
        mock_settings.memory_enabled = True
        mock_router.return_value = "NONE"

        from pile.context import summarize_turn
        summarize_turn("hello", "Hi there! How can I help you today? I'm here to assist.")

    mock_add.assert_not_called()


def test_summarize_turn_skips_short_response():
    """summarize_turn skips when agent response is too short."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model") as mock_router:
        mock_settings.memory_enabled = True

        from pile.context import summarize_turn
        summarize_turn("hi", "Hello!")

    mock_router.assert_not_called()


def test_summarize_turn_skips_when_memory_disabled():
    """summarize_turn does nothing when memory is disabled."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model") as mock_router:
        mock_settings.memory_enabled = False

        from pile.context import summarize_turn
        summarize_turn("Sprint?", "Sprint 5 has 68 tasks and is running from April 6 to April 12.")

    mock_router.assert_not_called()


def test_summarize_turn_skips_triage_agent():
    """summarize_turn skips when agent_name is TriageAgent."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model") as mock_router:
        mock_settings.memory_enabled = True

        from pile.context import summarize_turn
        summarize_turn(
            "Sprint?",
            "Sprint 5 has 68 tasks and is running from April 6 to April 12.",
            agent_name="TriageAgent",
        )

    mock_router.assert_not_called()


def test_summarize_turn_skips_null_router_response():
    """summarize_turn returns early when router returns None."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model") as mock_router, \
         patch("pile.memory.store.add_memory") as mock_add:
        mock_settings.memory_enabled = True
        mock_router.return_value = None

        from pile.context import summarize_turn
        summarize_turn("hello", "Hi there! How can I help you today? I'm here to assist with your request.")

    mock_add.assert_not_called()


def test_summarize_turn_handles_exception():
    """summarize_turn handles exceptions gracefully."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model", side_effect=RuntimeError("model error")):
        mock_settings.memory_enabled = True

        from pile.context import summarize_turn
        summarize_turn("test", "This is a long enough response for processing by the system.")  # should not raise


def test_summarize_turn_skips_when_parse_returns_empty():
    """summarize_turn returns early when _parse_facts returns empty list."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.client.call_router_model") as mock_router, \
         patch("pile.memory.store.add_memory") as mock_add:
        mock_settings.memory_enabled = True
        # Return only short lines that _parse_facts will skip
        mock_router.return_value = "- ok\n- hi"

        from pile.context import summarize_turn
        summarize_turn("test", "This is a response long enough to pass the minimum length check for processing.")

    mock_add.assert_not_called()


# ---- recall_facts ----

def test_recall_facts_returns_list():
    """recall_facts returns a list of fact strings."""
    with patch("pile.config.settings") as mock_settings, \
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


def test_recall_facts_returns_empty_when_disabled():
    """recall_facts returns empty list when memory disabled."""
    with patch("pile.config.settings") as mock_settings:
        mock_settings.memory_enabled = False

        from pile.context import recall_facts
        facts = recall_facts("test")

    assert facts == []


def test_recall_facts_returns_empty_on_exception():
    """recall_facts returns empty list on exception."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories", side_effect=RuntimeError("db error")):
        mock_settings.memory_enabled = True

        from pile.context import recall_facts
        facts = recall_facts("test")

    assert facts == []


def test_recall_facts_filters_empty_content():
    """recall_facts skips entries with empty content."""
    with patch("pile.config.settings") as mock_settings, \
         patch("pile.memory.store.search_memories") as mock_search:
        mock_settings.memory_enabled = True
        mock_search.return_value = [
            {"content": "", "distance": 0.3},
            {"content": "valid", "distance": 0.3},
        ]

        from pile.context import recall_facts
        facts = recall_facts("test")

    assert facts == ["valid"]
