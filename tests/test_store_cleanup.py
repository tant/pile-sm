"""Tests for session fact cleanup."""

from __future__ import annotations

import time
import pytest
from unittest.mock import patch, MagicMock


def test_cleanup_expired_facts_removes_old():
    """cleanup_expired_facts deletes session_facts older than max_age_days."""
    from pile.memory.store import cleanup_expired_facts

    mock_col = MagicMock()
    old_ts = time.time() - (8 * 86400)
    fresh_ts = time.time() - (2 * 86400)
    mock_col.get.return_value = {
        "ids": ["mem_1", "mem_2", "mem_3"],
        "metadatas": [
            {"type": "session_fact", "created_at": old_ts},
            {"type": "session_fact", "created_at": old_ts},
            {"type": "session_fact", "created_at": fresh_ts},
        ],
    }

    with patch("pile.memory.store._memories_collection", return_value=mock_col):
        removed = cleanup_expired_facts(max_age_days=7)

    assert removed == 2
    mock_col.delete.assert_called_once_with(ids=["mem_1", "mem_2"])


def test_cleanup_expired_facts_no_old():
    """cleanup_expired_facts returns 0 when nothing is expired."""
    from pile.memory.store import cleanup_expired_facts

    mock_col = MagicMock()
    fresh_ts = time.time() - (1 * 86400)
    mock_col.get.return_value = {
        "ids": ["mem_1"],
        "metadatas": [{"type": "session_fact", "created_at": fresh_ts}],
    }

    with patch("pile.memory.store._memories_collection", return_value=mock_col):
        removed = cleanup_expired_facts(max_age_days=7)

    assert removed == 0
    mock_col.delete.assert_not_called()


def test_cleanup_expired_facts_empty():
    """cleanup_expired_facts handles empty collection."""
    from pile.memory.store import cleanup_expired_facts

    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": [], "metadatas": []}

    with patch("pile.memory.store._memories_collection", return_value=mock_col):
        removed = cleanup_expired_facts(max_age_days=7)

    assert removed == 0
