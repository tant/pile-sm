"""Tests for pile.tools.memory_tools — tool wrapper functions."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# _safe_memory_call decorator
# ---------------------------------------------------------------------------

class TestSafeMemoryCall:
    def test_wraps_file_not_found(self):
        from pile.tools.memory_tools import _safe_memory_call

        @_safe_memory_call
        def boom():
            raise FileNotFoundError("gone")

        assert boom() == "Error: gone"

    def test_wraps_value_error(self):
        from pile.tools.memory_tools import _safe_memory_call

        @_safe_memory_call
        def boom():
            raise ValueError("bad input")

        assert boom() == "Error: bad input"

    def test_wraps_generic_exception(self):
        from pile.tools.memory_tools import _safe_memory_call

        @_safe_memory_call
        def boom():
            raise RuntimeError("oops")

        result = boom()
        assert "RuntimeError" in result
        assert "oops" in result

    def test_passes_through_on_success(self):
        from pile.tools.memory_tools import _safe_memory_call

        @_safe_memory_call
        def ok():
            return "fine"

        assert ok() == "fine"


# ---------------------------------------------------------------------------
# memory_remember
# ---------------------------------------------------------------------------

class TestMemoryRemember:
    def test_stores_and_returns_id(self):
        from pile.tools.memory_tools import memory_remember

        with patch("pile.memory.store.add_memory", return_value="mem_100") as mock_add:
            result = memory_remember("important fact")

        assert "mem_100" in result
        assert "important fact" in result
        mock_add.assert_called_once_with("important fact", memory_type="note", source="user")

    def test_custom_type(self):
        from pile.tools.memory_tools import memory_remember

        with patch("pile.memory.store.add_memory", return_value="mem_200"):
            result = memory_remember("decided X", memory_type="decision")

        assert "mem_200" in result

    def test_error_handling(self):
        from pile.tools.memory_tools import memory_remember

        with patch("pile.memory.store.add_memory", side_effect=RuntimeError("db down")):
            result = memory_remember("something")

        assert "Error" in result


# ---------------------------------------------------------------------------
# memory_forget
# ---------------------------------------------------------------------------

class TestMemoryForget:
    def test_deletes_matching_memories(self):
        from pile.tools.memory_tools import memory_forget

        matches = [
            {"id": "mem_1", "content": "fact A", "metadata": {"type": "note"}},
            {"id": "mem_2", "content": "fact B", "metadata": {"type": "decision"}},
        ]
        with patch("pile.memory.store.search_memories", return_value=matches), \
             patch("pile.memory.store.delete_memory") as mock_del:
            result = memory_forget("facts")

        assert "Deleted 2" in result
        assert mock_del.call_count == 2

    def test_no_matches(self):
        from pile.tools.memory_tools import memory_forget

        with patch("pile.memory.store.search_memories", return_value=[]):
            result = memory_forget("nothing here")

        assert "No matching" in result

    def test_custom_n_results(self):
        from pile.tools.memory_tools import memory_forget

        with patch("pile.memory.store.search_memories", return_value=[]) as mock_search:
            memory_forget("q", n_results=10)

        mock_search.assert_called_once_with("q", n_results=10)


# ---------------------------------------------------------------------------
# memory_search
# ---------------------------------------------------------------------------

class TestMemorySearch:
    def test_returns_formatted_memories_and_docs(self):
        from pile.tools.memory_tools import memory_search

        search_result = {
            "memories": [
                {"id": "m1", "content": "sprint status", "metadata": {"type": "note"}},
            ],
            "documents": [
                {"id": "d1", "content": "doc text " * 50, "metadata": {"doc_name": "spec.pdf", "page": 3}},
            ],
        }
        with patch("pile.memory.store.search_all", return_value=search_result):
            result = memory_search("sprint")

        assert "**Memories:**" in result
        assert "sprint status" in result
        assert "**Knowledge Base:**" in result
        assert "spec.pdf" in result

    def test_no_results(self):
        from pile.tools.memory_tools import memory_search

        with patch("pile.memory.store.search_all", return_value={"memories": [], "documents": []}):
            result = memory_search("nothing")

        assert result == "No results found."

    def test_only_memories(self):
        from pile.tools.memory_tools import memory_search

        search_result = {
            "memories": [{"id": "m1", "content": "x", "metadata": {"type": "note"}}],
            "documents": [],
        }
        with patch("pile.memory.store.search_all", return_value=search_result):
            result = memory_search("x")

        assert "**Memories:**" in result
        assert "Knowledge Base" not in result

    def test_only_documents(self):
        from pile.tools.memory_tools import memory_search

        search_result = {
            "memories": [],
            "documents": [{"id": "d1", "content": "chunk", "metadata": {"doc_name": "f.md", "page": 1}}],
        }
        with patch("pile.memory.store.search_all", return_value=search_result):
            result = memory_search("x")

        assert "Memories" not in result
        assert "**Knowledge Base:**" in result


# ---------------------------------------------------------------------------
# memory_ingest_document
# ---------------------------------------------------------------------------

class TestMemoryIngestDocument:
    def test_success(self):
        from pile.tools.memory_tools import memory_ingest_document

        ingest_result = {
            "doc_id": "abc123",
            "doc_name": "report.pdf",
            "pages": 5,
            "chunks": 12,
        }
        with patch("pile.memory.ingest.ingest_file", return_value=ingest_result):
            result = memory_ingest_document("/tmp/report.pdf")

        assert "report.pdf" in result
        assert "5 pages" in result
        assert "12 chunks" in result
        assert "abc123" in result

    def test_file_not_found(self):
        from pile.tools.memory_tools import memory_ingest_document

        with patch("pile.memory.ingest.ingest_file", side_effect=FileNotFoundError("not found")):
            result = memory_ingest_document("/missing.pdf")

        assert "Error" in result


# ---------------------------------------------------------------------------
# memory_list_documents
# ---------------------------------------------------------------------------

class TestMemoryListDocuments:
    def test_empty(self):
        from pile.tools.memory_tools import memory_list_documents

        with patch("pile.memory.store.list_documents", return_value=[]):
            result = memory_list_documents()

        assert "empty" in result.lower()

    def test_with_docs(self):
        from pile.tools.memory_tools import memory_list_documents

        docs = [
            {"doc_id": "a1", "doc_name": "spec.pdf", "source_path": "/docs/spec.pdf"},
            {"doc_id": "b2", "doc_name": "notes.md", "source_path": "/docs/notes.md"},
        ]
        with patch("pile.memory.store.list_documents", return_value=docs):
            result = memory_list_documents()

        assert "spec.pdf" in result
        assert "notes.md" in result
        assert "a1" in result


# ---------------------------------------------------------------------------
# memory_remove_document
# ---------------------------------------------------------------------------

class TestMemoryRemoveDocument:
    def test_removes_existing(self):
        from pile.tools.memory_tools import memory_remove_document

        with patch("pile.memory.store.remove_document", return_value=5):
            result = memory_remove_document("doc_abc")

        assert "Removed" in result
        assert "5 chunks" in result

    def test_not_found(self):
        from pile.tools.memory_tools import memory_remove_document

        with patch("pile.memory.store.remove_document", return_value=0):
            result = memory_remove_document("missing_id")

        assert "not found" in result.lower()
