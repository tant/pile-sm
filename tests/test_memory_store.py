"""Tests for pile.memory.store — vector store operations."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_collection(**overrides):
    """Return a MagicMock that behaves like a chromadb.Collection."""
    col = MagicMock()
    col.count.return_value = overrides.get("count", 0)
    col.get.return_value = overrides.get("get_result", {"ids": [], "metadatas": []})
    col.query.return_value = overrides.get("query_result", {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})
    return col


# ---------------------------------------------------------------------------
# add_memory
# ---------------------------------------------------------------------------

class TestAddMemory:
    def test_returns_id(self):
        from pile.memory.store import add_memory

        col = _mock_collection()
        with patch("pile.memory.store._memories_collection", return_value=col):
            mem_id = add_memory("test content", memory_type="note", source="user")

        assert mem_id.startswith("mem_")
        col.add.assert_called_once()
        call_kwargs = col.add.call_args
        assert call_kwargs[1]["documents"] == ["test content"]

    def test_default_params(self):
        from pile.memory.store import add_memory

        col = _mock_collection()
        with patch("pile.memory.store._memories_collection", return_value=col):
            add_memory("some fact")

        meta = col.add.call_args[1]["metadatas"][0]
        assert meta["type"] == "note"
        assert meta["source"] == "user"
        assert "created_at" in meta

    def test_custom_type_and_source(self):
        from pile.memory.store import add_memory

        col = _mock_collection()
        with patch("pile.memory.store._memories_collection", return_value=col):
            add_memory("decision X", memory_type="decision", source="agent")

        meta = col.add.call_args[1]["metadatas"][0]
        assert meta["type"] == "decision"
        assert meta["source"] == "agent"


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------

class TestDeleteMemory:
    def test_delete_success(self):
        from pile.memory.store import delete_memory

        col = _mock_collection()
        with patch("pile.memory.store._memories_collection", return_value=col):
            assert delete_memory("mem_123") is True

        col.delete.assert_called_once_with(ids=["mem_123"])

    def test_delete_failure_returns_false(self):
        from pile.memory.store import delete_memory

        col = _mock_collection()
        col.delete.side_effect = Exception("not found")
        with patch("pile.memory.store._memories_collection", return_value=col):
            assert delete_memory("mem_bad") is False


# ---------------------------------------------------------------------------
# cleanup_expired_facts (supplements test_store_cleanup.py)
# ---------------------------------------------------------------------------

class TestCleanupExpiredFacts:
    def test_missing_created_at_treated_as_old(self):
        """Entries without created_at should be treated as epoch 0 → expired."""
        from pile.memory.store import cleanup_expired_facts

        col = _mock_collection()
        col.get.return_value = {
            "ids": ["mem_no_ts"],
            "metadatas": [{"type": "session_fact"}],
        }
        with patch("pile.memory.store._memories_collection", return_value=col):
            removed = cleanup_expired_facts(max_age_days=1)

        assert removed == 1
        col.delete.assert_called_once_with(ids=["mem_no_ts"])


# ---------------------------------------------------------------------------
# search_memories
# ---------------------------------------------------------------------------

class TestSearchMemories:
    def test_empty_collection_returns_empty(self):
        from pile.memory.store import search_memories

        col = _mock_collection(count=0)
        with patch("pile.memory.store._memories_collection", return_value=col):
            assert search_memories("anything") == []

        col.query.assert_not_called()

    def test_returns_formatted_results(self):
        from pile.memory.store import search_memories

        col = _mock_collection(count=3)
        col.query.return_value = {
            "ids": [["mem_1", "mem_2"]],
            "documents": [["fact A", "fact B"]],
            "metadatas": [[{"type": "note"}, {"type": "decision"}]],
            "distances": [[0.1, 0.5]],
        }
        with patch("pile.memory.store._memories_collection", return_value=col):
            results = search_memories("test query", n_results=2)

        assert len(results) == 2
        assert results[0]["id"] == "mem_1"
        assert results[0]["content"] == "fact A"
        assert results[0]["distance"] == 0.1
        assert results[1]["metadata"]["type"] == "decision"

    def test_n_results_capped_to_count(self):
        from pile.memory.store import search_memories

        col = _mock_collection(count=2)
        col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        with patch("pile.memory.store._memories_collection", return_value=col):
            search_memories("q", n_results=100)

        called_n = col.query.call_args[1]["n_results"]
        assert called_n == 2


# ---------------------------------------------------------------------------
# add_document_chunks
# ---------------------------------------------------------------------------

class TestAddDocumentChunks:
    def test_basic_add(self):
        from pile.memory.store import add_document_chunks

        col = _mock_collection()
        with patch("pile.memory.store._documents_collection", return_value=col):
            n = add_document_chunks(
                chunks=["chunk0", "chunk1"],
                doc_id="abc",
                doc_name="doc.pdf",
                source_path="/tmp/doc.pdf",
            )

        assert n == 2
        call_kwargs = col.add.call_args[1]
        assert call_kwargs["ids"] == ["abc_chunk_0", "abc_chunk_1"]
        assert call_kwargs["documents"] == ["chunk0", "chunk1"]
        assert call_kwargs["metadatas"][0]["doc_id"] == "abc"
        assert call_kwargs["metadatas"][1]["chunk_index"] == 1

    def test_with_extra_metadatas(self):
        from pile.memory.store import add_document_chunks

        col = _mock_collection()
        with patch("pile.memory.store._documents_collection", return_value=col):
            add_document_chunks(
                chunks=["c0", "c1"],
                doc_id="x",
                doc_name="x.md",
                source_path="/x.md",
                metadatas=[{"page": 1}, {"page": 2}],
            )

        metas = col.add.call_args[1]["metadatas"]
        assert metas[0]["page"] == 1
        assert metas[1]["page"] == 2

    def test_metadatas_shorter_than_chunks(self):
        from pile.memory.store import add_document_chunks

        col = _mock_collection()
        with patch("pile.memory.store._documents_collection", return_value=col):
            add_document_chunks(
                chunks=["c0", "c1", "c2"],
                doc_id="y",
                doc_name="y.md",
                source_path="/y.md",
                metadatas=[{"page": 1}],
            )

        metas = col.add.call_args[1]["metadatas"]
        assert metas[0]["page"] == 1
        assert "page" not in metas[2]


# ---------------------------------------------------------------------------
# remove_document
# ---------------------------------------------------------------------------

class TestRemoveDocument:
    def test_removes_existing(self):
        from pile.memory.store import remove_document

        col = _mock_collection()
        col.get.return_value = {"ids": ["d_chunk_0", "d_chunk_1"]}
        with patch("pile.memory.store._documents_collection", return_value=col):
            n = remove_document("d")

        assert n == 2
        col.delete.assert_called_once_with(ids=["d_chunk_0", "d_chunk_1"])

    def test_returns_zero_when_not_found(self):
        from pile.memory.store import remove_document

        col = _mock_collection()
        col.get.return_value = {"ids": []}
        with patch("pile.memory.store._documents_collection", return_value=col):
            assert remove_document("missing") == 0

        col.delete.assert_not_called()


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------

class TestSearchDocuments:
    def test_empty_returns_empty(self):
        from pile.memory.store import search_documents

        col = _mock_collection(count=0)
        with patch("pile.memory.store._documents_collection", return_value=col):
            assert search_documents("q") == []

    def test_returns_results(self):
        from pile.memory.store import search_documents

        col = _mock_collection(count=5)
        col.query.return_value = {
            "ids": [["d_0"]],
            "documents": [["chunk text"]],
            "metadatas": [[{"doc_name": "f.pdf", "page": 1}]],
            "distances": [[0.3]],
        }
        with patch("pile.memory.store._documents_collection", return_value=col):
            results = search_documents("q")

        assert len(results) == 1
        assert results[0]["content"] == "chunk text"


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------

class TestListDocuments:
    def test_empty(self):
        from pile.memory.store import list_documents

        col = _mock_collection(count=0)
        with patch("pile.memory.store._documents_collection", return_value=col):
            assert list_documents() == []

    def test_deduplicates_by_doc_id(self):
        from pile.memory.store import list_documents

        col = _mock_collection(count=3)
        col.get.return_value = {
            "metadatas": [
                {"doc_id": "a", "doc_name": "a.pdf", "source_path": "/a.pdf"},
                {"doc_id": "a", "doc_name": "a.pdf", "source_path": "/a.pdf"},
                {"doc_id": "b", "doc_name": "b.md", "source_path": "/b.md"},
            ]
        }
        with patch("pile.memory.store._documents_collection", return_value=col):
            docs = list_documents()

        assert len(docs) == 2
        ids = {d["doc_id"] for d in docs}
        assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# search_all
# ---------------------------------------------------------------------------

class TestSearchAll:
    def test_combines_both(self):
        from pile.memory.store import search_all

        with patch("pile.memory.store.search_memories", return_value=[{"id": "m1"}]) as sm, \
             patch("pile.memory.store.search_documents", return_value=[{"id": "d1"}]) as sd:
            result = search_all("q", n_results=3)

        assert result["memories"] == [{"id": "m1"}]
        assert result["documents"] == [{"id": "d1"}]
        sm.assert_called_once_with("q", 3)
        sd.assert_called_once_with("q", 3)


# ---------------------------------------------------------------------------
# _format_results
# ---------------------------------------------------------------------------

class TestFormatResults:
    def test_empty_input(self):
        from pile.memory.store import _format_results

        assert _format_results(None) == []
        assert _format_results({}) == []
        assert _format_results({"ids": None}) == []

    def test_partial_fields(self):
        from pile.memory.store import _format_results

        results = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
        }
        items = _format_results(results)
        assert len(items) == 1
        assert items[0]["id"] == "id1"
        assert items[0]["content"] == "doc1"
        assert "metadata" not in items[0]
        assert "distance" not in items[0]
