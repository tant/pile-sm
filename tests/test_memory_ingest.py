"""Tests for pile.memory.ingest — text extraction and chunking."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from pile.memory.ingest import chunk_text, _split_long_text


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_single_short_paragraph(self):
        result = chunk_text("Hello world", chunk_size=100)
        assert result == ["Hello world"]

    def test_multiple_short_paragraphs_merged(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        result = chunk_text(text, chunk_size=100, overlap=0)
        assert len(result) == 1
        assert "Para one." in result[0]
        assert "Para three." in result[0]

    def test_paragraphs_split_when_exceeding_chunk_size(self):
        text = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
        result = chunk_text(text, chunk_size=60, overlap=0)
        assert len(result) == 3

    def test_overlap_carries_tail(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = chunk_text(text, chunk_size=20, overlap=5)
        assert len(result) >= 2
        # second chunk should contain overlap from first
        assert len(result[1]) > len("Second paragraph.")

    def test_long_paragraph_split_by_sentences(self):
        sentences = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = chunk_text(sentences, chunk_size=30, overlap=0)
        assert len(result) > 1

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n   ") == []


# ---------------------------------------------------------------------------
# _split_long_text
# ---------------------------------------------------------------------------

class TestSplitLongText:
    def test_basic_split(self):
        text = "One. Two. Three. Four. Five."
        result = _split_long_text(text, chunk_size=15, overlap=0)
        assert len(result) >= 2
        # all text should be present
        joined = " ".join(result)
        assert "One" in joined
        assert "Five" in joined

    def test_with_overlap(self):
        text = "Alpha. Beta. Gamma. Delta."
        result = _split_long_text(text, chunk_size=15, overlap=5)
        assert len(result) >= 2

    def test_single_sentence(self):
        result = _split_long_text("Just one sentence.", chunk_size=100, overlap=0)
        assert result == ["Just one sentence."]


# ---------------------------------------------------------------------------
# extract_text_from_markdown
# ---------------------------------------------------------------------------

class TestExtractTextFromMarkdown:
    def test_reads_file(self):
        from pile.memory.ingest import extract_text_from_markdown

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nSome content here.")
            f.flush()
            path = f.name

        try:
            pages = extract_text_from_markdown(path)
            assert len(pages) == 1
            assert pages[0]["page"] == 1
            assert "Title" in pages[0]["text"]
        finally:
            os.unlink(path)

    def test_empty_file(self):
        from pile.memory.ingest import extract_text_from_markdown

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            assert extract_text_from_markdown(path) == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_markdown_dispatch(self):
        from pile.memory.ingest import extract_text

        with patch("pile.memory.ingest.extract_text_from_markdown", return_value=[{"page": 1, "text": "hi"}]) as m:
            result = extract_text("/tmp/f.md")
            m.assert_called_once_with("/tmp/f.md")
            assert result == [{"page": 1, "text": "hi"}]

    def test_txt_dispatch(self):
        from pile.memory.ingest import extract_text

        with patch("pile.memory.ingest.extract_text_from_markdown", return_value=[]) as m:
            extract_text("/tmp/f.txt")
            m.assert_called_once()

    def test_pdf_dispatch(self):
        from pile.memory.ingest import extract_text

        with patch("pile.memory.ingest.extract_text_from_pdf", return_value=[]) as m:
            extract_text("/tmp/f.pdf")
            m.assert_called_once_with("/tmp/f.pdf")

    def test_unsupported_raises(self):
        from pile.memory.ingest import extract_text

        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text("/tmp/f.docx")


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------

class TestIngestFile:
    def test_file_not_found(self):
        from pile.memory.ingest import ingest_file

        with pytest.raises(FileNotFoundError):
            ingest_file("/nonexistent/path/file.md")

    def test_empty_document(self):
        from pile.memory.ingest import ingest_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            with patch("pile.memory.store.add_document_chunks") as mock_add:
                result = ingest_file(path)
                assert result["chunks"] == 0
                assert result["pages"] == 0
                mock_add.assert_not_called()
        finally:
            os.unlink(path)

    def test_successful_ingest(self):
        from pile.memory.ingest import ingest_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Hello\n\nWorld content here.")
            f.flush()
            path = f.name

        try:
            with patch("pile.memory.store.add_document_chunks", return_value=1) as mock_add:
                result = ingest_file(path)

                assert result["doc_name"] == os.path.basename(path)
                assert result["chunks"] == 1
                assert result["pages"] == 1
                mock_add.assert_called_once()

                call_kwargs = mock_add.call_args[1]
                assert call_kwargs["doc_name"] == os.path.basename(path)
                assert len(call_kwargs["chunks"]) >= 1
                assert call_kwargs["metadatas"][0]["page"] == 1
        finally:
            os.unlink(path)

    def test_multi_page_pdf(self):
        from pile.memory.ingest import ingest_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            path = f.name

        try:
            mock_pages = [
                {"page": 1, "text": "Page one content."},
                {"page": 2, "text": "Page two content."},
            ]
            with patch("pile.memory.ingest.extract_text_from_pdf", return_value=mock_pages), \
                 patch("pile.memory.store.add_document_chunks", return_value=2) as mock_add:
                result = ingest_file(path)
                assert result["pages"] == 2
                assert result["chunks"] == 2
        finally:
            os.unlink(path)
