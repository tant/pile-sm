"""Document ingestion — extract text from PDF/markdown and chunk for embedding."""

from __future__ import annotations

import hashlib
import os
from typing import Any


def extract_text_from_pdf(file_path: str) -> list[dict[str, Any]]:
    """Extract text from PDF, returning a list of {page, text} dicts."""
    import pymupdf

    pages = []
    with pymupdf.open(file_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def extract_text_from_markdown(file_path: str) -> list[dict[str, Any]]:
    """Read a markdown file, returning a single-page result."""
    with open(file_path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    return [{"page": 1, "text": text}]


def extract_text(file_path: str) -> list[dict[str, Any]]:
    """Auto-detect file type and extract text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext in (".md", ".markdown", ".txt"):
        return extract_text_from_markdown(file_path)
    raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .md, .markdown, .txt")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks by paragraph boundaries with overlap.

    Strategy: split by paragraphs first, then merge small paragraphs into
    chunks of ~chunk_size characters. If a paragraph exceeds chunk_size,
    split it by sentences.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Flush current buffer
            if current:
                chunks.append(current.strip())
                current = ""
            # Split long paragraph by sentences
            for sentence_chunk in _split_long_text(para, chunk_size, overlap):
                chunks.append(sentence_chunk)
        elif len(current) + len(para) + 2 > chunk_size:
            # Current buffer full — flush and start new
            if current:
                chunks.append(current.strip())
            # Overlap: keep tail of previous chunk
            if overlap > 0 and current:
                tail = current.strip()[-overlap:]
                current = tail + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long text block by sentence boundaries."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 > chunk_size and current:
            chunks.append(current.strip())
            if overlap > 0:
                tail = current.strip()[-overlap:]
                current = tail + " " + sent
            else:
                current = sent
        else:
            current = current + " " + sent if current else sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def ingest_file(file_path: str) -> dict[str, Any]:
    """Extract, chunk, and store a document. Returns ingestion summary."""
    from pile.memory.store import add_document_chunks

    file_path = os.path.expanduser(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc_name = os.path.basename(file_path)
    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]

    pages = extract_text(file_path)
    if not pages:
        return {"doc_id": doc_id, "doc_name": doc_name, "chunks": 0, "pages": 0}

    all_chunks: list[str] = []
    all_metas: list[dict[str, Any]] = []

    for page_data in pages:
        page_chunks = chunk_text(page_data["text"])
        for chunk in page_chunks:
            all_chunks.append(chunk)
            all_metas.append({"page": page_data["page"]})

    n_stored = add_document_chunks(
        chunks=all_chunks,
        doc_id=doc_id,
        doc_name=doc_name,
        source_path=file_path,
        metadatas=all_metas,
    )

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "chunks": n_stored,
        "pages": len(pages),
    }
