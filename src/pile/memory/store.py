"""ChromaDB vector store — singleton client with two collections (memories + documents)."""

from __future__ import annotations

import os
import time
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction

from pile.config import settings

_client: chromadb.ClientAPI | None = None
_embed_fn: EmbeddingFunction | None = None
_memories_col: chromadb.Collection | None = None
_documents_col: chromadb.Collection | None = None


def _get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client (singleton)."""
    global _client
    if _client is None:
        path = os.path.expanduser(settings.memory_store_path)
        os.makedirs(path, exist_ok=True)
        _client = chromadb.PersistentClient(path=path)
    return _client


def _embedding_fn() -> EmbeddingFunction:
    """Return a cached embedding function based on LLM provider config."""
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn

    if settings.llm_provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        _embed_fn = OpenAIEmbeddingFunction(
            api_base=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model_id,
        )
    else:
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        _embed_fn = OllamaEmbeddingFunction(
            url=f"{settings.ollama_host}/api/embed",
            model_name=settings.embedding_model_id,
        )
    return _embed_fn


def _memories_collection() -> chromadb.Collection:
    """Return cached memories collection (singleton)."""
    global _memories_col
    if _memories_col is None:
        _memories_col = _get_client().get_or_create_collection(
            name="memories",
            embedding_function=_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _memories_col


def _documents_collection() -> chromadb.Collection:
    """Return cached documents collection (singleton)."""
    global _documents_col
    if _documents_col is None:
        _documents_col = _get_client().get_or_create_collection(
            name="documents",
            embedding_function=_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _documents_col


# --- Memories (explicit remember / forget) ---


def add_memory(content: str, memory_type: str = "note", source: str = "user") -> str:
    """Store a memory. Returns the generated ID."""
    col = _memories_collection()
    mem_id = f"mem_{int(time.time() * 1000)}"
    col.add(
        ids=[mem_id],
        documents=[content],
        metadatas=[{"type": memory_type, "source": source, "created_at": time.time()}],
    )
    return mem_id


def delete_memory(memory_id: str) -> bool:
    """Delete a memory by ID. Returns True if deleted."""
    col = _memories_collection()
    try:
        col.delete(ids=[memory_id])
        return True
    except Exception:
        return False


def search_memories(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Semantic search in memories collection."""
    col = _memories_collection()
    if col.count() == 0:
        return []
    results = col.query(query_texts=[query], n_results=min(n_results, col.count()))
    return _format_results(results)


# --- Documents (ingested PDFs / markdown) ---


def add_document_chunks(
    chunks: list[str],
    doc_id: str,
    doc_name: str,
    source_path: str,
    metadatas: list[dict[str, Any]] | None = None,
) -> int:
    """Store document chunks. Returns number of chunks added."""
    col = _documents_collection()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    base_meta = {"doc_id": doc_id, "doc_name": doc_name, "source_path": source_path}
    metas = []
    for i, chunk in enumerate(chunks):
        meta = {**base_meta, "chunk_index": i}
        if metadatas and i < len(metadatas):
            meta.update(metadatas[i])
        metas.append(meta)
    col.add(ids=ids, documents=chunks, metadatas=metas)
    return len(chunks)


def remove_document(doc_id: str) -> int:
    """Remove all chunks for a document. Returns number of chunks removed."""
    col = _documents_collection()
    results = col.get(where={"doc_id": doc_id})
    if not results["ids"]:
        return 0
    col.delete(ids=results["ids"])
    return len(results["ids"])


def search_documents(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Semantic search in documents collection."""
    col = _documents_collection()
    if col.count() == 0:
        return []
    results = col.query(query_texts=[query], n_results=min(n_results, col.count()))
    return _format_results(results)


def list_documents() -> list[dict[str, Any]]:
    """List all unique documents in the store."""
    col = _documents_collection()
    if col.count() == 0:
        return []
    all_data = col.get(include=["metadatas"])
    seen: dict[str, dict[str, Any]] = {}
    for meta in all_data["metadatas"]:
        did = meta.get("doc_id", "")
        if did not in seen:
            seen[did] = {
                "doc_id": did,
                "doc_name": meta.get("doc_name", ""),
                "source_path": meta.get("source_path", ""),
            }
    return list(seen.values())


def search_all(query: str, n_results: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Search across both memories and documents."""
    return {
        "memories": search_memories(query, n_results),
        "documents": search_documents(query, n_results),
    }


# --- Helpers ---


def _format_results(results: dict) -> list[dict[str, Any]]:
    """Convert ChromaDB query results to a flat list of dicts."""
    items = []
    if not results or not results.get("ids"):
        return items
    for i, doc_id in enumerate(results["ids"][0]):
        item: dict[str, Any] = {"id": doc_id}
        if results.get("documents"):
            item["content"] = results["documents"][0][i]
        if results.get("metadatas"):
            item["metadata"] = results["metadatas"][0][i]
        if results.get("distances"):
            item["distance"] = results["distances"][0][i]
        items.append(item)
    return items
