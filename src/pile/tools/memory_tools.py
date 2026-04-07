"""Memory and knowledge base tools for remember, forget, search, and document management."""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import Field

from agent_framework import tool

logger = logging.getLogger("pile.tools.memory")


def _safe_memory_call(func):
    """Decorator to handle memory operation errors gracefully."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            return f"Error: {e}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.exception("Memory tool error in %s", func.__name__)
            return f"Error: {type(e).__name__}: {e}"

    return wrapper


@_safe_memory_call
def memory_remember(
    content: Annotated[str, Field(description="Information to remember")],
    memory_type: Annotated[str, Field(description="Type: decision, pattern, note")] = "note",
) -> str:
    """Save important information to long-term memory (decisions, patterns, notes)."""
    from pile.memory.store import add_memory

    mem_id = add_memory(content, memory_type=memory_type, source="user")
    return f"Remembered (ID: {mem_id}): {content}"


@tool(approval_mode="always_require")
@_safe_memory_call
def memory_forget(
    query: Annotated[str, Field(description="What to forget — describe the memory content in natural language")],
    n_results: Annotated[int, Field(description="Max number of matching memories to delete")] = 3,
) -> str:
    """Find and delete memories matching a query. Shows matches for user approval before deleting."""
    from pile.memory.store import delete_memory, search_memories

    matches = search_memories(query, n_results=n_results)
    if not matches:
        return "No matching memories found. Nothing to forget."

    deleted = []
    for item in matches:
        delete_memory(item["id"])
        meta = item.get("metadata", {})
        deleted.append(f"- [{item['id']}] ({meta.get('type', 'note')}) {item['content']}")

    return f"Deleted {len(deleted)} memories:\n" + "\n".join(deleted)


@_safe_memory_call
def memory_search(
    query: Annotated[str, Field(description="Search query in natural language")],
    n_results: Annotated[int, Field(description="Max results to return")] = 5,
) -> str:
    """Search across memories and knowledge base documents using semantic similarity."""
    from pile.memory.store import search_all

    results = search_all(query, n_results=n_results)

    parts: list[str] = []

    if results["memories"]:
        parts.append("**Memories:**")
        for item in results["memories"]:
            meta = item.get("metadata", {})
            parts.append(
                f"- [{item['id']}] ({meta.get('type', 'note')}) {item['content']}"
            )

    if results["documents"]:
        parts.append("\n**Knowledge Base:**")
        for item in results["documents"]:
            meta = item.get("metadata", {})
            source = meta.get("doc_name", "unknown")
            page = meta.get("page", "?")
            parts.append(f"- [{source} p.{page}] {item['content'][:300]}")

    if not parts:
        return "No results found."

    return "\n".join(parts)


@tool(approval_mode="always_require")
@_safe_memory_call
def memory_ingest_document(
    file_path: Annotated[str, Field(description="Absolute path to PDF or markdown file")],
) -> str:
    """Load a document (PDF, markdown, text) into the knowledge base. Requires user approval."""
    from pile.memory.ingest import ingest_file

    result = ingest_file(file_path)
    return (
        f"Ingested '{result['doc_name']}': "
        f"{result['pages']} pages, {result['chunks']} chunks stored. "
        f"(doc_id: {result['doc_id']})"
    )


@_safe_memory_call
def memory_list_documents() -> str:
    """List all documents currently in the knowledge base."""
    from pile.memory.store import list_documents

    docs = list_documents()
    if not docs:
        return "Knowledge base is empty. No documents ingested yet."

    lines = ["**Documents in knowledge base:**"]
    for doc in docs:
        lines.append(f"- {doc['doc_name']} (ID: {doc['doc_id']}, path: {doc['source_path']})")
    return "\n".join(lines)


@tool(approval_mode="always_require")
@_safe_memory_call
def memory_remove_document(
    doc_id: Annotated[str, Field(description="Document ID to remove")],
) -> str:
    """Remove a document and all its chunks from the knowledge base. Requires user approval."""
    from pile.memory.store import remove_document

    n = remove_document(doc_id)
    if n == 0:
        return f"Document {doc_id} not found in knowledge base."
    return f"Removed document {doc_id} ({n} chunks deleted)."
