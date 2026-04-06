"""Auto-recall and auto-learn — memory integration into the main workflow.

Auto-recall: before each agent run, search memory for relevant context.
Auto-learn: after agent encounters errors/corrections, compress and save the lesson.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("pile.context")


def recall(query: str, n_results: int = 3) -> str:
    """Search memory for context relevant to the query. Returns formatted hint or empty."""
    try:
        from pile.config import settings
        if not settings.memory_enabled:
            return ""

        from pile.memory.store import search_memories
        results = search_memories(query, n_results=n_results)
        if not results:
            return ""

        # Filter by relevance — ChromaDB cosine distance [0, 2].
        # < 0.8 means similarity > 0.6, fairly relevant.
        relevant = [r for r in results if r.get("distance", 2.0) < 0.8]
        if not relevant:
            return ""

        lines = []
        for r in relevant:
            content = r.get("content", "")
            if content:
                lines.append(f"- {content}")

        if not lines:
            return ""

        hint = "Relevant context from memory:\n" + "\n".join(lines)
        logger.info("Recall: %d memories found for '%s'", len(lines), query[:40])
        return hint

    except Exception as e:
        logger.warning("Recall failed: %s", e)
        return ""


def learn(query: str, lesson: str) -> None:
    """Compress a lesson and save to memory. Skips if similar lesson already exists."""
    try:
        from pile.config import settings
        if not settings.memory_enabled:
            return

        # Check for duplicate — if a similar memory already exists, skip
        from pile.memory.store import search_memories
        existing = search_memories(lesson, n_results=1)
        if existing and existing[0].get("distance", 2.0) < 0.3:
            logger.debug("Learn: similar memory already exists, skipping")
            return

        compressed = _compress(lesson)
        if not compressed or len(compressed) < 5:
            return

        from pile.memory.store import add_memory
        mem_id = add_memory(compressed, memory_type="auto_learn", source="system")
        logger.info("Learn: saved '%s' (id=%s)", compressed[:60], mem_id)

    except Exception as e:
        logger.warning("Learn failed: %s", e)


def _compress(text: str) -> str:
    """Compress a lesson into a short factual statement using the router model."""
    from pile.config import settings

    if not settings.router_model:
        # No router model — just truncate
        return text[:200].strip()

    try:
        import httpx

        prompt = (
            "Compress this into ONE short factual statement (under 50 words). "
            "Keep only the key fact, remove explanation.\n\n"
            f"{text}\n\nFact:"
        )

        if settings.llm_provider == "openai":
            resp = httpx.post(
                f"{settings.openai_base_url}/chat/completions",
                json={
                    "model": settings.router_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 80,
                    "temperature": 0,
                },
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                timeout=10.0,
            )
        else:
            resp = httpx.post(
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": settings.router_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": 80, "temperature": 0},
                },
                timeout=10.0,
            )

        resp.raise_for_status()
        data = resp.json()

        if settings.llm_provider == "openai":
            result = data["choices"][0]["message"]["content"]
        else:
            result = data["message"]["content"]

        return result.strip()

    except Exception as e:
        logger.warning("Compress failed: %s — saving raw", e)
        return text[:200].strip()
