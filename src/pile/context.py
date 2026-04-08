"""Auto-recall and auto-learn — memory integration into the main workflow.

Auto-recall: before each agent run, search memory for relevant context.
Auto-learn: after agent encounters errors/corrections, compress and save the lesson.
"""

from __future__ import annotations

import logging

from pile.config import settings
from pile.client import call_router_model

logger = logging.getLogger("pile.context")

# ChromaDB cosine distance thresholds [0, 2]. Lower = more similar.
RECALL_MAX_DISTANCE = 0.8    # similarity > 0.6 — fairly relevant
DEDUP_MAX_DISTANCE = 0.3     # similarity > 0.85 — near-duplicate


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

        lines = [
            f"- {r['content']}"
            for r in results
            if r.get("distance", 2.0) < RECALL_MAX_DISTANCE and r.get("content")
        ]
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

        from pile.memory.store import search_memories, add_memory
        existing = search_memories(lesson, n_results=1)
        if existing and existing[0].get("distance", 2.0) < DEDUP_MAX_DISTANCE:
            logger.debug("Learn: similar memory already exists, skipping")
            return

        compressed = _compress(lesson)
        if not compressed or len(compressed) < 5:
            return

        mem_id = add_memory(compressed, memory_type="auto_learn", source="system")
        logger.info("Learn: saved '%s' (id=%s)", compressed[:60], mem_id)

    except Exception as e:
        logger.warning("Learn failed: %s", e)


def _compress(text: str) -> str:
    """Compress a lesson into a short factual statement using the router model."""
    from pile.client import call_router_model

    prompt = (
        "Compress this into ONE short factual statement (under 50 words). "
        "Keep only the key fact, remove explanation.\n\n"
        f"{text}\n\nFact:"
    )

    result = call_router_model(prompt, max_tokens=80)
    if result:
        return result

    # No router model or call failed — truncate at word boundary
    truncated = text[:200]
    if len(text) > 200:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip()


# --- Session memory: extract and persist key facts per turn ---

EXTRACT_PROMPT = (
    "Extract only important facts from this conversation turn.\n"
    "Include: numbers, dates, decisions, issue IDs, names, statuses, assignments.\n"
    "Skip: greetings, generic explanations, filler text.\n"
    "Return one fact per line as bullet points. If nothing important, return \"NONE\".\n\n"
    "User: {user_msg}\n"
    "Agent: {agent_text}\n\n"
    "Facts:"
)

_SKIP_AGENTS = {"TriageAgent"}

MIN_RESPONSE_LENGTH = 20


def summarize_turn(user_msg: str, agent_text: str, agent_name: str = "") -> None:
    """Extract key facts from a conversation turn and store in memory.

    Skips if memory disabled, response too short, agent is in skip list,
    or router returns NONE. Deduplicates against existing memories.
    """
    try:
        if not settings.memory_enabled:
            return

        if len(agent_text.strip()) < MIN_RESPONSE_LENGTH:
            return

        if agent_name in _SKIP_AGENTS:
            return

        prompt = EXTRACT_PROMPT.format(user_msg=user_msg[:500], agent_text=agent_text[:1500])
        result = call_router_model(prompt, max_tokens=200)

        if not result or result.strip().upper() == "NONE":
            return

        facts = _parse_facts(result)
        if not facts:
            return

        from pile.memory.store import search_memories, add_memory
        stored_count = 0
        for fact in facts:
            existing = search_memories(fact, n_results=1)
            if existing and existing[0].get("distance", 2.0) < DEDUP_MAX_DISTANCE:
                logger.debug("Summarize: duplicate skipped — '%s'", fact[:50])
                continue
            add_memory(fact, memory_type="session_fact", source="session_summary")
            stored_count += 1

        if stored_count:
            logger.info("Summarize: stored %d facts from turn", stored_count)

    except Exception as e:
        logger.warning("Summarize turn failed: %s", e)


def recall_facts(query: str, n_results: int = 5) -> list[str]:
    """Return a list of recalled fact strings relevant to the query.

    Filters by RECALL_MAX_DISTANCE. Used by UI to show recalled context.
    """
    try:
        if not settings.memory_enabled:
            return []

        from pile.memory.store import search_memories
        results = search_memories(query, n_results=n_results)
        return [
            r["content"]
            for r in results
            if r.get("distance", 2.0) < RECALL_MAX_DISTANCE and r.get("content")
        ]
    except Exception:
        return []


def _parse_facts(text: str) -> list[str]:
    """Parse bullet-point facts from router model output."""
    facts = []
    for line in text.strip().split("\n"):
        line = line.strip().lstrip("-*").strip()
        if line and len(line) > 5 and line.upper() != "NONE":
            facts.append(line)
    return facts
