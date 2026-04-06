"""Smart router — keyword-based + embedding fallback for agent selection.

Phase 1: Keyword matching handles 70%+ queries instantly (<1ms).
Phase 2: Embedding similarity for ambiguous queries (no LLM call needed).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("pile.router")

# --- Phase 1: Keyword patterns → agent key (checked in order, first match wins) ---

_ROUTES: list[tuple[str, list[str]]] = [
    # High priority: memory and browser (before Jira patterns that might overlap)
    ("memory", [
        r"nh[oớ]\s+gi[uú]p", r"remember\b", r"qu[eê]n\b", r"forget\b",
        r"knowledge\b", r"t[aà]i\s+li[eệ]u",
        r"load\s+(file|document|pdf)", r"ingest\b",
        r"memory\b",
    ]),
    ("browser", [
        r"m[oở]\s+trang", r"open\s+(url|page|web)",
        r"login\s+v[aà]o", r"đ[aă]ng\s+nh[aậ]p",
        r"screenshot\b", r"scrape\b", r"browse\b",
        r"vnexpress|github\.com|gitlab\.com|atlassian",
        r"https?://",
    ]),
    # Curl/command generation
    ("jira_query", [
        r"cho\s+t[oô]i\s+l[eệ]nh", r"curl\b", r"l[eệ]nh\s+l[aấ]y",
        r"changelog\b", r"l[iị]ch\s+s[uử]",
    ]),
    # Jira write (create/update/transition)
    ("jira_write", [
        r"t[aạ]o\s+(issue|bug|task|story|epic|ticket)",
        r"create\s+(issue|bug|task|story|epic|ticket)",
        r"assign\b", r"chuy[eể]n\s+tr[aạ]ng\s+th[aá]i",
        r"transition\b", r"comment\b", r"link\s+issue",
        r"s[uử]a\s+(issue|bug|task)", r"update\s+issue",
        r"c[aậ]p\s+nh[aậ]t\s+(issue|bug|task)",
    ]),
    # Board
    ("board", [
        r"\bboard\b", r"li[eệ]t\s+k[eê]\s+.*board",
        r"board\s+config", r"c[aấ]u\s+h[iì]nh\s+board",
    ]),
    # Scrum (analysis/reports — BEFORE sprint to catch "tóm tắt sprint", "báo cáo sprint")
    ("scrum", [
        r"standup\b", r"velocity\b", r"workload\b",
        r"burndown\b", r"cycle\s+time", r"retro",
        r"sprint\s+review", r"sprint\s+planning",
        r"\bblock\b", r"blocke[rd]", r"b[iị]\s+block",
        r"qu[aá]\s+t[aả]i", r"overload",
        r"data\s+quality", r"thi[eế]u\s+th[oô]ng\s+tin",
        r"t[oổ]ng\s+h[oợ]p", r"b[aá]o\s+c[aá]o",
        r"t[oó]m\s+t[aắ]t", r"summary\b",
        r"stakeholder", r"meeting\s+prep",
        r"cho\s+s[eế]p", r"ph[aâ]n\s+t[ií]ch",
        r"ti[eế]n\s+đ[oộ]", r"progress\b",
    ]),
    # Sprint (management — create, move, view sprint data)
    ("sprint", [
        r"\bsprint\b", r"t[aạ]o\s+sprint",
        r"chuy[eể]n.*v[aà]o\s+sprint", r"move.*sprint",
        r"v[eề]\s+backlog", r"move.*backlog",
    ]),
    # Epic / Backlog
    ("epic", [
        r"\bepic\b", r"\bbacklog\b",
    ]),
    # Git
    ("git", [
        r"\bgit\b", r"\bcommit\b", r"\bbranch\b",
        r"\bdiff\b", r"\bblame\b", r"code\s+change",
    ]),
    # Jira query (broadest, lowest priority)
    ("jira_query", [
        r"t[iì]m\b", r"search\b", r"jql\b",
        r"issue\b", r"ticket\b",
        r"tr[aạ]ng\s+th[aá]i", r"status\b",
    ]),
]


def route_query(query: str) -> str | None:
    """Route by keyword. Returns agent key or None if ambiguous."""
    q = query.lower().strip()
    for agent_key, patterns in _ROUTES:
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                logger.debug("Keyword match: '%s' → %s", q[:40], agent_key)
                return agent_key
    return None


# --- Phase 2: Embedding-based fallback ---

# Agent descriptions for semantic matching
_AGENT_DESCRIPTIONS: dict[str, str] = {
    "jira_query": "Search Jira issues, view issue details, check issue status, get changelog history, generate curl API commands",
    "jira_write": "Create new Jira issues, update issue fields, change issue status, add comments, link issues together",
    "board": "List Jira boards, view board details with active sprint summary, check board configuration and columns",
    "sprint": "View sprint information and issues, create new sprints, move issues into sprints or back to backlog",
    "epic": "List epics on a board, view issues within an epic, check backlog items not in any sprint",
    "git": "View git commit history, compare branches with diffs, show file blame, list branches",
    "scrum": "Generate standup reports, analyze velocity and workload, sprint review, retrospective, cycle time analysis, blocker tracking",
    "memory": "Remember information for later, forget memories, search past decisions, load documents into knowledge base",
    "browser": "Open web pages, scrape content from URLs, login to websites, take screenshots",
}

_embedding_cache: dict[str, list[float]] | None = None


def _get_embeddings() -> dict[str, list[float]]:
    """Lazily compute and cache agent description embeddings."""
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache

    try:
        import httpx
        from pile.config import settings

        host = settings.ollama_host
        model = settings.embedding_model_id
        _embedding_cache = {}

        for agent_key, desc in _AGENT_DESCRIPTIONS.items():
            resp = httpx.post(
                f"{host}/api/embed",
                json={"model": model, "input": desc},
                timeout=15.0,
            )
            resp.raise_for_status()
            _embedding_cache[agent_key] = resp.json()["embeddings"][0]

        logger.info("Cached embeddings for %d agents", len(_embedding_cache))
        return _embedding_cache
    except Exception as e:
        logger.warning("Failed to compute embeddings: %s", e)
        return {}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def route_query_with_embedding(query: str) -> str:
    """Route using embedding similarity. Returns best matching agent key."""
    try:
        import httpx
        from pile.config import settings

        embeddings = _get_embeddings()
        if not embeddings:
            return "triage"

        # Embed the query
        resp = httpx.post(
            f"{settings.ollama_host}/api/embed",
            json={"model": settings.embedding_model_id, "input": query},
            timeout=15.0,
        )
        resp.raise_for_status()
        query_emb = resp.json()["embeddings"][0]

        # Find best match
        best_key = "triage"
        best_score = 0.0
        for agent_key, agent_emb in embeddings.items():
            score = _cosine_similarity(query_emb, agent_emb)
            if score > best_score:
                best_score = score
                best_key = agent_key

        logger.info("Embedding match: '%s' → %s (score=%.3f)", query[:40], best_key, best_score)
        return best_key
    except Exception as e:
        logger.warning("Embedding routing failed: %s", e)
        return "triage"


def smart_route(query: str) -> str:
    """Smart routing: keyword first, embedding fallback.

    Returns agent key (never None).
    """
    # Phase 1: keyword (instant)
    result = route_query(query)
    if result:
        return result

    # Phase 2: embedding similarity (needs Ollama, ~100ms)
    return route_query_with_embedding(query)
