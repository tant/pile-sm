"""Smart router — keyword-based + LLM classifier fallback for agent selection.

Phase 1: Keyword matching handles 70%+ queries instantly (<1ms).
Phase 2: LLM classifier (lightweight model) for ambiguous queries (~150ms).
Phase 3: Embedding similarity as last resort if no router model configured.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("pile.router")

# --- Phase 1: Keyword patterns → agent key (checked in order, first match wins) ---

_ROUTES: list[tuple[str, list[str]]] = [
    # Greetings → triage (cheapest, catch first)
    ("triage", [
        r"^(xin\s+)?ch[aà]o\b", r"^hello\b", r"^hi\b", r"^hey\b",
        r"^good\s+(morning|afternoon|evening)",
    ]),
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
    # Curl/command generation (unambiguous)
    ("jira_query", [
        r"cho\s+t[oô]i\s+l[eệ]nh", r"curl\b", r"l[eệ]nh\s+l[aấ]y",
        r"changelog\b", r"l[iị]ch\s+s[uử]",
    ]),
    # Jira write — only when user explicitly wants to MODIFY something
    ("jira_write", [
        r"t[aạ]o\s+(issue|bug|task|story|epic|ticket)",
        r"create\s+(issue|bug|task|story|epic|ticket)",
        r"assign\s+(cho\b|to\b|v[aà]o\b)",     # "assign cho X" = write
        r"chuy[eể]n\s+tr[aạ]ng\s+th[aá]i",
        r"transition\b", r"comment\b", r"link\s+issue",
        r"s[uử]a\s+(issue|bug|task)", r"update\s+issue",
        r"c[aậ]p\s+nh[aậ]t\s+(issue|bug|task)",
    ]),
    # Board
    ("board", [
        r"\bboard\b", r"li[eệ]t\s+k[eê]\s+.*board",
        r"board\s+config", r"c[aấ]u\s+h[iì]nh\s+board",
    ]),
    # Scrum (analysis/reports) — only UNAMBIGUOUS scrum patterns
    # Removed: progress\b, tiến độ, \bsprint\b overlap → let LLM handle ambiguous
    ("scrum", [
        r"standup\b", r"velocity\b", r"workload\b",
        r"burndown\b", r"cycle\s+time", r"retro",
        r"sprint\s+review", r"sprint\s+planning",
        r"blocke[rd]", r"b[iị]\s+block",
        r"qu[aá]\s+t[aả]i", r"overload",
        r"data\s+quality", r"thi[eế]u\s+th[oô]ng\s+tin",
        r"t[oổ]ng\s+h[oợ]p", r"b[aá]o\s+c[aá]o",
        r"t[oó]m\s+t[aắ]t", r"summary\b",
        r"stakeholder", r"meeting\s+prep",
        r"cho\s+s[eế]p",
    ]),
    # Sprint (management — only explicit sprint operations)
    ("sprint", [
        r"sprint\s+hi[eệ]n\s+t[aạ]i", r"sprint\s+n[aà]o",
        r"sprint\s+(có|co)\b", r"trong\s+sprint",
        r"t[aạ]o\s+sprint", r"create\s+sprint",
        r"chuy[eể]n.*v[aà]o\s+sprint", r"move.*sprint",
        r"v[eề]\s+backlog", r"move.*backlog",
        r"list.*sprints?\b", r"danh\s+s[aá]ch\s+sprint",
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
    # Issue key (TETRA-123) — but NOT if "epic" also present (Q65 case)
    # Checked late so "epic TETRA-893" matches epic first
    ("jira_query", [
        r"[A-Z]+-\d+",
    ]),
    # Jira query (broadest, lowest priority) — narrowed: removed "issue\b" standalone
    ("jira_query", [
        r"t[iì]m\b", r"search\b", r"jql\b",
        r"\bticket\b", r"\bbug\b",
        r"tr[aạ]ng\s+th[aá]i", r"status\b",
    ]),
]

_VALID_AGENTS = {
    "jira_query", "jira_write", "board", "sprint", "epic",
    "git", "scrum", "memory", "browser", "triage",
}


def route_query(query: str) -> str | None:
    """Route by keyword. Returns agent key or None if ambiguous."""
    q = query.lower().strip()
    for agent_key, patterns in _ROUTES:
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                logger.debug("Keyword match: '%s' → %s", q[:40], agent_key)
                return agent_key
    return None


# --- Phase 2: LLM classifier (lightweight router model) ---

_CLASSIFY_PROMPT = """\
Pick one agent for this query. Reply ONLY the agent name.

jira_query = search/view/list issues, who is assigned what, issue details, filter by status/type
jira_write = create/update issues, assign, transition status, comment
board = boards list/details/config
sprint = sprint info/issues, create/move sprint
epic = epics, backlog
git = commits, branches, diffs
scrum = standup, workload, velocity, review, retro, blockers, reports, team overview
memory = remember/forget, knowledge base
browser = open URL, scrape, screenshot

"Ai đang làm gì" / "X handle gì" → jira_query
"Team thế nào" / "có vấn đề gì" → scrum

{query}
Agent:"""


def route_query_with_llm(query: str) -> str:
    """Route using a lightweight LLM call. Returns agent key."""
    from pile.client import call_router_model

    prompt = _CLASSIFY_PROMPT.format(query=query)
    raw = call_router_model(prompt, max_tokens=20)

    if raw:
        agent_key = raw.lower().split()[0].strip(".,;:!\"'")
        if agent_key in _VALID_AGENTS:
            logger.info("LLM classify: '%s' → %s", query[:40], agent_key)
            return agent_key
        logger.warning("LLM classify returned invalid '%s'", raw[:30])

    return "triage"


# --- Phase 3: Embedding similarity (last resort) ---

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


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the configured provider."""
    import httpx
    from pile.config import settings

    if settings.llm_provider == "openai":
        resp = httpx.post(
            f"{settings.openai_base_url}/embeddings",
            json={"model": settings.embedding_model_id, "input": texts},
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    # Ollama / ollama-native
    results = []
    for text in texts:
        resp = httpx.post(
            f"{settings.ollama_host}/api/embed",
            json={"model": settings.embedding_model_id, "input": text},
            timeout=15.0,
        )
        resp.raise_for_status()
        results.append(resp.json()["embeddings"][0])
    return results


def _get_embeddings() -> dict[str, list[float]]:
    """Lazily compute and cache agent description embeddings."""
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache

    try:
        descs = list(_AGENT_DESCRIPTIONS.values())
        keys = list(_AGENT_DESCRIPTIONS.keys())
        vectors = _embed_texts(descs)
        _embedding_cache = dict(zip(keys, vectors))
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
        embeddings = _get_embeddings()
        if not embeddings:
            return "triage"

        query_emb = _embed_texts([query])[0]

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


# --- Smart route: keyword → LLM classify → embedding → triage ---


def smart_route(query: str) -> str:
    """Smart routing: keyword first, then LLM classifier, then embedding.

    Returns agent key (never None).
    """
    # Phase 1: keyword (instant)
    result = route_query(query)
    if result:
        return result

    # Phase 2: LLM classifier (if router model configured)
    from pile.config import settings
    if settings.router_model:
        return route_query_with_llm(query)

    # Phase 3: embedding similarity (last resort)
    return route_query_with_embedding(query)
