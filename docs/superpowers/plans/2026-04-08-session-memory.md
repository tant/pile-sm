# Session Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After each agent response, extract key facts and persist them in ChromaDB so they survive page refresh, with clear UI indication of recalled vs real-time data.

**Architecture:** Router model extracts facts from each turn (background task), deduplicates against existing memories, stores as `session_fact` type in ChromaDB. On new sessions, `recall()` picks them up automatically. UI shows a distinct "Recalled" step.

**Tech Stack:** Python 3.12, ChromaDB, llama-cpp (router model), Chainlit, asyncio

**Spec:** `docs/superpowers/specs/2026-04-08-session-memory-design.md`

---

### Task 1: Add `cleanup_expired_facts()` to store.py

**Files:**
- Modify: `src/pile/memory/store.py:70-82`
- Test: `tests/test_store_cleanup.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/test_store_cleanup.py`:

```python
"""Tests for session fact cleanup."""

from __future__ import annotations

import time
import pytest
from unittest.mock import patch, MagicMock


def test_cleanup_expired_facts_removes_old():
    """cleanup_expired_facts deletes session_facts older than max_age_days."""
    from pile.memory.store import cleanup_expired_facts

    mock_col = MagicMock()
    # Simulate 2 old facts and 1 fresh fact
    old_ts = time.time() - (8 * 86400)  # 8 days ago
    fresh_ts = time.time() - (2 * 86400)  # 2 days ago
    mock_col.get.return_value = {
        "ids": ["mem_1", "mem_2", "mem_3"],
        "metadatas": [
            {"type": "session_fact", "created_at": old_ts},
            {"type": "session_fact", "created_at": old_ts},
            {"type": "session_fact", "created_at": fresh_ts},
        ],
    }

    with patch("pile.memory.store._memories_collection", return_value=mock_col):
        removed = cleanup_expired_facts(max_age_days=7)

    assert removed == 2
    mock_col.delete.assert_called_once_with(ids=["mem_1", "mem_2"])


def test_cleanup_expired_facts_no_old():
    """cleanup_expired_facts returns 0 when nothing is expired."""
    from pile.memory.store import cleanup_expired_facts

    mock_col = MagicMock()
    fresh_ts = time.time() - (1 * 86400)
    mock_col.get.return_value = {
        "ids": ["mem_1"],
        "metadatas": [{"type": "session_fact", "created_at": fresh_ts}],
    }

    with patch("pile.memory.store._memories_collection", return_value=mock_col):
        removed = cleanup_expired_facts(max_age_days=7)

    assert removed == 0
    mock_col.delete.assert_not_called()


def test_cleanup_expired_facts_empty():
    """cleanup_expired_facts handles empty collection."""
    from pile.memory.store import cleanup_expired_facts

    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": [], "metadatas": []}

    with patch("pile.memory.store._memories_collection", return_value=mock_col):
        removed = cleanup_expired_facts(max_age_days=7)

    assert removed == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/test_store_cleanup.py -v`
Expected: FAIL — `ImportError: cannot import name 'cleanup_expired_facts'`

- [ ] **Step 3: Implement cleanup_expired_facts**

Add to `src/pile/memory/store.py` after `delete_memory()` (after line 92):

```python
def cleanup_expired_facts(max_age_days: int = 7) -> int:
    """Delete session_fact memories older than max_age_days. Returns count deleted."""
    col = _memories_collection()
    results = col.get(where={"type": "session_fact"}, include=["metadatas"])
    if not results["ids"]:
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    expired_ids = [
        results["ids"][i]
        for i, meta in enumerate(results["metadatas"])
        if meta.get("created_at", 0) < cutoff
    ]
    if expired_ids:
        col.delete(ids=expired_ids)
    return len(expired_ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/test_store_cleanup.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pile/memory/store.py tests/test_store_cleanup.py
git commit -m "feat: add cleanup_expired_facts for session memory TTL"
```

---

### Task 2: Add `summarize_turn()` and `recall_facts()` to context.py

**Files:**
- Modify: `src/pile/context.py:1-89`
- Test: `tests/test_context.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_context.py`:

```python
"""Tests for session memory summarization and recall."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


def test_summarize_turn_extracts_facts():
    """summarize_turn calls router model and stores non-duplicate facts."""
    stored = []

    def mock_add(content, memory_type, source):
        stored.append(content)
        return f"mem_{len(stored)}"

    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router, \
         patch("pile.memory.store.search_memories", return_value=[]), \
         patch("pile.memory.store.add_memory", side_effect=mock_add):
        mock_settings.memory_enabled = True
        mock_router.return_value = "- Sprint 5: 68 tasks\n- Done: 11"

        from pile.context import summarize_turn
        summarize_turn("Sprint thế nào?", "Sprint 5 có 68 tasks, 11 done.")

    assert len(stored) == 2
    assert "Sprint 5: 68 tasks" in stored[0]
    assert "Done: 11" in stored[1]


def test_summarize_turn_skips_duplicates():
    """summarize_turn skips facts that already exist in memory."""
    stored = []

    def mock_search(query, n_results=1):
        # First fact is a duplicate, second is new
        if "Sprint 5" in query:
            return [{"distance": 0.1, "content": "Sprint 5: 68 tasks"}]
        return []

    def mock_add(content, memory_type, source):
        stored.append(content)
        return "mem_1"

    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router, \
         patch("pile.memory.store.search_memories", side_effect=mock_search), \
         patch("pile.memory.store.add_memory", side_effect=mock_add):
        mock_settings.memory_enabled = True
        mock_router.return_value = "- Sprint 5: 68 tasks\n- Blocker: TETRA-101"

        from pile.context import summarize_turn
        summarize_turn("Sprint thế nào?", "Sprint 5 có 68 tasks. Blocker TETRA-101.")

    assert len(stored) == 1
    assert "TETRA-101" in stored[0]


def test_summarize_turn_skips_none_response():
    """summarize_turn returns early when router returns NONE."""
    stored = []

    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router, \
         patch("pile.memory.store.add_memory") as mock_add:
        mock_settings.memory_enabled = True
        mock_router.return_value = "NONE"

        from pile.context import summarize_turn
        summarize_turn("hello", "Hi there!")

    mock_add.assert_not_called()


def test_summarize_turn_skips_short_response():
    """summarize_turn skips when agent response is too short."""
    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router:
        mock_settings.memory_enabled = True

        from pile.context import summarize_turn
        summarize_turn("hi", "Hello!")

    mock_router.assert_not_called()


def test_summarize_turn_skips_when_memory_disabled():
    """summarize_turn does nothing when memory is disabled."""
    with patch("pile.context.settings") as mock_settings, \
         patch("pile.context.call_router_model") as mock_router:
        mock_settings.memory_enabled = False

        from pile.context import summarize_turn
        summarize_turn("Sprint?", "Sprint 5 has 68 tasks.")

    mock_router.assert_not_called()


def test_recall_facts_returns_list():
    """recall_facts returns a list of fact strings."""
    with patch("pile.context.settings") as mock_settings, \
         patch("pile.memory.store.search_memories") as mock_search:
        mock_settings.memory_enabled = True
        mock_search.return_value = [
            {"content": "Sprint 5: 68 tasks", "distance": 0.5, "metadata": {"type": "session_fact"}},
            {"content": "Tân: PR-PO Epic 4", "distance": 0.6, "metadata": {"type": "session_fact"}},
            {"content": "irrelevant", "distance": 0.9, "metadata": {"type": "session_fact"}},
        ]

        from pile.context import recall_facts
        facts = recall_facts("sprint status")

    assert len(facts) == 2
    assert "Sprint 5: 68 tasks" in facts
    assert "Tân: PR-PO Epic 4" in facts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/test_context.py -v`
Expected: FAIL — `ImportError: cannot import name 'summarize_turn'`

- [ ] **Step 3: Implement summarize_turn and recall_facts**

Add to `src/pile/context.py` after `_compress()` (after line 89):

```python
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

# Agents whose output is not worth summarizing
_SKIP_AGENTS = {"TriageAgent"}

MIN_RESPONSE_LENGTH = 50


def summarize_turn(user_msg: str, agent_text: str, agent_name: str = "") -> None:
    """Extract key facts from a conversation turn and store in memory.

    Skips if memory disabled, response too short, agent is in skip list,
    or router returns NONE. Deduplicates against existing memories.
    """
    try:
        from pile.config import settings
        if not settings.memory_enabled:
            return

        if len(agent_text.strip()) < MIN_RESPONSE_LENGTH:
            return

        if agent_name in _SKIP_AGENTS:
            return

        from pile.client import call_router_model
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
        from pile.config import settings
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
```

Also add the import at the top of `src/pile/context.py` (the `settings` import is already lazy, but we reference `call_router_model` — keep it lazy inside the function as already patterned).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/test_context.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Run all existing tests**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/ --ignore=tests/e2e -x`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pile/context.py tests/test_context.py
git commit -m "feat: add summarize_turn and recall_facts for session memory"
```

---

### Task 3: Emit recalled_context event in workflow

**Files:**
- Modify: `src/pile/workflows/interactive.py:278-283`

- [ ] **Step 1: Update `_run_query` to yield recalled_context event**

In `src/pile/workflows/interactive.py`, replace the auto-recall section (lines 278-283):

```python
            # --- Auto-recall: inject memory context ---
            enriched_message = message
            if agent_key not in ("triage", "memory"):
                memory_context = recall(message)
                if memory_context:
                    enriched_message = f"{message}\n\n{memory_context}"
```

With:

```python
            # --- Auto-recall: inject memory context ---
            enriched_message = message
            if agent_key not in ("triage", "memory"):
                memory_context = recall(message)
                if memory_context:
                    enriched_message = f"{message}\n\n{memory_context}"
                    # Emit recalled facts for UI display
                    from pile.context import recall_facts
                    facts = recall_facts(message)
                    if facts:
                        yield WorkflowEvent.emit("system", {"type": "recalled_context", "facts": facts})
```

Note: import `recall_facts` lazily inside the block to match existing import patterns in this file.

- [ ] **Step 2: Run existing tests**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/ --ignore=tests/e2e -x`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/pile/workflows/interactive.py
git commit -m "feat: emit recalled_context event for UI display"
```

---

### Task 4: Handle recalled_context in UI + background summarization

**Files:**
- Modify: `src/pile/ui/chainlit_app.py:171-297`

- [ ] **Step 1: Add recalled_context handler in `_run_workflow_once`**

In `src/pile/ui/chainlit_app.py`, inside the `elif event.type == "data" and isinstance(event.data, dict):` block (after the tool_end handler, before `elif event.type == "output":`), add:

```python
                elif tool_data.get("type") == "recalled_context":
                    facts = tool_data.get("facts", [])
                    if facts:
                        recall_step = cl.Step(name="Recalled from previous sessions", type="run")
                        recall_step.output = "\n".join(f"- {f}" for f in facts)
                        await recall_step.send()
                        await recall_step.update()
```

- [ ] **Step 2: Add background summarize_turn call after response**

In `src/pile/ui/chainlit_app.py`, after the `await _send_charts_if_any(msg)` call (line 295), add the background summarization:

```python
    # Background: extract and persist key facts from this turn
    if user_input and msg.content and len(msg.content) >= 50:
        import asyncio
        from pile.context import summarize_turn

        async def _bg_summarize():
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    summarize_turn,
                    user_input,
                    msg.content,
                )
            except Exception:
                pass  # Never break the response flow

        asyncio.create_task(_bg_summarize())
```

- [ ] **Step 3: Add cleanup call in on_chat_start**

In `src/pile/ui/chainlit_app.py`, inside `on_chat_start()`, after the health checks block (after line 87), add:

```python
    # Clean up expired session facts
    try:
        from pile.memory.store import cleanup_expired_facts
        removed = cleanup_expired_facts(max_age_days=7)
        if removed:
            logger.info("Cleaned up %d expired session facts", removed)
    except Exception:
        pass
```

- [ ] **Step 4: Verify module loads**

Run: `cd /Users/tantran/works/gg && python -c "from pile.ui.chainlit_app import summarize_args, AGENT_CONFIG; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Run all tests**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/ --ignore=tests/e2e -x`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pile/ui/chainlit_app.py
git commit -m "feat: show recalled context in UI, background summarization"
```

---

### Task 5: Integration test

**Files:** None (verification only)

- [ ] **Step 1: Test summarization works**

```bash
cd /Users/tantran/works/gg && python -c "
from pile.context import summarize_turn, recall_facts
summarize_turn('Sprint hiện tại thế nào?', 'Sprint 5 TETRA chạy từ 2026-04-06 đến 2026-04-12. Có 68 tasks: 35 To Do, 15 Testing, 1 In Progress, 11 Done. Tân làm PR-PO Epic 4.')
print('Stored. Now recalling...')
facts = recall_facts('sprint status')
print(f'Recalled {len(facts)} facts:')
for f in facts:
    print(f'  - {f}')
assert len(facts) > 0, 'No facts recalled'
print('PASS')
"
```

- [ ] **Step 2: Test dedup works**

```bash
cd /Users/tantran/works/gg && python -c "
from pile.context import summarize_turn, recall_facts
from pile.memory.store import search_memories

# Store same info twice
summarize_turn('Sprint?', 'Sprint 5 TETRA: 68 tasks, 35 To Do, 11 Done.')
summarize_turn('Sprint?', 'Sprint 5 TETRA: 68 tasks, 35 To Do, 11 Done.')

# Should not have duplicates
results = search_memories('Sprint 5', n_results=10)
session_facts = [r for r in results if r.get('metadata', {}).get('type') == 'session_fact']
print(f'Session facts found: {len(session_facts)}')
# Allow some variance but should not double
assert len(session_facts) < 8, f'Too many duplicates: {len(session_facts)}'
print('PASS')
"
```

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/ --ignore=tests/e2e -v`
Expected: All tests PASS

- [ ] **Step 4: Commit if any fixes needed**

```bash
git add -u
git commit -m "fix: integration fixes for session memory"
```
