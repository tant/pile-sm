# Session Memory — Persist Key Facts Across Chat Sessions

**Date:** 2026-04-08
**Status:** Draft
**Scope:** context.py, interactive.py, chainlit_app.py, store.py

## Problem

When a user refreshes the page or starts a new chat session, all conversation context is lost. The AgentSession objects are in-memory only. The user has to re-ask questions to get the same context back.

The system already has ChromaDB-backed persistent memory (`recall`/`learn`), but it only stores:
- User-saved memories (via memory tools)
- Auto-learned routing lessons (from fallback recovery)

It does not capture key facts (numbers, dates, decisions, statuses) from agent responses.

## Goals

- After each agent response, extract key facts and persist them in ChromaDB
- Deduplicate: don't store the same fact twice
- On new sessions, auto-recall relevant facts and show them clearly in UI
- User always knows which information comes from memory vs real-time data
- Session facts expire after 7 days (not permanent like user-saved memories)
- Runs in background — does not slow down the response

## Non-Goals

- Persisting full conversation history or raw AgentSession objects
- Replacing the existing `recall()`/`learn()` system
- Changing user-facing memory tools (remember/forget/search)

## Design

### 1. Summarization — Extract key facts per turn

After each agent response completes, send `(user_message, agent_response_text)` to the router model in a background task.

**Prompt:**
```
Extract only important facts from this conversation turn.
Include: numbers, dates, decisions, issue IDs, names, statuses, assignments.
Skip: greetings, generic explanations, filler text.
Return one fact per line as bullet points. If nothing important, return "NONE".

User: {user_message}
Agent: {agent_response_text}
```

**Skip summarization when:**
- Agent response is empty or error
- Response text < 50 chars
- Agent is TriageAgent (greetings only)

**Runs as:** `asyncio.create_task()` after `msg.update()` — does not block UI.

### 2. Storage — ChromaDB with dedup

Use the existing `memories` collection in ChromaDB. Each extracted fact is stored as a separate document.

**Before storing each fact:**
1. Embed the fact text
2. Search existing memories with cosine distance threshold 0.3 (same as `learn()` dedup)
3. If a near-duplicate exists → skip
4. If new → store with metadata:

```python
{
    "type": "session_fact",
    "source": "session_summary",
    "created_at": "2026-04-08T14:30:00",
}
```

**TTL:** Session facts (type `session_fact`) expire after 7 days. When `recall()` runs, it filters out expired session facts by checking `created_at` against current time. User-saved memories (type "decision", "pattern", "note") never expire.

**Why individual facts instead of one block?** Dedup is more precise. If the user asks about the sprint twice, individual facts like "Sprint 5: 68 tasks" will match and skip, instead of storing two different summary blocks containing the same information.

### 3. Recall — Show recalled context clearly in UI

#### 3a. Workflow emits recalled_context event

In `_run_query()`, after `recall()` returns memory context, yield a new event before streaming the agent:

```python
if memory_context:
    # Parse individual facts from memory_context string
    facts = [line.strip("- ").strip() for line in memory_context.split("\n") if line.strip().startswith("-")]
    yield WorkflowEvent.emit(agent_name, {"type": "recalled_context", "facts": facts})
```

#### 3b. UI renders recalled context as a distinct step

In `_run_workflow_once()`, handle the new event type:

```python
elif tool_data.get("type") == "recalled_context":
    facts = tool_data.get("facts", [])
    if facts:
        recall_step = cl.Step(name="Recalled from previous sessions", type="run")
        recall_step.output = "\n".join(f"- {f}" for f in facts)
        await recall_step.send()
        await recall_step.update()
```

This renders as:
```
▼ Recalled from previous sessions
  - Sprint 5 TETRA: 2026-04-06 to 2026-04-12, 68 tasks
  - Tân: PR-PO Epic 4 cho Maison
▼ Sprint
  > get_active_sprint (project=TETRA) — 1.2s
  New real-time results...
```

User sees the distinction: "Recalled" step = old data from memory, agent steps below = fresh data.

#### 3c. No change to recall() injection

`recall()` still injects context into `enriched_message` as before. The new event is purely for UI visibility — the agent already receives the recalled context in its prompt.

### 4. Expired fact cleanup

Add `cleanup_expired_facts(max_age_days=7)` to `store.py`. Called once at startup (`on_chat_start`) to remove stale session facts.

Query ChromaDB for all documents with `type=session_fact` and `created_at` older than 7 days, then delete them.

## Architecture

```
User sends message
    |
    v
+------------------------------------------+
| Workflow (_run_query)                     |
|                                          |
|  recall(message) -> memory_context       |
|  if memory_context:                      |
|    yield emit("recalled_context", facts) | <- UI shows step
|    enriched_message = message + context  |
|                                          |
|  stream agent -> yield events            |
|  yield executor_completed                |
+------------------+-----------------------+
                   |
                   v
+------------------------------------------+
| UI (_run_workflow_once)                   |
|                                          |
|  "recalled_context" -> cl.Step           | <- visible to user
|  Stream text + tool steps as before      |
|  After msg.update():                     |
|    asyncio.create_task(                  |
|      summarize_and_store(user, agent)    | <- background
|    )                                     |
+------------------------------------------+
                   |
                   v (background, non-blocking)
+------------------------------------------+
| summarize_and_store()                    |
|                                          |
|  router_model.extract(user + agent_text) |
|  parse bullet points -> individual facts |
|  for each fact:                          |
|    if not duplicate (cosine < 0.3):      |
|      store in ChromaDB (session_fact)    |
+------------------------------------------+
```

## Files Changed

| File | Change |
|------|--------|
| `src/pile/context.py` | Add `summarize_turn(user_msg, agent_text)` — extract facts via router model, dedup, store. Add `recall_facts(query)` returning list of fact strings (for UI event). |
| `src/pile/workflows/interactive.py` | Yield `recalled_context` event before agent stream when memory_context is non-empty |
| `src/pile/ui/chainlit_app.py` | Handle `recalled_context` event as cl.Step. Call `summarize_turn()` as background task after each response |
| `src/pile/memory/store.py` | Add `cleanup_expired_facts(max_age_days=7)` to delete old session_fact entries |

## Files Unchanged

- `recall()` — still works as-is, searches all memories including session_facts
- `learn()` — separate concern (recovery learning), unchanged
- `memory_tools.py` — user tools unchanged
- `middleware.py`, `engine.py`, `llm_client.py` — unrelated
- ChromaDB collections — reuse existing `memories` collection, no schema change
