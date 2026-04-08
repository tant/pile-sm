# Chainlit Streaming UI — Real-time Agent Visualization

**Date:** 2026-04-07
**Status:** Draft
**Scope:** middleware.py, interactive.py, chainlit_app.py

## Problem

The current Chainlit UI has several issues that make it feel unresponsive and opaque:

1. **No real streaming** — workflow runs `stream=False`, collects full output, yields once. User sees text appear all at once.
2. **Tool calls appear after agent completes** — user has no visibility into what the agent is doing while it runs.
3. **Tool steps not nested** — tool steps render flat alongside agent steps, no hierarchy.
4. **Empty steps flicker** — steps appear empty then get removed, causing visual noise.
5. **Redundant code** — `accumulated_text` in UI layer only used for logging (which belongs in workflow layer).

## Goals

- User sees text streaming token-by-token as agent generates it
- Tool calls appear in real-time: when they start and when they complete
- Hierarchical display: tool steps nested inside agent steps (expandable)
- Tool detail: collapsed shows name + key args + duration; expanded shows full args + result
- Recovery is transparent: if agent fails and fallback runs, user sees both in timeline
- No flicker, no duplicate content, no noise

## Non-Goals

- Changing the routing/recovery logic itself
- Modifying agent framework internals
- Adding new UI components beyond Chainlit's built-in Steps

## Architecture

```
User message
    |
    v
+------------------------------------------+
| Workflow Layer (interactive.py)           |
|                                          |
|  smart_route() -> agent_key              |
|  set tracker callbacks (on_tool_start,   |
|    on_tool_end)                          |
|  yield executor_invoked                  |
|  +-- stream agent -------------------+   |
|  |  token -> yield output event      |   |
|  |  tool start -> yield tool_start   |<-- tracker.on_tool_start
|  |  tool end -> yield tool_end       |<-- tracker.on_tool_end
|  +-----------------------------------+   |
|  detect_failure()                        |
|  if fail -> yield fallback events        |
|  yield executor_completed                |
+------------------+-----------------------+
                   | WorkflowEvents
                   v
+------------------------------------------+
| UI Layer (chainlit_app.py)               |
|                                          |
|  executor_invoked -> cl.Step(agent)      |
|  tool_start -> cl.Step(tool, parent_id)  |
|  tool_end -> update tool step            |
|  output -> msg.stream_token()            |
|  executor_completed -> update agent step |
+------------------------------------------+
```

## Design

### 1. Middleware Layer — Tool Event Callbacks

**File:** `src/pile/middleware.py`

Add optional async callbacks to `ToolCallTracker`:

```python
class ToolCallTracker(FunctionMiddleware):
    def __init__(self, on_tool_start=None, on_tool_end=None):
        self._calls: list[ToolCallRecord] = []
        self._seen_tools: dict[str, int] = {}
        self.on_tool_start = on_tool_start  # async (name, args) -> None
        self.on_tool_end = on_tool_end      # async (record) -> None
```

In `process()`:

```
1. Build args dict
2. Loop detection check (unchanged)
3. Call on_tool_start(tool_name, args) if set
4. await call_next()
5. Build ToolCallRecord with result + duration
6. Call on_tool_end(record) if set
7. Append to _calls (unchanged)
```

`drain()` stays for backward compatibility (CLI still uses it).

Callbacks are async to allow the workflow layer to put events on an asyncio.Queue without blocking.

### 2. Workflow Layer — Streaming + Tool Events

**File:** `src/pile/workflows/interactive.py`

#### 2a. `_execute_agent()` becomes an async generator

Instead of collecting `full_text` and returning it, yield each `AgentResponseUpdate` as it arrives:

```python
async def _execute_agent(self, agent_key, message):
    agent = self.agents.get(agent_key, self.agents["triage"])
    session = self._get_session(agent_key)
    self.tracker.drain()  # clear leftovers

    result_stream = agent.run(message, stream=True, session=session)
    async for update in result_stream:
        yield update
    await result_stream.get_final_response()
```

The caller collects `full_text` by accumulating `update.text` while also yielding WorkflowEvents.

#### 2b. Tool events via asyncio.Queue

Before running agent, wire up tracker callbacks to a queue:

```python
tool_events = asyncio.Queue()

async def on_start(name, args):
    await tool_events.put(("tool_start", name, args))

async def on_end(record):
    await tool_events.put(("tool_end", record))

self.tracker.on_tool_start = on_start
self.tracker.on_tool_end = on_end
```

In `_run_query()`, interleave text stream and tool events:

- After each `update` from agent stream, drain any pending tool events from queue (non-blocking `get_nowait()`)
- Also drain queue after stream ends (tools may complete after last text token)
- Yield both as WorkflowEvents in arrival order:
  - Text: `WorkflowEvent.output(agent_name, update)`
  - Tool start: `WorkflowEvent.emit(agent_name, {"type": "tool_start", "name": ..., "args": ...})`
  - Tool end: `WorkflowEvent.emit(agent_name, {"type": "tool_end", "record": ...})`
- Tool events that arrive between text tokens are yielded immediately before the next text token

#### 2c. Recovery on stream

```
1. Stream first agent -> yield text + tool events real-time
2. Accumulate full_text during streaming
3. Collect tool_calls from tracker after stream ends
4. detect_failure(full_text, tool_calls, agent_key)
5. If fail:
   - yield executor_completed for failed agent
   - yield executor_invoked for fallback agent
   - Stream fallback agent (repeat from step 1)
6. User sees both agents' output in timeline — nothing hidden
```

### 3. UI Layer — Nested Steps + Real-time Rendering

**File:** `src/pile/ui/chainlit_app.py`

#### 3a. Event handling map

| WorkflowEvent | UI Action |
|---|---|
| `executor_invoked` | Create `cl.Step(name=label, type=cfg_type)`, send |
| `tool_start` | Create `cl.Step(name=tool_name, type="tool", parent_id=agent_step.id)`, set `input` to summarized args, send |
| `tool_end` | Update tool step: set `output` to result summary, set `end` timestamp, update |
| `output` (text token) | `msg.stream_token(text)` |
| `executor_completed` | Update agent step |
| `executor_failed` | Update agent step with error |

#### 3b. Nested tool steps

Tool steps use `parent_id=agent_step.id` so they render inside the agent step:

```
> Sprint
  > get_active_sprint (project=TETRA, state=active) — 1.2s
  > get_sprint_issues (sprintId=42) — 0.8s
Sprint TETRA-42 is on day 8/10...
```

#### 3c. Tool input/output formatting

**Collapsed view:** `tool_name (key1=val1, key2=val2) — 1.2s`
- Max 3 key-value pairs
- Values truncated at 30 chars

**Expanded view (click):**
- Full JSON arguments
- Full result text

Helper function:

```python
def summarize_args(args: dict, max_keys=3, max_val_len=30) -> str:
    items = list(args.items())[:max_keys]
    parts = []
    for k, v in items:
        v_str = str(v)
        if len(v_str) > max_val_len:
            v_str = v_str[:max_val_len] + "..."
        parts.append(f"{k}={v_str}")
    summary = ", ".join(parts)
    if len(args) > max_keys:
        summary += f", +{len(args) - max_keys} more"
    return summary
```

#### 3d. Code cleanup

Remove from `_run_workflow_once()`:
- `accumulated_text` variable — logging is in workflow layer
- `tracker.drain()` loop in `executor_completed` — tools now rendered real-time
- `current_step.remove()` for empty steps — no more empty steps with streaming

### 4. Error Handling & Edge Cases

#### 4a. User presses Stop

```
1. CancelledError caught
2. If tool_step in progress -> update with "Cancelled", close
3. If agent_step in progress -> update with "Stopped by user", close
4. Main message -> update (keep already-streamed text)
5. Reset workflow running flag
```

No removal, no flicker. User sees output up to the point they stopped.

#### 4b. Tool timeout / error

- `on_tool_end` still fires with `record.result = "Error: ..."`
- Tool step shows error message in output
- Agent continues running (framework handles tool errors)

#### 4c. Recovery display

```
> Sprint (retrying...)
  > get_active_sprint — Error: 404
> Jira Query
  > search_issues (project=TETRA) — 0.9s
Results streaming here...
```

Failed agent step gets label suffix indicating retry. User sees full timeline.

#### 4d. Empty agent output

If agent streams no text but made tool calls:
- Agent step stays visible with tool calls nested inside
- Not removed — tool calls are meaningful output

## Files Changed

| File | Change |
|---|---|
| `src/pile/middleware.py` | Add `on_tool_start`, `on_tool_end` callbacks to ToolCallTracker |
| `src/pile/workflows/interactive.py` | `_execute_agent` as async generator with `stream=True`, yield tool events via queue, recovery on stream |
| `src/pile/ui/chainlit_app.py` | Nested steps via `parent_id`, handle `tool_start`/`tool_end` events, add `summarize_args()`, remove dead code |

## Files Unchanged

- `AGENT_CONFIG` dict — same labels and types
- `ToolCallRecord` dataclass — same fields
- `drain()` method — kept for CLI backward compatibility
- Recovery logic (`_detect_failure`, `_FALLBACK_CHAINS`) — same logic, different timing
- `_send_charts_if_any()` — unchanged
- Starters, file upload handling — unchanged
