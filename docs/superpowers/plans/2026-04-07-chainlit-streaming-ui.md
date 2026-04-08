# Chainlit Streaming UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Chainlit UI show real-time streaming text, real-time tool call progress, and nested hierarchical steps so users always know what the agent is doing.

**Architecture:** Three-layer change: (1) middleware gets callbacks for tool start/end events, (2) workflow streams tokens and yields tool events via asyncio.Queue, (3) UI renders nested steps with expandable tool details.

**Tech Stack:** Python 3.12, Chainlit, asyncio, agent_framework

**Spec:** `docs/superpowers/specs/2026-04-07-chainlit-streaming-ui-design.md`

---

### Task 1: Add tool event callbacks to ToolCallTracker

**Files:**
- Modify: `src/pile/middleware.py:24-42` (ToolCallTracker `__init__` and `process`)
- Test: `tests/test_middleware.py` (create)

- [ ] **Step 1: Write failing test for on_tool_start callback**

Create `tests/test_middleware.py`:

```python
"""Tests for ToolCallTracker middleware callbacks."""

from __future__ import annotations

import asyncio
import pytest

from pile.middleware import ToolCallTracker, ToolCallRecord


class FakeContext:
    """Minimal FunctionInvocationContext stub."""

    def __init__(self, name: str, arguments: dict):
        self.function = type("F", (), {"name": name})()
        self.arguments = arguments
        self.result = None


@pytest.mark.asyncio
async def test_on_tool_start_callback_fires():
    """on_tool_start is called with tool name and args before execution."""
    events: list[tuple] = []

    async def on_start(name, args):
        events.append(("start", name, args))

    tracker = ToolCallTracker(on_tool_start=on_start)

    ctx = FakeContext("get_sprint", {"board_id": 42})
    await tracker.process(ctx, call_next=asyncio.coroutine(lambda: None))

    assert len(events) == 1
    assert events[0] == ("start", "get_sprint", {"board_id": 42})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/test_middleware.py::test_on_tool_start_callback_fires -v`
Expected: FAIL — `TypeError: ToolCallTracker.__init__() got an unexpected keyword argument 'on_tool_start'`

- [ ] **Step 3: Write failing test for on_tool_end callback**

Append to `tests/test_middleware.py`:

```python
@pytest.mark.asyncio
async def test_on_tool_end_callback_fires():
    """on_tool_end is called with ToolCallRecord after execution."""
    events: list[ToolCallRecord] = []

    async def on_end(record):
        events.append(record)

    tracker = ToolCallTracker(on_tool_end=on_end)

    ctx = FakeContext("search_issues", {"project": "TETRA"})

    async def fake_call_next():
        ctx.result = '{"issues": []}'

    await tracker.process(ctx, call_next=fake_call_next)

    assert len(events) == 1
    assert events[0].name == "search_issues"
    assert events[0].result == '{"issues": []}'
    assert events[0].duration_ms >= 0
```

- [ ] **Step 4: Write failing test for callbacks being optional**

Append to `tests/test_middleware.py`:

```python
@pytest.mark.asyncio
async def test_callbacks_are_optional():
    """Tracker works without callbacks (backward compat)."""
    tracker = ToolCallTracker()

    ctx = FakeContext("get_board", {"id": 1})
    await tracker.process(ctx, call_next=asyncio.coroutine(lambda: None))

    calls = tracker.drain()
    assert len(calls) == 1
    assert calls[0].name == "get_board"
```

- [ ] **Step 5: Implement callbacks in ToolCallTracker**

Edit `src/pile/middleware.py`:

Change `__init__`:

```python
class ToolCallTracker(FunctionMiddleware):
    """Middleware that records tool calls and detects loops.

    Loop detection: if the same tool is called with the same arguments
    twice in a row, the second call is blocked and returns an error
    message instead of executing.

    Usage:
        tracker = ToolCallTracker()
        agent = client.as_agent(..., middleware=[tracker])

        for call in tracker.drain():
            print(f"{call.name}({call.arguments}) -> {call.result} [{call.duration_ms}ms]")
    """

    def __init__(self, on_tool_start=None, on_tool_end=None):
        self._calls: list[ToolCallRecord] = []
        self._seen_tools: dict[str, int] = {}
        self.on_tool_start = on_tool_start  # async (name, args) -> None
        self.on_tool_end = on_tool_end      # async (record) -> None
```

Change `process()` — add callback calls:

```python
    async def process(self, context: FunctionInvocationContext, call_next):
        args = dict(context.arguments) if hasattr(context.arguments, '__iter__') else {}
        tool_name = context.function.name

        # Loop detection: same tool called 2+ times in one agent run
        count = self._seen_tools.get(tool_name, 0)
        if count >= 2:
            logger.warning(
                "LOOP DETECTED: %s called %d times already — blocking",
                tool_name, count,
            )
            record = ToolCallRecord(
                name=tool_name,
                arguments=args,
                result="Error: tool called too many times.",
                duration_ms=0,
                timestamp=time.time(),
            )
            self._calls.append(record)
            if self.on_tool_end:
                await self.on_tool_end(record)
            context.result = f"You already called {tool_name} {count} times. Stop calling tools and analyze the data you have."
            return

        self._seen_tools[tool_name] = count + 1

        record = ToolCallRecord(
            name=context.function.name,
            arguments=args,
            timestamp=time.time(),
        )
        logger.info("CALL %s(%s)", context.function.name, args)

        if self.on_tool_start:
            await self.on_tool_start(tool_name, args)

        start = time.monotonic()
        await call_next()
        record.duration_ms = round((time.monotonic() - start) * 1000)

        result = context.result
        if isinstance(result, str):
            record.result = result[:200]
        elif result is not None:
            record.result = str(result)[:200]

        logger.info("DONE %s → %dms | %s", context.function.name, record.duration_ms, record.result[:100] if record.result else "")
        self._calls.append(record)

        if self.on_tool_end:
            await self.on_tool_end(record)
```

- [ ] **Step 6: Run all middleware tests**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/test_middleware.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_middleware.py src/pile/middleware.py
git commit -m "feat: add on_tool_start/on_tool_end callbacks to ToolCallTracker"
```

---

### Task 2: Add summarize_args helper

**Files:**
- Modify: `src/pile/ui/chainlit_app.py` (add helper function near top)
- Test: `tests/test_ui_helpers.py` (create)

- [ ] **Step 1: Write failing tests for summarize_args**

Create `tests/test_ui_helpers.py`:

```python
"""Tests for UI helper functions."""

from pile.ui.chainlit_app import summarize_args


def test_summarize_args_basic():
    args = {"project": "TETRA", "state": "active"}
    result = summarize_args(args)
    assert result == "project=TETRA, state=active"


def test_summarize_args_truncates_long_values():
    args = {"query": "a" * 50}
    result = summarize_args(args)
    assert result == "query=" + "a" * 30 + "..."


def test_summarize_args_max_3_keys():
    args = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5"}
    result = summarize_args(args)
    assert result == "a=1, b=2, c=3, +2 more"


def test_summarize_args_empty():
    assert summarize_args({}) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/test_ui_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'summarize_args'`

- [ ] **Step 3: Implement summarize_args**

Add to `src/pile/ui/chainlit_app.py` after the imports (before `AGENT_CONFIG`):

```python
def summarize_args(args: dict, max_keys: int = 3, max_val_len: int = 30) -> str:
    """Summarize tool arguments for display: 'key=val, key=val, +N more'."""
    if not args:
        return ""
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/test_ui_helpers.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ui_helpers.py src/pile/ui/chainlit_app.py
git commit -m "feat: add summarize_args helper for tool display"
```

---

### Task 3: Refactor workflow to stream text tokens

**Files:**
- Modify: `src/pile/workflows/interactive.py:181-206` (`_execute_agent` method)
- Modify: `src/pile/workflows/interactive.py:208-311` (`_run_query` method)

This is the core change. `_execute_agent` becomes an async generator yielding `AgentResponseUpdate` objects. A new `_stream_agent_events` method wraps it to yield `WorkflowEvent`s while interleaving tool events. Since async generators cannot `return` values, accumulated state (`full_text`, `tool_calls`) is stored on `self._last_full_text` / `self._last_tool_calls`.

- [ ] **Step 1: Add `import asyncio` at top of file**

Add to `src/pile/workflows/interactive.py` imports:

```python
import asyncio
```

- [ ] **Step 2: Change `_execute_agent` to async generator**

Replace the `_execute_agent` method in `src/pile/workflows/interactive.py` (lines 181-206):

```python
    async def _execute_agent(self, agent_key: str, message: str):
        """Run a single agent, yielding streaming updates.

        Yields AgentResponseUpdate objects as they arrive.
        Caller accumulates full_text for recovery detection.
        """
        agent = self.agents.get(agent_key, self.agents["triage"])
        session = self._get_session(agent_key)

        # Drain any leftover tool calls from previous runs.
        self.tracker.drain()

        result_stream = agent.run(message, stream=True, session=session)
        async for update in result_stream:
            yield update
        await result_stream.get_final_response()
```

- [ ] **Step 3: Add `_stream_agent_events` helper method**

Add new method to `RoutedWorkflow` class:

```python
    async def _stream_agent_events(self, agent_key: str, message: str, tool_events: asyncio.Queue):
        """Stream an agent run, yielding WorkflowEvents for text and tool calls.

        After iteration completes, read self._last_full_text and
        self._last_tool_calls for recovery detection.
        """
        from agent_framework._workflows._events import WorkflowEvent

        agent = self.agents.get(agent_key, self.agents["triage"])
        agent_name = agent.name
        self.last_agent_key = agent_key.replace("_scrum_prefetch", "scrum")
        logger.info("Route: '%s' -> %s", message[:50], agent_name)
        yield WorkflowEvent.executor_invoked(agent_name)

        self._last_full_text = ""

        async for update in self._execute_agent(agent_key, message):
            # Drain tool events that arrived during this iteration
            while not tool_events.empty():
                try:
                    evt = tool_events.get_nowait()
                    if evt[0] == "tool_start":
                        yield WorkflowEvent.emit(agent_name, {"type": "tool_start", "name": evt[1], "args": evt[2]})
                    elif evt[0] == "tool_end":
                        yield WorkflowEvent.emit(agent_name, {"type": "tool_end", "record": evt[1]})
                except asyncio.QueueEmpty:
                    break

            if update.text:
                self._last_full_text += update.text
                yield WorkflowEvent.output(agent_name, update)

        # Drain remaining tool events after stream ends
        while not tool_events.empty():
            try:
                evt = tool_events.get_nowait()
                if evt[0] == "tool_start":
                    yield WorkflowEvent.emit(agent_name, {"type": "tool_start", "name": evt[1], "args": evt[2]})
                elif evt[0] == "tool_end":
                    yield WorkflowEvent.emit(agent_name, {"type": "tool_end", "record": evt[1]})
            except asyncio.QueueEmpty:
                break

        self._last_tool_calls = self.tracker.drain()
```

- [ ] **Step 4: Replace `_run_query` with streaming version**

Replace `_run_query` method (lines 208-311) in `src/pile/workflows/interactive.py`:

```python
    async def _run_query(self, message: str, stream: bool = False):
        """Route and execute a user query, with recovery on failure."""
        from agent_framework._workflows._events import WorkflowEvent
        from agent_framework._types import AgentResponseUpdate, Content

        self._is_running = True
        agent_name = "unknown"
        try:
            # Check cache first (read-only queries)
            agent_key = smart_route(message)
            cached = get_cached(message)
            if cached and agent_key not in ("jira_write", "memory", "browser"):
                cached_text, cached_agent = cached
                logger.info("Cache hit: '%s' -> %s", message[:50], cached_agent)
                yield WorkflowEvent.executor_invoked(f"{cached_agent} (cached)")
                yield WorkflowEvent.output(cached_agent, AgentResponseUpdate(text=cached_text))
                yield WorkflowEvent.executor_completed(cached_agent)
                return

            # --- Prefetch data for scrum queries ---
            has_prefetch = False
            if agent_key == "scrum" and self.board_id:
                data = prefetch_scrum_data(message, self.board_id)
                if data:
                    has_prefetch = True
                    self.agents["_scrum_prefetch"] = create_scrum_agent(
                        self.client,
                        middleware=[self.tracker],
                        prefetch_data=data,
                    )
                    agent_key = "_scrum_prefetch"
                    self._sessions.pop("_scrum_prefetch", None)

            # --- Auto-recall: inject memory context ---
            enriched_message = message
            if agent_key not in ("triage", "memory"):
                memory_context = recall(message)
                if memory_context:
                    enriched_message = f"{message}\n\n{memory_context}"

            # --- Wire tool event callbacks ---
            tool_events: asyncio.Queue = asyncio.Queue()

            async def _on_tool_start(name, args):
                await tool_events.put(("tool_start", name, args))

            async def _on_tool_end(record):
                await tool_events.put(("tool_end", record))

            self.tracker.on_tool_start = _on_tool_start
            self.tracker.on_tool_end = _on_tool_end

            # --- Stream first agent ---
            async for event in self._stream_agent_events(agent_key, enriched_message, tool_events):
                yield event
            full_text = self._last_full_text
            tool_calls = self._last_tool_calls

            # --- Recovery check ---
            is_failure = (
                agent_key not in _NO_RETRY
                and _detect_failure(full_text, tool_calls, agent_key, has_prefetch)
            )

            if is_failure:
                fallback_key = _get_fallback(agent_key, set(self.agents.keys()))
                if fallback_key:
                    agent = self.agents.get(agent_key, self.agents["triage"])
                    logger.info(
                        "Recovery: %s failed (text=%d, tools=%d), retrying -> %s",
                        agent_key, len(full_text), len(tool_calls), fallback_key,
                    )
                    yield WorkflowEvent.executor_completed(agent.name)

                    # --- Fallback attempt ---
                    async for event in self._stream_agent_events(fallback_key, enriched_message, tool_events):
                        yield event
                    full_text = self._last_full_text
                    tool_calls = self._last_tool_calls

                    if full_text and len(full_text.strip()) > 20:
                        original_key = agent_key.replace("_scrum_prefetch", "scrum")
                        learn(
                            message,
                            f"Query '{message}' failed on {original_key}, "
                            f"succeeded on {fallback_key}.",
                        )

                    agent_key = fallback_key
                    is_failure = False

            # Cache read-only responses
            agent = self.agents.get(agent_key, self.agents["triage"])
            agent_name = agent.name
            if full_text and agent_key not in ("jira_write", "memory", "browser"):
                if not is_failure:
                    set_cached(message, full_text, agent_name)

            yield WorkflowEvent.executor_completed(agent_name)

        except Exception as e:
            logger.exception("Workflow error: %s", e)
            yield WorkflowEvent.executor_failed(agent_name, str(e))
        finally:
            self._is_running = False
            self.tracker.on_tool_start = None
            self.tracker.on_tool_end = None
```

- [ ] **Step 4: Run existing tests to check nothing is broken**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/ -v --ignore=tests/e2e -x`
Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pile/workflows/interactive.py
git commit -m "feat: stream text tokens and tool events in workflow"
```

---

### Task 4: Rewrite Chainlit UI to handle streaming + nested steps

**Files:**
- Modify: `src/pile/ui/chainlit_app.py:153-261` (`_run_workflow_once` function)

- [ ] **Step 1: Replace `_run_workflow_once` with streaming event handler**

Replace the entire `_run_workflow_once` function in `src/pile/ui/chainlit_app.py`:

```python
async def _run_workflow_once(workflow, *, user_input: str | None = None, responses: dict | None = None):
    """Stream a single workflow run, rendering steps in real-time."""
    import asyncio

    msg = cl.Message(content="", author="Pile SM")
    await msg.send()

    current_step: cl.Step | None = None
    active_tool_steps: dict[str, cl.Step] = {}  # tool_name -> step (for in-progress tools)

    if user_input is not None:
        run_iter = workflow.run(user_input, stream=True, include_status_events=True)
    else:
        run_iter = workflow.run(responses=responses, stream=True, include_status_events=True)

    try:
        async for event in run_iter:
            executor_id = getattr(event, "executor_id", None)

            if event.type == "executor_invoked" and executor_id:
                # Close previous agent step if any
                if current_step:
                    await current_step.update()
                    current_step = None

                cfg = AGENT_CONFIG.get(executor_id, {"type": "run", "label": executor_id})
                current_step = cl.Step(name=cfg["label"], type=cfg["type"])
                current_step.output = ""
                await current_step.send()

            elif event.type == "data" and isinstance(event.data, dict):
                # Tool events from workflow
                tool_data = event.data

                if tool_data.get("type") == "tool_start" and current_step:
                    tool_name = tool_data["name"]
                    tool_args = tool_data.get("args", {})
                    tool_step = cl.Step(
                        name=f"{tool_name} ({summarize_args(tool_args)})",
                        type="tool",
                        parent_id=current_step.id,
                    )
                    # input holds full args (visible when expanded)
                    tool_step.input = str(tool_args) if tool_args else ""
                    tool_step.output = ""
                    await tool_step.send()
                    active_tool_steps[tool_name] = tool_step

                elif tool_data.get("type") == "tool_end":
                    record = tool_data["record"]
                    tool_name = record.name
                    tool_step = active_tool_steps.pop(tool_name, None)
                    if tool_step:
                        duration = f"{record.duration_ms / 1000:.1f}s"
                        tool_step.output = record.result or ""
                        tool_step.name = f"{tool_name} — {duration}"
                        await tool_step.update()

            elif event.type == "output":
                text = getattr(event.data, "text", None) if not isinstance(event.data, list) else None
                if text:
                    if current_step:
                        current_step.output += text
                    await msg.stream_token(text)

            elif event.type == "executor_completed":
                # Close any remaining tool steps
                for tool_step in active_tool_steps.values():
                    await tool_step.update()
                active_tool_steps.clear()

                if current_step:
                    await current_step.update()
                    current_step = None

            elif event.type == "executor_failed":
                if current_step:
                    error_msg = str(event.data) if event.data else "Unknown error"
                    current_step.output += f"\n\n*Error: {error_msg}*"
                    await current_step.update()
                    current_step = None

            elif event.type == "request_info":
                pass  # Not used in current routing workflow

    except asyncio.CancelledError:
        workflow._reset_running_flag()
        # Close in-progress tool steps
        for tool_step in active_tool_steps.values():
            tool_step.output += "\n*Cancelled*"
            await tool_step.update()
        active_tool_steps.clear()
        # Close agent step
        if current_step:
            if current_step.output:
                current_step.output += "\n\n*Stopped by user*"
                await current_step.update()
            else:
                await current_step.remove()
        if not msg.content:
            msg.content = "*Stopped.*"
        await msg.update()
        return []
    except Exception as e:
        workflow._reset_running_flag()
        if current_step:
            current_step.output += f"\n\n*Error: {e}*"
            await current_step.update()
        if not msg.content:
            msg.content = f"*Error: {e}*"
        await msg.update()
        raise

    if current_step:
        await current_step.update()

    await msg.update()

    # Auto-detect numeric data and render charts
    await _send_charts_if_any(msg)

    return []
```

- [ ] **Step 2: Verify the app loads without errors**

Run: `cd /Users/tantran/works/gg && python -c "from pile.ui.chainlit_app import summarize_args, AGENT_CONFIG; print('OK')" `
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/pile/ui/chainlit_app.py
git commit -m "feat: render nested tool steps with real-time streaming"
```

---

### Task 5: Manual integration test

**Files:** None (verification only)

- [ ] **Step 1: Start the app**

Run: `cd /Users/tantran/works/gg && chainlit run src/pile/ui/chainlit_app.py -w`

- [ ] **Step 2: Test basic query**

In the browser, send: "Sprint hiện tại tiến độ thế nào?"

Verify:
- Agent step appears immediately (e.g. "Sprint")
- Tool steps appear nested inside agent step as they execute
- Tool steps show name + summarized args
- Tool steps update with duration when complete
- Text streams token-by-token into the main message
- No flickering, no empty steps appearing and disappearing

- [ ] **Step 3: Test recovery flow**

Send a query that triggers fallback (e.g. a query that misroutes).

Verify:
- First agent step appears, shows tool calls
- If it fails, second agent step appears below
- Both steps remain visible — nothing removed
- Final text streams from the successful agent

- [ ] **Step 4: Test stop button**

Send a query, click Stop while it's running.

Verify:
- In-progress tool steps show "Cancelled"
- Agent step shows "Stopped by user"
- Already-streamed text remains in message
- No errors in console

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/tantran/works/gg && python -m pytest tests/ -v --ignore=tests/e2e`
Expected: All tests PASS

- [ ] **Step 6: Final commit if any fixes needed**

```bash
git add -u
git commit -m "fix: integration fixes for streaming UI"
```
