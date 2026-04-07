"""Tests for ToolCallTracker middleware callbacks."""

from __future__ import annotations

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
    async def noop():
        pass

    await tracker.process(ctx, call_next=noop)

    assert len(events) == 1
    assert events[0] == ("start", "get_sprint", {"board_id": 42})


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


@pytest.mark.asyncio
async def test_callbacks_are_optional():
    """Tracker works without callbacks (backward compat)."""
    tracker = ToolCallTracker()

    ctx = FakeContext("get_board", {"id": 1})
    async def noop():
        pass

    await tracker.process(ctx, call_next=noop)

    calls = tracker.drain()
    assert len(calls) == 1
    assert calls[0].name == "get_board"
