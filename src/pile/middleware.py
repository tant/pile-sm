"""Middleware for tracking tool calls across agents."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from agent_framework import FunctionMiddleware, FunctionInvocationContext

logger = logging.getLogger("pile.tools")


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""
    name: str
    arguments: dict
    result: str | None = None
    duration_ms: float = 0
    timestamp: float = 0


class ToolCallTracker(FunctionMiddleware):
    """Middleware that records tool calls for UI visualization.

    Usage:
        tracker = ToolCallTracker()
        agent = client.as_agent(..., middleware=[tracker])

        # After agent runs, check tracker.calls
        for call in tracker.drain():
            print(f"{call.name}({call.arguments}) -> {call.result} [{call.duration_ms}ms]")
    """

    def __init__(self):
        self._calls: list[ToolCallRecord] = []

    async def process(self, context: FunctionInvocationContext, call_next):
        args = dict(context.arguments) if hasattr(context.arguments, '__iter__') else {}
        record = ToolCallRecord(
            name=context.function.name,
            arguments=args,
            timestamp=time.time(),
        )
        logger.info("CALL %s(%s)", context.function.name, args)

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

    def drain(self) -> list[ToolCallRecord]:
        """Return and clear all recorded calls."""
        calls = self._calls[:]
        self._calls.clear()
        return calls

    @property
    def calls(self) -> list[ToolCallRecord]:
        return self._calls
