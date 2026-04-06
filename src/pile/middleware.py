"""Middleware for tracking tool calls across agents."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

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

    def __init__(self):
        self._calls: list[ToolCallRecord] = []
        self._seen_tools: dict[str, int] = {}  # tool_name → call count

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
            context.result = f"You already called {tool_name} {count} times. Stop calling tools and analyze the data you have."
            return

        self._seen_tools[tool_name] = count + 1

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
        self._seen_tools.clear()
        return calls

    @property
    def calls(self) -> list[ToolCallRecord]:
        return self._calls
