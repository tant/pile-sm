"""Middleware for tracking tool calls across agents."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
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

    def __init__(
        self,
        on_tool_start: Callable[[str, dict], Awaitable[None]] | None = None,
        on_tool_end: Callable[[ToolCallRecord], Awaitable[None]] | None = None,
    ):
        self._calls: list[ToolCallRecord] = []
        self._seen_tools: dict[str, int] = {}  # tool_name → call count
        self._seen_exact: set[str] = set()  # "tool_name|args_hash" for exact dupe detection
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end

    async def process(self, context: FunctionInvocationContext, call_next):
        args = dict(context.arguments) if hasattr(context.arguments, '__iter__') else {}
        tool_name = context.function.name

        # Loop detection: block exact duplicate calls (same tool + same args)
        # or same tool called 4+ times (even with different args)
        call_key = f"{tool_name}|{sorted(args.items())}"
        count = self._seen_tools.get(tool_name, 0)
        is_exact_dupe = call_key in self._seen_exact

        if is_exact_dupe or count >= 3:
            reason = "exact same call" if is_exact_dupe else f"called {count} times"
            logger.warning("LOOP DETECTED: %s (%s) — blocking", tool_name, reason)
            record = ToolCallRecord(
                name=tool_name,
                arguments=args,
                result=f"Error: tool loop detected ({reason}).",
                duration_ms=0,
                timestamp=time.time(),
            )
            self._calls.append(record)
            if self.on_tool_end:
                await self.on_tool_end(record)
            context.result = f"Loop detected: {tool_name} ({reason}). Analyze the data you already have."
            return

        self._seen_tools[tool_name] = count + 1
        self._seen_exact.add(call_key)

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

    def drain(self) -> list[ToolCallRecord]:
        """Return and clear all recorded calls."""
        calls = self._calls[:]
        self._calls.clear()
        self._seen_tools.clear()
        self._seen_exact.clear()
        return calls

    @property
    def calls(self) -> list[ToolCallRecord]:
        return self._calls
