"""MAF-compatible chat client backed by llama-cpp-python."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Awaitable, ClassVar

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)
from agent_framework._types import ResponseStream, ChatResponseUpdate, UsageDetails

from pile.models.engine import chat_completion


# ---------------------------------------------------------------------------
# Helpers: MAF Message <-> llama-cpp dict conversion
# ---------------------------------------------------------------------------

def _messages_to_dicts(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert a sequence of MAF Messages into llama-cpp compatible dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        d: dict[str, Any] = {"role": msg.role}

        tool_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []
        tool_result: dict[str, Any] | None = None

        for c in msg.contents:
            if c.type == "text" and c.text:
                text_parts.append(c.text)
            elif c.type == "function_call":
                tool_calls.append({
                    "id": c.call_id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": c.arguments if isinstance(c.arguments, str) else "",
                    },
                })
            elif c.type == "function_result":
                tool_result = {
                    "role": "tool",
                    "tool_call_id": c.call_id,
                    "content": str(c.result) if c.result is not None else "",
                }

        if tool_result is not None:
            # Tool results become their own message with role=tool
            result.append(tool_result)
            continue

        d["content"] = " ".join(text_parts) if text_parts else None
        if tool_calls:
            d["tool_calls"] = tool_calls
        result.append(d)

    return result


def _parse_response(raw: dict[str, Any]) -> ChatResponse:
    """Convert a llama-cpp OpenAI-compatible response dict into a ChatResponse."""
    choice = raw["choices"][0]
    message = choice["message"]
    finish = choice.get("finish_reason", "stop")

    contents: list[Content] = []

    # Text content
    if message.get("content"):
        contents.append(Content.from_text(message["content"]))

    # Tool calls
    for tc in message.get("tool_calls", []) or []:
        fn = tc["function"]
        contents.append(
            Content.from_function_call(
                call_id=tc["id"],
                name=fn["name"],
                arguments=fn.get("arguments", ""),
            )
        )
        finish = "tool_calls"

    usage = raw.get("usage", {})
    usage_details: UsageDetails = {
        "input_token_count": usage.get("prompt_tokens"),
        "output_token_count": usage.get("completion_tokens"),
        "total_token_count": usage.get("total_tokens"),
    }

    return ChatResponse(
        messages=Message("assistant", contents),
        response_id=raw.get("id"),
        model=raw.get("model"),
        created_at=raw.get("created"),
        finish_reason=finish,
        usage_details=usage_details,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LlamaCppClient(
    FunctionInvocationLayer,
    ChatMiddlewareLayer,
    BaseChatClient,
):
    """Chat client bridging llama-cpp-python with MAF orchestration.

    Implements ``_inner_get_response`` so that MAF agents, tool-calling loops,
    handoffs, and middleware all work transparently.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "llama-cpp"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # ---- required abstract method ----------------------------------------

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        if stream:
            raise NotImplementedError("Streaming is not supported by LlamaCppClient")

        async def _call() -> ChatResponse:
            validated = await self._validate_options(options)
            dicts = _messages_to_dicts(messages)

            tools = validated.get("tools")
            max_tokens = validated.get("max_tokens", 2048)
            temperature = validated.get("temperature", 0.7)

            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: chat_completion(
                    messages=dicts,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            return _parse_response(raw)

        return _call()
