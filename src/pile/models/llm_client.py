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

from pile.models.engine import chat_completion, chat_completion_stream


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


def _parse_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen-style XML tool calls from model output text.

    Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    Returns list of {"name": ..., "arguments": {...}} dicts.
    """
    import re
    calls = []
    for block in re.finditer(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", text, re.DOTALL):
        name = block.group(1)
        params_text = block.group(2)
        args = {}
        for param in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", params_text, re.DOTALL):
            args[param.group(1)] = param.group(2)
        calls.append({"name": name, "arguments": args})
    return calls


def _parse_response(raw: dict[str, Any]) -> ChatResponse:
    """Convert a llama-cpp OpenAI-compatible response dict into a ChatResponse."""
    choice = raw["choices"][0]
    message = choice["message"]
    finish = choice.get("finish_reason", "stop")

    contents: list[Content] = []
    text = message.get("content") or ""

    # Check for structured tool_calls first (chatml-function-calling format)
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

    # If no structured tool calls, parse XML tool calls from text (Qwen native format)
    if not contents and "<tool_call>" in text:
        xml_calls = _parse_xml_tool_calls(text)
        for i, call in enumerate(xml_calls):
            import json as _json
            contents.append(
                Content.from_function_call(
                    call_id=f"call_{i}_{call['name']}",
                    name=call["name"],
                    arguments=_json.dumps(call["arguments"]),
                )
            )
        finish = "tool_calls"
        # Remove tool_call XML from text, keep any remaining text
        import re
        remaining = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
        if remaining:
            contents.insert(0, Content.from_text(remaining))
    elif text:
        contents.append(Content.from_text(text))

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
            return self._stream_response(messages=messages, options=options)

        async def _call() -> ChatResponse:
            validated = await self._validate_options(options)
            dicts = _messages_to_dicts(messages)

            # Convert MAF FunctionTool objects to OpenAI-compatible dicts
            raw_tools = validated.get("tools")
            tools = None
            if raw_tools:
                tools = [
                    t.to_json_schema_spec() if hasattr(t, "to_json_schema_spec") else t
                    for t in raw_tools
                ]

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

    def _stream_response(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        """Stream chat completion tokens from llama-cpp-python.

        Bridges the sync generator from llama-cpp to async by running it
        in a thread and feeding chunks through an asyncio.Queue.
        """
        import queue as thread_queue

        async def _generate():
            validated = await self._validate_options(options)
            dicts = _messages_to_dicts(messages)

            raw_tools = validated.get("tools")
            tools = None
            if raw_tools:
                tools = [
                    t.to_json_schema_spec() if hasattr(t, "to_json_schema_spec") else t
                    for t in raw_tools
                ]

            max_tokens = validated.get("max_tokens", 2048)
            temperature = validated.get("temperature", 0.7)

            # Bridge sync generator → async via thread + queue
            q: thread_queue.Queue = thread_queue.Queue()
            _sentinel = object()

            def _produce():
                try:
                    for chunk in chat_completion_stream(
                        messages=dicts,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ):
                        q.put(chunk)
                except Exception as exc:
                    q.put(exc)
                finally:
                    q.put(_sentinel)

            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, _produce)

            while True:
                # Poll queue without blocking the event loop
                while q.empty():
                    await asyncio.sleep(0.01)
                item = q.get_nowait()
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item

                choice = item.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    yield ChatResponseUpdate(
                        contents=[Content.from_text(text)],
                        role="assistant",
                        finish_reason=choice.get("finish_reason"),
                    )

        def _finalizer(updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
            all_text = "".join(
                c.text for u in updates for c in (u.contents or [])
                if c.type == "text" and c.text
            )
            # Parse tool calls from accumulated text
            tool_contents = []
            remaining_text = all_text
            if "<tool_call>" in all_text:
                xml_calls = _parse_xml_tool_calls(all_text)
                for i, call in enumerate(xml_calls):
                    import json as _json
                    tool_contents.append(
                        Content.from_function_call(
                            call_id=f"call_{i}_{call['name']}",
                            name=call["name"],
                            arguments=_json.dumps(call["arguments"]),
                        )
                    )
                import re
                remaining_text = re.sub(
                    r"<tool_call>.*?</tool_call>", "", all_text, flags=re.DOTALL
                ).strip()

            contents = []
            if remaining_text:
                contents.append(Content.from_text(remaining_text))
            contents.extend(tool_contents)

            finish = "tool_calls" if tool_contents else "stop"
            return ChatResponse(
                messages=Message("assistant", contents),
                finish_reason=finish,
            )

        return ResponseStream(stream=_generate(), finalizer=_finalizer)
