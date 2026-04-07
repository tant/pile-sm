"""Inference engine — wraps llama-cpp-python for chat, routing, and embedding."""

from __future__ import annotations

import json
import time
from typing import Any

from pile.models.logging import log_inference_call, log_inference_detail
from pile.models.manager import get_agent_model, get_router_model, get_embed_model


def _get_agent_model():
    return get_agent_model()


def _get_router_model():
    return get_router_model()


def _get_embed_model():
    return get_embed_model()


def _inject_no_think(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend /no_think to system message to disable Qwen thinking mode."""
    result = []
    for msg in messages:
        if msg.get("role") == "system" and not msg.get("content", "").startswith("/no_think"):
            result.append({**msg, "content": f"/no_think\n{msg.get('content', '')}"})
        else:
            result.append(msg)
    return result


def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> dict:
    """Run chat completion on the agent model. Returns OpenAI-compatible dict."""
    model = _get_agent_model()
    messages = _inject_no_think(messages)
    start = time.monotonic()

    tool_names = None
    if tools:
        tool_names = [
            t["function"]["name"] if isinstance(t, dict) else getattr(t, "name", str(t))
            for t in tools
        ]

    log_inference_detail(
        role="agent",
        direction="request",
        content=json.dumps(
            {"messages": messages, "tools": tool_names},
            ensure_ascii=False, indent=2,
        ),
    )

    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    result = model.create_chat_completion(**kwargs)

    latency_ms = int((time.monotonic() - start) * 1000)
    usage = result.get("usage", {})
    choice = result["choices"][0] if result.get("choices") else {}
    message = choice.get("message", {})
    tool_call_count = len(message.get("tool_calls", []))

    log_inference_call(
        role="agent",
        latency_ms=latency_ms,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        status="ok",
        tool_calls=tool_call_count if tool_call_count else None,
    )

    log_inference_detail(
        role="agent",
        direction="response",
        content=json.dumps(message, ensure_ascii=False, indent=2),
    )

    return result


def router_completion(prompt: str, max_tokens: int = 20) -> str | None:
    """Run classification on the router model. Returns response text or None."""
    model = _get_router_model()
    start = time.monotonic()

    log_inference_detail(role="router", direction="request", content=prompt)

    try:
        result = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        usage = result.get("usage", {})
        text = result["choices"][0]["message"]["content"].strip()

        log_inference_call(
            role="router",
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            status="ok",
        )
        log_inference_detail(role="router", direction="response", content=text)

        return text
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        log_inference_call(role="router", latency_ms=latency_ms, status="error", error=str(e))
        return None


def embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = _get_embed_model()
    start = time.monotonic()

    result = model.embed(texts)

    latency_ms = int((time.monotonic() - start) * 1000)
    log_inference_call(role="embedding", latency_ms=latency_ms, status="ok")

    return result
