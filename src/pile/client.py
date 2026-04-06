"""LLM client factory — creates the appropriate chat client based on config."""

from __future__ import annotations

import logging

import httpx

from pile.config import settings

logger = logging.getLogger("pile.client")

# Singleton httpx client for router model calls (reuses TCP connections).
_router_http: httpx.Client | None = None


def _router_client() -> httpx.Client:
    global _router_http
    if _router_http is None or _router_http.is_closed:
        _router_http = httpx.Client(timeout=10.0)
    return _router_http


def create_client():
    """Create an LLM chat client based on LLM_PROVIDER setting.

    - "ollama" (default): OpenAI-compat client pointed at Ollama /v1/ endpoint.
      This avoids the native OllamaChatClient bug with HandoffBuilder (#4402).
    - "openai": OpenAI-compat client for LM Studio or any compatible endpoint.
    - "ollama-native": Native Ollama client. Single-agent only, no workflows.

    Applies function invocation limits (max_iterations, max_function_calls) from config
    to prevent tool call loops with small models.
    """
    client = _create_raw_client()

    # Apply tool call limits to prevent infinite loops
    client.function_invocation_configuration["max_iterations"] = settings.agent_max_iterations
    client.function_invocation_configuration["max_function_calls"] = settings.agent_max_function_calls

    return client


def _create_raw_client():
    """Create the underlying LLM client without limits applied."""
    if settings.llm_provider in ("ollama", "openai"):
        from agent_framework.openai import OpenAIChatCompletionClient

        if settings.llm_provider == "ollama":
            return OpenAIChatCompletionClient(
                base_url=f"{settings.ollama_host}/v1/",
                model=settings.ollama_model_id,
                api_key="ollama",
            )
        return OpenAIChatCompletionClient(
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )

    if settings.llm_provider == "ollama-native":
        from agent_framework.ollama import OllamaChatClient

        return OllamaChatClient(
            host=settings.ollama_host,
            model_id=settings.ollama_model_id,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def call_router_model(prompt: str, max_tokens: int = 20) -> str | None:
    """Call the lightweight router model for classification/compression.

    Returns the response text, or None on failure. Uses the same provider
    as the main LLM but with ROUTER_MODEL model ID.
    """
    if not settings.router_model:
        return None

    client = _router_client()
    try:
        if settings.llm_provider in ("openai", "ollama"):
            resp = client.post(
                f"{settings.openai_base_url}/chat/completions"
                if settings.llm_provider == "openai"
                else f"{settings.ollama_host}/v1/chat/completions",
                json={
                    "model": settings.router_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
                headers={"Authorization": f"Bearer {settings.openai_api_key}"}
                if settings.llm_provider == "openai"
                else {},
            )
        else:
            # ollama-native
            resp = client.post(
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": settings.router_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0},
                },
            )

        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible format (both "openai" and "ollama" providers)
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        # Ollama native format
        return data["message"]["content"].strip()

    except Exception as e:
        logger.warning("Router model call failed: %s", e)
        return None
