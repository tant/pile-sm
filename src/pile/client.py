"""LLM client factory — creates the appropriate chat client based on config."""

from __future__ import annotations

from pile.config import settings


def create_client():
    """Create an LLM chat client based on LLM_PROVIDER setting.

    - "ollama" (default): OpenAI-compat client pointed at Ollama /v1/ endpoint.
      This avoids the native OllamaChatClient bug with HandoffBuilder (#4402).
    - "openai": OpenAI-compat client for LM Studio or any compatible endpoint.
    - "ollama-native": Native Ollama client. Single-agent only, no workflows.
    """
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
