"""LLM client factory — uses local llama-cpp-python inference."""

from __future__ import annotations

from pile.models.llm_client import LlamaCppClient
from pile.models.engine import router_completion

from pile.config import settings


def create_client() -> LlamaCppClient:
    """Create a LlamaCppClient with function invocation limits from config."""
    client = LlamaCppClient()
    client.function_invocation_configuration["max_iterations"] = settings.agent_max_iterations
    client.function_invocation_configuration["max_function_calls"] = settings.agent_max_function_calls
    return client


def call_router_model(prompt: str, max_tokens: int = 20) -> str | None:
    """Call the lightweight router model for classification/compression."""
    return router_completion(prompt, max_tokens=max_tokens)
