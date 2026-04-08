"""Tests for pile.client module."""

from __future__ import annotations

from unittest.mock import patch, MagicMock


def test_create_client_returns_configured_instance():
    """create_client returns LlamaCppClient with limits from settings."""
    mock_client_instance = MagicMock()
    mock_client_instance.function_invocation_configuration = {}

    with patch("pile.client.LlamaCppClient", return_value=mock_client_instance) as mock_cls, \
         patch("pile.client.settings") as mock_settings:
        mock_settings.agent_max_iterations = 10
        mock_settings.agent_max_function_calls = 25

        from pile.client import create_client
        client = create_client()

    mock_cls.assert_called_once()
    assert client.function_invocation_configuration["max_iterations"] == 10
    assert client.function_invocation_configuration["max_function_calls"] == 25


def test_call_router_model_delegates_to_router_completion():
    """call_router_model passes prompt and max_tokens to router_completion."""
    with patch("pile.client.router_completion", return_value="result") as mock_router:
        from pile.client import call_router_model
        result = call_router_model("test prompt", max_tokens=50)

    mock_router.assert_called_once_with("test prompt", max_tokens=50)
    assert result == "result"


def test_call_router_model_default_max_tokens():
    """call_router_model uses default max_tokens=20."""
    with patch("pile.client.router_completion", return_value="ok") as mock_router:
        from pile.client import call_router_model
        call_router_model("prompt")

    mock_router.assert_called_once_with("prompt", max_tokens=20)


def test_call_router_model_returns_none():
    """call_router_model returns None when router_completion returns None."""
    with patch("pile.client.router_completion", return_value=None):
        from pile.client import call_router_model
        result = call_router_model("prompt")

    assert result is None
