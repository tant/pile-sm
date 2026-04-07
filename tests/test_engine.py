"""Tests for inference engine functions."""

from unittest.mock import patch, MagicMock

import pytest


def _mock_llama_response(content="hello", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    """Build a fake llama-cpp response dict."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "test-id",
        "choices": [{"message": message, "finish_reason": "stop", "index": 0}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_basic(mock_get):
    """chat_completion should return response dict with content."""
    from pile.models.engine import chat_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_llama_response("test answer")
    mock_get.return_value = mock_model

    result = chat_completion([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "test answer"
    mock_model.create_chat_completion.assert_called_once()


@patch("pile.models.engine._get_router_model")
def test_router_completion(mock_get):
    """router_completion should return stripped text."""
    from pile.models.engine import router_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_llama_response("jira_query")
    mock_get.return_value = mock_model

    result = router_completion("Pick an agent for: find issues")

    assert result == "jira_query"


@patch("pile.models.engine._get_embed_model")
def test_embed(mock_get):
    """embed should return list of float vectors."""
    from pile.models.engine import embed

    mock_model = MagicMock()
    mock_model.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_get.return_value = mock_model

    result = embed(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
