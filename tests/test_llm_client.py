"""Tests for LlamaCppClient — MAF compatibility."""

import asyncio
from unittest.mock import patch, MagicMock

import pytest


def _mock_llama_response(content="hello", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {
        "id": "resp-1",
        "choices": [{"message": message, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "model": "test-model",
        "created": 1700000000,
    }


@patch("pile.models.engine.chat_completion")
def test_client_returns_chat_response(mock_chat):
    """LlamaCppClient should return a valid ChatResponse."""
    from pile.models.llm_client import LlamaCppClient
    from agent_framework import Message

    mock_chat.return_value = _mock_llama_response("test reply")

    client = LlamaCppClient()
    messages = [Message(role="user", contents=["hello"])]
    response = asyncio.get_event_loop().run_until_complete(
        client.get_response(messages)
    )

    assert response.messages is not None
    assert len(response.messages) > 0


@patch("pile.models.engine.chat_completion")
def test_client_handles_tool_calls(mock_chat):
    """LlamaCppClient should parse tool calls into Content objects."""
    from pile.models.llm_client import LlamaCppClient
    from agent_framework import Message

    mock_chat.return_value = _mock_llama_response(
        content=None,
        tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "jira_search", "arguments": '{"jql": "project=TETRA"}'},
        }],
    )

    client = LlamaCppClient()
    messages = [Message(role="user", contents=["find issues"])]
    response = asyncio.get_event_loop().run_until_complete(
        client.get_response(messages)
    )

    assert response.messages is not None
