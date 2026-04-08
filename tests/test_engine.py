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


# ---------------------------------------------------------------------------
# _inject_no_think
# ---------------------------------------------------------------------------


def test_inject_no_think_prepends_to_system():
    from pile.models.engine import _inject_no_think

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    result = _inject_no_think(messages)
    assert result[0]["content"] == "/no_think\nYou are helpful."
    assert result[1] == messages[1]


def test_inject_no_think_skips_already_tagged():
    from pile.models.engine import _inject_no_think

    messages = [{"role": "system", "content": "/no_think\nAlready tagged."}]
    result = _inject_no_think(messages)
    assert result[0]["content"] == "/no_think\nAlready tagged."


def test_inject_no_think_no_system_message():
    from pile.models.engine import _inject_no_think

    messages = [{"role": "user", "content": "hi"}]
    result = _inject_no_think(messages)
    assert result == messages


# ---------------------------------------------------------------------------
# chat_completion with tools
# ---------------------------------------------------------------------------


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_with_tools(mock_get):
    from pile.models.engine import chat_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_llama_response("ok")
    mock_get.return_value = mock_model

    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    result = chat_completion(
        [{"role": "user", "content": "find"}],
        tools=tools,
    )

    call_kwargs = mock_model.create_chat_completion.call_args[1]
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"
    assert result["choices"][0]["message"]["content"] == "ok"


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_without_tools(mock_get):
    from pile.models.engine import chat_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_llama_response("ok")
    mock_get.return_value = mock_model

    chat_completion([{"role": "user", "content": "hi"}])

    call_kwargs = mock_model.create_chat_completion.call_args[1]
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


# ---------------------------------------------------------------------------
# chat_completion_stream
# ---------------------------------------------------------------------------


def _make_stream_chunk(content=None, finish_reason=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    return {
        "choices": [{"delta": delta, "finish_reason": finish_reason, "index": 0}],
    }


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_stream_basic(mock_get):
    from pile.models.engine import chat_completion_stream

    chunks = [
        _make_stream_chunk("Hello"),
        _make_stream_chunk(" world"),
        _make_stream_chunk(finish_reason="stop"),
    ]
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = iter(chunks)
    mock_get.return_value = mock_model

    result = list(chat_completion_stream([{"role": "user", "content": "hi"}]))

    assert len(result) == 3
    assert result[0]["choices"][0]["delta"]["content"] == "Hello"
    assert result[1]["choices"][0]["delta"]["content"] == " world"
    call_kwargs = mock_model.create_chat_completion.call_args[1]
    assert call_kwargs["stream"] is True


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_stream_with_tools(mock_get):
    from pile.models.engine import chat_completion_stream

    chunks = [_make_stream_chunk("ok")]
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = iter(chunks)
    mock_get.return_value = mock_model

    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    list(chat_completion_stream(
        [{"role": "user", "content": "hi"}],
        tools=tools,
    ))

    call_kwargs = mock_model.create_chat_completion.call_args[1]
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["stream"] is True


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_stream_custom_params(mock_get):
    from pile.models.engine import chat_completion_stream

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = iter([])
    mock_get.return_value = mock_model

    list(chat_completion_stream(
        [{"role": "user", "content": "hi"}],
        max_tokens=512,
        temperature=0.1,
    ))

    call_kwargs = mock_model.create_chat_completion.call_args[1]
    assert call_kwargs["max_tokens"] == 512
    assert call_kwargs["temperature"] == 0.1


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_stream_logs_on_finish(mock_get):
    """Stream should log inference call after all chunks are consumed."""
    from pile.models.engine import chat_completion_stream

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = iter([_make_stream_chunk("hi")])
    mock_get.return_value = mock_model

    with patch("pile.models.engine.log_inference_call") as mock_log:
        list(chat_completion_stream([{"role": "user", "content": "hi"}]))
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["status"] == "ok"
        assert mock_log.call_args[1]["role"] == "agent"


# ---------------------------------------------------------------------------
# router_completion error handling
# ---------------------------------------------------------------------------


@patch("pile.models.engine._get_router_model")
def test_router_completion_error_returns_none(mock_get):
    from pile.models.engine import router_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.side_effect = RuntimeError("model crashed")
    mock_get.return_value = mock_model

    result = router_completion("classify this")
    assert result is None


@patch("pile.models.engine._get_router_model")
def test_router_completion_error_logs(mock_get):
    from pile.models.engine import router_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.side_effect = RuntimeError("bad")
    mock_get.return_value = mock_model

    with patch("pile.models.engine.log_inference_call") as mock_log:
        router_completion("classify this")
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["status"] == "error"
        assert "bad" in mock_log.call_args[1]["error"]


# ---------------------------------------------------------------------------
# embed edge cases
# ---------------------------------------------------------------------------


@patch("pile.models.engine._get_embed_model")
def test_embed_single_text(mock_get):
    from pile.models.engine import embed

    mock_model = MagicMock()
    mock_model.embed.return_value = [[0.1, 0.2]]
    mock_get.return_value = mock_model

    result = embed(["hello"])
    assert len(result) == 1
    mock_model.embed.assert_called_once_with(["hello"])


@patch("pile.models.engine._get_embed_model")
def test_embed_empty_list(mock_get):
    from pile.models.engine import embed

    mock_model = MagicMock()
    mock_model.embed.return_value = []
    mock_get.return_value = mock_model

    result = embed([])
    assert result == []


@patch("pile.models.engine._get_embed_model")
def test_embed_logs_inference(mock_get):
    from pile.models.engine import embed

    mock_model = MagicMock()
    mock_model.embed.return_value = [[0.1]]
    mock_get.return_value = mock_model

    with patch("pile.models.engine.log_inference_call") as mock_log:
        embed(["test"])
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["role"] == "embedding"
        assert mock_log.call_args[1]["status"] == "ok"


# ---------------------------------------------------------------------------
# chat_completion with tool_calls in response
# ---------------------------------------------------------------------------


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_logs_tool_call_count(mock_get):
    from pile.models.engine import chat_completion

    tool_calls = [
        {"id": "c1", "type": "function", "function": {"name": "fn1", "arguments": "{}"}},
        {"id": "c2", "type": "function", "function": {"name": "fn2", "arguments": "{}"}},
    ]
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_llama_response(
        content="", tool_calls=tool_calls,
    )
    mock_get.return_value = mock_model

    with patch("pile.models.engine.log_inference_call") as mock_log:
        chat_completion([{"role": "user", "content": "do stuff"}])
        assert mock_log.call_args[1]["tool_calls"] == 2


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_tool_names_from_dict(mock_get):
    """Tool names should be extracted from dict-style tool specs."""
    from pile.models.engine import chat_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_llama_response("ok")
    mock_get.return_value = mock_model

    tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]

    with patch("pile.models.engine.log_inference_detail") as mock_detail:
        chat_completion([{"role": "user", "content": "hi"}], tools=tools)
        request_call = mock_detail.call_args_list[0]
        import json
        logged = json.loads(request_call[1]["content"])
        assert logged["tools"] == ["my_tool"]
