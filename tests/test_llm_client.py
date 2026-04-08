"""Tests for LlamaCppClient — MAF compatibility."""

import asyncio
import json
from unittest.mock import patch, MagicMock

import pytest

from pile.models.llm_client import (
    _messages_to_dicts,
    _parse_xml_tool_calls,
    _parse_response,
    _parse_stream_output,
    LlamaCppClient,
)
from agent_framework import Content, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _get_contents(resp):
    """Extract Content list from a ChatResponse."""
    msgs = resp.messages
    if isinstance(msgs, list):
        return msgs[0].contents
    return msgs.contents


# ---------------------------------------------------------------------------
# _messages_to_dicts
# ---------------------------------------------------------------------------

class TestMessagesToDicts:
    def test_text_message(self):
        msgs = [Message(role="user", contents=["hello world"])]
        result = _messages_to_dicts(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello world"

    def test_multiple_text_parts_joined(self):
        msgs = [Message(role="user", contents=["part1", "part2"])]
        result = _messages_to_dicts(msgs)
        assert result[0]["content"] == "part1 part2"

    def test_function_call_content(self):
        fc = Content.from_function_call(
            call_id="call_1", name="search", arguments='{"q": "test"}'
        )
        msgs = [Message(role="assistant", contents=[fc])]
        result = _messages_to_dicts(msgs)
        assert len(result) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_1"
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"q": "test"}'

    def test_function_call_with_non_string_arguments(self):
        """When arguments is not a string, should default to empty string."""
        fc = Content.from_function_call(
            call_id="call_1", name="search", arguments=None
        )
        msgs = [Message(role="assistant", contents=[fc])]
        result = _messages_to_dicts(msgs)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == ""

    def test_function_result_becomes_tool_message(self):
        fr = Content.from_function_result(call_id="call_1", result="result data")
        msgs = [Message(role="tool", contents=[fr])]
        result = _messages_to_dicts(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "result data"

    def test_function_result_with_none_result(self):
        fr = Content.from_function_result(call_id="call_1", result=None)
        msgs = [Message(role="tool", contents=[fr])]
        result = _messages_to_dicts(msgs)
        assert result[0]["content"] == ""

    def test_empty_message_has_none_content(self):
        msgs = [Message(role="assistant", contents=[])]
        result = _messages_to_dicts(msgs)
        assert result[0]["content"] is None

    def test_mixed_text_and_tool_calls(self):
        text = Content.from_text("thinking...")
        fc = Content.from_function_call(
            call_id="call_1", name="search", arguments="{}"
        )
        msgs = [Message(role="assistant", contents=[text, fc])]
        result = _messages_to_dicts(msgs)
        assert result[0]["content"] == "thinking..."
        assert len(result[0]["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# _parse_xml_tool_calls
# ---------------------------------------------------------------------------

class TestParseXmlToolCalls:
    def test_single_tool_call(self):
        text = (
            '<tool_call><function=search>'
            '<parameter=query>test</parameter>'
            '</function></tool_call>'
        )
        calls = _parse_xml_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments"] == {"query": "test"}

    def test_multiple_parameters(self):
        text = (
            '<tool_call><function=create_issue>'
            '<parameter=title>Bug report</parameter>'
            '<parameter=priority>high</parameter>'
            '</function></tool_call>'
        )
        calls = _parse_xml_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["arguments"]["title"] == "Bug report"
        assert calls[0]["arguments"]["priority"] == "high"

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call><function=fn1><parameter=a>1</parameter></function></tool_call>'
            '<tool_call><function=fn2><parameter=b>2</parameter></function></tool_call>'
        )
        calls = _parse_xml_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "fn1"
        assert calls[1]["name"] == "fn2"

    def test_no_tool_calls(self):
        calls = _parse_xml_tool_calls("just some regular text")
        assert calls == []

    def test_no_parameters(self):
        text = '<tool_call><function=noop></function></tool_call>'
        calls = _parse_xml_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "noop"
        assert calls[0]["arguments"] == {}

    def test_whitespace_in_parameters(self):
        text = (
            '<tool_call>\n<function=search>\n'
            '<parameter=query>  spaced value  </parameter>\n'
            '</function>\n</tool_call>'
        )
        calls = _parse_xml_tool_calls(text)
        assert calls[0]["arguments"]["query"] == "spaced value"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_text_response(self):
        raw = _mock_llama_response("hello world")
        resp = _parse_response(raw)
        assert resp.finish_reason == "stop"
        contents = _get_contents(resp)
        text_contents = [c for c in contents if c.type == "text"]
        assert len(text_contents) == 1
        assert text_contents[0].text == "hello world"

    def test_empty_text_response(self):
        raw = _mock_llama_response(content="")
        resp = _parse_response(raw)
        contents = _get_contents(resp)
        text_contents = [c for c in contents if c.type == "text"]
        assert len(text_contents) == 0

    def test_structured_tool_calls(self):
        raw = _mock_llama_response(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "test"}'},
            }],
        )
        resp = _parse_response(raw)
        assert resp.finish_reason == "tool_calls"
        contents = _get_contents(resp)
        fc_contents = [c for c in contents if c.type == "function_call"]
        assert len(fc_contents) == 1
        assert fc_contents[0].name == "search"
        assert fc_contents[0].call_id == "call_1"

    def test_xml_tool_calls_in_text(self):
        text = (
            'Let me search for that. '
            '<tool_call><function=search>'
            '<parameter=query>test</parameter>'
            '</function></tool_call>'
        )
        raw = _mock_llama_response(content=text)
        resp = _parse_response(raw)
        assert resp.finish_reason == "tool_calls"
        contents = _get_contents(resp)
        fc_contents = [c for c in contents if c.type == "function_call"]
        assert len(fc_contents) == 1
        assert fc_contents[0].name == "search"
        assert fc_contents[0].arguments == json.dumps({"query": "test"})
        text_contents = [c for c in contents if c.type == "text"]
        assert len(text_contents) == 1
        assert "Let me search" in text_contents[0].text

    def test_xml_tool_call_only_no_extra_text(self):
        text = (
            '<tool_call><function=do_thing>'
            '<parameter=x>1</parameter>'
            '</function></tool_call>'
        )
        raw = _mock_llama_response(content=text)
        resp = _parse_response(raw)
        contents = _get_contents(resp)
        text_contents = [c for c in contents if c.type == "text"]
        assert len(text_contents) == 0
        fc_contents = [c for c in contents if c.type == "function_call"]
        assert len(fc_contents) == 1

    def test_usage_details_parsed(self):
        raw = _mock_llama_response("ok")
        resp = _parse_response(raw)
        assert resp.usage_details["input_token_count"] == 10
        assert resp.usage_details["output_token_count"] == 5
        assert resp.usage_details["total_token_count"] == 15

    def test_response_metadata(self):
        raw = _mock_llama_response("ok")
        resp = _parse_response(raw)
        assert resp.response_id == "resp-1"
        assert resp.model == "test-model"

    def test_tool_calls_with_missing_arguments(self):
        raw = _mock_llama_response(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "noop"},
            }],
        )
        resp = _parse_response(raw)
        contents = _get_contents(resp)
        fc_contents = [c for c in contents if c.type == "function_call"]
        assert fc_contents[0].arguments == ""

    def test_none_tool_calls_field(self):
        """When tool_calls is None, should treat as no tool calls."""
        raw = _mock_llama_response(content="plain text")
        raw["choices"][0]["message"]["tool_calls"] = None
        resp = _parse_response(raw)
        contents = _get_contents(resp)
        text_contents = [c for c in contents if c.type == "text"]
        assert len(text_contents) == 1

    def test_missing_usage(self):
        raw = _mock_llama_response("ok")
        del raw["usage"]
        resp = _parse_response(raw)
        assert resp.usage_details["input_token_count"] is None

    def test_missing_finish_reason(self):
        raw = _mock_llama_response("ok")
        del raw["choices"][0]["finish_reason"]
        resp = _parse_response(raw)
        assert resp.finish_reason == "stop"

    def test_multiple_structured_tool_calls(self):
        raw = _mock_llama_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "fn1", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "fn2", "arguments": '{"a": 1}'},
                },
            ],
        )
        resp = _parse_response(raw)
        contents = _get_contents(resp)
        fc_contents = [c for c in contents if c.type == "function_call"]
        assert len(fc_contents) == 2
        assert fc_contents[0].name == "fn1"
        assert fc_contents[1].name == "fn2"

    def test_xml_tool_call_id_format(self):
        """XML tool call IDs follow call_{index}_{name} format."""
        text = (
            '<tool_call><function=my_func>'
            '<parameter=x>1</parameter>'
            '</function></tool_call>'
        )
        raw = _mock_llama_response(content=text)
        resp = _parse_response(raw)
        contents = _get_contents(resp)
        fc = [c for c in contents if c.type == "function_call"][0]
        assert fc.call_id == "call_0_my_func"


# ---------------------------------------------------------------------------
# _parse_stream_output
# ---------------------------------------------------------------------------

class TestParseStreamOutput:
    def test_plain_text(self):
        contents = _parse_stream_output("just text")
        assert len(contents) == 1
        assert contents[0].type == "text"
        assert contents[0].text == "just text"

    def test_tool_call_xml(self):
        text = (
            '<tool_call><function=search>'
            '<parameter=q>hello</parameter>'
            '</function></tool_call>'
        )
        contents = _parse_stream_output(text)
        fc = [c for c in contents if c.type == "function_call"]
        assert len(fc) == 1
        assert fc[0].name == "search"

    def test_text_with_tool_call(self):
        text = (
            'Some preamble '
            '<tool_call><function=act>'
            '<parameter=x>1</parameter>'
            '</function></tool_call>'
        )
        contents = _parse_stream_output(text)
        text_parts = [c for c in contents if c.type == "text"]
        fc_parts = [c for c in contents if c.type == "function_call"]
        assert len(text_parts) == 1
        assert "Some preamble" in text_parts[0].text
        assert len(fc_parts) == 1

    def test_empty_text(self):
        contents = _parse_stream_output("")
        assert len(contents) == 0

    def test_tool_call_arguments_are_json(self):
        text = (
            '<tool_call><function=fn>'
            '<parameter=key>val</parameter>'
            '</function></tool_call>'
        )
        contents = _parse_stream_output(text)
        fc = [c for c in contents if c.type == "function_call"][0]
        parsed = json.loads(fc.arguments)
        assert parsed == {"key": "val"}

    def test_multiple_tool_calls_in_stream(self):
        text = (
            '<tool_call><function=a><parameter=x>1</parameter></function></tool_call>'
            '<tool_call><function=b><parameter=y>2</parameter></function></tool_call>'
        )
        contents = _parse_stream_output(text)
        fc = [c for c in contents if c.type == "function_call"]
        assert len(fc) == 2
        assert fc[0].name == "a"
        assert fc[1].name == "b"


# ---------------------------------------------------------------------------
# LlamaCppClient integration tests
# ---------------------------------------------------------------------------

class TestLlamaCppClient:
    @patch("pile.models.engine.chat_completion")
    def test_client_returns_chat_response(self, mock_chat):
        mock_chat.return_value = _mock_llama_response("test reply")
        client = LlamaCppClient()
        messages = [Message(role="user", contents=["hello"])]
        response = asyncio.run(client.get_response(messages))
        assert response.messages is not None
        assert len(response.messages) > 0

    @patch("pile.models.engine.chat_completion")
    def test_client_handles_tool_calls(self, mock_chat):
        mock_chat.return_value = _mock_llama_response(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "jira_search",
                    "arguments": '{"jql": "project=TETRA"}',
                },
            }],
        )
        client = LlamaCppClient()
        messages = [Message(role="user", contents=["find issues"])]
        response = asyncio.run(client.get_response(messages))
        assert response.messages is not None

    @patch("pile.models.engine.chat_completion")
    def test_client_with_xml_tool_call_response(self, mock_chat):
        """XML tool calls in text should be parsed into function_call contents."""
        text = (
            '<tool_call><function=get_weather>'
            '<parameter=city>Hanoi</parameter>'
            '</function></tool_call>'
        )
        mock_chat.return_value = _mock_llama_response(content=text)
        client = LlamaCppClient()
        messages = [Message(role="user", contents=["weather?"])]
        response = asyncio.run(client.get_response(messages))
        assert response.messages is not None
        assert len(response.messages) > 0


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


def _make_stream_chunk(content=None, finish_reason=None):
    """Build a fake streaming chunk dict."""
    delta = {}
    if content is not None:
        delta["content"] = content
    return {
        "choices": [{"delta": delta, "finish_reason": finish_reason, "index": 0}],
    }


async def _consume_stream(client, messages):
    """Consume a ResponseStream and return the final ChatResponse."""
    stream = client.get_response(messages, stream=True)
    updates = []
    async for update in stream:
        updates.append(update)
    return await stream.get_final_response()


class TestStreamResponse:
    """Tests for the streaming code path in LlamaCppClient._stream_response."""

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_plain_text(self, mock_stream):
        """Plain text chunks should be yielded as ChatResponseUpdates."""
        mock_stream.return_value = iter([
            _make_stream_chunk("Hello"),
            _make_stream_chunk(" world"),
        ])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["hi"])]

        response = asyncio.run(_consume_stream(client, messages))
        contents = _get_contents(response)
        text_parts = [c for c in contents if c.type == "text"]
        assert len(text_parts) >= 1
        combined = "".join(c.text for c in text_parts)
        assert "Hello" in combined
        assert "world" in combined

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_with_tool_call_xml_buffered(self, mock_stream):
        """Tool call XML in stream is buffered and not yielded as updates.

        The streaming generator strips tool_call XML blocks to avoid leaking
        raw markup to the UI. The finalizer only sees text from yielded updates.
        """
        xml = (
            '<tool_call><function=search>'
            '<parameter=q>test</parameter>'
            '</function></tool_call>'
        )
        mock_stream.return_value = iter([_make_stream_chunk(xml)])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["find"])]

        stream = client.get_response(messages, stream=True)
        updates = []

        async def _run():
            async for update in stream:
                updates.append(update)

        asyncio.run(_run())
        # Tool call XML is buffered and discarded from updates
        text_updates = [
            c.text for u in updates for c in (u.contents or [])
            if c.type == "text" and c.text
        ]
        combined = "".join(text_updates)
        assert "<tool_call>" not in combined

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_text_before_tool_call(self, mock_stream):
        """Text before a tool call tag should be yielded as update."""
        mock_stream.return_value = iter([
            _make_stream_chunk("Searching now "),
            _make_stream_chunk(
                '<tool_call><function=search>'
                '<parameter=q>test</parameter>'
                '</function></tool_call>'
            ),
        ])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["find"])]

        stream = client.get_response(messages, stream=True)
        updates = []

        async def _run():
            async for update in stream:
                updates.append(update)

        asyncio.run(_run())
        text_updates = [
            c.text for u in updates for c in (u.contents or [])
            if c.type == "text" and c.text
        ]
        combined = "".join(text_updates)
        assert "Searching now" in combined
        assert "<tool_call>" not in combined

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_empty_chunks(self, mock_stream):
        """Chunks with no content should not produce updates."""
        mock_stream.return_value = iter([
            _make_stream_chunk(),
            _make_stream_chunk("text"),
            _make_stream_chunk(),
        ])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["hi"])]

        response = asyncio.run(_consume_stream(client, messages))
        contents = _get_contents(response)
        text_parts = [c for c in contents if c.type == "text"]
        assert len(text_parts) >= 1
        assert text_parts[0].text == "text"

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_error_propagation(self, mock_stream):
        """Exceptions from the stream producer should be raised."""
        def _failing_stream(**kwargs):
            yield _make_stream_chunk("start")
            raise RuntimeError("model error")

        mock_stream.side_effect = lambda **kwargs: _failing_stream(**kwargs)

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["hi"])]

        with pytest.raises(RuntimeError, match="model error"):
            asyncio.run(_consume_stream(client, messages))

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_finalizer_plain_text(self, mock_stream):
        """Finalizer should produce text content for plain text stream."""
        mock_stream.return_value = iter([_make_stream_chunk("just text")])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["hi"])]

        response = asyncio.run(_consume_stream(client, messages))
        contents = _get_contents(response)
        text_parts = [c for c in contents if c.type == "text"]
        assert len(text_parts) == 1
        assert text_parts[0].text == "just text"

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_finalizer_with_tool_call_xml(self, mock_stream):
        """When tool call XML is in the stream, it is stripped from yielded updates.

        The streaming generator buffers tool_call blocks. The finalizer only
        reconstructs text from yielded updates, so tool calls are not present
        in the final response from streaming.
        """
        xml = (
            '<tool_call><function=act>'
            '<parameter=x>1</parameter>'
            '</function></tool_call>'
        )
        mock_stream.return_value = iter([_make_stream_chunk(xml)])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["do it"])]

        stream = client.get_response(messages, stream=True)
        updates = []

        async def _run():
            async for update in stream:
                updates.append(update)

        asyncio.run(_run())
        # No text updates should have been yielded for pure tool call XML
        text_updates = [
            c.text for u in updates for c in (u.contents or [])
            if c.type == "text" and c.text
        ]
        assert len(text_updates) == 0

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_mixed_text_and_tool_call_updates(self, mock_stream):
        """Text preceding tool call XML should appear in yielded updates."""
        mock_stream.return_value = iter([
            _make_stream_chunk("Here is the result: "),
            _make_stream_chunk(
                '<tool_call><function=a><parameter=x>1</parameter></function></tool_call>'
            ),
        ])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["multi"])]

        stream = client.get_response(messages, stream=True)
        updates = []

        async def _run():
            async for update in stream:
                updates.append(update)

        asyncio.run(_run())
        text_updates = [
            c.text for u in updates for c in (u.contents or [])
            if c.type == "text" and c.text
        ]
        combined = "".join(text_updates)
        assert "Here is the result" in combined

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_large_buffer_flush(self, mock_stream):
        """Buffer with '<' but no real tag should flush when > 20 chars."""
        long_text = "A" * 25 + " < not a tag"
        mock_stream.return_value = iter([_make_stream_chunk(long_text)])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["hi"])]

        response = asyncio.run(_consume_stream(client, messages))
        contents = _get_contents(response)
        text_parts = [c for c in contents if c.type == "text"]
        combined = "".join(c.text for c in text_parts)
        assert "< not a tag" in combined

    @patch("pile.models.llm_client.chat_completion_stream")
    def test_stream_tool_call_split_across_chunks(self, mock_stream):
        """Tool call XML split across multiple chunks should be buffered and stripped."""
        mock_stream.return_value = iter([
            _make_stream_chunk("<tool_call><function=search>"),
            _make_stream_chunk("<parameter=q>test</parameter>"),
            _make_stream_chunk("</function></tool_call>"),
        ])

        client = LlamaCppClient()
        messages = [Message(role="user", contents=["find"])]

        stream = client.get_response(messages, stream=True)
        updates = []

        async def _run():
            async for update in stream:
                updates.append(update)

        asyncio.run(_run())
        # All chunks were tool call XML, so no text updates should be yielded
        text_updates = [
            c.text for u in updates for c in (u.contents or [])
            if c.type == "text" and c.text
        ]
        combined = "".join(text_updates)
        assert "<tool_call>" not in combined


# ---------------------------------------------------------------------------
# _parse_stream_output additional cases
# ---------------------------------------------------------------------------


class TestParseStreamOutputAdditional:
    def test_only_whitespace_after_stripping_tool_calls(self):
        text = (
            '   <tool_call><function=fn>'
            '<parameter=a>1</parameter>'
            '</function></tool_call>   '
        )
        contents = _parse_stream_output(text)
        text_parts = [c for c in contents if c.type == "text"]
        assert len(text_parts) == 0
        fc_parts = [c for c in contents if c.type == "function_call"]
        assert len(fc_parts) == 1

    def test_tool_call_id_format(self):
        text = (
            '<tool_call><function=my_func>'
            '<parameter=x>val</parameter>'
            '</function></tool_call>'
        )
        contents = _parse_stream_output(text)
        fc = [c for c in contents if c.type == "function_call"][0]
        assert fc.call_id == "call_0_my_func"

    def test_no_tool_call_returns_text_only(self):
        contents = _parse_stream_output("regular response text")
        assert len(contents) == 1
        assert contents[0].type == "text"
        assert contents[0].text == "regular response text"
