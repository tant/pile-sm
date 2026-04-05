"""Tests for pile.tools.utils — ADF conversion helpers."""

from pile.tools.utils import extract_text, make_adf


class TestExtractText:
    def test_none_returns_empty(self):
        assert extract_text(None) == ""

    def test_empty_dict_returns_empty(self):
        assert extract_text({}) == ""

    def test_simple_text_node(self):
        node = {"type": "text", "text": "hello"}
        assert extract_text(node) == "hello"

    def test_paragraph_with_text(self):
        node = {
            "type": "paragraph",
            "content": [{"type": "text", "text": "hello world"}],
        }
        assert extract_text(node) == "hello world"

    def test_nested_content(self):
        node = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                },
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "third"}],
                },
            ],
        }
        result = extract_text(node)
        assert "first" in result
        assert "second" in result
        assert "third" in result

    def test_text_node_without_text_key(self):
        node = {"type": "text"}
        assert extract_text(node) == ""


class TestMakeAdf:
    def test_returns_valid_adf(self):
        result = make_adf("hello")
        assert result["type"] == "doc"
        assert result["version"] == 1
        assert len(result["content"]) == 1

        paragraph = result["content"][0]
        assert paragraph["type"] == "paragraph"
        assert paragraph["content"][0]["type"] == "text"
        assert paragraph["content"][0]["text"] == "hello"

    def test_preserves_special_characters(self):
        text = "line1\nline2 <tag> & 'quotes'"
        result = make_adf(text)
        assert result["content"][0]["content"][0]["text"] == text
