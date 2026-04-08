"""Tests for pile.cache — TTL cache for agent responses."""

import time
from unittest.mock import patch

from pile.cache import (
    DEFAULT_TTL,
    _cache,
    _cache_key,
    _normalize_query,
    clear_cache,
    get_cached,
    set_cached,
)


class TestNormalizeQuery:
    def test_lowercases(self):
        assert _normalize_query("Hello World") == "hello world"

    def test_strips_whitespace(self):
        assert _normalize_query("  hello  ") == "hello"

    def test_collapses_whitespace(self):
        assert _normalize_query("hello   world   foo") == "hello world foo"

    def test_empty_string(self):
        assert _normalize_query("") == ""


class TestCacheKey:
    def test_same_query_same_key(self):
        assert _cache_key("hello") == _cache_key("hello")

    def test_normalized_queries_match(self):
        assert _cache_key("Hello  World") == _cache_key("hello world")

    def test_different_queries_different_keys(self):
        assert _cache_key("hello") != _cache_key("world")


class TestGetSetCached:
    def setup_method(self):
        clear_cache()

    def teardown_method(self):
        clear_cache()

    def test_get_returns_none_when_empty(self):
        assert get_cached("anything") is None

    def test_set_and_get(self):
        set_cached("what is sprint 5?", "Sprint 5 has 10 issues.", "scrum")
        result = get_cached("what is sprint 5?")
        assert result is not None
        text, agent = result
        assert text == "Sprint 5 has 10 issues."
        assert agent == "scrum"

    def test_normalized_query_hits_cache(self):
        set_cached("What Is Sprint 5?", "Sprint 5 has 10 issues.", "scrum")
        result = get_cached("what   is  sprint   5?")
        assert result is not None

    def test_does_not_cache_short_responses(self):
        set_cached("query", "short", "agent")
        assert get_cached("query") is None

    def test_does_not_cache_empty_responses(self):
        set_cached("query", "", "agent")
        assert get_cached("query") is None

    def test_ttl_expiry(self):
        set_cached("query", "a valid response text here", "agent", ttl=0.01)
        time.sleep(0.02)
        assert get_cached("query") is None

    def test_ttl_not_expired(self):
        set_cached("query", "a valid response text here", "agent", ttl=10)
        result = get_cached("query")
        assert result is not None

    def test_eviction_at_max_entries(self):
        for i in range(105):
            set_cached(f"query-{i}", f"response text for query {i}", "agent")
        assert len(_cache) <= 100

    def test_custom_ttl(self):
        set_cached("query", "a valid response text here", "agent", ttl=1000)
        key = _cache_key("query")
        assert _cache[key].ttl == 1000

    def test_default_ttl(self):
        set_cached("query", "a valid response text here", "agent")
        key = _cache_key("query")
        assert _cache[key].ttl == DEFAULT_TTL


class TestClearCache:
    def setup_method(self):
        clear_cache()

    def test_clear_removes_all(self):
        set_cached("q1", "response one is long enough", "a1")
        set_cached("q2", "response two is long enough", "a2")
        clear_cache()
        assert get_cached("q1") is None
        assert get_cached("q2") is None
        assert len(_cache) == 0
