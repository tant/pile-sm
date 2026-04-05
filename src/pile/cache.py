"""Semantic cache for agent responses — avoid re-running identical or similar queries."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

_cache: dict[str, "_CacheEntry"] = {}
DEFAULT_TTL = 300  # 5 minutes


@dataclass
class _CacheEntry:
    response_text: str
    agent_key: str
    timestamp: float
    ttl: float


def _normalize_query(query: str) -> str:
    """Normalize query for cache key — lowercase, strip, collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", query.lower().strip())


def _cache_key(query: str) -> str:
    """Generate cache key from normalized query."""
    normalized = _normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()


def get_cached(query: str) -> tuple[str, str] | None:
    """Check cache for a matching query.

    Returns (response_text, agent_key) or None if not cached/expired.
    """
    key = _cache_key(query)
    entry = _cache.get(key)
    if entry is None:
        return None

    # Check TTL
    if time.time() - entry.timestamp > entry.ttl:
        del _cache[key]
        return None

    return entry.response_text, entry.agent_key


def set_cached(query: str, response_text: str, agent_key: str, ttl: float = DEFAULT_TTL):
    """Cache a query response."""
    if not response_text or len(response_text) < 10:
        return  # Don't cache empty/tiny responses

    key = _cache_key(query)
    _cache[key] = _CacheEntry(
        response_text=response_text,
        agent_key=agent_key,
        timestamp=time.time(),
        ttl=ttl,
    )

    # Evict old entries (keep max 100)
    if len(_cache) > 100:
        oldest_key = min(_cache, key=lambda k: _cache[k].timestamp)
        del _cache[oldest_key]


def clear_cache():
    """Clear all cached entries."""
    _cache.clear()
