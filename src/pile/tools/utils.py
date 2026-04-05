"""Utility functions for Jira data formatting."""

from __future__ import annotations


def extract_text(adf_node: dict | None) -> str:
    """Extract plain text from Atlassian Document Format."""
    if not adf_node:
        return ""
    if adf_node.get("type") == "text":
        return adf_node.get("text", "")
    text_parts = []
    for child in adf_node.get("content", []):
        text_parts.append(extract_text(child))
    return " ".join(text_parts).strip()


def make_adf(text: str) -> dict:
    """Convert plain text to Atlassian Document Format."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }
