"""Tests for UI helper functions."""

from pile.ui.chainlit_app import summarize_args


def test_summarize_args_basic():
    args = {"project": "TETRA", "state": "active"}
    result = summarize_args(args)
    assert result == "project=TETRA, state=active"


def test_summarize_args_truncates_long_values():
    args = {"query": "a" * 50}
    result = summarize_args(args)
    assert result == "query=" + "a" * 30 + "..."


def test_summarize_args_max_3_keys():
    args = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5"}
    result = summarize_args(args)
    assert result == "a=1, b=2, c=3, +2 more"


def test_summarize_args_empty():
    assert summarize_args({}) == ""
