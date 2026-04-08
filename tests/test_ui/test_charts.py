"""Tests for pile.ui.charts — chart detection and building."""

import plotly.graph_objects as go

from pile.ui.charts import (
    ChartData,
    build_chart,
    detect_charts,
    _detect_status_distribution,
    _detect_workload,
    _detect_velocity,
    _detect_time_metrics,
)


class TestDetectStatusDistribution:
    def test_detects_colon_format(self):
        text = "Done: 12, In Progress: 5, To Do: 3"
        result = _detect_status_distribution(text)
        assert result is not None
        assert result.chart_type == "pie"
        assert "Done" in result.labels
        assert 12.0 in result.values

    def test_detects_parentheses_format(self):
        text = "Done (10), In Progress (4), To Do (2)"
        result = _detect_status_distribution(text)
        assert result is not None
        assert len(result.labels) == 3

    def test_detects_dash_format(self):
        text = "Done - 8\nIn Progress - 3\nBlocked - 1"
        result = _detect_status_distribution(text)
        assert result is not None
        assert "Blocked" in result.labels

    def test_returns_none_for_single_status(self):
        text = "Done: 5"
        result = _detect_status_distribution(text)
        assert result is None

    def test_returns_none_for_no_match(self):
        result = _detect_status_distribution("no status data here")
        assert result is None

    def test_ignores_time_units(self):
        text = "Done: 3 days, In Progress: 5 hours"
        result = _detect_status_distribution(text)
        assert result is None

    def test_title(self):
        text = "Done: 10, In Progress: 5"
        result = _detect_status_distribution(text)
        assert result.title == "Issue Status Distribution"


class TestDetectWorkload:
    def test_detects_member_issues(self):
        text = "- Alice: 5 issues\n- Bob: 3 issues"
        result = _detect_workload(text)
        assert result is not None
        assert result.chart_type == "hbar"
        assert "Alice" in result.labels
        assert 5.0 in result.values

    def test_detects_story_points(self):
        text = "- Alice: 5 issues (13 pts)\n- Bob: 3 issues (8 pts)"
        result = _detect_workload(text)
        assert result is not None
        assert result.secondary_values == [13.0, 8.0]
        assert result.secondary_label == "Story Points"

    def test_returns_none_for_single_member(self):
        text = "- Alice: 5 issues"
        result = _detect_workload(text)
        assert result is None

    def test_ignores_long_names(self):
        text = "- " + "A" * 35 + ": 5 issues\n- Bob: 3 issues"
        result = _detect_workload(text)
        # Long name filtered out, only 1 member left
        assert result is None

    def test_returns_none_for_no_match(self):
        result = _detect_workload("nothing workload here")
        assert result is None


class TestDetectVelocity:
    def test_detects_sprint_velocity(self):
        text = "Sprint 1: 20 pts\nSprint 2: 25 pts\nSprint 3: 30 pts"
        result = _detect_velocity(text)
        assert result is not None
        assert result.chart_type == "bar"
        assert len(result.labels) == 3

    def test_returns_none_for_single_sprint(self):
        text = "Sprint 1: 20 pts"
        result = _detect_velocity(text)
        assert result is None

    def test_returns_none_for_no_match(self):
        result = _detect_velocity("no velocity data")
        assert result is None


class TestDetectTimeMetrics:
    def test_detects_days(self):
        text = "- Development: 3.5 days\n- Testing: 2.0 days\n- Review: 1.0 days"
        result = _detect_time_metrics(text)
        assert result is not None
        assert result.chart_type == "bar"
        assert "Development" in result.labels
        assert 3.5 in result.values

    def test_converts_hours_to_days(self):
        text = "- Development: 48 hours\n- Testing: 24 hours"
        result = _detect_time_metrics(text)
        assert result is not None
        assert result.values[0] == 2.0  # 48h / 24
        assert result.values[1] == 1.0  # 24h / 24

    def test_returns_none_for_single_stage(self):
        text = "- Development: 3 days"
        result = _detect_time_metrics(text)
        assert result is None

    def test_returns_none_for_no_match(self):
        result = _detect_time_metrics("no time data")
        assert result is None


class TestDetectCharts:
    def test_returns_multiple_charts(self):
        text = (
            "Done: 10, In Progress: 5, To Do: 3\n"
            "- Alice: 5 issues\n- Bob: 3 issues"
        )
        charts = detect_charts(text)
        assert len(charts) >= 2

    def test_returns_empty_for_no_data(self):
        charts = detect_charts("just a plain text response")
        assert charts == []


class TestBuildChart:
    def test_build_pie(self):
        data = ChartData(
            chart_type="pie",
            title="Test Pie",
            labels=["A", "B", "C"],
            values=[10, 20, 30],
        )
        fig = build_chart(data)
        assert isinstance(fig, go.Figure)

    def test_build_bar(self):
        data = ChartData(
            chart_type="bar",
            title="Test Bar",
            labels=["X", "Y"],
            values=[5, 15],
        )
        fig = build_chart(data)
        assert isinstance(fig, go.Figure)

    def test_build_bar_with_secondary(self):
        data = ChartData(
            chart_type="bar",
            title="Test Bar",
            labels=["X", "Y"],
            values=[5, 15],
            secondary_values=[3, 8],
            secondary_label="Points",
        )
        fig = build_chart(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_build_hbar(self):
        data = ChartData(
            chart_type="hbar",
            title="Test HBar",
            labels=["A", "B"],
            values=[10, 20],
        )
        fig = build_chart(data)
        assert isinstance(fig, go.Figure)

    def test_build_hbar_with_secondary(self):
        data = ChartData(
            chart_type="hbar",
            title="Test HBar",
            labels=["A", "B"],
            values=[10, 20],
            secondary_values=[5, 8],
            secondary_label="Story Points",
        )
        fig = build_chart(data)
        assert len(fig.data) == 2

    def test_unknown_type_defaults_to_bar(self):
        data = ChartData(
            chart_type="unknown",
            title="Fallback",
            labels=["X"],
            values=[10],
        )
        fig = build_chart(data)
        assert isinstance(fig, go.Figure)

    def test_dark_template_applied(self):
        data = ChartData(chart_type="bar", title="T", labels=["A"], values=[1])
        fig = build_chart(data)
        assert fig.layout.template.layout.to_plotly_json() is not None
