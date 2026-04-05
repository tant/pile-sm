"""Auto-detect numeric data in agent output and build Plotly charts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import plotly.graph_objects as go

# Consistent color palette
COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
STATUS_COLORS = {"Done": "#00CC96", "In Progress": "#636EFA", "To Do": "#EF553B", "Blocked": "#AB63FA"}


@dataclass
class ChartData:
    chart_type: str  # "pie", "bar", "hbar"
    title: str
    labels: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    secondary_values: list[float] = field(default_factory=list)  # e.g., story points alongside issue counts
    secondary_label: str = ""


def detect_charts(text: str) -> list[ChartData]:
    """Scan agent output text for chartable numeric patterns. Returns chart specs."""
    charts: list[ChartData] = []

    for detector in [
        _detect_status_distribution,
        _detect_workload,
        _detect_velocity,
        _detect_time_metrics,
    ]:
        result = detector(text)
        if result:
            charts.append(result)

    return charts


def build_chart(data: ChartData) -> go.Figure:
    """Build a Plotly figure from chart data."""
    builders = {
        "pie": _build_pie,
        "bar": _build_bar,
        "hbar": _build_hbar,
    }
    builder = builders.get(data.chart_type, _build_bar)
    fig = builder(data)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(size=13),
    )
    return fig


# --- Detectors ---


def _detect_status_distribution(text: str) -> ChartData | None:
    """Detect issue status counts: Done: 12, In Progress: 5, To Do: 3."""
    statuses = {}

    # Pattern: "Status: N" or "Status (N)" or "- Status: N issues"
    for status_name in ["Done", "In Progress", "To Do", "Blocked", "In Review", "Ready", "Open", "Closed"]:
        # Only match integer counts NOT followed by decimals or time units
        patterns = [
            rf"(?i){re.escape(status_name)}\s*[:：]\s*(\d+)(?!\.\d)(?!\s*(?:days?|hours?|hrs?|mins?))",
            rf"(?i){re.escape(status_name)}\s*\((\d+)(?!\.\d)\)",
            rf"(?i){re.escape(status_name)}\s*[-–]\s*(\d+)(?!\.\d)(?!\s*(?:days?|hours?|hrs?|mins?))",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                statuses[status_name] = float(m.group(1))
                break

    if len(statuses) < 2:
        return None

    return ChartData(
        chart_type="pie",
        title="Issue Status Distribution",
        labels=list(statuses.keys()),
        values=list(statuses.values()),
    )


def _detect_workload(text: str) -> ChartData | None:
    """Detect per-member workload: - Name: N issues (M pts) or Name: N issues."""
    members: dict[str, float] = {}
    points: dict[str, float] = {}

    # Pattern: "- Name: N issues" or "Name: N issues (M pts/points/story points)"
    pattern = r"[-•]\s*(.+?):\s*(\d+)\s*(?:issues?|tasks?|items?)"
    pts_pattern = r"\((\d+(?:\.\d+)?)\s*(?:pts?|points?|story points?|sp)\)"

    for m in re.finditer(pattern, text, re.IGNORECASE):
        name = m.group(1).strip().rstrip("*").strip()
        if len(name) > 30 or len(name) < 2:
            continue
        count = float(m.group(2))
        members[name] = count

        # Check for story points in the same line context
        line_end = text[m.end():m.end() + 50]
        pts_m = re.search(pts_pattern, line_end, re.IGNORECASE)
        if pts_m:
            points[name] = float(pts_m.group(1))

    if len(members) < 2:
        return None

    data = ChartData(
        chart_type="hbar",
        title="Workload Distribution",
        labels=list(members.keys()),
        values=list(members.values()),
    )
    if points:
        data.secondary_values = [points.get(name, 0) for name in members]
        data.secondary_label = "Story Points"
    return data


def _detect_velocity(text: str) -> ChartData | None:
    """Detect velocity data: Sprint N: X pts, Sprint M: Y pts."""
    sprints: dict[str, float] = {}

    # Pattern: "Sprint N: X pts" or "Sprint Name: X story points"
    pattern = r"(?:Sprint\s+[\w#]+)\s*[:：]\s*(\d+(?:\.\d+)?)\s*(?:pts?|points?|story points?|sp)"
    for m in re.finditer(pattern, text, re.IGNORECASE):
        label = text[m.start():m.start() + 30].split(":")[0].strip().rstrip("*").strip()
        sprints[label] = float(m.group(1))

    if len(sprints) < 2:
        return None

    return ChartData(
        chart_type="bar",
        title="Sprint Velocity",
        labels=list(sprints.keys()),
        values=list(sprints.values()),
    )


def _detect_time_metrics(text: str) -> ChartData | None:
    """Detect time/duration metrics: Stage: X.Y days, Phase: N hours."""
    stages: dict[str, float] = {}

    # Pattern: "- Stage: X.Y days" or "Stage: X hours"
    pattern = r"[-•]\s*(.+?):\s*(\d+(?:\.\d+)?)\s*(days?|hours?|hrs?)"
    for m in re.finditer(pattern, text, re.IGNORECASE):
        name = m.group(1).strip().rstrip("*").strip()
        if len(name) > 30 or len(name) < 2:
            continue
        value = float(m.group(2))
        unit = m.group(3).lower()
        if unit.startswith("hour") or unit.startswith("hr"):
            value = value / 24  # normalize to days
        stages[name] = value

    if len(stages) < 2:
        return None

    return ChartData(
        chart_type="bar",
        title="Cycle Time by Stage",
        labels=list(stages.keys()),
        values=list(stages.values()),
    )


# --- Chart builders ---


def _build_pie(data: ChartData) -> go.Figure:
    colors = [STATUS_COLORS.get(label, COLORS[i % len(COLORS)]) for i, label in enumerate(data.labels)]
    fig = go.Figure(data=[go.Pie(
        labels=data.labels,
        values=data.values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo="label+value+percent",
        textposition="outside",
    )])
    fig.update_layout(title=data.title, showlegend=False)
    return fig


def _build_bar(data: ChartData) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.labels,
        y=data.values,
        marker_color=COLORS[:len(data.labels)],
        text=[f"{v:g}" for v in data.values],
        textposition="outside",
    ))
    if data.secondary_values:
        fig.add_trace(go.Bar(
            x=data.labels,
            y=data.secondary_values,
            name=data.secondary_label,
            marker_color=COLORS[1],
            text=[f"{v:g}" for v in data.secondary_values],
            textposition="outside",
        ))
        fig.update_layout(barmode="group")
    fig.update_layout(title=data.title)
    return fig


def _build_hbar(data: ChartData) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=data.labels,
        x=data.values,
        orientation="h",
        marker_color=COLORS[:len(data.labels)],
        text=[f"{v:g} issues" for v in data.values],
        textposition="outside",
        name="Issues",
    ))
    if data.secondary_values:
        fig.add_trace(go.Bar(
            y=data.labels,
            x=data.secondary_values,
            orientation="h",
            marker_color=[COLORS[1]] * len(data.labels),
            text=[f"{v:g} pts" for v in data.secondary_values],
            textposition="outside",
            name=data.secondary_label,
        ))
        fig.update_layout(barmode="group")
    fig.update_layout(title=data.title)
    return fig
