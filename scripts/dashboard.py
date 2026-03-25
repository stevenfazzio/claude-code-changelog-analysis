"""Generate an interactive HTML dashboard from enriched changelog data."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "enriched.parquet"
OUTPUT_DIR = ROOT / "docs"
OUTPUT_PATH = OUTPUT_DIR / "index.html"

# Shared styling
COLORS = px.colors.qualitative.Set2
LAYOUT = dict(
    template="plotly_white",
    font_family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    margin=dict(l=40, r=40, t=50, b=40),
)
TOP_CATEGORIES = ["cli", "config", "performance", "mcp", "agents", "ide"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    return df


# ── KPI Cards ────────────────────────────────────────────────────────────────


def make_kpi_cards(df: pd.DataFrame) -> str:
    stats = [
        ("Total Entries", f"{len(df):,}"),
        ("Versions", f"{df['version'].nunique():,}"),
        (
            "Date Range",
            f"{df['date'].min():%b %Y} – {df['date'].max():%b %Y}",
        ),
        ("Top Category", df["category"].value_counts().index[0]),
        ("Bugfix %", f"{(df['change_type'] == 'bugfix').mean():.0%}"),
        ("User-Facing %", f"{df['user_facing'].mean():.0%}"),
    ]
    cards = "\n".join(
        f'<div class="kpi"><div class="kpi-value">{val}</div>'
        f'<div class="kpi-label">{label}</div></div>'
        for label, val in stats
    )
    return f'<div class="kpi-row">{cards}</div>'


# ── Section 1: Timeline Trends ───────────────────────────────────────────────


def make_release_cadence(df: pd.DataFrame) -> go.Figure:
    weekly = df.groupby("week")["version"].nunique().reset_index()
    weekly.columns = ["week", "versions"]
    fig = px.bar(weekly, x="week", y="versions", color_discrete_sequence=COLORS)
    fig.update_layout(
        title="Release Cadence (versions per week)",
        xaxis_title="",
        yaxis_title="Versions",
        **LAYOUT,
    )
    return fig


def make_entries_histogram(df: pd.DataFrame) -> go.Figure:
    per_version = df.groupby("version").size().reset_index(name="entries")
    fig = px.histogram(
        per_version,
        x="entries",
        marginal="box",
        nbins=40,
        color_discrete_sequence=COLORS,
    )
    fig.update_layout(
        title="Entries Per Version",
        xaxis_title="Number of entries",
        yaxis_title="Count of versions",
        **LAYOUT,
    )
    return fig


def make_category_trends(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2.loc[~df2["category"].isin(TOP_CATEGORIES), "category"] = "other"
    monthly = df2.groupby(["month", "category"]).size().reset_index(name="count")
    order = TOP_CATEGORIES + ["other"]
    monthly["category"] = pd.Categorical(monthly["category"], categories=order, ordered=True)
    monthly = monthly.sort_values(["month", "category"])
    fig = px.area(
        monthly,
        x="month",
        y="count",
        color="category",
        color_discrete_sequence=COLORS,
    )
    fig.update_layout(
        title="Category Trends (monthly)",
        xaxis_title="",
        yaxis_title="Entries",
        **LAYOUT,
    )
    return fig


# ── Section 2: Distributions ─────────────────────────────────────────────────


def make_category_dist(df: pd.DataFrame) -> go.Figure:
    counts = df["category"].value_counts().sort_values()
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        color_discrete_sequence=COLORS,
        text=counts.values,
    )
    fig.update_layout(
        title="Category Distribution",
        xaxis_title="Entries",
        yaxis_title="",
        **LAYOUT,
    )
    fig.update_traces(textposition="outside")
    return fig


def make_change_type_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["change_type"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.4,
            marker_colors=COLORS[: len(counts)],
        )
    )
    fig.update_layout(title="Change Type", **LAYOUT)
    return fig


def make_complexity_donut(df: pd.DataFrame) -> go.Figure:
    order = ["minor", "moderate", "major"]
    counts = df["complexity"].value_counts().reindex(order).dropna()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.4,
            marker_colors=COLORS[: len(counts)],
        )
    )
    fig.update_layout(title="Complexity", **LAYOUT)
    return fig


# ── Section 3: Deeper Analysis ───────────────────────────────────────────────


def make_bugfix_ratio(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby("month").agg(
        total=("change_type", "size"),
        bugfixes=("change_type", lambda s: (s == "bugfix").sum()),
    )
    monthly["pct"] = monthly["bugfixes"] / monthly["total"] * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["pct"],
            name="Bugfix %",
            mode="lines+markers",
            marker_color=COLORS[0],
        )
    )
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly["total"],
            name="Total entries",
            marker_color=COLORS[1],
            opacity=0.4,
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Bugfix Ratio Over Time",
        yaxis=dict(title="Bugfix %", side="left"),
        yaxis2=dict(title="Total entries", side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.12),
        **LAYOUT,
    )
    return fig


def make_heatmap(df: pd.DataFrame) -> go.Figure:
    ct = pd.crosstab(df["category"], df["change_type"], normalize="index") * 100
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    fig = px.imshow(
        ct,
        text_auto=".0f",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(
        title="Category × Change Type (row %)",
        xaxis_title="Change Type",
        yaxis_title="",
        **LAYOUT,
    )
    return fig


def make_major_changes(df: pd.DataFrame) -> go.Figure:
    major = df[df["complexity"] == "major"].copy()
    major["short_text"] = major["text"].str[:80] + "…"
    fig = px.scatter(
        major,
        x="date",
        y="category",
        color="category",
        hover_data={"short_text": True, "version": True, "date": False, "category": False},
        color_discrete_sequence=COLORS,
    )
    fig.update_traces(marker_size=10)
    fig.update_layout(
        title="Major Changes Timeline",
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        **LAYOUT,
    )
    return fig


# ── HTML Assembly ─────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f7; color: #1d1d1f;
}
header {
    background: #0d1117; color: #fff; padding: 2rem 2rem 1.5rem;
}
header h1 { font-size: 1.8rem; font-weight: 700; }
header p { opacity: 0.7; margin-top: 0.3rem; }
.kpi-row {
    display: flex; gap: 1rem; padding: 1.5rem 2rem; flex-wrap: wrap;
}
.kpi {
    flex: 1; min-width: 140px; background: #fff; border-radius: 8px;
    padding: 1.2rem; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-top: 3px solid #66b2ff;
}
.kpi-value { font-size: 1.5rem; font-weight: 700; color: #0d1117; }
.kpi-label { font-size: 0.85rem; color: #666; margin-top: 0.3rem; }
.section { padding: 0.5rem 2rem 1rem; }
.section h2 { font-size: 1.2rem; margin-bottom: 0.8rem; color: #333; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.chart-card {
    background: #fff; border-radius: 8px; padding: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    min-height: 350px;
}
.chart-card.full { grid-column: span 2; }
footer {
    text-align: center; padding: 1.5rem; color: #999; font-size: 0.8rem;
}
@media (max-width: 900px) {
    .chart-grid { grid-template-columns: 1fr; }
    .chart-card.full { grid-column: span 1; }
}
"""


def _chart_html(fig: go.Figure, include_js: str | bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_js)


def render_html(
    kpi_html: str,
    cadence: go.Figure,
    histogram: go.Figure,
    trends: go.Figure,
    cat_dist: go.Figure,
    type_donut: go.Figure,
    complexity_donut: go.Figure,
    bugfix: go.Figure,
    heatmap: go.Figure,
    major: go.Figure,
) -> str:
    # First chart includes plotly.js CDN, rest skip it
    def card(fig, full=False, first=False):
        cls = "chart-card full" if full else "chart-card"
        js = "cdn" if first else False
        return f'<div class="{cls}">{_chart_html(fig, js)}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Code Changelog Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Claude Code Changelog Dashboard</h1>
  <p>Trends and patterns from the Claude Code changelog</p>
</header>

{kpi_html}

<div class="section">
  <h2>Timeline Trends</h2>
  <div class="chart-grid">
    {card(cadence, full=True, first=True)}
    {card(histogram)}
    {card(trends)}
  </div>
</div>

<div class="section">
  <h2>Distributions</h2>
  <div class="chart-grid">
    {card(cat_dist, full=True)}
    {card(type_donut)}
    {card(complexity_donut)}
  </div>
</div>

<div class="section">
  <h2>Deeper Analysis</h2>
  <div class="chart-grid">
    {card(bugfix, full=True)}
    {card(heatmap)}
    {card(major)}
  </div>
</div>

<footer>Generated {datetime.now():%Y-%m-%d %H:%M} · Data from anthropics/claude-code CHANGELOG.md</footer>
</body>
</html>"""


def main():
    print(f"Reading {INPUT_PATH}")
    df = load_data()

    print("Building charts…")
    html = render_html(
        kpi_html=make_kpi_cards(df),
        cadence=make_release_cadence(df),
        histogram=make_entries_histogram(df),
        trends=make_category_trends(df),
        cat_dist=make_category_dist(df),
        type_donut=make_change_type_donut(df),
        complexity_donut=make_complexity_donut(df),
        bugfix=make_bugfix_ratio(df),
        heatmap=make_heatmap(df),
        major=make_major_changes(df),
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(html)
    print(f"Dashboard written to {OUTPUT_PATH} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
