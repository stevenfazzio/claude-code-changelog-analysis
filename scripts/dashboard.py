"""Generate multi-page HTML site from enriched changelog data."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "enriched.parquet"
OUTPUT_DIR = ROOT / "docs"

# Shared styling — muted, Tufte-inspired palette
COLORS = ["#5b7b6f", "#8b7355", "#7a6f8a", "#9b7c6b", "#6b8f8a", "#a89b7b", "#7b8fa0"]
FONT = '"IBM Plex Mono", monospace'
LAYOUT = dict(
    template="plotly_white",
    font_family=FONT,
    font_color="#2c2c2c",
    font_size=11,
    margin=dict(l=40, r=40, t=45, b=35),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title_font_size=13,
    title_font_family='"Newsreader", Georgia, serif',
    title_font_color="#2c2c2c",
    xaxis=dict(gridcolor="#e8e5de", gridwidth=1),
    yaxis=dict(gridcolor="#e8e5de", gridwidth=1),
)
TOP_CATEGORIES = ["cli", "config", "performance", "mcp", "agents", "ide"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    return df


# ── Shared HTML Components ───────────────────────────────────────────────────

SHARED_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: "IBM Plex Mono", monospace;
    background: #faf9f6; color: #2c2c2c;
    line-height: 1.5;
}
.wrapper {
    max-width: 1100px; margin: 0 auto; padding: 0 2rem;
}
h1 {
    font-family: "Newsreader", Georgia, serif;
    font-size: 1.6rem; font-weight: 400; color: #2c2c2c;
    padding-top: 2.5rem;
}
.subtitle {
    font-size: 0.8rem; color: #888; margin-top: 0.2rem;
}
.nav {
    font-size: 0.82rem;
    padding: 0.8rem 0 0.2rem;
}
.nav a {
    color: #888;
    text-decoration: none;
}
.nav a:hover {
    color: #2c2c2c;
}
.nav a.active {
    color: #2c2c2c;
    font-weight: 600;
}
.nav .sep {
    color: #ccc;
    margin: 0 0.6rem;
}
hr {
    border: none; border-top: 1px solid #d5d0c8;
    margin: 1rem 0 1.5rem;
}
.kpi-row {
    font-size: 0.82rem; line-height: 2;
    padding-bottom: 0.5rem;
}
.kpi { white-space: nowrap; }
.kpi-label { color: #888; }
.kpi-value { font-weight: 600; color: #2c2c2c; }
.section { padding: 1.5rem 0 0.5rem; }
.section h2 {
    font-family: "Newsreader", Georgia, serif;
    font-size: 1.1rem; font-weight: 400; color: #555;
    letter-spacing: 0.02em;
    margin-bottom: 0.8rem;
}
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.chart-card {
    border: 1px solid #e0ddd5;
    padding: 0.5rem;
    min-height: 350px;
    background: #faf9f6;
}
.chart-card.full { grid-column: span 2; }
footer {
    text-align: center; padding: 2.5rem 0 2rem;
    color: #aaa; font-size: 0.75rem;
}
/* Tabulator overrides */
.tabulator { font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; border: 1px solid #e0ddd5; }
.tabulator .tabulator-header { font-family: "Newsreader", Georgia, serif; font-weight: 400; }
.tabulator .tabulator-header .tabulator-col { background: #faf9f6; border-color: #e0ddd5; overflow: visible; }
.tabulator .tabulator-header .tabulator-col .tabulator-header-filter { overflow: visible; }
.tabulator .tabulator-header .tabulator-col .tabulator-header-filter input[type="date"] { display: block; }
.tabulator .tabulator-tableholder .tabulator-table .tabulator-row { border-color: #eae7e0; }
.tabulator .tabulator-tableholder .tabulator-table .tabulator-row:hover { background: #f4f2ec; }
/* Stub page */
.stub {
    text-align: center; padding: 6rem 0;
}
.stub p {
    color: #888; font-size: 0.85rem; margin-top: 0.5rem;
}
@media (max-width: 900px) {
    .chart-grid { grid-template-columns: 1fr; }
    .chart-card.full { grid-column: span 1; }
    .wrapper { padding: 0 1rem; }
}
"""


def nav_html(active: str) -> str:
    pages = [("Explorer", "index.html"), ("Analysis", "analysis.html"), ("Map", "map.html")]
    links = []
    for label, href in pages:
        cls = ' class="active"' if label.lower() == active else ""
        links.append(f'<a href="{href}"{cls}>{label}</a>')
    return '<nav class="nav">' + '<span class="sep">&middot;</span>'.join(links) + "</nav>"


def page_shell(title: str, nav_active: str, body_content: str, extra_head: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Newsreader:opsz,wght@6..72,400&display=swap" rel="stylesheet">
<style>{SHARED_CSS}</style>
{extra_head}
</head>
<body>
<div class="wrapper">

<h1>Claude Code Changelog</h1>
<p class="subtitle">Trends and patterns from the Claude Code changelog</p>
{nav_html(nav_active)}
<hr>

{body_content}

<footer>Generated {datetime.now():%Y-%m-%d %H:%M} &middot; Data from anthropics/claude-code CHANGELOG.md</footer>
</div>
</body>
</html>"""


# ── KPI Cards ────────────────────────────────────────────────────────────────


def make_kpi_cards(df: pd.DataFrame) -> str:
    stats = [
        ("Entries", f"{len(df):,}"),
        ("Versions", f"{df['version'].nunique():,}"),
        (
            "Range",
            f"{df['date'].min():%b %Y} – {df['date'].max():%b %Y}",
        ),
        ("Top Category", df["category"].value_counts().index[0]),
        ("Bugfix %", f"{(df['change_type'] == 'bugfix').mean():.0%}"),
        ("User-Facing %", f"{df['user_facing'].mean():.0%}"),
    ]
    items = " &middot; ".join(
        f'<span class="kpi"><span class="kpi-label">{label}</span> '
        f'<span class="kpi-value">{val}</span></span>'
        for label, val in stats
    )
    return f'<div class="kpi-row">{items}</div>'


# ── Explorer Page ────────────────────────────────────────────────────────────


def _explorer_data_json(df: pd.DataFrame) -> str:
    cols = ["version", "date", "text", "category", "change_type", "complexity", "user_facing", "is_vscode", "is_breaking"]
    export = df[cols].copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    export["user_facing"] = export["user_facing"].map({True: "Yes", False: "No"})
    export["is_vscode"] = export["is_vscode"].map({True: "Yes", False: "No"})
    export["is_breaking"] = export["is_breaking"].map({True: "Yes", False: "No"})
    return export.to_json(orient="records")


def render_explorer_page(df: pd.DataFrame) -> str:
    data_json = _explorer_data_json(df)

    extra_head = """
<link href="https://unpkg.com/tabulator-tables@6.3.1/dist/css/tabulator.min.css" rel="stylesheet">
<script src="https://unpkg.com/tabulator-tables@6.3.1/dist/js/tabulator.min.js"></script>
"""

    body = f"""{make_kpi_cards(df)}

<div id="table"></div>

<script>
function minMaxFilterEditor(cell, onRendered, success, cancel) {{
  var container = document.createElement("span");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.gap = "1px";

  var from = document.createElement("input");
  from.type = "date";
  from.style.width = "100%";
  from.style.fontSize = "0.65rem";
  from.style.padding = "1px 2px";
  from.style.fontFamily = "IBM Plex Mono, monospace";

  var to = document.createElement("input");
  to.type = "date";
  to.style.width = "100%";
  to.style.fontSize = "0.65rem";
  to.style.padding = "1px 2px";
  to.style.fontFamily = "IBM Plex Mono, monospace";

  function update() {{
    success({{from: from.value, to: to.value}});
  }}
  from.addEventListener("change", update);
  to.addEventListener("change", update);

  var row = function(label, input) {{
    var r = document.createElement("span");
    r.style.display = "flex";
    r.style.alignItems = "center";
    r.style.gap = "2px";
    var s = document.createElement("span");
    s.textContent = label;
    s.style.fontSize = "0.6rem";
    s.style.color = "#888";
    s.style.flexShrink = "0";
    r.appendChild(s);
    input.style.flex = "1";
    input.style.minWidth = "0";
    r.appendChild(input);
    return r;
  }};
  container.appendChild(row("\u2265", from));
  container.appendChild(row("\u2264", to));

  return container;
}}

function minMaxFilterFunction(headerValue, rowValue) {{
  if (headerValue.from && rowValue < headerValue.from) return false;
  if (headerValue.to && rowValue > headerValue.to) return false;
  return true;
}}

var table = new Tabulator("#table", {{
  data: {data_json},
  layout: "fitColumns",
  height: "75vh",
  initialSort: [{{column: "date", dir: "desc"}}],
  columns: [
    {{title: "Version", field: "version", headerFilter: "list",
      headerFilterParams: {{valuesLookup: true, multiselect: true, sort: "desc"}},
      headerFilterFunc: "in"}},
    {{title: "Date", field: "date", width: 150, headerFilter: minMaxFilterEditor, headerFilterFunc: minMaxFilterFunction, headerFilterLiveFilter: false}},
    {{title: "Entry", field: "text", minWidth: 200, widthGrow: 3, headerFilter: "input",
      formatter: "textarea"}},
    {{title: "Category", field: "category", headerFilter: "list",
      headerFilterParams: {{valuesLookup: true, multiselect: true, sort: "asc"}},
      headerFilterFunc: "in"}},
    {{title: "Type", field: "change_type", headerFilter: "list",
      headerFilterParams: {{valuesLookup: true, multiselect: true, sort: "asc"}},
      headerFilterFunc: "in"}},
    {{title: "Complexity", field: "complexity", headerFilter: "list",
      headerFilterParams: {{multiselect: true,
        values: ["minor", "moderate", "major"]}},
      headerFilterFunc: "in"}},
    {{title: "User-facing", field: "user_facing", headerFilter: "list",
      headerFilterParams: {{values: ["Yes", "No"]}}}},
    {{title: "VS Code", field: "is_vscode", headerFilter: "list",
      headerFilterParams: {{values: ["Yes", "No"]}}}},
    {{title: "Breaking", field: "is_breaking", headerFilter: "list",
      headerFilterParams: {{values: ["Yes", "No"]}}}},
  ],
}});
</script>
"""

    return page_shell("Claude Code Changelog — Explorer", "explorer", body, extra_head=extra_head)


# ── Analysis Page (Charts) ───────────────────────────────────────────────────


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
            opacity=0.35,
            yaxis="y2",
        )
    )
    layout_no_axes = {k: v for k, v in LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        title="Bugfix Ratio Over Time",
        xaxis=dict(gridcolor="#e8e5de", gridwidth=1),
        yaxis=dict(title="Bugfix %", side="left", gridcolor="#e8e5de"),
        yaxis2=dict(title="Total entries", side="right", overlaying="y", gridcolor="#e8e5de"),
        legend=dict(orientation="h", y=1.12),
        **layout_no_axes,
    )
    return fig


def make_heatmap(df: pd.DataFrame) -> go.Figure:
    ct = pd.crosstab(df["category"], df["change_type"], normalize="index") * 100
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    fig = px.imshow(
        ct,
        text_auto=".0f",
        color_continuous_scale=[[0, "#faf9f6"], [1, "#5b7b6f"]],
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


def _chart_html(fig: go.Figure, include_js: str | bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_js)


def render_analysis_page(df: pd.DataFrame) -> str:
    def card(fig, full=False, first=False):
        cls = "chart-card full" if full else "chart-card"
        js = "cdn" if first else False
        return f'<div class="{cls}">{_chart_html(fig, js)}</div>'

    cadence = make_release_cadence(df)
    histogram = make_entries_histogram(df)
    trends = make_category_trends(df)
    cat_dist = make_category_dist(df)
    type_donut = make_change_type_donut(df)
    complexity_donut = make_complexity_donut(df)
    bugfix = make_bugfix_ratio(df)
    heatmap = make_heatmap(df)
    major = make_major_changes(df)

    body = f"""{make_kpi_cards(df)}

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
</div>"""

    return page_shell("Claude Code Changelog — Analysis", "analysis", body)


# ── Map Page (Stub) ──────────────────────────────────────────────────────────


def render_map_page() -> str:
    body = """<div class="stub">
  <h2>Map</h2>
  <p>Semantic map visualization coming soon.</p>
</div>"""
    return page_shell("Claude Code Changelog — Map", "map", body)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"Reading {INPUT_PATH}")
    df = load_data()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Building explorer page…")
    explorer = render_explorer_page(df)
    (OUTPUT_DIR / "index.html").write_text(explorer)
    print(f"  → index.html ({len(explorer):,} bytes)")

    print("Building analysis page…")
    analysis = render_analysis_page(df)
    (OUTPUT_DIR / "analysis.html").write_text(analysis)
    print(f"  → analysis.html ({len(analysis):,} bytes)")

    print("Building map page…")
    map_page = render_map_page()
    (OUTPUT_DIR / "map.html").write_text(map_page)
    print(f"  → map.html ({len(map_page):,} bytes)")

    print("Done — 3 pages written to docs/")


if __name__ == "__main__":
    main()
