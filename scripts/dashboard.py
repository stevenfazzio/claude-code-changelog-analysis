"""Generate multi-page HTML site from enriched changelog data."""

import json
from datetime import datetime
from pathlib import Path

import glasbey
import numpy as np
import pandas as pd

from html import escape as html_escape

from nav import NAV_CSS, PLAUSIBLE_EVENTS_SCRIPT, PLAUSIBLE_SCRIPT, nav_html

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "enriched.parquet"
OUTPUT_DIR = ROOT / "docs"
DATA_DIR = OUTPUT_DIR / "data"

# ── Unified color palette ────────────────────────────────────────────────────
CATEGORIES = [
    "terminal", "input", "slash_commands", "sessions",
    "mcp", "voice", "auth", "ide", "hooks", "permissions",
    "performance", "agents", "plugins", "config", "api", "sdk", "other",
]
CATEGORY_COLORS = dict(zip(
    CATEGORIES,
    glasbey.create_palette(
        palette_size=len(CATEGORIES),
        lightness_bounds=(40, 60),
        chroma_bounds=(25, 50),
    ),
))

TYPE_COLORS = {
    "feature": "#2d7d46",
    "bugfix": "#c4382a",
    "improvement": "#2e6eb8",
    "breaking": "#d4710e",
    "internal": "#777777",
}

COMPLEXITY_COLORS = {
    "minor": "#888888",
    "moderate": "#d4910e",
    "major": "#c4382a",
}

AUDIENCES = ["interactive_user", "sdk_developer", "admin", "extension_developer"]

AUDIENCE_COLORS = {
    "interactive_user": "#18756e",     # teal
    "sdk_developer": "#592c85",        # violet
    "admin": "#72721c",                # olive
    "extension_developer": "#852c59",  # rose
}


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    return df


# ── Data File Generation ────────────────────────────────────────────────────


def generate_entries_json(df: pd.DataFrame) -> list[dict]:
    cols = ["date", "text", "category", "change_type", "complexity", "audience"]
    export = df[cols].copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d").replace("NaT", "")
    records = export.to_dict(orient="records")
    # Replace any remaining NaN/None values with empty strings for JSON safety
    for rec in records:
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[k] = ""
    return records


def _monthly_trends(df: pd.DataFrame, col: str) -> dict:
    """Build {months, series} for a given column, grouped monthly."""
    order = df[col].value_counts().index.tolist()
    monthly = df.groupby(["month", col]).size().reset_index(name="count")
    months_sorted = sorted(m for m in df["month"].unique() if pd.notna(m))
    months_str = [m.strftime("%Y-%m-%d") for m in months_sorted]
    series = {}
    for val in order:
        val_data = monthly[monthly[col] == val].set_index("month")["count"]
        series[val] = [int(val_data.get(m, 0)) for m in months_sorted]
    return {"months": months_str, "series": series}


def generate_analysis_json(df: pd.DataFrame) -> dict:
    # Dimension trends (monthly) — category, change_type, complexity, audience
    dimension_trends = {
        dim: _monthly_trends(df, dim)
        for dim in ("category", "change_type", "complexity", "audience")
    }

    # Release cadence (weekly)
    weekly = df.groupby("week")["version"].nunique().reset_index()
    weekly.columns = ["week", "versions"]
    weekly = weekly.sort_values("week")

    # Category distribution
    cat_counts = df["category"].value_counts().sort_values()

    # Change type counts
    type_counts = df["change_type"].value_counts()

    # Complexity counts
    complexity_order = ["minor", "moderate", "major"]
    complexity_counts = df["complexity"].value_counts().reindex(complexity_order).dropna()

    # Audience counts
    audience_counts = df["audience"].value_counts()

    # Bugfix ratio
    monthly_agg = df.groupby("month").agg(
        total=("change_type", "size"),
        bugfixes=("change_type", lambda s: (s == "bugfix").sum()),
    )
    monthly_agg["pct"] = (monthly_agg["bugfixes"] / monthly_agg["total"] * 100).round(1)
    monthly_agg = monthly_agg.sort_index()

    # Heatmap (category x change_type, row-normalized %)
    ct = pd.crosstab(df["category"], df["change_type"], normalize="index") * 100
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    heatmap_data = []
    for yi, cat in enumerate(ct.index):
        for xi, ctype in enumerate(ct.columns):
            heatmap_data.append([xi, yi, round(float(ct.loc[cat, ctype]), 1)])

    # Major changes
    major = df[df["complexity"] == "major"].copy()
    major = major.dropna(subset=["date"]).sort_values("date")
    major_list = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "category": row["category"],
            "text": row["text"][:200] + "..." if len(row["text"]) > 200 else row["text"],
        }
        for _, row in major.iterrows()
    ]

    # KPIs
    kpi = {
        "entries": int(len(df)),
        "versions": int(df["version"].nunique()),
        "date_min": df["date"].min().strftime("%b %Y"),
        "date_max": df["date"].max().strftime("%b %Y"),
        "top_category": df["category"].value_counts().index[0],
        "bugfix_pct": int(round((df["change_type"] == "bugfix").mean() * 100)),
        "top_audience": df["audience"].value_counts().index[0],
    }

    return {
        "dimension_trends": dimension_trends,
        "release_cadence": {
            "weeks": [w.strftime("%Y-%m-%d") for w in weekly["week"]],
            "versions": weekly["versions"].tolist(),
        },
        "category_dist": {
            "categories": cat_counts.index.tolist(),
            "counts": cat_counts.values.tolist(),
        },
        "change_type": {
            "labels": type_counts.index.tolist(),
            "values": [int(v) for v in type_counts.values],
        },
        "complexity": {
            "labels": complexity_counts.index.tolist(),
            "values": [int(v) for v in complexity_counts.values],
        },
        "audience": {
            "labels": audience_counts.index.tolist(),
            "values": [int(v) for v in audience_counts.values],
        },
        "bugfix_ratio": {
            "months": [m.strftime("%Y-%m-%d") for m in monthly_agg.index],
            "pct": monthly_agg["pct"].tolist(),
            "total": [int(v) for v in monthly_agg["total"].values],
        },
        "heatmap": {
            "categories": ct.index.tolist(),
            "change_types": ct.columns.tolist(),
            "data": heatmap_data,
        },
        "major_changes": major_list,
        "kpi": kpi,
        "colors": {
            "category": CATEGORY_COLORS,
            "type": TYPE_COLORS,
            "complexity": COMPLEXITY_COLORS,
            "audience": AUDIENCE_COLORS,
        },
    }


# ── Shared HTML Components ───────────────────────────────────────────────────

TABULATOR_CSS = """
/* Tabulator overrides */
.tabulator { font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; border: 1px solid #e0ddd5; }
.tabulator .tabulator-header { font-family: "Newsreader", Georgia, serif; font-weight: 400; }
.tabulator .tabulator-header .tabulator-col { background: #faf9f6; border-color: #e0ddd5; overflow: visible; }
.tabulator .tabulator-header .tabulator-col .tabulator-header-filter { overflow: visible; }
.tabulator .tabulator-header .tabulator-col .tabulator-header-filter input[type="date"] { display: block; }
.tabulator .tabulator-tableholder .tabulator-table .tabulator-row { border-color: #eae7e0; background: transparent; }
.tabulator .tabulator-tableholder .tabulator-table .tabulator-row.tabulator-row-even { background: transparent; }
.tabulator .tabulator-tableholder .tabulator-table .tabulator-row.date-band { background: #f5f3ed; }
.tabulator .tabulator-tableholder .tabulator-table .tabulator-row:hover { background: #ece9e0; }
/* Table pills & indicators */
.pill {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 0.72rem; font-weight: 600;
    line-height: 1.4; white-space: nowrap;
}
.field-indicator, .complexity-indicator { font-weight: 600; white-space: nowrap; }
/* Tabulator responsive collapse */
.tabulator-responsive-collapse { padding: 8px 12px; }
.tabulator-responsive-collapse table { font-size: 0.75rem; width: 100%; }
.tabulator-responsive-collapse table td { padding: 2px 8px; vertical-align: top; }
.tabulator-responsive-collapse table td:first-child { font-weight: 600; color: #888; white-space: nowrap; }
.tabulator-responsive-collapse-toggle { display: inline-flex; align-items: center; justify-content: center; }
@media (max-width: 600px) {
    .tabulator { font-size: 0.72rem; }
    .tabulator .tabulator-header .tabulator-col .tabulator-header-filter input[type="date"] {
        font-size: 0.6rem; padding: 0 1px;
    }
    .tabulator .tabulator-header .tabulator-col .tabulator-header-filter input[type="search"],
    .tabulator .tabulator-header .tabulator-col .tabulator-header-filter input[type="text"] {
        font-size: 0.65rem; padding: 1px 2px;
    }
}
"""

TAILWIND_CONFIG = """
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  theme: {
    extend: {
      colors: {
        cream: '#faf9f6',
        'cream-dark': '#f4f2ec',
        border: '#e0ddd5',
        'border-light': '#e8e5de',
        'text-primary': '#2c2c2c',
        'text-secondary': '#888',
        'text-muted': '#aaa',
        divider: '#d5d0c8',
      },
      fontFamily: {
        mono: ['"IBM Plex Mono"', 'monospace'],
        serif: ['"Newsreader"', 'Georgia', 'serif'],
      },
      maxWidth: {
        site: '1100px',
      },
    },
  },
}
</script>
"""



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
{PLAUSIBLE_SCRIPT}
{PLAUSIBLE_EVENTS_SCRIPT}
{TAILWIND_CONFIG}
{NAV_CSS}
{extra_head}
</head>
<body class="font-mono bg-cream text-text-primary leading-relaxed text-[11px]">
{nav_html(nav_active)}
<div class="max-w-site mx-auto px-8 max-md:px-4">

{body_content}

<footer class="text-center py-10 text-text-muted text-xs"><span id="generated-at"></span>Data from anthropics/claude-code CHANGELOG.md</footer>
<script>fetch('data/meta.json').then(r=>r.json()).then(d=>{{document.getElementById('generated-at').textContent='Generated '+d.generated_at+' \u00b7 '}}).catch(()=>{{}})</script>
</div>
</body>
</html>"""


# ── Explorer Page ────────────────────────────────────────────────────────────


def render_explorer_page(df: pd.DataFrame) -> str:
    extra_head = f"""
<link href="https://unpkg.com/tabulator-tables@6.3.1/dist/css/tabulator.min.css" rel="stylesheet">
<script src="https://unpkg.com/tabulator-tables@6.3.1/dist/js/tabulator.min.js"></script>
<style>{TABULATOR_CSS}</style>
"""

    cat_colors_js = json.dumps(CATEGORY_COLORS)
    type_colors_js = json.dumps(TYPE_COLORS)
    complexity_colors_js = json.dumps(COMPLEXITY_COLORS)
    audience_colors_js = json.dumps(AUDIENCE_COLORS)

    body = f"""<div id="kpi-row" class="text-[0.82rem] leading-8 pb-2 flex flex-wrap gap-x-2 max-sm:text-[0.75rem]"></div>

<div id="table"></div>

<script>
var CAT_COLORS = {cat_colors_js};
var TYPE_COLORS = {type_colors_js};
var TYPE_ICONS = {{feature:"\u2726", bugfix:"\u2715", improvement:"\u2191", breaking:"\u26a0", internal:"\u2699"}};
var COMPLEXITY_COLORS = {complexity_colors_js};
var COMPLEXITY_DOTS = {{minor:"\u25cf\u25cb\u25cb", moderate:"\u25cf\u25cf\u25cb", major:"\u25cf\u25cf\u25cf"}};
var AUDIENCE_COLORS = {audience_colors_js};
var AUDIENCE_ICONS = {{interactive_user:"\u25c9", extension_developer:"\u2b21", admin:"\u26ed", sdk_developer:"\u2692"}};
</script>

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

function updateKpis(rows) {{
  var n = rows.length;
  if (n === 0) {{
    document.getElementById("kpi-row").innerHTML = '<span class="whitespace-nowrap"><span class="text-text-secondary">Entries</span> <span class="font-semibold text-text-primary">0</span></span>';
    return;
  }}
  var dates = rows.map(function(r) {{ return r.getData().date; }}).filter(Boolean).sort();
  var cats = {{}};
  var bugfixes = 0;
  var audiences = {{}};
  rows.forEach(function(r) {{
    var d = r.getData();
    cats[d.category] = (cats[d.category] || 0) + 1;
    if (d.change_type === "bugfix") bugfixes++;
    audiences[d.audience] = (audiences[d.audience] || 0) + 1;
  }});
  var topCat = Object.keys(cats).sort(function(a, b) {{ return cats[b] - cats[a]; }})[0];
  var topAudience = Object.keys(audiences).sort(function(a, b) {{ return audiences[b] - audiences[a]; }})[0] || "";
  var minDate = new Date(dates[0]);
  var maxDate = new Date(dates[dates.length - 1]);
  var months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  var range = months[minDate.getMonth()] + " " + minDate.getDate() + ", " + minDate.getFullYear() + " \u2013 " + months[maxDate.getMonth()] + " " + maxDate.getDate() + ", " + maxDate.getFullYear();
  var bugPct = Math.round(bugfixes / n * 100) + "%";
  var stats = [
    ["Entries", n.toLocaleString()],
    ["Range", range],
    ["Top Category", topCat],
    ["Bugfix %", bugPct],
    ["Top Audience", topAudience.replace(/_/g, " ")],
  ];
  var sep = '<span class="text-border">&middot;</span>';
  document.getElementById("kpi-row").innerHTML = stats.map(function(s) {{
    return '<span class="whitespace-nowrap"><span class="text-text-secondary">' + s[0] + '</span> <span class="font-semibold text-text-primary">' + s[1] + '</span></span>';
  }}).join(" " + sep + " ");
}}

fetch("data/entries.json")
  .then(function(r) {{ return r.json(); }})
  .then(function(data) {{
    data.sort(function(a, b) {{ return a.date < b.date ? 1 : a.date > b.date ? -1 : 0; }});
    var table = new Tabulator("#table", {{
      data: data,
      layout: "fitColumns",
      responsiveLayout: "collapse",
      responsiveLayoutCollapseStartOpen: false,
      height: "75vh",
      initialSort: [{{column: "date", dir: "desc"}}],
      columns: [
        {{formatter: "responsiveCollapse", width: 30, minWidth: 30, hozAlign: "center",
          resizable: false, headerSort: false, responsive: 0}},
        {{title: "Date", field: "date", minWidth: 90, width: 120, responsive: 0,
          headerFilter: minMaxFilterEditor, headerFilterFunc: minMaxFilterFunction,
          headerFilterLiveFilter: false}},
        {{title: "Entry", field: "text", widthGrow: 5, minWidth: 150, responsive: 0,
          headerFilter: "input", formatter: "textarea"}},
        {{title: "Category", field: "category", minWidth: 120, width: 130, responsive: 1,
          headerFilter: "list",
          headerFilterParams: {{valuesLookup: true, multiselect: true, sort: "asc"}},
          headerFilterFunc: "in",
          formatter: function(cell) {{
            var v = cell.getValue();
            var c = CAT_COLORS[v] || "#888";
            return '<span class="pill" style="background:' + c + '18;color:' + c + ';border:1px solid ' + c + '30;">' + v.replace(/_/g, ' ') + '</span>';
          }}
        }},
        {{title: "Type", field: "change_type", width: 140, responsive: 2,
          headerFilter: "list",
          headerFilterParams: {{valuesLookup: true, multiselect: true, sort: "asc"}},
          headerFilterFunc: "in",
          formatter: function(cell) {{
            var v = cell.getValue();
            var color = TYPE_COLORS[v] || "#888";
            var icon = TYPE_ICONS[v] || "";
            return '<span class="field-indicator" style="color:' + color + ';">' + icon + ' ' + v + '</span>';
          }}
        }},
        {{title: "Audience", field: "audience", width: 160, responsive: 3,
          headerFilter: "list",
          headerFilterParams: {{multiselect: true,
            values: {json.dumps(AUDIENCES)}}},
          headerFilterFunc: "in",
          formatter: function(cell) {{
            var v = cell.getValue();
            var color = AUDIENCE_COLORS[v] || "#888";
            var icon = AUDIENCE_ICONS[v] || "";
            return '<span class="field-indicator" style="color:' + color + ';">' + icon + ' ' + v.replace(/_/g, ' ') + '</span>';
          }}
        }},
        {{title: "Complexity", field: "complexity", width: 130, responsive: 3,
          headerFilter: "list",
          headerFilterParams: {{multiselect: true,
            values: ["minor", "moderate", "major"]}},
          headerFilterFunc: "in",
          formatter: function(cell) {{
            var v = cell.getValue();
            var dots = COMPLEXITY_DOTS[v] || "";
            var color = COMPLEXITY_COLORS[v] || "#888";
            return '<span class="complexity-indicator" style="color:' + color + ';">' + dots + ' ' + v + '</span>';
          }}
        }},
      ],
    }});

    function applyDateBanding() {{
      var rows = table.getRows("active");
      var band = false;
      var prevDate = null;
      rows.forEach(function(row) {{
        var d = row.getData().date;
        if (d !== prevDate) {{ band = !band; prevDate = d; }}
        var el = row.getElement();
        if (band) {{ el.classList.add("date-band"); }}
        else {{ el.classList.remove("date-band"); }}
      }});
    }}

    var _tableReady = false;
    table.on("dataFiltered", function(filters, rows) {{
      updateKpis(rows);
      if (!_tableReady) return;
      var hf = table.getHeaderFilters();
      hf.forEach(function(f) {{
        if (f.value && String(f.value).length > 0) {{
          track('Filter Applied', {{column: f.field, value: String(f.value)}});
        }}
      }});
    }});

    table.on("dataSorted", function(sorters, rows) {{
      applyDateBanding();
      if (!_tableReady || !sorters.length) return;
      track('Table Sorted', {{column: sorters[0].field, direction: sorters[0].dir}});
    }});
    table.on("renderComplete", applyDateBanding);

    table.on("tableBuilt", function() {{
      updateKpis(table.getRows("active"));
      _tableReady = true;
    }});
  }});
</script>
"""

    return page_shell("Claude Code Changelog \u2014 Explorer", "explorer", body, extra_head=extra_head)


# ── Analysis Page (ECharts) ──────────────────────────────────────────────────

ECHARTS_JS = """
<script>
var BASE_TEXT = {fontFamily: '"IBM Plex Mono", monospace', color: '#2c2c2c', fontSize: 11};
var TITLE_TEXT = {fontFamily: '"Newsreader", Georgia, serif', fontSize: 13, color: '#2c2c2c', fontWeight: 'normal'};

function initChart(id, option) {
  var dom = document.getElementById(id);
  if (!dom) return null;
  var chart = echarts.init(dom);
  chart.setOption(option);
  return chart;
}

var charts = [];
window.addEventListener('resize', function() {
  charts.forEach(function(c) { if (c) c.resize(); });
});

fetch('data/analysis.json')
  .then(function(r) { return r.json(); })
  .then(function(data) {
    var cc = data.colors.category;
    var tc = data.colors.type;
    var xc = data.colors.complexity;
    var ac = data.colors.audience;

    // KPIs
    var k = data.kpi;
    var kpis = [
      ['Entries', k.entries.toLocaleString()],
      ['Versions', k.versions.toLocaleString()],
      ['Range', k.date_min + ' \\u2013 ' + k.date_max],
      ['Top Category', k.top_category],
      ['Bugfix %', k.bugfix_pct + '%'],
      ['Top Audience', (k.top_audience || '').replace(/_/g, ' ')],
    ];
    var sep = '<span class="text-border">\\u00b7</span>';
    document.getElementById('kpi-row').innerHTML = kpis.map(function(s) {
      return '<span class="whitespace-nowrap"><span class="text-text-secondary">' + s[0] + '</span> <span class="font-semibold text-text-primary">' + s[1] + '</span></span>';
    }).join(' ' + sep + ' ');

    // 1. Dimension Trends (stacked area) with dimension pills + count/% toggle
    var dimColors = {category: cc, change_type: tc, complexity: xc, audience: ac};
    var dimLabels = {category: 'Category', change_type: 'Change Type', complexity: 'Complexity', audience: 'Audience'};
    var activeDim = 'category';
    var trendMode = 'count';

    function getTrendData() { return data.dimension_trends[activeDim]; }

    function getMonthlyTotals() {
      var td = getTrendData();
      var keys = Object.keys(td.series);
      return td.months.map(function(m, i) {
        var total = 0;
        keys.forEach(function(k) { total += td.series[k][i]; });
        return total;
      });
    }

    function makeTrendSeries(mode) {
      var td = getTrendData();
      var colors = dimColors[activeDim];
      var totals = getMonthlyTotals();
      return Object.keys(td.series).map(function(key) {
        return {
          name: key.replace(/_/g, ' '),
          type: 'line',
          stack: 'total',
          areaStyle: {opacity: 0.7},
          symbol: 'none',
          lineStyle: {width: 1},
          itemStyle: {color: colors[key]},
          data: td.months.map(function(m, i) {
            var val = td.series[key][i];
            if (mode === 'pct') val = totals[i] > 0 ? Math.round(val / totals[i] * 1000) / 10 : 0;
            return [m, val];
          }),
        };
      });
    }

    // Build pill buttons
    var pillContainer = document.getElementById('dim-pills');
    Object.keys(dimLabels).forEach(function(dim) {
      var btn = document.createElement('button');
      btn.textContent = dimLabels[dim];
      btn.dataset.dim = dim;
      btn.className = 'text-[0.7rem] font-mono px-2 py-0.5 rounded cursor-pointer transition-colors '
        + (dim === activeDim
          ? 'bg-text-primary text-cream border border-text-primary'
          : 'bg-cream text-text-secondary border border-border hover:text-text-primary hover:border-divider');
      btn.addEventListener('click', function() { switchDimension(dim); });
      pillContainer.appendChild(btn);
    });

    var trendChart = initChart('chart-dimension-trends', {
      textStyle: BASE_TEXT,
      tooltip: {trigger: 'axis', confine: true, order: 'seriesDesc'},
      legend: {type: 'scroll', bottom: 0, textStyle: {fontSize: 10, fontFamily: '"IBM Plex Mono", monospace'}},
      grid: {left: 40, right: 20, top: 30, bottom: 50},
      xAxis: {type: 'time', axisLine: {lineStyle: {color: '#e8e5de'}}, axisLabel: {fontSize: 10}, splitLine: {lineStyle: {color: '#e8e5de'}}},
      yAxis: {type: 'value', name: 'Entries', nameGap: 10, axisLine: {show: false}, splitLine: {lineStyle: {color: '#e8e5de'}}},
      series: makeTrendSeries('count'),
    });
    charts.push(trendChart);

    function switchDimension(dim) {
      if (dim === activeDim) return;
      activeDim = dim;
      // Update pill styles
      pillContainer.querySelectorAll('button').forEach(function(btn) {
        if (btn.dataset.dim === dim) {
          btn.className = 'text-[0.7rem] font-mono px-2 py-0.5 rounded cursor-pointer transition-colors bg-text-primary text-cream border border-text-primary';
        } else {
          btn.className = 'text-[0.7rem] font-mono px-2 py-0.5 rounded cursor-pointer transition-colors bg-cream text-text-secondary border border-border hover:text-text-primary hover:border-divider';
        }
      });
      trendChart.setOption({
        yAxis: {name: trendMode === 'count' ? 'Entries' : '% of Entries', max: trendMode === 'pct' ? 100 : null},
        series: makeTrendSeries(trendMode),
      }, {replaceMerge: ['series']});
      track('Dimension Switch', {dimension: dim});
    }

    // Wire up count/% toggle
    var toggleBtn = document.getElementById('trend-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', function() {
        trendMode = trendMode === 'count' ? 'pct' : 'count';
        toggleBtn.textContent = trendMode === 'count' ? 'Show %' : 'Show #';
        trendChart.setOption({
          yAxis: {name: trendMode === 'count' ? 'Entries' : '% of Entries', max: trendMode === 'pct' ? 100 : null},
          series: makeTrendSeries(trendMode),
        });
        track('Trend Toggle', {mode: trendMode});
      });
    }

    // 2. Release Cadence
    charts.push(initChart('chart-release-cadence', {
      title: {text: 'Release Cadence (versions per week)', textStyle: TITLE_TEXT},
      textStyle: BASE_TEXT,
      tooltip: {trigger: 'axis', confine: true},
      grid: {left: 40, right: 20, top: 45, bottom: 35},
      xAxis: {type: 'time', axisLine: {lineStyle: {color: '#e8e5de'}}, axisLabel: {fontSize: 10}, splitLine: {lineStyle: {color: '#e8e5de'}}},
      yAxis: {type: 'value', name: 'Versions', axisLine: {show: false}, splitLine: {lineStyle: {color: '#e8e5de'}}},
      series: [{
        type: 'bar',
        itemStyle: {color: '#6b7280'},
        data: data.release_cadence.weeks.map(function(w, i) {
          return [w, data.release_cadence.versions[i]];
        }),
      }],
    }));

    // 3. Category Distribution (horizontal bar)
    charts.push(initChart('chart-category-dist', {
      title: {text: 'Category Distribution', textStyle: TITLE_TEXT},
      textStyle: BASE_TEXT,
      tooltip: {trigger: 'axis', confine: true, axisPointer: {type: 'shadow'}},
      grid: {left: 100, right: 60, top: 45, bottom: 20},
      xAxis: {type: 'value', name: 'Entries', splitLine: {lineStyle: {color: '#e8e5de'}}},
      yAxis: {type: 'category', data: data.category_dist.categories, axisLabel: {fontSize: 10}},
      series: [{
        type: 'bar',
        data: data.category_dist.counts.map(function(v, i) {
          return {value: v, itemStyle: {color: cc[data.category_dist.categories[i]] || '#888'}};
        }),
        label: {show: true, position: 'right', fontSize: 10, fontFamily: '"IBM Plex Mono", monospace'},
      }],
    }));

    // 4. Change Type Donut
    charts.push(initChart('chart-change-type', {
      title: {text: 'Change Type', textStyle: TITLE_TEXT},
      textStyle: BASE_TEXT,
      tooltip: {trigger: 'item', confine: true, formatter: '{b}: {c} ({d}%)'},
      legend: {bottom: 0, textStyle: {fontSize: 10, fontFamily: '"IBM Plex Mono", monospace'}},
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '45%'],
        avoidLabelOverlap: false,
        label: {show: false},
        data: data.change_type.labels.map(function(l, i) {
          return {value: data.change_type.values[i], name: l, itemStyle: {color: tc[l] || '#888'}};
        }),
      }],
    }));

    // 5. Complexity Donut
    charts.push(initChart('chart-complexity', {
      title: {text: 'Complexity', textStyle: TITLE_TEXT},
      textStyle: BASE_TEXT,
      tooltip: {trigger: 'item', confine: true, formatter: '{b}: {c} ({d}%)'},
      legend: {bottom: 0, textStyle: {fontSize: 10, fontFamily: '"IBM Plex Mono", monospace'}},
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '45%'],
        avoidLabelOverlap: false,
        label: {show: false},
        data: data.complexity.labels.map(function(l, i) {
          return {value: data.complexity.values[i], name: l, itemStyle: {color: xc[l] || '#888'}};
        }),
      }],
    }));

    // 6. Audience Donut
    charts.push(initChart('chart-audience', {
      title: {text: 'Audience', textStyle: TITLE_TEXT},
      textStyle: BASE_TEXT,
      tooltip: {trigger: 'item', confine: true, formatter: function(p) {
        return p.name.replace(/_/g, ' ') + ': ' + p.value + ' (' + p.percent + '%)';
      }},
      legend: {bottom: 0, textStyle: {fontSize: 10, fontFamily: '"IBM Plex Mono", monospace'}, formatter: function(n) { return n.replace(/_/g, ' '); }},
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '45%'],
        avoidLabelOverlap: false,
        label: {show: false},
        data: data.audience.labels.map(function(l, i) {
          return {value: data.audience.values[i], name: l, itemStyle: {color: ac[l] || '#888'}};
        }),
      }],
    }));

    // 7. Bugfix Ratio (dual axis)
    charts.push(initChart('chart-bugfix-ratio', {
      title: {text: 'Bugfix Ratio Over Time', textStyle: TITLE_TEXT},
      textStyle: BASE_TEXT,
      tooltip: {trigger: 'axis', confine: true},
      legend: {data: ['Bugfix %', 'Total entries'], top: 25, textStyle: {fontSize: 10, fontFamily: '"IBM Plex Mono", monospace'}},
      grid: {left: 50, right: 50, top: 60, bottom: 35},
      xAxis: {type: 'time', axisLine: {lineStyle: {color: '#e8e5de'}}, splitLine: {lineStyle: {color: '#e8e5de'}}},
      yAxis: [
        {type: 'value', name: 'Bugfix %', splitLine: {lineStyle: {color: '#e8e5de'}}},
        {type: 'value', name: 'Total entries', splitLine: {show: false}},
      ],
      series: [
        {
          name: 'Bugfix %',
          type: 'line',
          yAxisIndex: 0,
          symbol: 'circle',
          symbolSize: 6,
          itemStyle: {color: tc['bugfix']},
          data: data.bugfix_ratio.months.map(function(m, i) {
            return [m, data.bugfix_ratio.pct[i]];
          }),
        },
        {
          name: 'Total entries',
          type: 'bar',
          yAxisIndex: 1,
          itemStyle: {color: '#a89b7b', opacity: 0.35},
          data: data.bugfix_ratio.months.map(function(m, i) {
            return [m, data.bugfix_ratio.total[i]];
          }),
        },
      ],
    }));

    // 7. Heatmap (Category x Change Type)
    charts.push(initChart('chart-heatmap', {
      title: {text: 'Category \\u00d7 Change Type (row %)', textStyle: TITLE_TEXT},
      textStyle: BASE_TEXT,
      tooltip: {confine: true, formatter: function(p) { return p.data[2].toFixed(0) + '%'; }},
      grid: {left: 100, right: 60, top: 45, bottom: 60},
      xAxis: {type: 'category', data: data.heatmap.change_types, axisLabel: {fontSize: 10, interval: 0, rotate: 45}, splitArea: {show: true}},
      yAxis: {type: 'category', data: data.heatmap.categories, axisLabel: {fontSize: 10}, splitArea: {show: true}},
      visualMap: {
        min: 0, max: 100,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        inRange: {color: ['#faf9f6', '#c4382a']},
        textStyle: {fontSize: 10},
        show: false,
      },
      series: [{
        type: 'heatmap',
        data: data.heatmap.data,
        label: {show: true, fontSize: 10, formatter: function(p) { return p.data[2].toFixed(0) + '%'; }},
        emphasis: {itemStyle: {shadowBlur: 5, shadowColor: 'rgba(0,0,0,0.2)'}},
      }],
    }));

    // 8. Major Changes Timeline (scatter)
    var majorCategories = [...new Set(data.major_changes.map(function(d) { return d.category; }))].sort();
    charts.push(initChart('chart-major-changes', {
      title: {text: 'Major Changes Timeline', textStyle: TITLE_TEXT},
      textStyle: BASE_TEXT,
      tooltip: {
        confine: true,
        formatter: function(p) {
          var d = data.major_changes[p.dataIndex];
          return '<b>' + d.date + '</b><br><div style="max-width:300px;white-space:normal;word-wrap:break-word;">' + d.text + '</div>';
        },
      },
      grid: {left: 100, right: 20, top: 45, bottom: 35},
      xAxis: {type: 'time', axisLine: {lineStyle: {color: '#e8e5de'}}, splitLine: {lineStyle: {color: '#e8e5de'}}},
      yAxis: {type: 'category', data: majorCategories, axisLabel: {fontSize: 10}},
      series: [{
        type: 'scatter',
        symbolSize: 10,
        data: data.major_changes.map(function(d) {
          return {
            value: [d.date, d.category],
            itemStyle: {color: cc[d.category] || '#888'},
          };
        }),
      }],
    }));
  });
</script>
"""


def render_analysis_page() -> str:
    extra_head = """
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
"""

    body = f"""<div id="kpi-row" class="text-[0.82rem] leading-8 pb-2 flex flex-wrap gap-x-2 max-sm:text-[0.75rem]">
  <span class="text-text-secondary">Loading...</span>
</div>

<div class="pt-6 pb-2">
  <h2 class="font-serif text-[1.1rem] font-normal text-text-secondary tracking-wide mb-3">Timeline Trends</h2>
  <div class="grid grid-cols-1 gap-4">
    <div class="border border-border p-2 bg-cream">
      <div class="flex flex-wrap items-center gap-1 mb-1 px-1">
        <div id="dim-pills" class="flex gap-1"></div>
        <div class="flex-1"></div>
        <button id="trend-toggle" class="text-[0.7rem] font-mono px-2 py-0.5 border border-border rounded text-text-secondary hover:text-text-primary hover:border-divider cursor-pointer bg-cream">Show %</button>
      </div>
      <div id="chart-dimension-trends" class="h-[400px]"></div>
    </div>
    <div class="border border-border p-2 bg-cream"><div id="chart-release-cadence" class="h-[350px]"></div></div>
  </div>
</div>

<div class="pt-6 pb-2">
  <h2 class="font-serif text-[1.1rem] font-normal text-text-secondary tracking-wide mb-3">Distributions</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <div class="border border-border p-2 bg-cream md:col-span-2"><div id="chart-category-dist" class="h-[400px]"></div></div>
    <div class="border border-border p-2 bg-cream"><div id="chart-change-type" class="h-[350px]"></div></div>
    <div class="border border-border p-2 bg-cream"><div id="chart-complexity" class="h-[350px]"></div></div>
    <div class="border border-border p-2 bg-cream md:col-span-2"><div id="chart-audience" class="h-[350px]"></div></div>
  </div>
</div>

<div class="pt-6 pb-2">
  <h2 class="font-serif text-[1.1rem] font-normal text-text-secondary tracking-wide mb-3">Deeper Analysis</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <div class="border border-border p-2 bg-cream md:col-span-2"><div id="chart-bugfix-ratio" class="h-[350px]"></div></div>
    <div class="border border-border p-2 bg-cream"><div id="chart-heatmap" class="h-[400px]"></div></div>
    <div class="border border-border p-2 bg-cream"><div id="chart-major-changes" class="h-[400px]"></div></div>
  </div>
</div>

{ECHARTS_JS}
"""

    return page_shell("Claude Code Changelog \u2014 Analysis", "analysis", body, extra_head=extra_head)


# ── About Page ───────────────────────────────────────────────────────────────


def _enrichment_prompt_html() -> str:
    """Return the enrichment system prompt as HTML-escaped text."""
    from enrich import SYSTEM_PROMPT
    return html_escape(SYSTEM_PROMPT)


def render_about_page() -> str:
    extra_head = """<style>
.about h2 { font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem; font-weight: 400; margin: 2.5rem 0 0.75rem; color: #2c2c2c; }
.about h2:first-child { margin-top: 0; }
.about p { margin: 0.6rem 0; color: #444; line-height: 1.8; font-size: 0.82rem; }
.about ul { margin: 0.4rem 0 0.6rem 1.2rem; color: #444; font-size: 0.82rem; line-height: 1.8; }
.about li { margin: 0.15rem 0; }
.about code { background: #f0ece4; padding: 1px 5px; border-radius: 3px; font-size: 0.78rem; }
.about a { color: #2e6eb8; text-decoration: underline; text-underline-offset: 3px; }
.about a:hover { color: #1a4a7a; }
.about .dim { color: #888; }
.about hr { border: none; border-top: 1px solid #d5d0c8; margin: 2.5rem 0; }
.about details { margin: 0.6rem 0; }
.about summary { cursor: pointer; color: #2e6eb8; font-size: 0.82rem; }
.about summary:hover { color: #1a4a7a; }
.about pre.prompt { background: #f0ece4; padding: 16px; border-radius: 6px; font-size: 0.75rem; line-height: 1.7; overflow-x: auto; white-space: pre-wrap; word-break: break-word; color: #444; margin-top: 0.5rem; }
</style>"""

    body = """\
<div class="about max-w-[680px] mx-auto pb-12">

<h2>What is this?</h2>
<p>
  This site tracks every change to
  <a href="https://github.com/anthropics/claude-code">Claude Code</a>,
  Anthropic's official CLI for Claude. It automatically fetches the
  CHANGELOG.md from the repository, parses each entry, enriches it with
  LLM-derived classifications, and presents the results as an interactive,
  browsable site.
</p>

<h2>Data source</h2>
<p>
  Raw data comes from two places: the
  <a href="https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md">CHANGELOG.md</a>
  file in the Claude Code repository (fetched via the <code>gh</code> CLI),
  and version publication dates from the npm registry. Each changelog entry is
  associated with its version number and release date.
</p>

<h2>Enrichment pipeline</h2>
<p>
  After parsing, every entry is classified by Claude (Opus) along four
  dimensions:
</p>
<ul>
  <li><strong>Category</strong> &mdash; the area of Claude Code affected:
    <span class="dim">terminal, input, slash_commands, sessions, mcp, voice, auth,
    ide, hooks, permissions, performance, agents, plugins, config, api, sdk,
    other</span></li>
  <li><strong>Change type</strong> &mdash;
    <span class="dim">feature, bugfix, improvement, breaking, internal</span></li>
  <li><strong>Complexity</strong> &mdash;
    <span class="dim">minor, moderate, major</span></li>
  <li><strong>Audience</strong> &mdash; who the change matters most to:
    <span class="dim">interactive_user, sdk_developer, admin,
    extension_developer</span></li>
</ul>
<p>
  Both enrichment and embedding stages are incremental &mdash; only new or
  changed entries are processed on each run.
</p>
<details>
  <summary>View the full classification prompt</summary>
  <pre class="prompt">PROMPT_PLACEHOLDER</pre>
</details>

<h2>Embeddings &amp; the Map</h2>
<p>
  Each entry is embedded into a 512-dimensional vector using
  <a href="https://cohere.com/">Cohere</a>'s embed-v4.0 model.
  These high-dimensional vectors are then projected into 2D with
  <a href="https://umap-learn.readthedocs.io/">UMAP</a> for the
  <a href="map.html">Map</a> visualization. Semantic topic labels on the map
  are generated by
  <a href="https://github.com/TutteInstitute/Toponymy">Toponymy</a>
  producing hierarchical labels from coarse themes down
  to fine-grained topics.
</p>

<h2>Site generation</h2>
<p>
  The site is fully static &mdash; all pages are generated by Python scripts
  and served via GitHub Pages. Data is pre-computed into JSON files and
  rendered client-side:
</p>
<ul>
  <li><a href="index.html">Explorer</a> &mdash;
    interactive table powered by <a href="https://tabulator.info/">Tabulator</a></li>
  <li><a href="analysis.html">Analysis</a> &mdash;
    charts built with <a href="https://echarts.apache.org/">ECharts</a></li>
  <li><a href="map.html">Map</a> &mdash;
    2D scatter plot via <a href="https://github.com/TutteInstitute/datamapplot">DataMapPlot</a>
    (deck.gl + d3)</li>
</ul>
<p>
  The pipeline runs automatically and regenerates the site whenever the
  upstream changelog is updated.
</p>

<hr>

<p>
  This is an independent community project, not affiliated with or endorsed by
  Anthropic. Built by Steven Fazzio. Source code is on
  <a href="https://github.com/stevenfazzio/claude-code-changelog-analysis">GitHub</a>
  &mdash; feel free to
  <a href="https://github.com/stevenfazzio/claude-code-changelog-analysis/issues">open an issue</a>
  if you spot a bug or have a suggestion.
</p>
<p>
  Released under the
  <a href="https://github.com/stevenfazzio/claude-code-changelog-analysis/blob/main/LICENSE">MIT License</a>
  &copy; 2026 Steven Fazzio.
</p>

</div>"""

    body = body.replace("PROMPT_PLACEHOLDER", _enrichment_prompt_html())
    return page_shell("Claude Code Changelog — About", "about", body, extra_head=extra_head)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"Reading {INPUT_PATH}")
    df = load_data()
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print("Writing data files...")
    meta = {"generated_at": f"{datetime.now():%Y-%m-%d %H:%M}"}
    meta_json = json.dumps(meta, separators=(",", ":"))
    (DATA_DIR / "meta.json").write_text(meta_json)
    print(f"  \u2192 data/meta.json ({len(meta_json):,} bytes)")

    entries_data = generate_entries_json(df)
    entries_json = json.dumps(entries_data, separators=(",", ":"))
    (DATA_DIR / "entries.json").write_text(entries_json)
    print(f"  \u2192 data/entries.json ({len(entries_json):,} bytes)")

    analysis_data = generate_analysis_json(df)
    analysis_json = json.dumps(analysis_data, separators=(",", ":"))
    (DATA_DIR / "analysis.json").write_text(analysis_json)
    print(f"  \u2192 data/analysis.json ({len(analysis_json):,} bytes)")

    print("Building explorer page...")
    explorer = render_explorer_page(df)
    (OUTPUT_DIR / "index.html").write_text(explorer)
    print(f"  \u2192 index.html ({len(explorer):,} bytes)")

    print("Building analysis page...")
    analysis = render_analysis_page()
    (OUTPUT_DIR / "analysis.html").write_text(analysis)
    print(f"  \u2192 analysis.html ({len(analysis):,} bytes)")

    print("Building about page...")
    about = render_about_page()
    (OUTPUT_DIR / "about.html").write_text(about)
    print(f"  \u2192 about.html ({len(about):,} bytes)")

    print("Done \u2014 3 pages + 3 data files written to docs/")


if __name__ == "__main__":
    main()
