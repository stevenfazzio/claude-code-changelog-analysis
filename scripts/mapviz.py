"""Generate interactive DataMapPlot visualization for the map page."""

import json
import re
from html import escape
from pathlib import Path

import datamapplot

from nav import NAV_CSS, PLAUSIBLE_EVENTS_SCRIPT, PLAUSIBLE_SCRIPT, nav_html
import glasbey
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "map_data.parquet"
OUTPUT_PATH = ROOT / "docs" / "map.html"
FILTER_PANEL_HTML = ROOT / "docs" / "filter_panel.html"

# ── Color palettes (must match dashboard.py) ─────────────────────────────────

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

AUDIENCE_COLORS = {
    "interactive_user": "#7c3aed",     # violet
    "extension_developer": "#e85d04",  # orange
    "admin": "#0e7490",                # teal
    "sdk_developer": "#be185d",        # pink
}

# Marker sizes by complexity
COMPLEXITY_SIZES = {"minor": 4, "moderate": 7, "major": 12}


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} entries from {INPUT_PATH}")

    coords = df[["umap_x", "umap_y"]].values

    # ── Label layers (coarsest first) ────────────────────────────────────────
    label_columns = sorted(c for c in df.columns if c.startswith("label_layer_"))
    topic_name_vectors = [df[c].values for c in label_columns]
    print(f"Using {len(label_columns)} label layer(s)")

    # ── Hover text ───────────────────────────────────────────────────────────
    # Truncate entry text for tooltip display
    MAX_HOVER_CHARS = 400
    hover_text = []
    for t in df["text"]:
        t = str(t).strip()
        if len(t) > MAX_HOVER_CHARS:
            t = t[:MAX_HOVER_CHARS].rsplit(" ", 1)[0] + "…"
        hover_text.append(escape(t))

    hover_text_html_template = (
        '<div class="hc">'
        '  <div class="hc-header">'
        '    <span class="hc-date">{date}</span>'
        '    <span class="hc-version">{version}</span>'
        '  </div>'
        '  <div class="hc-body">{hover_text}</div>'
        '  <div class="hc-footer">'
        '    <span class="hc-label">category</span>'
        '    <span class="hc-cat" style="background:{category_color}18;color:{category_color};border-color:{category_color}30">{category}</span>'
        '    <span class="hc-label">change type</span>'
        '    <span class="hc-type" style="color:{type_color}">{type_icon} {change_type}</span>'
        '    <span class="hc-label">audience</span>'
        '    <span class="hc-audience" style="color:{audience_color}">{audience_icon} {audience}</span>'
        '    <span class="hc-label">complexity</span>'
        '    <span class="hc-complexity" style="color:{complexity_color}">{complexity_dots} {complexity}</span>'
        '  </div>'
        '</div>'
    )

    # ── Extra point data for hover template ──────────────────────────────────
    def _esc(values):
        return [escape(str(v)) for v in values]

    dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").fillna("").tolist()
    categories = df["category"].fillna("other").values
    change_types = df["change_type"].fillna("").values
    complexities = df["complexity"].fillna("minor").values

    category_colors = [CATEGORY_COLORS.get(c, "#888888") for c in categories]
    type_colors = [TYPE_COLORS.get(t, "#777777") for t in change_types]

    type_icons = {
        "feature": "✦", "bugfix": "✕", "improvement": "↑",
        "breaking": "⚠", "internal": "⚙",
    }
    complexity_dots = {
        "minor": "●○○", "moderate": "●●○", "major": "●●●",
    }
    audience_icons = {
        "interactive_user": "◉", "extension_developer": "⬡",
        "admin": "⛭", "sdk_developer": "⚒",
    }

    extra_data = pd.DataFrame({
        "version": _esc(df["version"].values),
        "date": dates,
        "category": _esc(categories),
        "category_color": category_colors,
        "change_type": _esc(change_types),
        "type_color": type_colors,
        "type_icon": [type_icons.get(t, "") for t in change_types],
        "complexity": _esc(complexities),
        "complexity_color": [COMPLEXITY_COLORS.get(c, "#888888") for c in complexities],
        "complexity_dots": [complexity_dots.get(c, "●○○") for c in complexities],
        "audience": _esc(
            [v.replace("_", " ") for v in df["audience"].fillna("interactive_user").values]
        ),
        "audience_color": [
            AUDIENCE_COLORS.get(a, "#7c3aed")
            for a in df["audience"].fillna("interactive_user").values
        ],
        "audience_icon": [
            audience_icons.get(a, "◉")
            for a in df["audience"].fillna("interactive_user").values
        ],
    })

    # ── Marker sizes (by complexity) ─────────────────────────────────────────
    marker_sizes = np.array([COMPLEXITY_SIZES.get(c, 4) for c in complexities], dtype=float)

    # ── Colormaps ────────────────────────────────────────────────────────────

    # 1. Category (categorical)
    all_rawdata = [categories]
    all_metadata = [
        {
            "field": "category",
            "description": "Category",
            "kind": "categorical",
            "color_mapping": CATEGORY_COLORS,
        },
    ]

    # 2. Change Type (categorical)
    all_rawdata.append(change_types)
    all_metadata.append({
        "field": "change_type",
        "description": "Change Type",
        "kind": "categorical",
        "color_mapping": TYPE_COLORS,
    })

    # 3. Complexity (categorical)
    all_rawdata.append(complexities)
    all_metadata.append({
        "field": "complexity",
        "description": "Complexity",
        "kind": "categorical",
        "color_mapping": COMPLEXITY_COLORS,
    })

    # 4. Audience (categorical)
    audiences = df["audience"].fillna("interactive_user").values
    all_rawdata.append(audiences)
    all_metadata.append({
        "field": "audience",
        "description": "Audience",
        "kind": "categorical",
        "color_mapping": AUDIENCE_COLORS,
    })

    # 5. Date (temporal)
    entry_dates = pd.to_datetime(df["date"], utc=True).values
    all_rawdata.append(entry_dates)
    all_metadata.append({
        "field": "date",
        "description": "Date",
        "kind": "datetime",
        "cmap": "viridis",
    })

    # ── Tooltip CSS ──────────────────────────────────────────────────────────
    tooltip_css = """
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 400;
        color: #2c2c2c !important;
        background: #faf9f6f0 !important;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 12px;
        backdrop-filter: blur(16px) saturate(1.2);
        -webkit-backdrop-filter: blur(16px) saturate(1.2);
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.10),
            0 2px 8px rgba(0, 0, 0, 0.04),
            inset 0 0 0 1px rgba(255, 255, 255, 0.5);
        max-width: 420px;
        padding: 0 !important;
        overflow: hidden;
    """

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Newsreader:opsz,wght@6..72,400&display=swap');

    body { background: #faf9f6 !important; }
    #title-container { font-family: 'Newsreader', Georgia, serif !important; }
    #main-title { font-weight: 400 !important; letter-spacing: -0.01em; }

    .hc {
        display: flex;
        flex-direction: column;
    }
    .hc-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 10px 16px 8px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    }
    .hc-date {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        color: #2c2c2c;
        letter-spacing: -0.01em;
    }
    .hc-version {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: #999;
        letter-spacing: 0.02em;
    }
    .hc-body {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 13.5px;
        line-height: 1.55;
        color: #3a3a3a;
        padding: 12px 16px;
    }
    .hc-footer {
        display: grid;
        grid-template-columns: auto 1fr auto 1fr;
        align-items: center;
        gap: 4px 8px;
        padding: 10px 16px 12px;
        border-top: 1px solid rgba(0, 0, 0, 0.05);
        background: rgba(0, 0, 0, 0.015);
    }
    .hc-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 8.5px;
        color: #aaa;
        text-transform: lowercase;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }
    /* Category: filled badge */
    .hc-cat {
        display: inline-flex;
        align-items: center;
        justify-self: start;
        padding: 2px 8px;
        border-radius: 5px;
        border: 1px solid;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
        line-height: 1.6;
    }
    .hc-cat:empty { display: none; }
    /* Type: icon + colored text, no badge */
    .hc-type {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
        justify-self: start;
    }
    .hc-type:empty { display: none; }
    /* Complexity: dots + text, no badge */
    .hc-complexity {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
        justify-self: start;
    }
    .hc-complexity:empty { display: none; }
    /* Audience: colored text, no badge */
    .hc-audience {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
        justify-self: start;
    }
    .hc-audience:empty { display: none; }

    /* Align colormap dropdown labels by giving swatch blocks a consistent width */
    .color-map-option .color-swatch {
        display: inline-block;
        min-width: 72px;
    }
    """

    # Increase scroll-to-zoom speed
    custom_js = """
    datamap.deckgl.setProps({controller: {scrollZoom: {speed: 0.05, smooth: true}}});
    """

    # ── Generate DataMapPlot ─────────────────────────────────────────────────
    print("Generating DataMapPlot...")
    fig = datamapplot.create_interactive_plot(
        coords,
        *topic_name_vectors,
        hover_text=hover_text,
        hover_text_html_template=hover_text_html_template,
        marker_size_array=marker_sizes,
        extra_point_data=extra_data,
        colormap_rawdata=all_rawdata,
        colormap_metadata=all_metadata,
        title="",
        sub_title="",
        enable_search=True,
        search_field="hover_text",
        custom_js=custom_js,
        custom_css=custom_css,
        tooltip_css=tooltip_css,
        font_family="IBM Plex Mono",
        font_weight=600,
        darkmode=False,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.save(str(OUTPUT_PATH))
    print(f"Saved raw DataMapPlot to {OUTPUT_PATH}")

    # ── Post-process: inject nav bar ─────────────────────────────────────────
    _inject_nav(OUTPUT_PATH)
    print("Injected navigation bar")

    _inject_filter_panel(OUTPUT_PATH, df)
    print("Injected filter panel")


def _inject_nav(html_path):
    """Add shared site navigation bar to DataMapPlot-generated HTML."""
    html = Path(html_path).read_text()

    # Fixed-position wrapper CSS for the map page (nav content comes from shared module)
    # Hides the <hr> and replaces it with border-bottom + padding for identical spacing.
    fixed_wrapper_css = """<style>
.site-nav-fixed {
  position: fixed; top: 0; left: 0; right: 0; z-index: 200;
  background: rgba(250, 249, 246, 0.92);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  border-bottom: 1px solid #d5d0c8;
  pointer-events: auto;
}
.site-nav-fixed .site-nav { padding-bottom: 16px; }
.site-nav-fixed .site-nav-hr { display: none; }
/* Override DataMapPlot default that hides colormap selector on mobile */
@media screen and (max-width: 768px) {
  .bottom-left { display: flex !important; }
  /* Fix mobile 100vh bug: dvh excludes browser chrome (URL bar, nav) */
  #deck-container { height: 100dvh !important; }
  .stack.bottom-left { padding-bottom: 4px !important; }
  #colormap-selector-container { max-width: calc(100vw - 24px); }
  .color-map-options { max-height: 50dvh !important; }
}
</style>"""

    # JS to measure actual nav height and offset the map content accordingly
    nav_resize_js = """<script>
(function() {
  var nav = document.querySelector('.site-nav-fixed');
  function adjust() {
    var h = nav.offsetHeight;
    document.body.style.paddingTop = h + 'px';
    var deck = document.querySelector('[style*="z-index: -1"]');
    if (deck) { deck.style.top = h + 'px'; deck.style.height = 'calc(100% - ' + h + 'px)'; }
    var cw = document.querySelector('.content-wrapper');
    var vh = CSS.supports('height', '100dvh') ? 'dvh' : 'vh';
    if (cw) { cw.style.minHeight = 'calc(100' + vh + ' - ' + h + 'px)'; }
  }
  adjust();
  window.addEventListener('resize', adjust);
})();
</script>"""

    nav_block = f'<div class="site-nav-fixed">{nav_html("map")}</div>'

    # Fix empty <title> tag left by clearing DataMapPlot's title param
    html = html.replace("<title></title>", "<title>Claude Code Changelog — Map</title>", 1)

    # Remove DataMapPlot's @font-face blocks that conflict with our font weights
    html = re.sub(
        r'@font-face\s*\{[^}]*font-family:\s*[\'"]IBM Plex Mono[\'"][^}]*\}\s*',
        '',
        html,
    )

    # Inject correct fonts, viewport meta, and nav CSS into <head>
    head_inject = (
        '<link href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;600&'
        'family=Newsreader:opsz,wght@6..72,400&display=swap" rel="stylesheet">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'{PLAUSIBLE_SCRIPT}\n'
        f'{PLAUSIBLE_EVENTS_SCRIPT}\n'
        f'{NAV_CSS}\n'
        f'{fixed_wrapper_css}\n'
    )
    html = html.replace("</head>", f'{head_inject}</head>', 1)

    # Inject nav after <body> tag, plus JS to dynamically offset map content
    html = html.replace("<body>", f"<body>{nav_block}", 1)
    html = html.replace("</body>", f"{nav_resize_js}</body>", 1)

    Path(html_path).write_text(html)


def _inject_filter_panel(html_path, df):
    """Inject the advanced filter panel into DataMapPlot-generated HTML."""
    html = Path(html_path).read_text()

    # 1. Dispatch datamapReady event after metadata finishes loading
    html = re.sub(
        r"(updateProgressBar\('meta-data-progress', 100\);\s*)(checkAllDataLoaded\(\);)",
        r"\1window.dispatchEvent(new CustomEvent('datamapReady', "
        r"{ detail: { datamap, hoverData } }));\n          \2",
        html,
        count=1,
    )

    # 2. Fix grid-template-rows to prevent filter panel from stretching the row
    html = html.replace(
        "grid-template-rows:1fr 1fr",
        "grid-template-rows:minmax(0,1fr) minmax(0,1fr)",
        1,
    )

    # 3. Compute filter config
    entry_dates = pd.to_datetime(df["date"], utc=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    date_days = ((entry_dates - epoch).dt.days)
    min_date = int(date_days.dropna().min())
    max_date = int(date_days.dropna().max())

    audience_vals = sorted(df["audience"].fillna("interactive_user").unique().tolist())
    audience_vals = [v for v in audience_vals if v]

    filter_config = {
        "totalCount": len(df),
        "categories": sorted(CATEGORIES),
        "changeTypes": sorted(TYPE_COLORS.keys()),
        "complexities": ["minor", "moderate", "major"],
        "audiences": audience_vals,
        "ranges": {
            "date": {
                "min": min_date,
                "max": max_date,
            },
        },
        "colormapFieldToFilterId": {
            "category": "filter-category",
            "change_type": "filter-change-type",
            "complexity": "filter-complexity",
        },
        "filterIdToColormapField": {
            "filter-category": "category",
            "filter-change-type": "change_type",
            "filter-complexity": "complexity",
        },
    }

    # 4. Read and split template by <!-- SECTION: xxx --> markers
    template = FILTER_PANEL_HTML.read_text()
    sections = re.split(r"<!-- SECTION: (\w+) -->", template)
    section_map = {}
    for i in range(1, len(sections), 2):
        section_map[sections[i]] = sections[i + 1].strip()

    # 5. Replace config placeholder in JS section
    js_section = section_map["js"].replace(
        "__FILTER_CONFIG_JSON__", json.dumps(filter_config)
    )

    # 6. Inject CSS before </head>
    html = html.replace("</head>", section_map["css"] + "\n</head>", 1)

    # 7. Inject HTML after search-container div
    search_pattern = re.compile(
        r'(<div id="search-container" class="container-box[^"]*">\s*'
        r"<input[^/]*/>\s*</div>)"
    )
    match = search_pattern.search(html)
    if match:
        insert_pos = match.end()
        html = html[:insert_pos] + "\n      " + section_map["html"] + "\n" + html[insert_pos:]

    # 8. Inject JS before </html>
    html = html.replace("</html>", js_section + "\n</html>", 1)

    Path(html_path).write_text(html)


if __name__ == "__main__":
    main()
