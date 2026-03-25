"""Generate interactive DataMapPlot visualization for the map page."""

import re
from html import escape
from pathlib import Path

import datamapplot

from nav import NAV_CSS, nav_html
import glasbey
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "map_data.parquet"
OUTPUT_PATH = ROOT / "docs" / "map.html"

# ── Color palettes (must match dashboard.py) ─────────────────────────────────

CATEGORIES = [
    "cli", "config", "mcp", "agents", "other", "performance",
    "ide", "permissions", "voice", "auth", "hooks", "plugins", "api",
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
    MAX_HOVER_CHARS = 200
    hover_text = []
    for t in df["text"]:
        t = str(t).strip()
        if len(t) > MAX_HOVER_CHARS:
            t = t[:MAX_HOVER_CHARS].rsplit(" ", 1)[0] + "…"
        hover_text.append(t)

    hover_text_html_template = (
        '<div class="hc">'
        '  <div class="hc-text">{hover_text}</div>'
        '  <div class="hc-meta">'
        '    <span class="hc-chip">{version}</span>'
        '    <span class="hc-chip">{date}</span>'
        '  </div>'
        '  <div class="hc-classify">'
        '    <span class="hc-cat" style="background:{category_color}30">{category}</span>'
        '    <span class="hc-type" style="background:{type_color}30">{change_type}</span>'
        '    <span class="hc-complexity">{complexity}</span>'
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

    extra_data = pd.DataFrame({
        "version": _esc(df["version"].values),
        "date": dates,
        "category": _esc(categories),
        "category_color": category_colors,
        "change_type": _esc(change_types),
        "type_color": type_colors,
        "complexity": _esc(complexities),
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

    # 4. Date (temporal)
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
        font-size: 12px;
        font-weight: 400;
        color: #2c2c2c !important;
        background: linear-gradient(135deg, #faf9f6e8, #f4f2ece8) !important;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 10px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04);
        max-width: 360px;
        padding: 0 !important;
        overflow: hidden;
    """

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Newsreader:opsz,wght@6..72,400&display=swap');

    body { background: #faf9f6 !important; }
    #title-container { font-family: 'Newsreader', Georgia, serif !important; }
    #main-title { font-weight: 400 !important; letter-spacing: -0.01em; }

    .hc {
        padding: 12px 14px 10px;
    }
    .hc-text {
        font-size: 12px;
        line-height: 1.5;
        color: #2c2c2c;
        margin-bottom: 8px;
    }
    .hc-meta {
        display: flex;
        gap: 6px;
        margin-bottom: 8px;
    }
    .hc-chip {
        display: inline-flex;
        align-items: center;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        background: rgba(0, 0, 0, 0.04);
        color: #555;
        white-space: nowrap;
    }
    .hc-chip:empty { display: none; }
    .hc-classify {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .hc-cat, .hc-type {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #2c2c2c;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .hc-cat:empty, .hc-type:empty { display: none; }
    .hc-complexity {
        font-size: 10px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .hc-complexity:empty { display: none; }
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
    var vh = document.querySelector('[style*="100vh"]');
    if (vh) { vh.style.height = 'calc(100vh - ' + h + 'px)'; }
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
        f'{NAV_CSS}\n'
        f'{fixed_wrapper_css}\n'
    )
    html = html.replace("</head>", f'{head_inject}</head>', 1)

    # Inject nav after <body> tag, plus JS to dynamically offset map content
    html = html.replace("<body>", f"<body>{nav_block}", 1)
    html = html.replace("</body>", f"{nav_resize_js}</body>", 1)

    Path(html_path).write_text(html)


if __name__ == "__main__":
    main()
