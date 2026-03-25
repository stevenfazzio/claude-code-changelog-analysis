# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas>=2.2",
#     "pyarrow>=18.0",
#     "plotly>=5.24",
#     "numpy>=1.26",
#     "umap-learn>=0.5",
#     "datamapplot>=0.5",
#     "cohere>=5.0",
#     "python-dotenv>=1.0",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path

    return mo, pd, np, px, go, Path


@app.cell
def _(pd, np, Path):
    DATA_DIR = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(DATA_DIR / "embeddings.parquet")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    embeddings = df[emb_cols].values
    meta = df.drop(columns=emb_cols)

    meta["date"] = pd.to_datetime(meta["date"])
    meta["month"] = meta["date"].dt.tz_localize(None).dt.to_period("M").astype(str)

    return DATA_DIR, df, emb_cols, embeddings, meta


@app.cell
def _(mo):
    mo.md(
        """
        # Claude Code Changelog Explorer

        Analyzing **{n_entries}** changelog entries across **{n_versions}** versions,
        from {date_min} to {date_max}.
        """.format(
            n_entries="1,706",
            n_versions="245",
            date_min="Apr 2025",
            date_max="Mar 2026",
        )
    )
    return


@app.cell
def _(mo):
    tab = mo.ui.tabs(
        {
            "Release Timeline": "timeline",
            "Category Trends": "trends",
            "Embedding Map": "map",
            "Semantic Search": "search",
            "Version Detail": "detail",
        }
    )
    tab
    return (tab,)


# ── Release Timeline ──────────────────────────────────────────────


@app.cell
def _(meta, pd, px, mo, tab):
    mo.stop(tab.value != "timeline")

    version_stats = (
        meta.groupby(["version", "date", "change_type"])
        .size()
        .reset_index(name="count")
    )
    version_stats = version_stats.sort_values("date")

    fig_timeline = px.bar(
        version_stats,
        x="date",
        y="count",
        color="change_type",
        color_discrete_map={
            "feature": "#2196F3",
            "bugfix": "#F44336",
            "improvement": "#4CAF50",
            "breaking": "#FF9800",
            "internal": "#9E9E9E",
        },
        title="Changes per Release Over Time",
        labels={"date": "Release Date", "count": "Number of Changes", "change_type": "Type"},
        hover_data=["version"],
    )
    fig_timeline.update_layout(barmode="stack", height=500)
    fig_timeline
    return (fig_timeline,)


@app.cell
def _(meta, px, mo, tab):
    mo.stop(tab.value != "timeline")

    entries_per_version = (
        meta.groupby(["version", "date"])
        .size()
        .reset_index(name="count")
        .sort_values("date")
    )

    fig_size = px.scatter(
        entries_per_version,
        x="date",
        y="count",
        hover_data=["version"],
        title="Release Size (entries per version)",
        labels={"date": "Release Date", "count": "Entries"},
        trendline="lowess",
    )
    fig_size.update_layout(height=400)
    fig_size
    return (fig_size,)


# ── Category Trends ───────────────────────────────────────────────


@app.cell
def _(meta, pd, px, mo, tab):
    mo.stop(tab.value != "trends")

    cat_by_month = (
        meta.groupby(["month", "category"])
        .size()
        .reset_index(name="count")
        .sort_values("month")
    )

    fig_trends = px.area(
        cat_by_month,
        x="month",
        y="count",
        color="category",
        title="Feature Area Trends Over Time",
        labels={"month": "Month", "count": "Changes", "category": "Category"},
    )
    fig_trends.update_layout(height=600)
    fig_trends
    return (fig_trends,)


@app.cell
def _(meta, px, mo, tab):
    mo.stop(tab.value != "trends")

    fig_cat_bar = px.histogram(
        meta,
        x="category",
        color="change_type",
        color_discrete_map={
            "feature": "#2196F3",
            "bugfix": "#F44336",
            "improvement": "#4CAF50",
            "breaking": "#FF9800",
            "internal": "#9E9E9E",
        },
        title="Change Types by Category",
        labels={"category": "Category", "count": "Changes"},
    )
    fig_cat_bar.update_layout(height=500, barmode="stack")
    fig_cat_bar.update_xaxes(categoryorder="total descending")
    fig_cat_bar
    return (fig_cat_bar,)


# ── Embedding Map ─────────────────────────────────────────────────


@app.cell
def _(embeddings, np, mo, tab):
    mo.stop(tab.value != "map")

    import umap

    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.001,
        random_state=42,
    )
    coords_2d = reducer.fit_transform(embeddings)

    return (coords_2d,)


@app.cell
def _(coords_2d, meta, px, mo, tab):
    mo.stop(tab.value != "map")

    map_df = meta.copy()
    map_df["x"] = coords_2d[:, 0]
    map_df["y"] = coords_2d[:, 1]
    map_df["short_text"] = map_df["text"].str[:80] + "..."

    fig_map = px.scatter(
        map_df,
        x="x",
        y="y",
        color="category",
        hover_data=["version", "short_text", "change_type"],
        title="Changelog Entry Embedding Map (UMAP)",
        labels={"x": "", "y": ""},
        opacity=0.7,
    )
    fig_map.update_layout(height=700)
    fig_map.update_xaxes(showticklabels=False)
    fig_map.update_yaxes(showticklabels=False)
    fig_map
    return (fig_map,)


# ── Semantic Search ───────────────────────────────────────────────


@app.cell
def _(mo, tab):
    mo.stop(tab.value != "search")

    search_input = mo.ui.text(
        placeholder="Search changelog entries semantically...",
        label="Query",
        full_width=True,
    )
    search_input
    return (search_input,)


@app.cell
def _(search_input, embeddings, meta, np, pd, mo, tab):
    mo.stop(tab.value != "search")
    mo.stop(not search_input.value)

    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv
    import cohere as _cohere

    _load_dotenv(_Path.home() / ".config" / "data-apis" / ".env")
    _load_dotenv(override=True)

    co = _cohere.ClientV2()
    query_resp = co.embed(
        model="embed-v4.0",
        texts=[search_input.value],
        input_type="search_query",
        embedding_types=["float"],
        output_dimension=512,
    )
    query_emb = np.array(query_resp.embeddings.float_[0])

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_emb)
    similarities = embeddings @ query_emb / (norms * query_norm + 1e-10)

    top_k = 20
    top_idx = np.argsort(similarities)[::-1][:top_k]

    results = meta.iloc[top_idx].copy()
    results["similarity"] = similarities[top_idx]

    mo.ui.table(
        results[["version", "date", "category", "change_type", "text", "similarity"]],
        label=f"Top {top_k} results for: {search_input.value}",
    )
    return


# ── Version Detail ────────────────────────────────────────────────


@app.cell
def _(meta, mo, tab):
    mo.stop(tab.value != "detail")

    versions = sorted(meta["version"].unique(), key=lambda v: meta[meta["version"] == v]["date"].iloc[0], reverse=True)
    version_picker = mo.ui.dropdown(
        options=versions,
        value=versions[0],
        label="Select version",
    )
    version_picker
    return (version_picker,)


@app.cell
def _(version_picker, meta, mo, tab):
    mo.stop(tab.value != "detail")

    v = version_picker.value
    entries = meta[meta["version"] == v].sort_values("entry_index")

    release_date = entries["date"].iloc[0].strftime("%Y-%m-%d") if len(entries) > 0 else "?"
    n = len(entries)

    mo.vstack([
        mo.md(f"### Version {v} — {release_date} ({n} entries)"),
        mo.ui.table(
            entries[["entry_index", "text", "prefix", "category", "change_type", "complexity", "user_facing"]],
            label=f"v{v} entries",
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
