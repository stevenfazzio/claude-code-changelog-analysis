# Claude Code Changelog Analysis

Toolkit for analyzing trends in the [Claude Code](https://github.com/anthropics/claude-code) changelog. Fetches `CHANGELOG.md` from the upstream repo, parses entries, enriches them with LLM-derived classifications, generates embeddings, and produces an interactive dashboard.

**[View the dashboard](https://stevenfazzio.github.io/claude-code-changelog-analysis/)**

## Pipeline

Each stage reads the previous stage's output from `data/` and supports incremental updates where applicable.

```
CHANGELOG.md + blame.json -> raw_entries.parquet -> enriched.parquet -> embeddings.parquet
                                                          |
                                                   docs/index.html
```

| Stage | Command | Description |
|-------|---------|-------------|
| Fetch | `uv run python scripts/fetch.py` | Fetches CHANGELOG.md and git blame data via `gh` CLI |
| Parse | `uv run python scripts/parse.py` | Parses changelog + blame into structured entries |
| Enrich | `uv run python scripts/enrich.py` | Classifies entries using Claude Haiku |
| Embed | `uv run python scripts/embed.py` | Generates 512-dim Cohere embeddings |
| Dashboard | `uv run python scripts/dashboard.py` | Builds interactive HTML dashboard |

Run all stages at once:

```sh
uv run python scripts/run_pipeline.py
```

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```sh
uv sync
```

Create a `.env` file (or place one at `~/.config/data-apis/.env`):

```
ANTHROPIC_API_KEY=...   # for enrich stage
CO_API_KEY=...          # for embed stage
```

The fetch stage requires the [GitHub CLI](https://cli.github.com/) (`gh`) to be authenticated.

## Dashboard

The dashboard is an interactive Plotly page generated at `docs/index.html` and served via GitHub Pages. It includes:

- Release cadence and category trends over time
- Distribution breakdowns by category, change type, and complexity
- Bugfix ratio analysis, category/change-type heatmap, and major changes timeline
