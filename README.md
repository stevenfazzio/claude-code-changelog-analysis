# Claude Code Changelog Analysis

A site for staying up-to-date on the latest changes to [Claude Code](https://github.com/anthropics/claude-code). Fetches `CHANGELOG.md` from the upstream repo, parses entries, enriches them with LLM-derived classifications, generates embeddings, and produces a browsable site.

**[View the dashboard](https://stevenfazzio.github.io/claude-code-changelog-analysis/)**

## Pipeline

Each stage reads the previous stage's output from `data/` and supports incremental updates where applicable.

```
CHANGELOG.md + versions.json -> raw_entries.parquet -> enriched.parquet -> embeddings.parquet
                                                          |
                                                   docs/ (index.html, analysis.html, map.html)
```

| Stage | Command | Description |
|-------|---------|-------------|
| Fetch | `uv run python scripts/fetch.py` | Fetches CHANGELOG.md via `gh` CLI and version dates from npm |
| Parse | `uv run python scripts/parse.py` | Parses changelog with version dates into structured entries |
| Enrich | `uv run python scripts/enrich.py` | Classifies entries using Claude Haiku |
| Embed | `uv run python scripts/embed.py` | Generates 512-dim Cohere embeddings |
| Dashboard | `uv run python scripts/dashboard.py` | Builds 3-page site in `docs/` |

Run all stages at once:

```sh
uv run python scripts/run_pipeline.py
```

## Setup

Requires Python 3.12+, [uv](https://github.com/astral-sh/uv), and [Git LFS](https://git-lfs.com/) (data files are stored with LFS).

```sh
uv sync
```

Create a `.env` file (see `.env.example`):

```
ANTHROPIC_API_KEY=...   # for enrich stage
CO_API_KEY=...          # for embed stage
```

The fetch stage requires the [GitHub CLI](https://cli.github.com/) (`gh`) to be authenticated.

## Site

The site is served via GitHub Pages from the `docs/` directory. Generated HTML and data files are gitignored and built/committed automatically by the [CI workflow](.github/workflows/update.yml), so `docs/` will appear mostly empty after a fresh clone.

It includes three pages:

- **Explorer** — Filterable table of all changelog entries with multiselect and date range filters
- **Analysis** — Release cadence charts, category trends, distribution breakdowns, and heatmaps
- **Map** — Semantic map of changes using Cohere embeddings with advanced filters for category, change type, complexity, and version range
