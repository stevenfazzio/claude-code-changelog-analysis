# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A site for staying up-to-date on the latest changes to Claude Code. Fetches CHANGELOG.md from `anthropics/claude-code`, parses entries, enriches them with LLM-derived classifications, generates embeddings, and produces a browsable site. Secondary goals include identifying trends, discovering notable changes, finding groups of similar changes, and exploring the space of changes.

## Pipeline

Run the full pipeline: `uv run python scripts/run_pipeline.py`

Run individual stages:
- `uv run python scripts/fetch.py` — fetches CHANGELOG.md via `gh` CLI (requires GitHub auth) and version dates from npm registry
- `uv run python scripts/parse.py` — parses changelog with version dates into `data/raw_entries.parquet`
- `uv run python scripts/enrich.py` — classifies entries using Claude Opus (`ANTHROPIC_API_KEY` required)
- `uv run python scripts/embed.py` — generates Cohere embeddings (`CO_API_KEY` required)
- `uv run python scripts/reduce.py` — UMAP 2D reduction + Toponymy topic labeling (`ANTHROPIC_API_KEY` + `CO_API_KEY`)
- `uv run python scripts/mapviz.py` — generates interactive DataMapPlot visualization
- `uv run python scripts/dashboard.py` — generates 3-page site in `docs/` (Explorer, Analysis, Map)

Standalone analysis:
- `uv run python scripts/explore_taxonomy.py` — runs Toponymy with EVoC clustering on full 512-dim embeddings (no UMAP reduction) to discover natural semantic structure; compares raw vs bugfix-direction-removed clustering; outputs `data/taxonomy_exploration.md`

Each stage reads the previous stage's output from `data/`. The enrich and embed stages support incremental updates — they skip entries that already exist in their output files.

## Data Flow

```
CHANGELOG.md + versions.json → raw_entries.parquet → enriched.parquet → embeddings.parquet → map_data.parquet
                                                         ↓                                       ↓
                                                  docs/ (index.html, analysis.html, map.html)
```

Key columns added at each stage:
- **parse**: version, text, date, prefix, is_vscode, is_breaking
- **enrich**: category, change_type, complexity, audience
- **embed**: emb_0 through emb_511 (512-dimensional Cohere embed-v4.0 vectors)

## Deployment

The site is served via GitHub Pages from the `docs/` directory on the `main` branch.

## Branching

Files in `docs/` are generated output. On feature branches, only commit changes to scripts — do not commit regenerated `docs/` files. The auto-update job on `main` will regenerate them after merge. Run the generation scripts locally to verify your changes, but leave the output uncommitted.

A `.gitattributes` merge strategy auto-resolves conflicts in `docs/map.html`. New clones need a one-time setup:

```
git config merge.ours.driver true
```

## Environment

- Python 3.12+, managed with `uv`
- Data files (`data/*.parquet`) are stored with Git LFS
- API keys loaded via `python-dotenv`: first from `~/.config/data-apis/.env`, then project-local `.env`
- `gh` CLI must be authenticated for the fetch stage

## Categories and Enums

The enrich stage classifies entries along these dimensions:
- **category**: terminal, input, slash_commands, sessions, mcp, voice, auth, ide, hooks, permissions, performance, agents, plugins, config, api, sdk, other
- **change_type**: feature, bugfix, improvement, breaking, internal
- **complexity**: minor, moderate, major
- **audience**: interactive_user, sdk_developer, admin, extension_developer
