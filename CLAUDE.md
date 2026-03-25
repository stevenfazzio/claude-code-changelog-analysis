# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Toolkit for analyzing trends in the Claude Code changelog. Fetches CHANGELOG.md from `anthropics/claude-code`, parses entries, enriches them with LLM-derived classifications, generates embeddings, and produces visualizations.

## Pipeline

Run the full pipeline: `uv run python scripts/run_pipeline.py`

Run individual stages:
- `uv run python scripts/fetch.py` — fetches CHANGELOG.md via `gh` CLI (requires GitHub auth) and version dates from npm registry
- `uv run python scripts/parse.py` — parses changelog with version dates into `data/raw_entries.parquet`
- `uv run python scripts/enrich.py` — classifies entries using Claude Haiku (`ANTHROPIC_API_KEY` required)
- `uv run python scripts/embed.py` — generates Cohere embeddings (`CO_API_KEY` required)
- `uv run python scripts/dashboard.py` — generates 3-page site in `docs/` (Explorer, Analysis, Map)

Each stage reads the previous stage's output from `data/`. The enrich and embed stages support incremental updates — they skip entries that already exist in their output files.

## Data Flow

```
CHANGELOG.md + versions.json → raw_entries.parquet → enriched.parquet → embeddings.parquet
                                                         ↓
                                                  docs/ (index.html, analysis.html, map.html)
```

Key columns added at each stage:
- **parse**: version, text, date, prefix, is_vscode, is_breaking
- **enrich**: category, change_type, complexity, user_facing
- **embed**: emb_0 through emb_511 (512-dimensional Cohere embed-v4.0 vectors)

## Environment

- Python 3.12+, managed with `uv`
- API keys loaded via `python-dotenv`: first from `~/.config/data-apis/.env`, then project-local `.env`
- `gh` CLI must be authenticated for the fetch stage

## Categories and Enums

The enrich stage classifies entries into: cli, mcp, voice, auth, ide, hooks, permissions, performance, agents, plugins, config, api, other. Change types: feature, bugfix, improvement, breaking, internal. Complexity: minor, moderate, major.
