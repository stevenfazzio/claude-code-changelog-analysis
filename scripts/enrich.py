"""Enrich changelog entries with LLM-derived classifications."""

import json
import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

# Load centralized credentials, then project-local overrides
load_dotenv(Path.home() / ".config" / "data-apis" / ".env")
load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw_entries.parquet"
OUTPUT_PATH = ROOT / "data" / "enriched.parquet"

CATEGORIES = [
    "terminal", "input", "slash_commands", "sessions",
    "mcp", "voice", "auth", "ide", "hooks", "permissions",
    "performance", "agents", "plugins", "config", "api", "sdk", "other",
]
CHANGE_TYPES = ["feature", "bugfix", "improvement", "breaking", "internal"]
COMPLEXITIES = ["minor", "moderate", "major"]
AUDIENCES = ["interactive_user", "sdk_developer", "admin", "extension_developer"]

BATCH_SIZE = 20

SYSTEM_PROMPT = """\
You classify changelog entries for Claude Code (an AI coding assistant CLI tool).

For each entry, output a JSON array of objects with these fields:
- "index": the entry's position in the input list (0-based)
- "category": one of %s
- "change_type": one of %s
- "complexity": one of %s
- "audience": one of %s

## Category guidelines

Pick the MOST SPECIFIC category that applies:
- "terminal" = terminal rendering, display, cursor, flickering, UI layout, spinner, progress indicators
- "input" = keyboard handling, text entry, vim mode, readline keybindings, CJK/IME, clipboard/paste
- "slash_commands" = slash command system, autocomplete, custom commands (/help, /config, /resume etc.)
- "sessions" = session resume, session management, remote control, conversation compaction, history
- "mcp" = MCP server management, MCP tool behavior, MCP OAuth, MCP resources
- "voice" = voice mode, audio, microphone, push-to-talk
- "auth" = authentication, OAuth, API keys, login (NOT MCP OAuth — that's "mcp")
- "ide" = VSCode extension, IDE integrations, editor features
- "hooks" = hooks system (SessionEnd, PreToolUse, PostCompact, etc.), hook configuration
- "permissions" = permission prompts, allow/deny, sandbox, tool approval
- "performance" = speed, memory, startup time, caching, token usage optimization
- "agents" = subagents (Explore, Plan, etc.), background tasks/agents, worktrees, task tools, agent teams
- "plugins" = plugin system, marketplace, plugin install/discover, skills
- "config" = settings, env vars, configuration files, managed settings, .claude/ files, CLAUDE.md
- "api" = API integration, model providers, Bedrock, Vertex, model selection, rate limits
- "sdk" = SDK features, SDK messages, SDK configuration, non-interactive/print mode (-p)
- "other" = doesn't fit any category above

Boundary rules (use these to resolve ambiguity):
- Env vars and settings → "config"; terminal display and rendering → "terminal"
- Keyboard/text input behavior → "input"; slash command system → "slash_commands"
- Session resume/management → "sessions"; general conversation flow → "sessions"
- Specific subagent behavior or worktrees → "agents"; general tool behavior → "other"
- Plugin/skill system features → "plugins"; general features → "other"
- MCP server/tool management → "mcp"; general tool usage → "other"
- SDK and -p/print mode → "sdk"; interactive CLI features → use specific category

## Change type guidelines

- "feature" = entirely new capability that didn't exist before
- "improvement" = existing capability made noticeably better (faster, cleaner, more robust, better UX)
- "bugfix" = something was broken/incorrect and is now fixed (look for "Fixed", "Fix" prefixes)
- "breaking" = backwards-incompatible change (look for "Breaking" prefix or explicit breaking notes)
- "internal" = refactoring, test changes, or infra with no user-visible effect

## Complexity guidelines

- "minor" = small tweak: single flag, typo fix, one-line behavior change, simple bugfix
- "moderate" = meaningful change: new command, new integration, workflow change, multi-component fix
- "major" = large scope: new subsystem, architectural change, major new feature, breaking change

## Audience guidelines

Who is the primary target of this change?
- "interactive_user" = someone using Claude Code interactively in a terminal (most changes)
- "sdk_developer" = someone building on the SDK, using -p/print mode, or programmatic APIs
- "admin" = system administrators configuring managed settings, permissions, or deployment
- "extension_developer" = someone building plugins, hooks, skills, or MCP integrations

Return ONLY the JSON array, no other text.""" % (
    json.dumps(CATEGORIES),
    json.dumps(CHANGE_TYPES),
    json.dumps(COMPLEXITIES),
    json.dumps(AUDIENCES),
)


DEFAULT_MODEL = "claude-opus-4-6"

FALLBACK_RESULT = {
    "category": "other",
    "change_type": "internal",
    "complexity": "minor",
    "audience": "interactive_user",
}


def enrich_batch(client: anthropic.Anthropic, entries: list[str], model: str = DEFAULT_MODEL) -> list[dict]:
    """Classify a batch of entries using the specified model."""
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(entries))

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": numbered}],
    )

    text = next(b.text for b in response.content if b.type == "text")
    # Extract JSON from response (handle potential markdown code blocks)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    results = json.loads(text)
    return results


def run_enrichment(model: str, input_path: Path, output_path: Path) -> pd.DataFrame:
    """Run enrichment on all entries, with incremental support.

    Returns the full enriched DataFrame.
    """
    df = pd.read_parquet(input_path)

    # Check for existing enriched data to enable incremental updates
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        already_enriched = set(existing["text"].tolist())
        to_enrich = df[~df["text"].isin(already_enriched)]
        print(f"Found {len(existing)} already-enriched entries, {len(to_enrich)} new entries to enrich")
        if len(to_enrich) == 0:
            print("Nothing to do.")
            return existing
    else:
        existing = None
        to_enrich = df
        print(f"Enriching {len(to_enrich)} entries with {model}...")

    client = anthropic.Anthropic()

    all_results = []
    texts = to_enrich["text"].tolist()

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} entries)...")

        try:
            results = enrich_batch(client, batch, model=model)
            # Map results back by index
            result_map = {r["index"]: r for r in results}
            for i in range(len(batch)):
                if i in result_map:
                    all_results.append(result_map[i])
                else:
                    all_results.append({"index": i, **FALLBACK_RESULT})
        except Exception as e:
            print(f"    Error: {e}, using fallback for batch")
            for i in range(len(batch)):
                all_results.append({"index": i, **FALLBACK_RESULT})

    # Attach enrichment columns to the entries being enriched
    enriched_new = to_enrich.copy()
    for field in ("category", "change_type", "complexity", "audience"):
        enriched_new[field] = [r.get(field, FALLBACK_RESULT[field]) for r in all_results]

    # Merge with existing if incremental
    if existing is not None:
        enriched = pd.concat([existing, enriched_new], ignore_index=True)
    else:
        enriched = enriched_new

    # Sort to match original order
    enriched = enriched.sort_values(["date", "version", "entry_index"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)
    print(f"\nSaved {len(enriched)} enriched entries to {output_path}")

    # Summary
    print(f"\nCategory distribution:")
    print(enriched["category"].value_counts().to_string())
    print(f"\nChange type distribution:")
    print(enriched["change_type"].value_counts().to_string())

    return enriched


def main():
    run_enrichment(DEFAULT_MODEL, INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
