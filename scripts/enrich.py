"""Enrich changelog entries with LLM-derived classifications."""

import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw_entries.parquet"
OUTPUT_PATH = ROOT / "data" / "enriched.parquet"

CATEGORIES = [
    "terminal", "input", "skills", "sessions",
    "mcp", "voice", "auth", "ide", "hooks", "permissions",
    "performance", "agents", "plugins", "config", "api", "sdk", "other",
]
CHANGE_TYPES = ["feature", "bugfix", "improvement", "breaking"]
COMPLEXITIES = ["minor", "moderate", "major"]
AUDIENCES = ["interactive_user", "sdk_developer", "admin", "extension_developer"]

BATCH_SIZE = 20

SYSTEM_PROMPT = """\
<task>
You classify changelog entries for Claude Code (an AI coding assistant CLI tool).
For each entry in the input list, determine its category, change type, complexity, and audience.
Call the classify_entries tool with your classifications.
</task>

<category-guidelines>
Pick the MOST SPECIFIC category that applies:
- "terminal" = terminal rendering, display, cursor, flickering, UI layout, spinner, progress indicators
- "input" = keyboard handling, text entry, vim mode, readline keybindings, CJK/IME, clipboard/paste
- "skills" = skills framework, slash commands (built-in and custom), .claude/commands/, skill discovery/install, skill frontmatter and behavior
- "sessions" = session resume, session management, remote control, conversation compaction, history
- "mcp" = MCP server management, MCP tool behavior, MCP OAuth, MCP resources
- "voice" = voice mode, audio, microphone, push-to-talk
- "auth" = authentication, OAuth, API keys, login (NOT MCP OAuth — that's "mcp")
- "ide" = VSCode extension, IDE integrations, editor features
- "hooks" = hooks system (SessionEnd, PreToolUse, PostCompact, etc.), hook configuration
- "permissions" = permission prompts, allow/deny, sandbox, tool approval
- "performance" = speed, memory, startup time, caching, token usage optimization
- "agents" = subagents (Explore, Plan, etc.), background tasks/agents, worktrees, task tools, agent teams
- "plugins" = plugin system, marketplace, plugin install/uninstall/discover/validate, plugin configuration, plugin trust/permissions
- "config" = settings, env vars, configuration files, managed settings, .claude/ files, CLAUDE.md
- "api" = API integration, model providers, Bedrock, Vertex, model selection, rate limits
- "sdk" = SDK features, SDK messages, SDK configuration, non-interactive/print mode (-p)
- "other" = doesn't fit any category above
</category-guidelines>

<boundary-rules>
- Env vars and settings → "config"; terminal display and rendering → "terminal"
- Keyboard/text input behavior → "input"; skill/slash command behavior → "skills"
- Session resume/management → "sessions"; general conversation flow → "sessions"
- Specific subagent behavior or worktrees → "agents"; general tool behavior → "other"
- MCP server/tool management → "mcp"; general tool usage → "other"
- SDK and -p/print mode → "sdk"; interactive CLI features → use specific category
- Plugins are composed of skills, agents, hooks, and MCP servers. If a change is about exactly one of those plugin types, use that specific category ("skills", "agents", "hooks", or "mcp"). If a change is about the plugin system in general or about multiple plugin types, use "plugins".
</boundary-rules>

<change-type-guidelines>
- "feature" = entirely new capability that didn't exist before
- "improvement" = existing capability made noticeably better (faster, cleaner, more robust, better UX)
- "bugfix" = something was broken/incorrect and is now fixed (look for "Fixed", "Fix" prefixes)
- "breaking" = backwards-incompatible change (look for "Breaking" prefix, "Deprecated", "Removed", or explicit breaking notes)
</change-type-guidelines>

<complexity-guidelines>
- "minor" = small tweak: single flag, typo fix, one-line behavior change, simple bugfix
- "moderate" = meaningful change: new command, new integration, workflow change, multi-component fix
- "major" = large scope: new subsystem, architectural change, major new feature, breaking change
</complexity-guidelines>

<audience-guidelines>
Who is the primary target of this change?
- "interactive_user" = someone using Claude Code interactively in a terminal (most changes)
- "sdk_developer" = someone building on the SDK, using -p/print mode, or programmatic APIs
- "admin" = system administrators configuring managed settings, permissions, or deployment
- "extension_developer" = someone building plugins, hooks, skills, or MCP integrations
</audience-guidelines>

<examples>
Entry: "Fixed ghost text flickering when typing slash commands mid-input"
-> category="terminal", change_type="bugfix", complexity="minor", audience="interactive_user"
Reasoning: Display flickering is terminal rendering; "Fixed" indicates bugfix; single UI glitch is minor.

Entry: "Added support for running skills and slash commands in a forked sub-agent context using `context: fork` in skill frontmatter"
-> category="skills", change_type="feature", complexity="moderate", audience="extension_developer"
Reasoning: Skill frontmatter feature; new capability; targets skill authors.

Entry: "Fixed plugin path resolution for file-based marketplace sources"
-> category="plugins", change_type="bugfix", complexity="minor", audience="extension_developer"
Reasoning: Plugin marketplace infrastructure, not a specific plugin type; "Fixed" indicates bugfix.

Entry: "Improved memory usage by 3x for large conversations"
-> category="performance", change_type="improvement", complexity="major", audience="interactive_user"
Reasoning: Memory optimization; 3x improvement on existing capability; large impact.

Entry: "SDK: Renamed `total_cost` to `total_cost_usd`"
-> category="sdk", change_type="breaking", complexity="minor", audience="sdk_developer"
Reasoning: SDK API rename is breaking; small scope but incompatible; targets SDK consumers.
</examples>"""

CLASSIFY_TOOL = {
    "name": "classify_entries",
    "description": "Submit classifications for a batch of changelog entries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "The entry's position in the input list (0-based)",
                        },
                        "category": {
                            "type": "string",
                            "enum": CATEGORIES,
                        },
                        "change_type": {
                            "type": "string",
                            "enum": CHANGE_TYPES,
                        },
                        "complexity": {
                            "type": "string",
                            "enum": COMPLEXITIES,
                        },
                        "audience": {
                            "type": "string",
                            "enum": AUDIENCES,
                        },
                    },
                    "required": ["index", "category", "change_type", "complexity", "audience"],
                },
            },
        },
        "required": ["classifications"],
    },
}


DEFAULT_MODEL = "claude-opus-4-6"

FALLBACK_RESULT = {
    "category": "other",
    "change_type": "improvement",
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
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_entries"},
        messages=[{"role": "user", "content": numbered}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return tool_use.input["classifications"]


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
