"""Parse CHANGELOG.md into structured parquet using npm version dates."""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
VERSIONS_PATH = ROOT / "data" / "versions.json"
OUTPUT_PATH = ROOT / "data" / "raw_entries.parquet"

VERSION_RE = re.compile(r"^## (\d+\.\d+\.\d+)")
ENTRY_RE = re.compile(r"^- (.+)")
PREFIX_RE = re.compile(
    r"^(Added|Fixed|Improved|Changed|Removed|Updated|Disabled|Reduced|Simplified|"
    r"Suppressed|Resuming|Breaking change|New)\b",
    re.IGNORECASE,
)
VSCODE_RE = re.compile(r"^\[VSCode\]\s*")


def parse_changelog(changelog_path: Path, version_dates: dict[str, str]) -> list[dict]:
    """Parse changelog into structured entries."""
    lines = changelog_path.read_text().splitlines()
    entries = []
    current_version = None
    entry_index = 0

    for line in lines:
        version_match = VERSION_RE.match(line)
        if version_match:
            current_version = version_match.group(1)
            entry_index = 0
            continue

        if current_version is None:
            continue

        entry_match = ENTRY_RE.match(line)
        if not entry_match:
            continue

        text = entry_match.group(1).strip()

        # Detect prefix — strip leading [VSCode], **Security:**, backtick expressions, etc.
        stripped = text
        stripped = re.sub(r"^\[VSCode\]\s*", "", stripped)
        stripped = re.sub(r"^\*\*\w+:\*\*\s*", "", stripped)
        stripped = re.sub(r"^`[^`]+`\s*", "", stripped)
        prefix_match = PREFIX_RE.match(stripped)
        prefix = prefix_match.group(1) if prefix_match else None
        # Normalize prefix casing
        if prefix:
            prefix = prefix.capitalize()
            if prefix == "Breaking change":
                prefix = "Breaking"

        # Detect [VSCode] tag
        is_vscode = bool(VSCODE_RE.search(text))

        # Detect breaking change
        is_breaking = "breaking change" in text.lower() or prefix == "Breaking"

        # Get date from npm version publish time
        date = version_dates.get(current_version)

        entries.append(
            {
                "version": current_version,
                "entry_index": entry_index,
                "text": text,
                "date": date,
                "prefix": prefix,
                "is_vscode": is_vscode,
                "is_breaking": is_breaking,
            }
        )
        entry_index += 1

    return entries


def main():
    print("Loading version dates...")
    version_dates = json.loads(VERSIONS_PATH.read_text())

    print("Parsing changelog...")
    entries = parse_changelog(CHANGELOG_PATH, version_dates)

    # Warn about versions missing from npm
    versions_without_dates = {e["version"] for e in entries if e["date"] is None}
    if versions_without_dates:
        print(
            f"  Warning: {len(versions_without_dates)} versions have no npm publish date: "
            f"{sorted(versions_without_dates)}"
        )

    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["date"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} entries across {df['version'].nunique()} versions to {OUTPUT_PATH}")

    # Summary stats
    print(f"\nPrefix distribution:")
    print(df["prefix"].value_counts().to_string())
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"VSCode entries: {df['is_vscode'].sum()}")
    print(f"Breaking changes: {df['is_breaking'].sum()}")


if __name__ == "__main__":
    main()
