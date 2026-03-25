"""Fetch CHANGELOG.md and version dates from the Claude Code GitHub repo."""

import base64
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
VERSIONS_PATH = DATA_DIR / "versions.json"

OWNER = "anthropics"
REPO = "claude-code"
FILE_PATH = "CHANGELOG.md"
NPM_REGISTRY_URL = "https://registry.npmjs.org/@anthropic-ai/claude-code"


def gh_api(endpoint: str) -> dict:
    """Call the GitHub CLI REST API."""
    result = subprocess.run(
        ["gh", "api", endpoint], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def fetch_changelog() -> str:
    """Fetch CHANGELOG.md content via GitHub API."""
    print("Fetching CHANGELOG.md...")
    data = gh_api(f"repos/{OWNER}/{REPO}/contents/{FILE_PATH}")
    content = base64.b64decode(data["content"]).decode("utf-8")
    CHANGELOG_PATH.write_text(content)
    print(f"  Saved {len(content):,} bytes to {CHANGELOG_PATH}")
    return content


def fetch_version_dates() -> dict[str, str]:
    """Fetch publish dates for each version from the npm registry."""
    print("Fetching version dates from npm registry...")
    with urllib.request.urlopen(NPM_REGISTRY_URL) as resp:
        data = json.loads(resp.read().decode())

    time_map = data["time"]
    versions = {k: v for k, v in time_map.items() if k not in ("created", "modified")}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_PATH.write_text(json.dumps(versions, indent=2))
    print(f"  Saved {len(versions)} version dates to {VERSIONS_PATH}")
    return versions


def main():
    fetch_changelog()
    fetch_version_dates()
    print("Done.")


if __name__ == "__main__":
    main()
