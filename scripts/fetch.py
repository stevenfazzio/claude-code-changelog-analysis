"""Fetch CHANGELOG.md and git blame data from the Claude Code GitHub repo."""

import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
BLAME_PATH = DATA_DIR / "blame.json"

OWNER = "anthropics"
REPO = "claude-code"
FILE_PATH = "CHANGELOG.md"
BRANCH = "main"


def gh_api(endpoint: str, *, graphql: str | None = None) -> dict:
    """Call the GitHub CLI API."""
    cmd = ["gh", "api"]
    if graphql:
        cmd += ["graphql", "-f", f"query={graphql}"]
    else:
        cmd.append(endpoint)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def fetch_changelog() -> str:
    """Fetch CHANGELOG.md content via GitHub API."""
    print("Fetching CHANGELOG.md...")
    data = gh_api(f"repos/{OWNER}/{REPO}/contents/{FILE_PATH}")
    content = base64.b64decode(data["content"]).decode("utf-8")
    CHANGELOG_PATH.write_text(content)
    print(f"  Saved {len(content):,} bytes to {CHANGELOG_PATH}")
    return content


def fetch_blame() -> list[dict]:
    """Fetch blame data via GitHub GraphQL API."""
    print("Fetching blame data...")
    query = """
    {
      repository(owner: "%s", name: "%s") {
        object(expression: "%s") {
          ... on Commit {
            blame(path: "%s") {
              ranges {
                startingLine
                endingLine
                commit {
                  committedDate
                  message
                }
              }
            }
          }
        }
      }
    }
    """ % (OWNER, REPO, BRANCH, FILE_PATH)

    data = gh_api("graphql", graphql=query)
    ranges = data["data"]["repository"]["object"]["blame"]["ranges"]

    # Simplify: keep only the fields we need
    blame = [
        {
            "start_line": r["startingLine"],
            "end_line": r["endingLine"],
            "date": r["commit"]["committedDate"],
            "message": r["commit"]["message"].split("\n")[0],
        }
        for r in ranges
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BLAME_PATH.write_text(json.dumps(blame, indent=2))
    print(f"  Saved {len(blame)} blame ranges to {BLAME_PATH}")
    return blame


def main():
    fetch_changelog()
    fetch_blame()
    print("Done.")


if __name__ == "__main__":
    main()
