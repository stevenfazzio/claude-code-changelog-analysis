"""Run the full data pipeline: fetch → parse → enrich → embed."""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

STEPS = [
    ("Fetch", "fetch.py"),
    ("Parse", "parse.py"),
    ("Enrich", "enrich.py"),
    ("Embed", "embed.py"),
    ("Dashboard", "dashboard.py"),
]


def run_step(name: str, script: str) -> bool:
    """Run a pipeline step, return True if successful."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  {name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"\n  {name} completed in {elapsed:.1f}s")
    return True


def main():
    print("Claude Code Changelog Analysis Pipeline")
    print(f"Working directory: {ROOT}")

    start = time.time()
    for name, script in STEPS:
        if not run_step(name, script):
            print(f"\nPipeline failed at step: {name}")
            sys.exit(1)

    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
