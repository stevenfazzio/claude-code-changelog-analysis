"""Run enrichment with multiple models for comparison."""

import argparse
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.enrich import run_enrichment

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw_entries.parquet"
OUTPUT_DIR = ROOT / "data" / "eval"

MODELS = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}

# Rough cost estimates for 1,822 entries (92 batches)
COST_ESTIMATES = {
    "haiku": "$0.30",
    "sonnet": "$1.10",
    "opus": "$5.50",
}


def main():
    parser = argparse.ArgumentParser(description="Run enrichment with different models")
    parser.add_argument(
        "--model",
        choices=[*MODELS.keys(), "all"],
        default="all",
        help="Which model to run (default: all)",
    )
    args = parser.parse_args()

    models_to_run = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    print("=== Model Comparison: Enrichment ===\n")
    print("Estimated costs:")
    for name in models_to_run:
        print(f"  {name:8s} {COST_ESTIMATES[name]}")
    total = sum(float(COST_ESTIMATES[n].strip("$")) for n in models_to_run)
    print(f"  {'total':8s} ${total:.2f}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model_id in models_to_run.items():
        output_path = OUTPUT_DIR / f"enriched_{name}.parquet"
        print(f"\n{'='*60}")
        print(f"Running {name} ({model_id})")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

        start = time.time()
        run_enrichment(model_id, INPUT_PATH, output_path)
        elapsed = time.time() - start
        print(f"\n{name} completed in {elapsed:.0f}s")

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
