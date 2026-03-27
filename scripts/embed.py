"""Generate Cohere embeddings for changelog entries."""

import os
import time
from pathlib import Path

import cohere
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "enriched.parquet"
OUTPUT_PATH = ROOT / "data" / "embeddings.parquet"

MODEL = "embed-v4.0"
DIMENSIONS = 512
INPUT_TYPE = "clustering"
BATCH_SIZE = 96  # Cohere limit


def main():
    df = pd.read_parquet(INPUT_PATH)

    # Check for existing embeddings to enable incremental updates
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        already_embedded = set(existing["text"].tolist())
        to_embed_mask = ~df["text"].isin(already_embedded)
        to_embed = df[to_embed_mask]
        print(f"Found {len(existing)} already-embedded entries, {len(to_embed)} new entries to embed")
        if len(to_embed) == 0:
            print("Nothing to do.")
            return
    else:
        existing = None
        to_embed = df
        print(f"Embedding {len(to_embed)} entries...")

    co = cohere.ClientV2()  # Uses CO_API_KEY

    texts = to_embed["text"].tolist()
    all_embeddings = []

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)...")

        for attempt in range(3):
            try:
                response = co.embed(
                    model=MODEL,
                    texts=batch,
                    input_type=INPUT_TYPE,
                    embedding_types=["float"],
                    output_dimension=DIMENSIONS,
                )
                all_embeddings.extend(response.embeddings.float_)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/3 after {wait}s: {e}")
                time.sleep(wait)

    # Build output dataframe
    embedding_cols = [f"emb_{i}" for i in range(DIMENSIONS)]
    emb_array = np.array(all_embeddings)

    emb_df = pd.DataFrame(emb_array, columns=embedding_cols, index=to_embed.index)
    embedded_new = pd.concat([to_embed, emb_df], axis=1)

    # Merge with existing if incremental
    if existing is not None:
        result = pd.concat([existing, embedded_new], ignore_index=True)
    else:
        result = embedded_new

    # Sort to match original order
    result = result.sort_values(["date", "version", "entry_index"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result)} entries with {DIMENSIONS}d embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
