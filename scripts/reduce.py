"""Reduce embeddings to 2D with UMAP and generate topic labels with Toponymy."""

import os
from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd
import umap
from dotenv import load_dotenv

# Monkey-patch fast_hdbscan for numpy 2.x compatibility: structured array
# attribute access (e.g., arr.idx_start) was removed in numpy 2.0; use arr['idx_start'].
import fast_hdbscan.numba_kdtree as _nkd

def _kdtree_to_numba_patched(sklearn_kdtree):
    data, idx_array, node_data, node_bounds = sklearn_kdtree.get_arrays()
    return _nkd.NumbaKDTree(
        data, idx_array,
        node_data["idx_start"], node_data["idx_end"],
        node_data["radius"], node_data["is_leaf"],
        node_bounds,
    )

_nkd.kdtree_to_numba = _kdtree_to_numba_patched

# Also patch parallel_boruvka: fast_hdbscan 0.3 added a required n_threads arg
# but toponymy 0.4 doesn't pass it.
import fast_hdbscan.boruvka as _boruvka
import toponymy.clustering as _tc

_orig_boruvka = _boruvka.parallel_boruvka

def _boruvka_patched(tree, n_threads=1, **kwargs):
    return _orig_boruvka(tree, n_threads, **kwargs)

_tc.parallel_boruvka = _boruvka_patched

from toponymy import Toponymy, ToponymyClusterer
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

# Load centralized credentials, then project-local overrides
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "embeddings.parquet"
OUTPUT_PATH = ROOT / "data" / "map_data.parquet"


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} entries from {INPUT_PATH}")

    # Check for existing output with same row count (skip if unchanged)
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        if len(existing) == len(df):
            print("map_data.parquet already up to date, skipping.")
            return
        print(f"Row count changed ({len(existing)} → {len(df)}), recomputing.")

    # Extract embedding matrix
    emb_cols = [f"emb_{i}" for i in range(512)]
    embeddings = df[emb_cols].values.astype(np.float32)
    print(f"Embedding matrix: {embeddings.shape}")

    # ── UMAP reduction ───────────────────────────────────────────────────────
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    print(f"UMAP complete: x=[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}], "
          f"y=[{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")

    # ── Toponymy clustering + labeling ───────────────────────────────────────
    print("Running Toponymy clustering...")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    co_key = os.environ.get("CO_API_KEY")
    if not anthropic_key or not co_key:
        raise EnvironmentError("ANTHROPIC_API_KEY and CO_API_KEY required for Toponymy labeling")

    llm = AsyncAnthropicNamer(api_key=anthropic_key, model="claude-sonnet-4-20250514")
    embedder = CohereEmbedder(api_key=co_key, model="embed-v4.0")
    clusterer = ToponymyClusterer(min_clusters=4)

    np.random.seed(42)

    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    # Use entry text as documents for Toponymy
    documents = df["text"].tolist()

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Claude Code changelog entries",
        corpus_description="changelog entries from Claude Code, Anthropic's AI coding assistant CLI",
        exemplar_delimiters=['    * "', '"\n'],
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )

    # Extract per-document labels (layer 0 = finest, last = coarsest)
    # DataMapPlot expects coarsest first, so reverse
    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("No cluster layers found")
    print(f"Toponymy produced {n_layers} cluster layer(s)")

    # ── Build output dataframe ───────────────────────────────────────────────
    result = df.copy()
    result["umap_x"] = coords[:, 0]
    result["umap_y"] = coords[:, 1]

    for i, layer in enumerate(reversed(topic_model.cluster_layers_)):
        col = f"label_layer_{i}"
        result[col] = layer.topic_name_vector
        unique_labels = sorted(set(layer.topic_name_vector))
        print(f"  {col}: {len(unique_labels)} topics — {unique_labels[:5]}{'...' if len(unique_labels) > 5 else ''}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
