"""Explore natural taxonomy of changelog entries using Toponymy + EVoC.

Runs Toponymy with EVoCClusterer on full 512-dim embeddings (no dimension
reduction) to discover semantic structure, then repeats with the bugfix
direction removed to see what additional structure emerges.

Outputs a readable markdown file for human review.
"""

import os
from datetime import datetime
from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

# Monkey-patch fast_hdbscan for numpy 2.x compatibility: structured array
# attribute access (e.g., arr.idx_start) was removed in numpy 2.0.
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

# Patch parallel_boruvka: fast_hdbscan 0.3+ requires n_threads arg
# but toponymy 0.4 doesn't pass it.
import fast_hdbscan.boruvka as _boruvka
import toponymy.clustering as _tc

_orig_boruvka = _boruvka.parallel_boruvka


def _boruvka_patched(tree, n_threads=1, **kwargs):
    return _orig_boruvka(tree, n_threads, **kwargs)


_tc.parallel_boruvka = _boruvka_patched

from toponymy import Toponymy
from toponymy.clustering import EVoCClusterer
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

load_dotenv(Path.home() / ".config" / "data-apis" / ".env")
load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "embeddings.parquet"
OUTPUT_PATH = ROOT / "data" / "taxonomy_exploration.md"


def remove_bugfix_direction(embeddings: np.ndarray, is_bugfix: np.ndarray) -> np.ndarray:
    """Remove the bugfix/non-bugfix linear direction from embeddings.

    Fits logistic regression to find the separating hyperplane, then projects
    all vectors onto the orthogonal complement of the weight vector.
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(embeddings, is_bugfix)
    w = clf.coef_[0]
    w_hat = w / np.linalg.norm(w)
    residual = embeddings - (embeddings @ w_hat[:, None]) * w_hat[None, :]
    return residual


def run_toponymy(
    documents: list[str],
    embeddings: np.ndarray,
    anthropic_key: str,
    co_key: str,
    label: str,
) -> list:
    """Run Toponymy with EVoCClusterer and return cluster layers."""
    print(f"\n{'='*60}")
    print(f"Running Toponymy: {label}")
    print(f"{'='*60}")

    llm = AsyncAnthropicNamer(api_key=anthropic_key, model="claude-opus-4-6-20250414")
    embedder = CohereEmbedder(api_key=co_key, model="embed-v4.0")
    clusterer = EVoCClusterer(min_clusters=4)

    np.random.seed(42)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Claude Code changelog entries",
        corpus_description="changelog entries from Claude Code, Anthropic's AI coding assistant CLI",
        exemplar_delimiters=['    * "', '"\n'],
        lowest_detail_level=0.3,
        highest_detail_level=1.0,
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=embeddings,  # ignored by EVoCClusterer, but required by API
    )

    n_layers = len(topic_model.cluster_layers_)
    print(f"Produced {n_layers} cluster layer(s)")
    return topic_model.cluster_layers_


def format_layers(layers: list, section_title: str) -> str:
    """Format cluster layers into readable markdown."""
    lines = [f"## {section_title}\n"]

    for i, layer in enumerate(reversed(layers)):
        n_topics = len(layer.topic_names)
        n_entries = len(layer.cluster_labels)
        lines.append(f"### Layer {i} ({n_topics} topics, {n_entries} entries)\n")

        # Count entries per topic
        topic_counts = {}
        for label_idx in range(len(layer.topic_names)):
            count = int(np.sum(layer.cluster_labels == label_idx))
            topic_counts[label_idx] = count

        # Sort by count descending
        sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])

        for topic_idx, count in sorted_topics:
            name = layer.topic_names[topic_idx]
            lines.append(f"- **{name}** ({count} entries)")

            # Add exemplars if available
            if hasattr(layer, "exemplars") and layer.exemplars is not None:
                exemplars = layer.exemplars[topic_idx]
                for ex in exemplars[:3]:
                    # Truncate long exemplars
                    ex_text = ex.strip()
                    if len(ex_text) > 120:
                        ex_text = ex_text[:117] + "..."
                    lines.append(f'  - "{ex_text}"')

        lines.append("")

    return "\n".join(lines)


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} entries from {INPUT_PATH}")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    co_key = os.environ.get("CO_API_KEY")
    if not anthropic_key or not co_key:
        raise EnvironmentError("ANTHROPIC_API_KEY and CO_API_KEY required")

    emb_cols = [f"emb_{i}" for i in range(512)]
    embeddings = df[emb_cols].values.astype(np.float32)
    documents = df["text"].tolist()
    is_bugfix = (df["change_type"] == "bugfix").values.astype(int)

    print(f"Embedding matrix: {embeddings.shape}")
    print(f"Bugfix entries: {is_bugfix.sum()} / {len(is_bugfix)} ({is_bugfix.mean():.1%})")

    # Run 1: Raw embeddings
    raw_layers = run_toponymy(documents, embeddings, anthropic_key, co_key, "Raw Embeddings")

    # Remove bugfix direction
    print("\nRemoving bugfix direction via logistic regression...")
    residual_embeddings = remove_bugfix_direction(embeddings, is_bugfix)
    print(f"Residual embedding matrix: {residual_embeddings.shape}")

    # Run 2: Residual embeddings
    residual_layers = run_toponymy(
        documents, residual_embeddings, anthropic_key, co_key, "Bugfix-Direction Removed"
    )

    # Write output
    bugfix_count = int(is_bugfix.sum())
    output = f"""# Taxonomy Exploration

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Entries: {len(df)}
Bugfix entries: {bugfix_count} ({bugfix_count/len(df):.1%})

{format_layers(raw_layers, "Raw Embeddings")}
{format_layers(residual_layers, "Bugfix-Direction Removed")}
"""

    OUTPUT_PATH.write_text(output)
    print(f"\nSaved taxonomy exploration to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
