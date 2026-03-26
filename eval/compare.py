"""Compare enrichment results across models and produce analysis reports."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"

MODEL_NAMES = ["haiku", "sonnet", "opus"]
FIELDS = ["category", "change_type", "complexity", "user_facing"]
PAIRS = [("haiku", "sonnet"), ("haiku", "opus"), ("sonnet", "opus")]

COMPLEXITY_MAP = {"minor": 1, "moderate": 2, "major": 3}


def load_results() -> dict[str, pd.DataFrame]:
    """Load enrichment results for all models."""
    results = {}
    for name in MODEL_NAMES:
        path = EVAL_DIR / f"enriched_{name}.parquet"
        if path.exists():
            results[name] = pd.read_parquet(path)
            print(f"Loaded {name}: {len(results[name])} entries")
        else:
            print(f"WARNING: {path} not found, skipping {name}")
    return results


def build_merged(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all model results into a single DataFrame with prefixed columns."""
    base_name = next(iter(results))
    base = results[base_name][["version", "entry_index", "text"]].copy()

    for name, df in results.items():
        for field in FIELDS:
            base[f"{name}_{field}"] = df[field].values

    return base


def agreement_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise and 3-way agreement rates per field."""
    rows = []
    for field in FIELDS:
        row = {"field": field}
        for a, b in PAIRS:
            agree = (merged[f"{a}_{field}"] == merged[f"{b}_{field}"]).mean()
            row[f"{a}-{b}"] = agree
        # All-3 agreement
        all_agree = True
        for a, b in PAIRS:
            all_agree = all_agree & (merged[f"{a}_{field}"] == merged[f"{b}_{field}"])
        row["all-3"] = all_agree.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("field")


def majority_vote(merged: pd.DataFrame, field: str) -> pd.Series:
    """Compute majority vote for a field. Returns 'no_majority' when all 3 disagree."""
    cols = [f"{m}_{field}" for m in MODEL_NAMES]

    def vote(row):
        counts = Counter(row[col] for col in cols)
        winner, count = counts.most_common(1)[0]
        return winner if count >= 2 else "no_majority"

    return merged.apply(vote, axis=1)


def complexity_stats(merged: pd.DataFrame) -> dict[str, float]:
    """Mean complexity score per model."""
    return {
        name: merged[f"{name}_complexity"].map(COMPLEXITY_MAP).mean()
        for name in MODEL_NAMES
    }


def find_disagreements(merged: pd.DataFrame) -> pd.DataFrame:
    """Find all rows where at least one field disagrees across models."""
    any_disagree = pd.Series(False, index=merged.index)
    for field in FIELDS:
        for a, b in PAIRS:
            any_disagree |= merged[f"{a}_{field}"] != merged[f"{b}_{field}"]

    result = merged[any_disagree].copy()
    for field in FIELDS:
        result[f"majority_{field}"] = majority_vote(result, field)
    return result


def top_disagreement_patterns(merged: pd.DataFrame, field: str, top_n: int = 10) -> list[dict]:
    """Find the most common disagreement patterns for a field."""
    patterns = []
    for _, row in merged.iterrows():
        vals = tuple(row[f"{m}_{field}"] for m in MODEL_NAMES)
        if len(set(vals)) > 1:
            patterns.append(vals)

    counts = Counter(patterns)
    return [
        {"haiku": p[0], "sonnet": p[1], "opus": p[2], "count": c}
        for p, c in counts.most_common(top_n)
    ]


def print_terminal_report(merged: pd.DataFrame):
    """Print analysis summary to terminal."""
    n = len(merged)
    print(f"\n{'='*60}")
    print(f"Model Comparison: {n} entries")
    print(f"{'='*60}")

    # Agreement table
    agree = agreement_table(merged)
    print(f"\nPer-field agreement rates:")
    print(f"{'':16s} {'haiku-sonnet':>14s} {'haiku-opus':>14s} {'sonnet-opus':>14s} {'all-3':>14s}")
    for field in FIELDS:
        row = agree.loc[field]
        print(f"  {field:14s} {row['haiku-sonnet']:13.1%} {row['haiku-opus']:13.1%} {row['sonnet-opus']:13.1%} {row['all-3']:13.1%}")

    # Complexity bias
    stats = complexity_stats(merged)
    print(f"\nMean complexity score (minor=1, moderate=2, major=3):")
    for name, score in stats.items():
        print(f"  {name:8s} {score:.3f}")

    # No-majority counts
    print(f"\nEntries with no majority vote:")
    for field in FIELDS:
        votes = majority_vote(merged, field)
        no_maj = (votes == "no_majority").sum()
        print(f"  {field:14s} {no_maj:>4d}")

    # Top disagreement patterns
    for field in FIELDS:
        patterns = top_disagreement_patterns(merged, field, top_n=5)
        if patterns:
            print(f"\nTop disagreement patterns ({field}):")
            print(f"  {'haiku':>14s} {'sonnet':>14s} {'opus':>14s} {'count':>6s}")
            for p in patterns:
                print(f"  {str(p['haiku']):>14s} {str(p['sonnet']):>14s} {str(p['opus']):>14s} {p['count']:>6d}")

    # Overall disagreement rate
    disagree = find_disagreements(merged)
    print(f"\nTotal entries with any disagreement: {len(disagree)} / {n} ({len(disagree)/n:.1%})")


def save_disagreements_csv(merged: pd.DataFrame):
    """Save disagreement rows to CSV for manual review."""
    disagree = find_disagreements(merged)
    disagree = disagree.sort_values(["version", "entry_index"])

    # Reorder columns for readability
    cols = ["version", "entry_index", "text"]
    for field in FIELDS:
        for name in MODEL_NAMES:
            cols.append(f"{name}_{field}")
        cols.append(f"majority_{field}")

    output_path = EVAL_DIR / "disagreements.csv"
    disagree[cols].to_csv(output_path, index=False)
    print(f"\nSaved {len(disagree)} disagreement rows to {output_path}")


def build_html_report(merged: pd.DataFrame):
    """Generate interactive HTML report with Plotly."""
    agree = agreement_table(merged)

    # --- Figure 1: Agreement heatmap ---
    pair_labels = ["haiku-sonnet", "haiku-opus", "sonnet-opus", "all-3"]
    z = [[agree.loc[field, pair] for pair in pair_labels] for field in FIELDS]

    fig_agree = go.Figure(data=go.Heatmap(
        z=z,
        x=pair_labels,
        y=FIELDS,
        text=[[f"{v:.1%}" for v in row] for row in z],
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=0.5,
        zmax=1.0,
    ))
    fig_agree.update_layout(title="Agreement Rates by Field and Model Pair", height=300)

    # --- Figure 2: Confusion matrices for category ---
    confusion_figs = []
    for a, b in PAIRS:
        ct = pd.crosstab(merged[f"{a}_category"], merged[f"{b}_category"])
        # Normalize by row
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        cats = sorted(set(ct.index) | set(ct.columns))
        ct_norm = ct_norm.reindex(index=cats, columns=cats, fill_value=0)

        fig = go.Figure(data=go.Heatmap(
            z=ct_norm.values,
            x=ct_norm.columns.tolist(),
            y=ct_norm.index.tolist(),
            text=ct.reindex(index=cats, columns=cats, fill_value=0).values,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
        ))
        fig.update_layout(
            title=f"Category: {a} vs {b} (row-normalized, counts shown)",
            xaxis_title=b,
            yaxis_title=a,
            height=500,
            width=600,
        )
        confusion_figs.append(fig)

    # --- Figure 3: Distribution comparison per field ---
    dist_figs = []
    for field in FIELDS:
        all_vals = sorted(set(
            val for name in MODEL_NAMES
            for val in merged[f"{name}_{field}"].unique()
        ), key=str)

        fig = go.Figure()
        for name in MODEL_NAMES:
            counts = merged[f"{name}_{field}"].value_counts()
            fig.add_trace(go.Bar(
                name=name,
                x=[str(v) for v in all_vals],
                y=[counts.get(v, 0) for v in all_vals],
            ))
        fig.update_layout(
            title=f"Distribution: {field}",
            barmode="group",
            height=400,
        )
        dist_figs.append(fig)

    # --- Build HTML ---
    html_parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>Model Comparison Report</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.0.min.js'></script>",
        "<style>",
        "body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "h1, h2 { color: #333; }",
        ".chart { margin: 30px 0; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 13px; }",
        "th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }",
        "th { background: #f5f5f5; position: sticky; top: 0; }",
        ".disagree { background: #fff3cd; }",
        ".table-wrapper { max-height: 600px; overflow-y: auto; }",
        "</style>",
        "</head><body>",
        "<h1>Model Comparison Report</h1>",
        f"<p>{len(merged)} entries compared across {len(MODEL_NAMES)} models</p>",
    ]

    # Agreement heatmap
    html_parts.append("<h2>Agreement Rates</h2>")
    html_parts.append("<div class='chart' id='agree'></div>")
    html_parts.append(f"<script>Plotly.newPlot('agree', {fig_agree.to_json()}.data, {fig_agree.to_json()}.layout)</script>")

    # Confusion matrices
    html_parts.append("<h2>Category Confusion Matrices</h2>")
    for i, (fig, (a, b)) in enumerate(zip(confusion_figs, PAIRS)):
        div_id = f"confusion_{i}"
        html_parts.append(f"<div class='chart' id='{div_id}'></div>")
        html_parts.append(f"<script>Plotly.newPlot('{div_id}', {fig.to_json()}.data, {fig.to_json()}.layout)</script>")

    # Distribution charts
    html_parts.append("<h2>Field Distributions</h2>")
    for i, (fig, field) in enumerate(zip(dist_figs, FIELDS)):
        div_id = f"dist_{i}"
        html_parts.append(f"<div class='chart' id='{div_id}'></div>")
        html_parts.append(f"<script>Plotly.newPlot('{div_id}', {fig.to_json()}.data, {fig.to_json()}.layout)</script>")

    # Disagreement table
    disagree = find_disagreements(merged).sort_values(["version", "entry_index"])
    html_parts.append(f"<h2>Disagreements ({len(disagree)} entries)</h2>")
    html_parts.append("<div class='table-wrapper'><table>")
    html_parts.append("<tr><th>Version</th><th>Text</th>")
    for field in FIELDS:
        for name in MODEL_NAMES:
            html_parts.append(f"<th>{name}<br>{field}</th>")
        html_parts.append(f"<th>majority<br>{field}</th>")
    html_parts.append("</tr>")

    for _, row in disagree.iterrows():
        html_parts.append("<tr>")
        text_short = str(row["text"])[:80] + ("..." if len(str(row["text"])) > 80 else "")
        html_parts.append(f"<td>{row['version']}</td><td title='{_escape_html(str(row['text']))}'>{_escape_html(text_short)}</td>")
        for field in FIELDS:
            majority = row[f"majority_{field}"]
            for name in MODEL_NAMES:
                val = row[f"{name}_{field}"]
                cls = " class='disagree'" if str(val) != str(majority) else ""
                html_parts.append(f"<td{cls}>{val}</td>")
            html_parts.append(f"<td><b>{majority}</b></td>")
        html_parts.append("</tr>")

    html_parts.append("</table></div>")
    html_parts.append("</body></html>")

    output_path = EVAL_DIR / "comparison_report.html"
    output_path.write_text("\n".join(html_parts))
    print(f"Saved HTML report to {output_path}")


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#39;").replace('"', "&quot;")


def main():
    results = load_results()
    if len(results) < 2:
        print("Need at least 2 model results to compare. Run eval/run_models.py first.")
        return

    # Verify row counts match
    counts = {name: len(df) for name, df in results.items()}
    if len(set(counts.values())) > 1:
        print(f"WARNING: Row counts differ: {counts}")

    merged = build_merged(results)
    print_terminal_report(merged)
    save_disagreements_csv(merged)
    build_html_report(merged)


if __name__ == "__main__":
    main()
