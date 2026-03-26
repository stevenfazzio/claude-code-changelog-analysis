"""Compare enrichment results across models and prompt versions."""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAMES = ["haiku", "sonnet", "opus"]
FIELDS = ["category", "change_type", "complexity", "user_facing"]
PAIRS = [("haiku", "sonnet"), ("haiku", "opus"), ("sonnet", "opus")]

COMPLEXITY_MAP = {"minor": 1, "moderate": 2, "major": 3}


# ---------------------------------------------------------------------------
# Loading & merging
# ---------------------------------------------------------------------------

def load_results(eval_dir: Path) -> dict[str, pd.DataFrame]:
    """Load enrichment results for all models from a directory."""
    results = {}
    for name in MODEL_NAMES:
        path = eval_dir / f"enriched_{name}.parquet"
        if path.exists():
            results[name] = pd.read_parquet(path)
            print(f"Loaded {name}: {len(results[name])} entries from {eval_dir.name}/")
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


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def agreement_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise and 3-way agreement rates per field."""
    rows = []
    for field in FIELDS:
        row = {"field": field}
        for a, b in PAIRS:
            agree = (merged[f"{a}_{field}"] == merged[f"{b}_{field}"]).mean()
            row[f"{a}-{b}"] = agree
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


# ---------------------------------------------------------------------------
# Cross-model comparison (terminal + CSV + HTML)
# ---------------------------------------------------------------------------

def print_terminal_report(merged: pd.DataFrame):
    """Print analysis summary to terminal."""
    n = len(merged)
    print(f"\n{'='*60}")
    print(f"Model Comparison: {n} entries")
    print(f"{'='*60}")

    agree = agreement_table(merged)
    print(f"\nPer-field agreement rates:")
    print(f"{'':16s} {'haiku-sonnet':>14s} {'haiku-opus':>14s} {'sonnet-opus':>14s} {'all-3':>14s}")
    for field in FIELDS:
        row = agree.loc[field]
        print(f"  {field:14s} {row['haiku-sonnet']:13.1%} {row['haiku-opus']:13.1%} {row['sonnet-opus']:13.1%} {row['all-3']:13.1%}")

    stats = complexity_stats(merged)
    print(f"\nMean complexity score (minor=1, moderate=2, major=3):")
    for name, score in stats.items():
        print(f"  {name:8s} {score:.3f}")

    print(f"\nEntries with no majority vote:")
    for field in FIELDS:
        votes = majority_vote(merged, field)
        no_maj = (votes == "no_majority").sum()
        print(f"  {field:14s} {no_maj:>4d}")

    for field in FIELDS:
        patterns = top_disagreement_patterns(merged, field, top_n=5)
        if patterns:
            print(f"\nTop disagreement patterns ({field}):")
            print(f"  {'haiku':>14s} {'sonnet':>14s} {'opus':>14s} {'count':>6s}")
            for p in patterns:
                print(f"  {str(p['haiku']):>14s} {str(p['sonnet']):>14s} {str(p['opus']):>14s} {p['count']:>6d}")

    disagree = find_disagreements(merged)
    print(f"\nTotal entries with any disagreement: {len(disagree)} / {n} ({len(disagree)/n:.1%})")


def save_disagreements_csv(merged: pd.DataFrame, output_path: Path):
    """Save disagreement rows to CSV for manual review."""
    disagree = find_disagreements(merged)
    disagree = disagree.sort_values(["version", "entry_index"])

    cols = ["version", "entry_index", "text"]
    for field in FIELDS:
        for name in MODEL_NAMES:
            cols.append(f"{name}_{field}")
        cols.append(f"majority_{field}")

    disagree[cols].to_csv(output_path, index=False)
    print(f"\nSaved {len(disagree)} disagreement rows to {output_path}")


def build_html_report(merged: pd.DataFrame, output_path: Path, title: str = "Model Comparison Report",
                      extra_html: str = ""):
    """Generate interactive HTML report with Plotly."""
    agree = agreement_table(merged)

    pair_labels = ["haiku-sonnet", "haiku-opus", "sonnet-opus", "all-3"]
    z = [[agree.loc[field, pair] for pair in pair_labels] for field in FIELDS]

    fig_agree = go.Figure(data=go.Heatmap(
        z=z, x=pair_labels, y=FIELDS,
        text=[[f"{v:.1%}" for v in row] for row in z],
        texttemplate="%{text}", colorscale="RdYlGn", zmin=0.5, zmax=1.0,
    ))
    fig_agree.update_layout(title="Agreement Rates by Field and Model Pair", height=300)

    confusion_figs = []
    for a, b in PAIRS:
        ct = pd.crosstab(merged[f"{a}_category"], merged[f"{b}_category"])
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        cats = sorted(set(ct.index) | set(ct.columns))
        ct_norm = ct_norm.reindex(index=cats, columns=cats, fill_value=0)

        fig = go.Figure(data=go.Heatmap(
            z=ct_norm.values, x=ct_norm.columns.tolist(), y=ct_norm.index.tolist(),
            text=ct.reindex(index=cats, columns=cats, fill_value=0).values,
            texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
        ))
        fig.update_layout(
            title=f"Category: {a} vs {b} (row-normalized, counts shown)",
            xaxis_title=b, yaxis_title=a, height=500, width=600,
        )
        confusion_figs.append(fig)

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
        fig.update_layout(title=f"Distribution: {field}", barmode="group", height=400)
        dist_figs.append(fig)

    html_parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.0.min.js'></script>",
        "<style>",
        "body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "h1, h2, h3 { color: #333; }",
        ".chart { margin: 30px 0; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 13px; }",
        "th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }",
        "th { background: #f5f5f5; position: sticky; top: 0; }",
        ".disagree { background: #fff3cd; }",
        ".improved { background: #d4edda; }",
        ".regressed { background: #f8d7da; }",
        ".table-wrapper { max-height: 600px; overflow-y: auto; }",
        ".delta-positive { color: #28a745; } .delta-negative { color: #dc3545; }",
        "</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>{len(merged)} entries compared across {len(MODEL_NAMES)} models</p>",
    ]

    # Extra HTML (before/after section) goes first if present
    if extra_html:
        html_parts.append(extra_html)

    html_parts.append("<h2>Agreement Rates</h2>")
    _add_plotly_chart(html_parts, "agree", fig_agree)

    html_parts.append("<h2>Category Confusion Matrices</h2>")
    for i, (fig, (a, b)) in enumerate(zip(confusion_figs, PAIRS)):
        _add_plotly_chart(html_parts, f"confusion_{i}", fig)

    html_parts.append("<h2>Field Distributions</h2>")
    for i, (fig, field) in enumerate(zip(dist_figs, FIELDS)):
        _add_plotly_chart(html_parts, f"dist_{i}", fig)

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

    html_parts.append("</table></div></body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts))
    print(f"Saved HTML report to {output_path}")


def _add_plotly_chart(html_parts: list[str], div_id: str, fig: go.Figure):
    fig_json = fig.to_json()
    html_parts.append(f"<div class='chart' id='{div_id}'></div>")
    html_parts.append(f"<script>Plotly.newPlot('{div_id}', {fig_json}.data, {fig_json}.layout)</script>")


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#39;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Before/after comparison
# ---------------------------------------------------------------------------

def before_after_analysis(v1_dir: Path, v2_dir: Path) -> tuple[str, str]:
    """Compare v1 and v2 results. Returns (terminal_report, html_section)."""
    v1 = load_results(v1_dir)
    v2 = load_results(v2_dir)

    common_models = sorted(set(v1) & set(v2))
    if not common_models:
        return "No common models between v1 and v2.", ""

    terminal_lines = []
    html_parts = []

    terminal_lines.append(f"\n{'='*60}")
    terminal_lines.append("Before/After Comparison (v1 prompt vs v2 prompt)")
    terminal_lines.append(f"{'='*60}")

    html_parts.append("<h2>Before/After: Prompt Improvement Impact</h2>")

    # --- Per-model churn ---
    terminal_lines.append(f"\nPer-model churn (% entries that changed classification):")
    terminal_lines.append(f"  {'model':10s} {'category':>10s} {'change_type':>12s} {'complexity':>12s} {'user_facing':>12s} {'any_field':>10s}")

    churn_data = {}  # model -> field -> rate
    for name in common_models:
        df1, df2 = v1[name], v2[name]
        churn = {}
        for field in FIELDS:
            changed = (df1[field].astype(str) != df2[field].astype(str)).mean()
            churn[field] = changed
        churn["any"] = any_field_changed(df1, df2)
        churn_data[name] = churn
        terminal_lines.append(
            f"  {name:10s} {churn['category']:>9.1%} {churn['change_type']:>11.1%} "
            f"{churn['complexity']:>11.1%} {churn['user_facing']:>11.1%} {churn['any']:>9.1%}"
        )

    # HTML churn table
    html_parts.append("<h3>Per-Model Churn (% entries changed)</h3>")
    html_parts.append("<table><tr><th>Model</th>")
    for field in FIELDS:
        html_parts.append(f"<th>{field}</th>")
    html_parts.append("<th>any field</th></tr>")
    for name in common_models:
        html_parts.append(f"<tr><td><b>{name}</b></td>")
        for field in FIELDS:
            val = churn_data[name][field]
            html_parts.append(f"<td>{val:.1%}</td>")
        html_parts.append(f"<td>{churn_data[name]['any']:.1%}</td></tr>")
    html_parts.append("</table>")

    # --- Agreement rate comparison (v1 vs v2) ---
    v1_merged = build_merged(v1)
    v2_merged = build_merged(v2)
    v1_agree = agreement_table(v1_merged)
    v2_agree = agreement_table(v2_merged)

    terminal_lines.append(f"\nAgreement rates: v1 → v2 (delta)")
    terminal_lines.append(f"  {'field':14s} {'pair':>14s} {'v1':>8s} {'v2':>8s} {'delta':>8s}")
    for field in FIELDS:
        for pair_key in ["haiku-sonnet", "haiku-opus", "sonnet-opus", "all-3"]:
            old = v1_agree.loc[field, pair_key]
            new = v2_agree.loc[field, pair_key]
            delta = new - old
            sign = "+" if delta >= 0 else ""
            terminal_lines.append(f"  {field:14s} {pair_key:>14s} {old:>7.1%} {new:>7.1%} {sign}{delta:>6.1%}")

    # HTML agreement comparison
    pair_labels = ["haiku-sonnet", "haiku-opus", "sonnet-opus", "all-3"]
    html_parts.append("<h3>Agreement Rates: v1 vs v2</h3>")
    html_parts.append("<table><tr><th>Field</th><th>Pair</th><th>v1</th><th>v2</th><th>Delta</th></tr>")
    for field in FIELDS:
        for pair_key in pair_labels:
            old = v1_agree.loc[field, pair_key]
            new = v2_agree.loc[field, pair_key]
            delta = new - old
            cls = "delta-positive" if delta > 0.005 else ("delta-negative" if delta < -0.005 else "")
            sign = "+" if delta >= 0 else ""
            html_parts.append(
                f"<tr><td>{field}</td><td>{pair_key}</td>"
                f"<td>{old:.1%}</td><td>{new:.1%}</td>"
                f"<td class='{cls}'>{sign}{delta:.1%}</td></tr>"
            )
    html_parts.append("</table>")

    # --- Complexity bias comparison ---
    v1_cx = {name: v1_merged[f"{name}_complexity"].map(COMPLEXITY_MAP).mean() for name in common_models}
    v2_cx = {name: v2_merged[f"{name}_complexity"].map(COMPLEXITY_MAP).mean() for name in common_models}

    terminal_lines.append(f"\nMean complexity: v1 → v2")
    for name in common_models:
        delta = v2_cx[name] - v1_cx[name]
        sign = "+" if delta >= 0 else ""
        terminal_lines.append(f"  {name:8s} {v1_cx[name]:.3f} → {v2_cx[name]:.3f} ({sign}{delta:.3f})")

    # --- Convergence: spread between models ---
    terminal_lines.append(f"\nModel spread (max - min agreement across pairs):")
    for field in FIELDS:
        v1_vals = [v1_agree.loc[field, f"{a}-{b}"] for a, b in PAIRS]
        v2_vals = [v2_agree.loc[field, f"{a}-{b}"] for a, b in PAIRS]
        v1_spread = max(v1_vals) - min(v1_vals)
        v2_spread = max(v2_vals) - min(v2_vals)
        terminal_lines.append(f"  {field:14s} v1={v1_spread:.1%}  v2={v2_spread:.1%}")

    terminal_report = "\n".join(terminal_lines)
    html_section = "\n".join(html_parts)
    return terminal_report, html_section


def any_field_changed(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Fraction of entries where any classification field changed."""
    changed = pd.Series(False, index=df1.index)
    for field in FIELDS:
        changed |= df1[field].astype(str) != df2[field].astype(str)
    return changed.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare enrichment results across models")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Directory with model results (default: data/eval/)")
    parser.add_argument("--before-after", action="store_true",
                        help="Run before/after comparison between two prompt versions")
    parser.add_argument("--v1-dir", type=Path, default=None,
                        help="v1 results directory (for --before-after)")
    parser.add_argument("--v2-dir", type=Path, default=None,
                        help="v2 results directory (for --before-after)")
    args = parser.parse_args()

    # Resolve default directories
    default_eval_dir = ROOT / "data" / "eval"
    eval_dir = args.dir if args.dir else default_eval_dir
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir

    # Before/after mode
    ba_terminal = ""
    ba_html = ""
    if args.before_after:
        v1_dir = args.v1_dir or default_eval_dir / "v1"
        v2_dir = args.v2_dir or default_eval_dir / "v2"
        if not v1_dir.is_absolute():
            v1_dir = ROOT / v1_dir
        if not v2_dir.is_absolute():
            v2_dir = ROOT / v2_dir

        ba_terminal, ba_html = before_after_analysis(v1_dir, v2_dir)
        print(ba_terminal)

        # Use v2 as the main comparison dir
        eval_dir = v2_dir

    # Cross-model comparison
    results = load_results(eval_dir)
    if len(results) < 2:
        print("Need at least 2 model results to compare. Run eval/run_models.py first.")
        return

    counts = {name: len(df) for name, df in results.items()}
    if len(set(counts.values())) > 1:
        print(f"WARNING: Row counts differ: {counts}")

    merged = build_merged(results)
    print_terminal_report(merged)

    output_dir = eval_dir
    save_disagreements_csv(merged, output_dir / "disagreements.csv")

    title = "Model Comparison Report (v2)" if args.before_after else "Model Comparison Report"
    build_html_report(merged, output_dir / "comparison_report.html", title=title, extra_html=ba_html)


if __name__ == "__main__":
    main()
