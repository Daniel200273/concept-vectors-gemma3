#!/usr/bin/env python3
"""Aggregate evaluation metrics across concept validation summary tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_summary_tables(root: Path) -> pd.DataFrame:
    records = []
    for concept_dir in sorted(root.iterdir()):
        if not concept_dir.is_dir():
            continue
        csv_path = concept_dir / "summary_table.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["concept"] = concept_dir.name
        records.append(df)
    if not records:
        raise FileNotFoundError(
            f"No summary_table.csv files found under {root}."
        )
    combined = pd.concat(records, ignore_index=True)
    # Normalize perturbation names to lowercase for consistency.
    combined["perturbation_type"] = combined["perturbation_type"].str.lower()
    return combined


def aggregate_by_configuration(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        [
            "combination_name",
            "perturbation_type",
            "num_vectors_ablated",
            "actual_vector_count",
            "vector_count",
        ],
        dropna=False,
    )

    config_stats = grouped.agg(
        num_concepts=("concept", "nunique"),
        mean_combined_specificity=("combined_specificity", "mean"),
        median_combined_specificity=("combined_specificity", "median"),
        std_combined_specificity=("combined_specificity", "std"),
        min_combined_specificity=("combined_specificity", "min"),
        max_combined_specificity=("combined_specificity", "max"),
        mean_bleu_specificity=("bleu_specificity", "mean"),
        std_bleu_specificity=("bleu_specificity", "std"),
        mean_rouge_specificity=("rouge_specificity", "mean"),
        std_rouge_specificity=("rouge_specificity", "std"),
        mean_sbert_specificity=("sbert_specificity", "mean"),
        std_sbert_specificity=("sbert_specificity", "std"),
        mean_concept_bleu=("concept_bleu", "mean"),
        mean_unrelated_bleu=("unrelated_bleu", "mean"),
        mean_concept_rouge=("concept_rouge", "mean"),
        mean_unrelated_rouge=("unrelated_rouge", "mean"),
        mean_concept_sbert=("concept_sbert", "mean"),
        mean_unrelated_sbert=("unrelated_sbert", "mean"),
    ).reset_index()

    config_stats["delta_bleu"] = (
        config_stats["mean_concept_bleu"] - config_stats["mean_unrelated_bleu"]
    )
    config_stats["delta_rouge"] = (
        config_stats["mean_concept_rouge"] - config_stats["mean_unrelated_rouge"]
    )
    config_stats["delta_sbert"] = (
        config_stats["mean_concept_sbert"] - config_stats["mean_unrelated_sbert"]
    )
    return config_stats


def aggregate_by_noise(df: pd.DataFrame) -> pd.DataFrame:
    noise_stats = df.groupby("perturbation_type", dropna=False).agg(
        num_records=("concept", "count"),
        num_concepts=("concept", "nunique"),
        mean_combined_specificity=("combined_specificity", "mean"),
        median_combined_specificity=("combined_specificity", "median"),
        std_combined_specificity=("combined_specificity", "std"),
        mean_bleu_specificity=("bleu_specificity", "mean"),
        mean_rouge_specificity=("rouge_specificity", "mean"),
        mean_sbert_specificity=("sbert_specificity", "mean"),
        mean_concept_bleu=("concept_bleu", "mean"),
        mean_unrelated_bleu=("unrelated_bleu", "mean"),
        mean_concept_rouge=("concept_rouge", "mean"),
        mean_unrelated_rouge=("unrelated_rouge", "mean"),
        mean_concept_sbert=("concept_sbert", "mean"),
        mean_unrelated_sbert=("unrelated_sbert", "mean"),
    ).reset_index()

    noise_stats["delta_bleu"] = (
        noise_stats["mean_concept_bleu"] - noise_stats["mean_unrelated_bleu"]
    )
    noise_stats["delta_rouge"] = (
        noise_stats["mean_concept_rouge"] - noise_stats["mean_unrelated_rouge"]
    )
    noise_stats["delta_sbert"] = (
        noise_stats["mean_concept_sbert"] - noise_stats["mean_unrelated_sbert"]
    )
    return noise_stats


def _configure_plot_style() -> None:
    """Apply a consistent plotting style with graceful fallback."""
    # Prefer a clean white-grid style suitable for thesis figures.
    preferred_styles = [
        "seaborn-whitegrid",
        "seaborn-v0_8-whitegrid",
        "seaborn-v0_8-darkgrid",
        "seaborn-v0_8",
        "seaborn",
        "ggplot",
    ]
    for style in preferred_styles:
        try:
            plt.style.use(style)
            return
        except OSError:
            continue
    # Fall back to default style but tweak a few rcParams for readability.
    plt.rcParams.update(
        {
            # Use pure white axes/figure background for clean export to thesis
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#4c4c4c",
            # lighter grid lines
            "grid.color": "#e6e6e6",
            "grid.alpha": 0.9,
            "grid.linestyle": "--",
        }
    )


def _build_palette(categories: pd.Series) -> dict[str | float | None, tuple[float, float, float, float]]:
    unique = list(dict.fromkeys(categories.fillna("missing")))
    cmap = plt.get_cmap("tab10")
    return {
        (None if cat == "missing" else cat): cmap(i % cmap.N)
        for i, cat in enumerate(unique)
    }
def plot_specificity_distributions(
    df: pd.DataFrame,
    config_stats: pd.DataFrame,
    noise_stats: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> tuple[Path, Path, Path, Path]:
    if df.empty:
        raise ValueError("No data available to plot distributions.")

    combined_scores = df["combined_specificity"].dropna()
    if combined_scores.empty:
        raise ValueError("No combined specificity scores available.")

    hist_threshold = 0.2

    # Draw histogram manually so we can color bins above a threshold differently.
    bins = 50
    counts, bin_edges = np.histogram(combined_scores, bins=bins)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2.0
    colors = ["#4c72b0" if c <= hist_threshold else "#d62728" for c in bin_centers]

    fig_hist = plt.figure(figsize=(12, 5))
    ax_hist = fig_hist.add_subplot(1, 1, 1)
    # Force white background for histogram figure/axes so it matches thesis style
    fig_hist.patch.set_facecolor("white")
    ax_hist.set_facecolor("white")
    ax_hist.set_axisbelow(True)
    ax_hist.bar(
        bin_edges[:-1],
        counts,
        width=bin_widths,
        align="edge",
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )
    ax_hist.axvline(
        combined_scores.mean(),
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean {combined_scores.mean():.3f}",
    )
    ax_hist.axvline(
        combined_scores.median(),
        color="#ff7f0e",
        linestyle=":",
        linewidth=2.2,
        label=f"Median {combined_scores.median():.3f}",
    )
    ax_hist.set_title("Distribution of Combined Specificity across All Concepts")
    ax_hist.set_xlabel("Combined Specificity")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.35, linestyle="--")
    ax_hist.legend(loc="upper right")

    # Annotate percentage of points above the 'specific' threshold
    pct_specific = float((combined_scores >= hist_threshold).mean() * 100.0)
    annot_text = f"{pct_specific:.2f}% ≥ {hist_threshold}"
    ax_hist.text(
        0.02,
        0.95,
        annot_text,
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        color="#d62728" if pct_specific > 0 else "#4c4c4c",
    )

    # Save histogram figure
    hist_path = output_dir / "specificity_histogram.png"
    fig_hist.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close(fig_hist)

    # PLOT 1: Perturbation type analysis for high specificity scores (PIE CHART)
    df_high = df[df["combined_specificity"] >= hist_threshold].copy()
    
    perturbation_high_counts = df_high["perturbation_type"].value_counts()
    
    # Define consistent colors for perturbation types (matching other plots)
    perturbation_colors = {
        'zero': '#1f77b4',      # blue
        'gaussian': '#ff7f0e',  # orange
        'uniform': '#2ca02c',   # green
        'mean': '#d62728',      # red
        'random': '#9467bd',    # purple
    }
    
    # Create pie chart
    fig_pert = plt.figure(figsize=(10, 8))
    fig_pert.patch.set_facecolor("white")
    ax_pert = fig_pert.add_subplot(1, 1, 1)
    ax_pert.set_facecolor("white")
    
    # Map colors to perturbation types
    colors = [perturbation_colors.get(pt, '#7f7f7f') for pt in perturbation_high_counts.index]
    
    # Create 3D-like pie chart with shadow and explode
    wedges, texts, autotexts = ax_pert.pie(
        perturbation_high_counts.values,
        labels=perturbation_high_counts.index,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*perturbation_high_counts.sum())})',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 11, 'weight': 'bold'},
        pctdistance=0.85,
        shadow=True,
        explode=[0.05] * len(perturbation_high_counts)
    )
    
    # Make percentage text white for better contrast
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    ax_pert.set_title(f"Perturbation Types in High Specificity Scores (≥{hist_threshold})", 
                      fontsize=14, weight='bold', pad=20)
    
    fig_pert.tight_layout()
    pert_path = output_dir / "perturbation_high_specificity.png"
    fig_pert.savefig(pert_path, dpi=300, bbox_inches="tight")
    plt.close(fig_pert)
    
    # PLOT 2: Single-layer vs Multi-layer comparison (PIE CHART)
    # Determine if configuration is single or multi-layer based on num_vectors_ablated
    df_analysis = df[df["combined_specificity"] >= hist_threshold].copy()
    df_analysis["layer_type"] = df_analysis["num_vectors_ablated"].apply(
        lambda x: "Single-layer" if pd.notna(x) and x == 1 else "Multi-layer" if pd.notna(x) and x > 1 else "Unknown"
    )
    
    # Filter out unknown
    df_analysis = df_analysis[df_analysis["layer_type"] != "Unknown"]
    
    if not df_analysis.empty:
        layer_counts = df_analysis["layer_type"].value_counts()
        
        fig_layer = plt.figure(figsize=(10, 8))
        fig_layer.patch.set_facecolor("white")
        ax_layer = fig_layer.add_subplot(1, 1, 1)
        ax_layer.set_facecolor("white")
        
        # Use distinct colors for single vs multi (matching color scheme)
        colors = ['#3498db', '#e67e22']  # Bright blue and orange
        
        # Create 3D-like pie chart with shadow and explode
        wedges, texts, autotexts = ax_layer.pie(
            layer_counts.values,
            labels=layer_counts.index,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*layer_counts.sum())})',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 12, 'weight': 'bold'},
            pctdistance=0.85,
            shadow=True,
            explode=[0.05] * len(layer_counts)
        )
        
        # Make percentage text white for better contrast
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        ax_layer.set_title(f"Single-layer vs Multi-layer in High Specificity Scores (≥{hist_threshold})", 
                          fontsize=14, weight='bold', pad=20)
        
        fig_layer.tight_layout()
        layer_path = output_dir / "single_vs_multi_layer.png"
        fig_layer.savefig(layer_path, dpi=300, bbox_inches="tight")
        plt.close(fig_layer)
    else:
        # Create placeholder if no data
        fig_layer = plt.figure(figsize=(10, 4))
        fig_layer.patch.set_facecolor("white")
        fig_layer.text(0.5, 0.5, "No layer configuration data available", ha="center", va="center")
        layer_path = output_dir / "single_vs_multi_layer.png"
        fig_layer.savefig(layer_path, dpi=200, bbox_inches="tight")
        plt.close(fig_layer)

    # PLOT 3: Top 50 configs - Perturbation type breakdown by layer type (STACKED BAR CHART)
    # Get top 50 configurations by mean combined specificity
    top50_configs = config_stats.nlargest(50, "mean_combined_specificity")
    
    if not top50_configs.empty:
        # Determine layer type from combination_name
        def get_layer_type(name):
            if pd.isna(name):
                return "Unknown"
            name_str = str(name)
            if "L1_" in name_str:
                return "Single-layer (L1)"
            elif "L2_" in name_str or "L5_" in name_str:
                return "Multi-layer (L2/L5)"
            return "Unknown"
        
        top50_configs["layer_type"] = top50_configs["combination_name"].apply(get_layer_type)
        
        # Count by perturbation type and layer type
        summary = top50_configs.groupby(["perturbation_type", "layer_type"]).size().unstack(fill_value=0)
        
        # Ensure we have both layer types as columns
        for col in ["Single-layer (L1)", "Multi-layer (L2/L5)"]:
            if col not in summary.columns:
                summary[col] = 0
        
        # Sort by total count descending
        summary["total"] = summary.sum(axis=1)
        summary = summary.sort_values("total", ascending=True).drop("total", axis=1)
        
        # Create stacked horizontal bar chart
        fig_stacked = plt.figure(figsize=(12, 6))
        fig_stacked.patch.set_facecolor("white")
        ax_stacked = fig_stacked.add_subplot(1, 1, 1)
        ax_stacked.set_facecolor("white")
        
        # Colors matching the layer type pie chart
        layer_colors = {
            "Single-layer (L1)": "#3498db",      # Blue
            "Multi-layer (L2/L5)": "#e67e22"     # Orange
        }
        
        # Create horizontal stacked bars
        y_pos = np.arange(len(summary))
        left = np.zeros(len(summary))
        
        for col in ["Single-layer (L1)", "Multi-layer (L2/L5)"]:
            if col in summary.columns:
                bars = ax_stacked.barh(
                    y_pos, 
                    summary[col], 
                    left=left, 
                    label=col,
                    color=layer_colors[col],
                    alpha=0.9,
                    edgecolor='white',
                    linewidth=1.5
                )
                
                # Add count labels on bars
                for i, (bar, val) in enumerate(zip(bars, summary[col])):
                    if val > 0:
                        x_pos = left[i] + val / 2
                        ax_stacked.text(
                            x_pos, 
                            bar.get_y() + bar.get_height() / 2,
                            f'{int(val)}',
                            ha='center',
                            va='center',
                            fontsize=10,
                            weight='bold',
                            color='white'
                        )
                
                left += summary[col].values
        
        # Customize appearance
        ax_stacked.set_yticks(y_pos)
        ax_stacked.set_yticklabels(summary.index, fontsize=11, weight='bold')
        ax_stacked.set_xlabel('Number of Configurations', fontsize=12, weight='bold')
        ax_stacked.set_title('Top 50 Configurations: Perturbation Types by Layer Configuration', 
                            fontsize=14, weight='bold', pad=20)
        ax_stacked.legend(loc='lower right', fontsize=10, framealpha=0.95)
        ax_stacked.grid(axis='x', alpha=0.3, linestyle='--')
        ax_stacked.set_axisbelow(True)
        
        # Add total count on the right side
        for i, (idx, row) in enumerate(summary.iterrows()):
            total = row.sum()
            ax_stacked.text(
                total + 0.5,
                i,
                f'{int(total)}',
                ha='left',
                va='center',
                fontsize=10,
                weight='bold',
                color='#2c3e50'
            )
        
        fig_stacked.tight_layout()
        stacked_path = output_dir / "top50_perturbation_layer_breakdown.png"
        fig_stacked.savefig(stacked_path, dpi=300, bbox_inches="tight")
        plt.close(fig_stacked)
    else:
        # Create placeholder
        fig_stacked = plt.figure(figsize=(10, 4))
        fig_stacked.patch.set_facecolor("white")
        fig_stacked.text(0.5, 0.5, "No top 50 configuration data available", ha="center", va="center")
        stacked_path = output_dir / "top50_perturbation_layer_breakdown.png"
        fig_stacked.savefig(stacked_path, dpi=200, bbox_inches="tight")
        plt.close(fig_stacked)

    return hist_path, pert_path, layer_path, stacked_path


def main(
    root: Path, output_dir: Path, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _configure_plot_style()
    df = load_summary_tables(root)
    config_stats = aggregate_by_configuration(df)
    noise_stats = aggregate_by_noise(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config_aggregates.csv"
    noise_path = output_dir / "noise_aggregates.csv"
    raw_path = output_dir / "all_summary_rows.csv"

    df.sort_values(["concept", "combination_name"]).to_csv(raw_path, index=False)
    config_stats.sort_values("mean_combined_specificity").to_csv(
        config_path, index=False
    )
    noise_stats.sort_values("mean_combined_specificity").to_csv(
        noise_path, index=False
    )

    best_configs = config_stats.nlargest(top_n, "mean_combined_specificity")

    print("Aggregated", len(df), "rows across", df["concept"].nunique(), "concepts.")
    print()
    print("Top configurations by mean combined specificity (higher is better):")
    cols_to_show = [
        "combination_name",
        "perturbation_type",
        "num_concepts",
        "mean_combined_specificity",
        "std_combined_specificity",
        "delta_bleu",
        "delta_rouge",
        "delta_sbert",
    ]
    print(best_configs[cols_to_show].to_string(index=False, float_format="{:.4f}".format))

    print()
    print("Noise type aggregates:")
    noise_cols = [
        "perturbation_type",
        "num_records",
        "num_concepts",
        "mean_combined_specificity",
        "std_combined_specificity",
        "delta_bleu",
        "delta_rouge",
        "delta_sbert",
        "mean_concept_bleu",
        "mean_unrelated_bleu",
        "mean_concept_rouge",
        "mean_unrelated_rouge",
        "mean_concept_sbert",
        "mean_unrelated_sbert",
    ]
    print(
        noise_stats[noise_cols]
        .sort_values("mean_combined_specificity", ascending=False)
        .to_string(index=False, float_format="{:.4f}".format)
    )

    best_noise = noise_stats.nlargest(1, "mean_combined_specificity")
    if not best_noise.empty:
        row = best_noise.iloc[0]
        print()
        print(
            "Most robust noise type:"
            f" {row['perturbation_type']} (mean specificity {row['mean_combined_specificity']:.4f},"
            f" affecting {int(row['num_concepts'])} concepts across {int(row['num_records'])} evaluations)"
        )

    variability = config_stats[[
        "combination_name",
        "perturbation_type",
        "mean_combined_specificity",
        "std_combined_specificity",
    ]].sort_values("std_combined_specificity")

    if not variability.empty:
        print()
        print("Most stable configurations (lowest std dev):")
        print(
            variability.head(3).to_string(
                index=False,
                float_format="{:.4f}".format,
            )
        )
        print()
        print("Most variable configurations (highest std dev):")
        print(
            variability.tail(3).to_string(
                index=False,
                float_format="{:.4f}".format,
            )
        )

    summary_lines = [
        "# Concept Validation Aggregate Summary",
        "",
        f"*Total concepts analyzed:* {df['concept'].nunique()}",
        f"*Total evaluations:* {len(df)}",
        "",
    ]

    if not best_configs.empty:
        top_row = best_configs.iloc[0]
        summary_lines.extend(
            [
                "## Best configuration by specificity",
                (
                    f"- **Combination:** {top_row['combination_name']}"
                    f" ({top_row['perturbation_type']})"
                ),
                (
                    f"- **Mean combined specificity:** {top_row['mean_combined_specificity']:.4f}"
                ),
                (
                    f"- **Metric deltas (BLEU / ROUGE / SBERT):**"
                    f" {top_row['delta_bleu']:.4f} / {top_row['delta_rouge']:.4f} / {top_row['delta_sbert']:.4f}"
                ),
                "",
            ]
        )

    if not best_noise.empty:
        row = best_noise.iloc[0]
        summary_lines.extend(
            [
                "## Most reliable noise type",
                f"- **Noise:** {row['perturbation_type']}",
                f"- **Mean combined specificity:** {row['mean_combined_specificity']:.4f}",
                f"- **Concept coverage:** {int(row['num_concepts'])} concepts",
                f"- **Average metric deltas (BLEU/ROUGE/SBERT):** {row['delta_bleu']:.4f} / {row['delta_rouge']:.4f} / {row['delta_sbert']:.4f}",
                "",
            ]
        )

    if not variability.empty:
        summary_lines.append("## Stability highlights")
        summary_lines.append("- **Most stable configs (lowest std dev):**")
        for _, row in variability.head(3).iterrows():
            summary_lines.append(
                (
                    f"  - {row['combination_name']} ({row['perturbation_type']}):"
                    f" mean {row['mean_combined_specificity']:.4f}, std {row['std_combined_specificity']:.4f}"
                )
            )
        summary_lines.append("- **Most variable configs (highest std dev):**")
        for _, row in variability.tail(3).iterrows():
            summary_lines.append(
                (
                    f"  - {row['combination_name']} ({row['perturbation_type']}):"
                    f" mean {row['mean_combined_specificity']:.4f}, std {row['std_combined_specificity']:.4f}"
                )
            )
        summary_lines.append("")

    summary_path = output_dir / "summary_report.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print(f"Wrote per-configuration stats to {config_path}")
    print(f"Wrote per-noise stats to {noise_path}")
    print(f"Wrote raw combined rows to {raw_path}")
    print(f"Wrote textual summary to {summary_path}")

    return config_stats, noise_stats, df


def cli(root: Path, output_dir: Path, top_n: int) -> int:
    config_stats, noise_stats, df = main(root, output_dir, top_n)

    top_configs = config_stats.nlargest(top_n, "mean_combined_specificity")
    top_path = output_dir / f"top_{top_n}_configurations.csv"
    top_configs.to_csv(top_path, index=False)

    print(f"Wrote top {len(top_configs)} configurations to {top_path}")

    plot_paths: list[tuple[str, Path]] = []
    try:
        hist_path, pert_path, layer_path, stacked_path = plot_specificity_distributions(
            df, config_stats, noise_stats, output_dir, top_n
        )
        plot_paths.append(("Specificity histogram", hist_path))
        plot_paths.append(("Perturbation types with high specificity", pert_path))
        plot_paths.append(("Single-layer vs Multi-layer comparison", layer_path))
        plot_paths.append(("Top 50 configs perturbation/layer breakdown", stacked_path))
    except ValueError as exc:
        print(f"Skipping specificity distribution plot: {exc}")

    if plot_paths:
        print()
        print("Generated plots:")
        for label, path in plot_paths:
            print(f" - {label}: {path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("val-plots"),
        help="Root directory containing concept subfolders with summary_table.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory to write aggregate CSV files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top configurations to report and plot",
    )
    args = parser.parse_args()

    try:
        sys.exit(cli(args.root, args.output, args.top_n))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
