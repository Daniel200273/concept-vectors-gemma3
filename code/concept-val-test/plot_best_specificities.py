#!/usr/bin/env python3
"""
Plot Best Specificity Scores Across Concepts

This script analyzes all summary_table.csv files in concept folders under val-plots/
and creates a visualization showing the configuration with the highest combined
specificity score for each concept.

For each concept, it:
1. Finds the configuration with max(sbert_specificity + rouge_specificity + bleu_specificity)
2. Plots three bars showing each individual specificity score
3. Labels with concept name and configuration details

Usage:
    python plot_best_specificities.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import re
from matplotlib.patches import Patch


def collect_config_rows(val_plots_dir):
    """Load all configuration rows across concepts, ensuring combined specificity is present."""

    if not os.path.exists(val_plots_dir):
        return pd.DataFrame()

    concept_folders = [
        d for d in os.listdir(val_plots_dir)
        if os.path.isdir(os.path.join(val_plots_dir, d))
    ]

    data_frames = []

    for concept_folder in concept_folders:
        csv_path = os.path.join(val_plots_dir, concept_folder, "summary_table.csv")
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠️  Skipping {concept_folder}: could not read CSV ({e})")
            continue

        required_metric_cols = {'sbert_specificity', 'rouge_specificity', 'bleu_specificity'}
        if not required_metric_cols.issubset(df.columns):
            print(f"⚠️  Skipping {concept_folder}: missing metric columns for combined specificity")
            continue

        if 'combined_specificity' not in df.columns:
            df['combined_specificity'] = (
                0.5 * df['sbert_specificity']
                + 0.25 * df['bleu_specificity']
                + 0.25 * df['rouge_specificity']
            )

        if 'perturbation_type' not in df.columns:
            df['perturbation_type'] = np.nan

        if 'combination_name' not in df.columns:
            df['combination_name'] = np.nan

        df['concept_name'] = concept_folder
        data_frames.append(df)

    if not data_frames:
        return pd.DataFrame()

    return pd.concat(data_frames, ignore_index=True)

def load_concept_data(val_plots_dir):
    """
    Load data from all concept folders and find best configuration for each concept.
    
    Args:
        val_plots_dir: Path to val-plots directory
        
    Returns:
        List of dictionaries with concept data and best configurations
    """
    concept_data = []
    
    # Get all subdirectories (concept folders)
    concept_folders = [d for d in os.listdir(val_plots_dir) 
                      if os.path.isdir(os.path.join(val_plots_dir, d))]
    
    print(f"Found {len(concept_folders)} concept folders")
    
    for concept_folder in sorted(concept_folders):
        csv_path = os.path.join(val_plots_dir, concept_folder, "summary_table.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️  No summary_table.csv found in {concept_folder}")
            continue
            
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Check if required columns exist
            required_cols = ['sbert_specificity', 'rouge_specificity', 'bleu_specificity']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️  Missing required columns in {concept_folder}")
                print(f"    Available columns: {list(df.columns)}")
                continue
            
            # Calculate combined specificity score using the project's weighted formula
            # combined = 0.5 * SBERT + 0.25 * BLEU + 0.25 * ROUGE
            df['combined_specificity'] = (
                0.5 * df['sbert_specificity']
                + 0.25 * df['bleu_specificity']
                + 0.25 * df['rouge_specificity']
            )
            
            # Find row with highest combined specificity
            best_idx = df['combined_specificity'].idxmax()
            best_row = df.loc[best_idx]
            
            # Helper: clean combination names like
            # 'L5_V10_SCALE_scale_50vec' -> 'L5_V10_SCALE'
            def clean_combination_name(name):
                if pd.isna(name):
                    return name
                name_str = str(name)
                m = re.match(r'^([A-Z0-9_]+)', name_str)
                return m.group(1) if m else name_str

            # Use only the cleaned combination_name as the config identifier
            if 'combination_name' in df.columns:
                config_name = clean_combination_name(best_row['combination_name']) or f"row_{best_idx}"
            else:
                config_name = f"row_{best_idx}"
            
            concept_data.append({
                'concept_name': concept_folder,
                'config_name': config_name,
                'sbert_specificity': best_row['sbert_specificity'],
                'rouge_specificity': best_row['rouge_specificity'], 
                'bleu_specificity': best_row['bleu_specificity'],
                'combined_specificity': best_row['combined_specificity'],
                'num_configs': len(df)
            })
            
            print(f"✅ {concept_folder}: best config = {config_name} "
                  f"(combined: {best_row['combined_specificity']:.3f})")
            
        except Exception as e:
            print(f"❌ Error processing {concept_folder}: {e}")
            continue
    
    return concept_data

def create_specificity_plot(concept_data, output_path=None, top_n=None):
    """
    Create a grouped bar plot showing specificity scores for each concept.

    Args:
        concept_data: List of dictionaries with best config per concept.
        output_path: Optional path to save the plot image.
        top_n: If provided, limit to top N concepts by combined specificity.
    """
    if not concept_data:
        print("No data to plot!")
        return
    
    # Sort concepts by combined specificity (descending)
    concept_data_sorted = sorted(concept_data, key=lambda x: x['combined_specificity'], reverse=True)

    # Optionally limit to top N concepts
    if top_n is not None:
        concept_data_sorted = concept_data_sorted[:top_n]
    
    # Prepare data for plotting
    concepts = [d['concept_name'] for d in concept_data_sorted]
    configs = [d['config_name'] for d in concept_data_sorted]
    sbert_scores = [d['sbert_specificity'] for d in concept_data_sorted]
    rouge_scores = [d['rouge_specificity'] for d in concept_data_sorted]
    bleu_scores = [d['bleu_specificity'] for d in concept_data_sorted]
    
    # Create labels combining concept and simplified config (capslock only)
    simplified_configs = []
    for config in configs:
        # Extract capslock portion (like L1_V20_SCALE) from full config name
        parts = config.split('_')
        capslock_parts = [p for p in parts if p.isupper() or p.isdigit() or any(c.isupper() for c in p)]
        simplified_config = '_'.join(capslock_parts) if capslock_parts else config
        simplified_configs.append(simplified_config)
    
    labels = [f"{concept}\n{config}" for concept, config in zip(concepts, simplified_configs)]
    
    # Set up the plot
    n_concepts = len(concepts)
    x = np.arange(n_concepts)
    width = 0.25
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(max(12, n_concepts * 0.8), 8))
    
    # Create grouped bars
    bars1 = ax.bar(x - width, sbert_scores, width, label='SBERT Specificity', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, rouge_scores, width, label='ROUGE Specificity', 
                   color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, bleu_scores, width, label='BLEU Specificity', 
                   color='#F18F01', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Concept and Best Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Specificity Score', fontsize=12, fontweight='bold')
    ax.set_title('Best Specificity Scores by Concept\n(Highest Combined Score Configuration)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0.01:  # Only label significant values
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1, sbert_scores)
    add_value_labels(bars2, rouge_scores)
    add_value_labels(bars3, bleu_scores)
    
    # Adjust layout
    plt.tight_layout()
    
    # (Summary box removed by request)
    
    # Save plot if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    return fig, ax

def create_perturbation_type_distribution_plot(val_plots_dir, output_path=None, threshold=0.25):
    """Visualize how often each perturbation type appears and highlight high-specificity configs."""

    combined_df = collect_config_rows(val_plots_dir)

    if combined_df.empty:
        print("No data available to summarize perturbation types")
        return
    combined_df['perturbation_type'] = (
        combined_df['perturbation_type']
        .fillna('unknown')
        .astype(str)
        .str.strip()
        .str.lower()
    )

    type_mapping = {
        'raw': 'Raw',
        'raw text': 'Raw',
        'baseline': 'Baseline',
        'ablation': 'Ablation',
        'gaussian': 'Gaussian',
        'gaussian_noise': 'Gaussian',
        'noise': 'Gaussian',
        'normal': 'Gaussian',
        'scale': 'Scale',
        'scaling': 'Scale',
        'scale_up': 'Scale',
        'mix': 'Mix',
        'mixing': 'Mix',
        'unknown': 'Unknown'
    }

    combined_df['perturbation_type'] = combined_df['perturbation_type'].map(
        lambda x: type_mapping.get(x, x.title())
    )

    high_specificity_df = combined_df[combined_df['combined_specificity'] > threshold]

    if high_specificity_df.empty:
        print(f"No configurations found with combined specificity > {threshold:.2f}")
        return

    high_counts = high_specificity_df.groupby('perturbation_type').size()

    order = high_counts.sort_values(ascending=False).index.tolist()
    high_counts = high_counts.loc[order]

    x = np.arange(len(high_counts))

    fig, ax = plt.subplots(figsize=(max(10, len(high_counts) * 1.2), 6))

    color_map = {
        'Gaussian': '#2E86AB',  # Blue
        'Scale': '#F18F01',     # Orange
        'Ablation': '#27AE60'   # Green
    }

    bar_colors = [color_map.get(pt, '#7f8c8d') for pt in order]

    bars = ax.bar(
        x,
        high_counts.values,
        color=bar_colors,
        edgecolor='white',
        label=f"Combined > {threshold:.2f}"
    )

    for bar, count in zip(bars, high_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.5,
            f"{int(count)}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Number of High-Specificity Configurations', fontsize=12, fontweight='bold')
    ax.set_xlabel('Perturbation Type', fontsize=12, fontweight='bold')
    ax.set_title(
        'High-Specificity Configurations by Perturbation Type\n(Combined > 0.25)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ylim_max = max(high_counts.values) * 1.15 + 1
    ax.set_ylim(0, ylim_max)

    legend_elements = [
        Patch(facecolor=color_map['Gaussian'], edgecolor='white', label='Gaussian'),
        Patch(facecolor=color_map['Scale'], edgecolor='white', label='Scale'),
        Patch(facecolor=color_map['Ablation'], edgecolor='white', label='Ablation')
    ]

    if any(pt not in color_map for pt in order):
        legend_elements.append(
            Patch(facecolor='#7f8c8d', edgecolor='white', label='Other')
        )

    ax.legend(handles=legend_elements, fontsize=11, title='Perturbation Type')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Perturbation type distribution saved to: {output_path}")

    plt.show()

    return fig, ax


def create_layer_performance_plot(val_plots_dir, output_path=None, threshold=0.25):
    """Compare average specificity metrics for single-layer vs multi-layer configurations."""

    combined_df = collect_config_rows(val_plots_dir)

    if combined_df.empty:
        print("No data available to compare layer performance")
        return

    high_specificity_df = combined_df[combined_df['combined_specificity'] > threshold].copy()

    if high_specificity_df.empty:
        print(f"No configurations found with combined specificity > {threshold:.2f}")
        return

    def classify_layer(name):
        if pd.isna(name):
            return 'Unknown'
        name_str = str(name).upper()
        if name_str.startswith('L1_'):
            return 'Single-layer (L1)'
        if name_str.startswith('L2_') or name_str.startswith('L5_'):
            return 'Multi-layer (L2/L5)'
        return 'Unknown'

    high_specificity_df['layer_category'] = high_specificity_df['combination_name'].apply(classify_layer)

    recognized_df = high_specificity_df[high_specificity_df['layer_category'].isin([
        'Single-layer (L1)', 'Multi-layer (L2/L5)'
    ])].copy()

    if recognized_df.empty:
        print("No high-specificity configurations with identifiable layer categories")
        return

    metrics = ['sbert_specificity', 'rouge_specificity', 'bleu_specificity', 'combined_specificity']
    metric_labels = ['SBERT', 'ROUGE', 'BLEU', 'Combined']

    layer_counts = recognized_df.groupby('layer_category').size()
    avg_metrics = recognized_df.groupby('layer_category')[metrics].mean()

    layer_order = ['Single-layer (L1)', 'Multi-layer (L2/L5)']
    layer_counts = layer_counts.reindex(layer_order, fill_value=0)
    avg_metrics = avg_metrics.reindex(layer_order)

    valid_layers = [layer for layer in layer_order if layer_counts.get(layer, 0) > 0]

    if not valid_layers:
        print("Insufficient data to compare layer categories")
        return

    avg_metrics = avg_metrics.loc[valid_layers]
    layer_counts = layer_counts.loc[valid_layers]

    x = np.arange(len(metrics))
    total_width = 0.8
    n_layers = len(valid_layers)
    bar_width = total_width / n_layers

    layer_colors = {
        'Single-layer (L1)': '#2E86AB',
        'Multi-layer (L2/L5)': '#F18F01'
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_containers = []

    for idx, layer in enumerate(valid_layers):
        offset = (-total_width / 2) + (idx + 0.5) * bar_width
        scores = avg_metrics.loc[layer, metrics].fillna(0).values
        bars = ax.bar(
            x + offset,
            scores,
            width=bar_width,
            color=layer_colors.get(layer, '#7f8c8d'),
            edgecolor='white',
            label=f"{layer} (n={int(layer_counts[layer])})"
        )
        bar_containers.append((bars, scores))

    for bars, scores in bar_containers:
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.01,
                f"{score:.3f}",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Average Specificity Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title(
        'High-Specificity Performance by Layer Configuration\n(Combined > 0.25)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    max_score = max(scores.max() for _, scores in bar_containers)
    ax.set_ylim(0, min(1.05, max_score * 1.2))

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Layer performance plot saved to: {output_path}")

    plt.show()

    return fig, ax


def print_summary_table(concept_data):
    """Print a summary table of the results"""
    if not concept_data:
        return
    
    print("\n" + "="*100)
    print("SUMMARY: Best Configurations by Concept")
    print("="*100)
    
    # Sort by combined specificity
    sorted_data = sorted(concept_data, key=lambda x: x['combined_specificity'], reverse=True)
    
    print(f"{'Rank':<4} {'Concept':<30} {'Config':<30} {'SBERT':<8} {'ROUGE':<8} {'BLEU':<8} {'Combined':<9}")
    print("-" * 110)

    for i, data in enumerate(sorted_data, 1):
        print(f"{i:<4} {data['concept_name']:<30} {data['config_name']:<30} "
              f"{data['sbert_specificity']:<8.3f} {data['rouge_specificity']:<8.3f} "
              f"{data['bleu_specificity']:<8.3f} {data['combined_specificity']:<9.3f}")
    # Show global range for context
    print("-" * 100)
    print(f"Range: {sorted_data[-1]['combined_specificity']:.3f} to {sorted_data[0]['combined_specificity']:.3f}")

def main():
    """Main function to run the analysis and create plots"""
    # Set up paths
    script_dir = Path(__file__).parent
    val_plots_dir = script_dir / "val-plots"
    output_path = script_dir / "best_specificity_scores.png"
    
    print("🔍 Analyzing concept specificity scores...")
    print(f"📂 Looking in: {val_plots_dir}")
    
    # Check if val-plots directory exists
    if not val_plots_dir.exists():
        print(f"❌ Directory not found: {val_plots_dir}")
        print("Please run this script from the concept-val-test directory")
        return
    
    # Load data from all concept folders
    concept_data = load_concept_data(val_plots_dir)
    
    if not concept_data:
        print("❌ No valid data found!")
        return
    
    # Print summary table
    print_summary_table(concept_data)
    
    # Create and display plot
    print(f"\n📊 Creating specificity plot...")
    fig, ax = create_specificity_plot(concept_data, output_path, top_n=15)
    
    # Create perturbation type distribution plot across all configurations
    print(f"\n📊 Summarizing perturbation types across all configurations...")
    perturbation_output_path = script_dir / "perturbation_type_distribution.png"
    create_perturbation_type_distribution_plot(val_plots_dir, perturbation_output_path)

    # Compare high-specificity performance by layer depth
    print(f"\n📊 Comparing high-specificity performance by layer depth...")
    layer_output_path = script_dir / "layer_performance_high_specificity.png"
    create_layer_performance_plot(val_plots_dir, layer_output_path)
    
    print(f"\n✅ Analysis complete!")
    print(f"   - Analyzed {len(concept_data)} concepts")
    print(f"   - Plot saved to: {output_path}")
    print(f"   - Perturbation distribution saved to: {perturbation_output_path}")
    print(f"   - Layer performance comparison saved to: {layer_output_path}")

if __name__ == "__main__":
    # Set style for better-looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    main()