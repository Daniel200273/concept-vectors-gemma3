#!/usr/bin/env python3
"""
Debug script to analyze candidate_vectors.npy
Analyzes min/max values and trends across layers for the extracted vectors.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_vectors_and_metadata():
    """Load the candidate vectors and their metadata."""
    vectors_path = Path("extracted_vectors/candidate_vectors.npy")
    metadata_path = Path("extracted_vectors/candidate_vectors_metadata.json")
    
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vectors_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print("Loading vectors...")
    vectors = np.load(vectors_path)
    
    print("Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return vectors, metadata

def analyze_global_statistics(vectors):
    """Analyze global statistics of all vectors."""
    print("\n" + "="*60)
    print("GLOBAL VECTOR STATISTICS")
    print("="*60)
    
    print(f"Vector shape: {vectors.shape}")
    print(f"Total elements: {vectors.size:,}")
    
    # Process in chunks to avoid memory issues with very large arrays
    chunk_size = 1000000  # 1M elements at a time
    
    if vectors.size <= chunk_size:
        # Small enough to process all at once
        global_min = np.min(vectors)
        global_max = np.max(vectors)
        global_mean = np.mean(vectors)
        global_std = np.std(vectors)
        global_median = np.median(vectors)
    else:
        # Process in chunks
        print("Processing large array in chunks...")
        flat_vectors = vectors.flatten()
        
        global_min = float('inf')
        global_max = float('-inf')
        sum_vals = 0.0
        sum_sq = 0.0
        
        for i in range(0, len(flat_vectors), chunk_size):
            chunk = flat_vectors[i:i+chunk_size]
            global_min = min(global_min, np.min(chunk))
            global_max = max(global_max, np.max(chunk))
            sum_vals += np.sum(chunk)
            sum_sq += np.sum(chunk**2)
        
        global_mean = sum_vals / len(flat_vectors)
        global_std = np.sqrt(sum_sq / len(flat_vectors) - global_mean**2)
        global_median = np.median(flat_vectors[::100])  # Sample for median
    
    print(f"Global minimum: {global_min:.6f}")
    print(f"Global maximum: {global_max:.6f}")
    print(f"Global mean: {global_mean:.6f}")
    print(f"Global std: {global_std:.6f}")
    print(f"Global median: {global_median:.6f}")
    print(f"Value range: {global_max - global_min:.6f}")
    
    # Check for unusual values (using sampling for large arrays)
    sample_size = min(100000, vectors.size)
    sample_indices = np.random.choice(vectors.size, sample_size, replace=False)
    sample_vectors = vectors.flatten()[sample_indices]
    
    num_zeros = np.sum(sample_vectors == 0)
    num_positive = np.sum(sample_vectors > 0)
    num_negative = np.sum(sample_vectors < 0)
    
    print(f"\nValue distribution (sampled):")
    print(f"Zeros: {num_zeros:,} ({100 * num_zeros / sample_size:.2f}%)")
    print(f"Positive: {num_positive:,} ({100 * num_positive / sample_size:.2f}%)")
    print(f"Negative: {num_negative:,} ({100 * num_negative / sample_size:.2f}%)")
    
    # Check for extreme values
    percentiles = [0.1, 1, 5, 95, 99, 99.9]
    print(f"\nPercentiles (sampled):")
    for p in percentiles:
        val = np.percentile(sample_vectors, p)
        print(f"  {p:5.1f}%: {val:.6f}")

def analyze_layer_statistics(vectors, metadata):
    """Analyze statistics for each layer."""
    print("\n" + "="*60)
    print("LAYER-WISE STATISTICS")
    print("="*60)
    
    num_layers = metadata['metadata']['total_layers']
    vectors_per_layer = metadata['metadata']['vector_dimension']  # This should be 6912
    vector_dim = metadata['metadata']['vector_dimension']  # This should be 1152
    
    # Actually, let me correct this based on the metadata
    vectors_per_layer = 6912  # From the metadata, each layer has 6912 vectors
    vector_dim = 1152  # Each vector has 1152 dimensions
    
    layer_stats = []
    
    for layer_idx in range(num_layers):
        # Calculate the slice for this layer
        start_idx = layer_idx * vectors_per_layer
        end_idx = start_idx + vectors_per_layer
        
        layer_vectors = vectors[start_idx:end_idx]
        
        # Calculate statistics
        layer_min = np.min(layer_vectors)
        layer_max = np.max(layer_vectors)
        layer_mean = np.mean(layer_vectors)
        layer_std = np.std(layer_vectors)
        layer_norm_mean = np.mean(np.linalg.norm(layer_vectors, axis=1))
        
        # Store stats
        stats = {
            'layer': layer_idx,
            'min': layer_min,
            'max': layer_max,
            'mean': layer_mean,
            'std': layer_std,
            'range': layer_max - layer_min,
            'mean_norm': layer_norm_mean,
            'num_vectors': len(layer_vectors)
        }
        layer_stats.append(stats)
        
        print(f"Layer {layer_idx:2d}: min={layer_min:8.4f}, max={layer_max:8.4f}, "
              f"mean={layer_mean:8.4f}, std={layer_std:8.4f}, "
              f"range={layer_max - layer_min:8.4f}, mean_norm={layer_norm_mean:8.4f}")
    
    return layer_stats

def analyze_trends(layer_stats):
    """Analyze trends across layers."""
    print("\n" + "="*60)
    print("TREND ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(layer_stats)
    
    # Calculate correlations with layer index
    metrics = ['min', 'max', 'mean', 'std', 'range', 'mean_norm']
    
    print("Correlation with layer depth:")
    for metric in metrics:
        corr = df['layer'].corr(df[metric])
        print(f"  {metric:10s}: {corr:7.4f}")
    
    # Find layers with extreme values
    print(f"\nLayers with extreme values:")
    for metric in metrics:
        min_layer = df.loc[df[metric].idxmin(), 'layer']
        max_layer = df.loc[df[metric].idxmax(), 'layer']
        min_val = df[metric].min()
        max_val = df[metric].max()
        print(f"  {metric:10s}: min at layer {min_layer:2d} ({min_val:.4f}), "
              f"max at layer {max_layer:2d} ({max_val:.4f})")
    
    # Look for patterns in early vs late layers
    early_layers = df[df['layer'] < 9]  # First third
    middle_layers = df[(df['layer'] >= 9) & (df['layer'] < 18)]  # Middle third
    late_layers = df[df['layer'] >= 18]  # Last third
    
    print(f"\nLayer group comparisons:")
    print(f"{'Metric':12s} {'Early (0-8)':>12s} {'Middle (9-17)':>14s} {'Late (18-25)':>13s}")
    print("-" * 60)
    
    for metric in metrics:
        early_mean = early_layers[metric].mean()
        middle_mean = middle_layers[metric].mean()
        late_mean = late_layers[metric].mean()
        print(f"{metric:12s} {early_mean:12.4f} {middle_mean:14.4f} {late_mean:13.4f}")

def create_visualizations(layer_stats, vectors, metadata):
    """Create intuitive visualizations of the data."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    print(f"Created/using plots directory: {plots_dir}")
    
    df = pd.DataFrame(layer_stats)
    
    # Set up the plotting style for better readability
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#048A81']
    
    # Create main analysis figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # 1. QUANTIZATION DISCOVERY - Show the discrete nature (use smaller sample)
    ax1 = fig.add_subplot(gs[0, :])
    sample_size = min(10000, vectors.size)  # Limit sample size
    sample_indices = np.random.choice(vectors.size, sample_size, replace=False)
    sample_vectors = vectors.flatten()[sample_indices]
    unique_vals = np.unique(sample_vectors)
    
    ax1.hist(sample_vectors, bins=min(100, len(unique_vals)), alpha=0.7, color=colors[0], edgecolor='black', linewidth=0.5)
    ax1.set_title('Quantized Value Distribution (8-bit)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Weight Values')
    ax1.set_ylabel('Frequency')
    
    # Clear sample data from memory
    del sample_vectors, sample_indices
    
    # 2. LAYER CONSISTENCY - Box plot to show how similar layers are
    ax2 = fig.add_subplot(gs[1, 0])
    norm_data = [df['mean_norm'].values]
    bp = ax2.boxplot(norm_data, patch_artist=True, labels=['All Layers'])
    bp['boxes'][0].set_facecolor(colors[1])
    ax2.set_title('Vector Norm Consistency', fontweight='bold')
    ax2.set_ylabel('Mean Vector Norm')
    
    # 3. STABILITY INDICATOR - Standard deviation consistency
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(range(len(df)), df['std'], color=colors[2], alpha=0.8, width=0.8)
    ax3.set_title('Weight Stability by Layer', fontweight='bold')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_xticks(range(0, len(df), 5))
    ax3.set_xticklabels(range(0, len(df), 5))
    
    # 4. ARCHITECTURE UNIFORMITY - Show how changes would be minimal
    ax4 = fig.add_subplot(gs[1, 2])
    layer_groups = ['Early\n(0-8)', 'Middle\n(9-17)', 'Late\n(18-25)']
    early_norm = df[df['layer'] < 9]['mean_norm'].mean()
    middle_norm = df[(df['layer'] >= 9) & (df['layer'] < 18)]['mean_norm'].mean()
    late_norm = df[df['layer'] >= 18]['mean_norm'].mean()
    group_norms = [early_norm, middle_norm, late_norm]
    
    bars = ax4.bar(layer_groups, group_norms, color=[colors[3], colors[4], colors[5]], alpha=0.8)
    ax4.set_title('Architecture Uniformity', fontweight='bold')
    ax4.set_ylabel('Mean Vector Norm')
    
    # Add value labels on bars
    for bar, norm in zip(bars, group_norms):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{norm:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. OUTLIER DETECTION - Highlight unusual layers
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(df['layer'], df['mean_norm'], c=colors[0], s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax5.plot(df['layer'], df['mean_norm'], 'k--', alpha=0.5, linewidth=1)
    ax5.set_title('Layer-wise Vector Norms', fontweight='bold')
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Mean Vector Norm')
    ax5.grid(True, alpha=0.3)
    
    # 6. SCALE COMPARISON
    ax6 = fig.add_subplot(gs[2, 1])
    quantization_step = 0.078125
    typical_std = df['std'].mean()
    
    categories = ['Quantization\nStep', 'Weight\nVariation']
    values = [quantization_step, typical_std]
    colors_bar = [colors[0], colors[2]]
    
    bars = ax6.bar(categories, values, color=colors_bar, alpha=0.8)
    ax6.set_title('Scale Comparison', fontweight='bold')
    ax6.set_ylabel('Magnitude')
    ax6.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. MIN/MAX RANGE
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.fill_between(df['layer'], df['min'], df['max'], alpha=0.3, color=colors[1], label='Value Range')
    ax7.plot(df['layer'], df['mean'], color=colors[1], linewidth=2, label='Mean')
    ax7.set_title('Value Range by Layer', fontweight='bold')
    ax7.set_xlabel('Layer')
    ax7.set_ylabel('Value')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Gemma-3-1B Weight Analysis', fontsize=16, fontweight='bold', y=0.95)
    
    # Save the main plot
    output_path = plots_dir / "intuitive_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Main visualization saved as: {output_path}")
    plt.close()  # Close the figure to free memory
    
    # Create a separate detailed quantization plot
    print("Creating detailed quantization analysis...")
    plt.figure(figsize=(12, 8))
    
    # Sample data for detailed quantization analysis (use smaller sample)
    sample_size = min(50000, vectors.size)
    sample_indices = np.random.choice(vectors.size, sample_size, replace=False)
    sample_data = vectors.flatten()[sample_indices]
    unique_values = np.unique(sample_data)
    
    plt.subplot(2, 2, 1)
    plt.hist(sample_data, bins=len(unique_values), alpha=0.7, color=colors[0], edgecolor='black')
    plt.title('Full Value Distribution')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    value_counts = pd.Series(sample_data).value_counts().sort_index()
    plt.bar(range(len(value_counts)), value_counts.values, color=colors[1])
    plt.title(f'Unique Values (Total: {len(unique_values)})')
    plt.xlabel('Value Index')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 3)
    # Show the exact quantization steps
    step_size = unique_values[1] - unique_values[0] if len(unique_values) > 1 else 0
    plt.plot(unique_values, 'o-', color=colors[2])
    plt.title(f'Quantization Steps (Δ = {step_size:.6f})')
    plt.xlabel('Step Index')
    plt.ylabel('Value')
    
    plt.subplot(2, 2, 4)
    # Show how this compares to IEEE 754 expectations
    theoretical_8bit = np.linspace(-0.078125, 0.078125, 256)
    plt.plot(theoretical_8bit, label='8-bit expectation', alpha=0.7)
    plt.plot(unique_values, 'o', label='Actual values', markersize=3)
    plt.title('Quantization Verification')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.suptitle('Detailed Quantization Analysis', fontsize=14, y=0.98)
    
    quantization_path = plots_dir / "quantization_analysis.png"
    plt.savefig(quantization_path, dpi=300, bbox_inches='tight')
    print(f"Quantization analysis saved as: {quantization_path}")
    plt.close()  # Close the figure to free memory
    
    print("All visualizations saved successfully!")

def save_analysis_results(layer_stats, vectors, metadata):
    """Save analysis results to a JSON file."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Prepare results dictionary
    results = {
        'analysis_summary': {
            'total_vectors': len(vectors),
            'vector_dimension': vectors.shape[1] if len(vectors.shape) > 1 else 1,
            'num_layers': metadata['metadata']['total_layers'],
            'global_statistics': {
                'min': float(np.min(vectors)),
                'max': float(np.max(vectors)),
                'mean': float(np.mean(vectors)),
                'std': float(np.std(vectors)),
                'median': float(np.median(vectors))
            }
        },
        'layer_statistics': layer_stats,
        'metadata': metadata['metadata']
    }
    
    # Convert numpy types to Python types for JSON serialization
    for layer_stat in results['layer_statistics']:
        for key, value in layer_stat.items():
            if isinstance(value, (np.integer, np.floating)):
                layer_stat[key] = float(value)
    
    # Save to file
    output_path = "debug_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis results saved to: {output_path}")

def main():
    """Main analysis function."""
    print("Candidate Vectors Debug Analysis")
    print("=" * 60)
    
    try:
        # Load data
        print("Step 1/6: Loading data...")
        vectors, metadata = load_vectors_and_metadata()
        
        # Global analysis
        print("Step 2/6: Global analysis...")
        analyze_global_statistics(vectors)
        
        # Layer-wise analysis
        print("Step 3/6: Layer-wise analysis...")
        layer_stats = analyze_layer_statistics(vectors, metadata)
        
        # Trend analysis
        print("Step 4/6: Trend analysis...")
        analyze_trends(layer_stats)
        
        # Create visualizations
        print("Step 5/6: Creating visualizations...")
        create_visualizations(layer_stats, vectors, metadata)
        
        # Save results
        print("Step 6/6: Saving results...")
        save_analysis_results(layer_stats, vectors, metadata)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Files generated:")
        print("  - plots/intuitive_analysis.png (main insights visualization)")
        print("  - plots/quantization_analysis.png (detailed quantization study)")
        print("  - debug_analysis_results.json (detailed results)")
        
        # Final cleanup
        plt.close('all')  # Close all remaining figures
        print("\nAll operations completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        plt.close('all')

if __name__ == "__main__":
    main()
