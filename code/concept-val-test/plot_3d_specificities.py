#!/usr/bin/env python3
"""
Create a 3D scatter plot of specificity scores for a chosen concept.

The script reads `val-plots/<CONCEPT>/summary_table.csv` and plots one point per
configuration using the three specificity metrics:
 - BLEU specificity (x)
 - ROUGE specificity (y)
 - SBERT specificity (z)

Point size is proportional to the `combined_specificity` column. The generated
figure is saved in the same concept folder as `<concept>_specificities_3d.png`.

Set the `CONCEPT_NAME` global variable below to pick the concept folder.

Usage:
    python plot_3d_specificities.py

"""
from pathlib import Path
import sys
import math

# Choose the concept folder name (must match a folder in val-plots/)
CONCEPT_NAME = "Ducati"
# Initial 3D view angles (elevation less = more side-on, azimuth rotates around)
# Lower elevation (e.g. 10-20) gives a less top-down perspective
VIEW_ELEV = 15
VIEW_AZIM = 10

def main():
    # Use non-interactive backend to allow running on headless clusters
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        pass

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
    import seaborn as sns

    script_dir = Path(__file__).parent
    val_plots_dir = script_dir / "val-plots"

    concept_dir = val_plots_dir / CONCEPT_NAME
    csv_path = concept_dir / "summary_table.csv"

    if not concept_dir.exists():
        print(f"❌ Concept folder not found: {concept_dir}")
        sys.exit(1)
    if not csv_path.exists():
        print(f"❌ summary_table.csv not found for concept '{CONCEPT_NAME}': {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Check required columns
    required = ['bleu_specificity', 'rouge_specificity', 'sbert_specificity', 'combined_specificity']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns in {csv_path}: {missing}")
        sys.exit(1)

    # Extract scores
    x = df['bleu_specificity'].to_numpy(dtype=float)
    y = df['rouge_specificity'].to_numpy(dtype=float)
    z = df['sbert_specificity'].to_numpy(dtype=float)
    combined = df['combined_specificity'].to_numpy(dtype=float)

    # Compute marker sizes from combined specificity. Map to a reasonable point size range.
    # Handle constant or negative values gracefully.
    if math.isfinite(combined.max()) and math.isfinite(combined.min()) and combined.max() != combined.min():
        norm = (combined - combined.min()) / (combined.max() - combined.min())
    else:
        norm = np.full_like(combined, 0.5)

    sizes = 50 + norm * 450  # sizes in points^2 for matplotlib scatter

    # Color by perturbation type if available, else use a single color palette
    if 'perturbation_type' in df.columns:
        categories = df['perturbation_type'].fillna('unknown').astype(str)
        unique_cats = categories.unique()
        palette = dict(zip(unique_cats, sns.color_palette(n_colors=len(unique_cats))))
        colors = categories.map(palette)
    else:
        colors = sns.color_palette('husl', n_colors=1)[0]

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter: draw normal points and star-shaped points for high combined specificity
    threshold = 0.3
    try:
        mask = (combined > threshold)
    except Exception:
        mask = np.zeros_like(combined, dtype=bool)

    # Robustly detect per-point color arrays (lists, numpy arrays, or pandas Series)
    per_point_colors = False
    try:
        if hasattr(colors, '__len__') and len(colors) == len(x):
            per_point_colors = True
            # convert to numpy array of color tuples
            colors_arr = np.array(list(colors))
        else:
            colors_arr = None
    except Exception:
        colors_arr = None
        per_point_colors = False

    # Non-star points
    if (~mask).any():
        if per_point_colors:
            cols_non = list(colors_arr[~mask])
            ax.scatter(x[~mask], y[~mask], z[~mask], s=sizes[~mask], c=cols_non, alpha=0.8, edgecolor='k')
        else:
            ax.scatter(x[~mask], y[~mask], z[~mask], s=sizes[~mask], c=colors, alpha=0.8, edgecolor='k')

    # Star points (highlighted when combined_specificity > threshold)
    if mask.any():
        star_kwargs = dict(marker='*', s=sizes[mask], linewidths=0.6, alpha=0.95, edgecolor='k')
        if per_point_colors:
            cols_star = list(colors_arr[mask])
            ax.scatter(x[mask], y[mask], z[mask], c=cols_star, **star_kwargs)
        else:
            ax.scatter(x[mask], y[mask], z[mask], c=colors, **star_kwargs)

    # Labels and title
    ax.set_xlabel('BLEU Specificity')
    ax.set_ylabel('ROUGE Specificity')
    ax.set_zlabel('SBERT Specificity')
    ax.set_title(f"3D Specificity Scatter — {CONCEPT_NAME}")

    # Ensure axes include zero for clarity and add light reference planes through the origin
    def extended_limits(values):
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if math.isclose(vmin, vmax):
            delta = abs(vmin) * 0.1 if vmin != 0 else 0.5
            vmin -= delta
            vmax += delta
        if vmin > 0:
            vmin = 0.0
        if vmax < 0:
            vmax = 0.0
        padding = max((vmax - vmin) * 0.05, 0.05)
        return vmin - padding, vmax + padding

    xlim = extended_limits(x)
    ylim = extended_limits(y)
    zlim = extended_limits(z)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # (Removed semi-transparent zero planes to reduce visual clutter.)

    # Draw axes with distinct styles for negative/positive directions to clarify sign
    neg_kwargs = dict(color='black', linewidth=1.2, linestyle='--', alpha=0.7)
    pos_kwargs = dict(color='red', linewidth=2.2, linestyle='-', alpha=0.9)

    # X axis: negative segment (xlim[0] -> 0), positive segment (0 -> xlim[1])
    ax.plot([xlim[0], 0.0], [0, 0], [0, 0], **neg_kwargs)
    ax.plot([0.0, xlim[1]], [0, 0], [0, 0], **pos_kwargs)

    # Y axis: negative segment (ylim[0] -> 0), positive segment (0 -> ylim[1])
    ax.plot([0, 0], [ylim[0], 0.0], [0, 0], **neg_kwargs)
    ax.plot([0, 0], [0.0, ylim[1]], [0, 0], **pos_kwargs)

    # Z axis: negative segment (zlim[0] -> 0), positive segment (0 -> zlim[1])
    ax.plot([0, 0], [0, 0], [zlim[0], 0.0], **neg_kwargs)
    ax.plot([0, 0], [0, 0], [0.0, zlim[1]], **pos_kwargs)

    # Add simple arrows on the positive direction of each axis to indicate +
    # Use explicit shaft lines + triangular tip markers for robust rendering
    try:
        def arrow_length(span):
            return max(span * 0.08, 0.05)

        x_span_pos = max(xlim[1], 0.0) - 0.0
        y_span_pos = max(ylim[1], 0.0) - 0.0
        z_span_pos = max(zlim[1], 0.0) - 0.0

        arrow_x = arrow_length(x_span_pos)
        arrow_y = arrow_length(y_span_pos)
        arrow_z = arrow_length(z_span_pos)

        # Helper to draw shaft + tip
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        def draw_shaft_and_tip(tail, tip, axis):
            # tail, tip: scalar positions along axis
            # Make arrows smaller: thinner shaft and shorter tip
            shaft_lw = 1.0
            # tip geometry (scaled down)
            tip_length = max(arrow_length(abs(tip - tail)) * 0.5, 0.03)
            tip_radius = tip_length * 0.45

            # draw shaft
            if axis == 'x':
                ax.plot([tail, tip], [0, 0], [0, 0], color='red', linewidth=shaft_lw)
            elif axis == 'y':
                ax.plot([0, 0], [tail, tip], [0, 0], color='red', linewidth=shaft_lw)
            else:  # 'z'
                ax.plot([0, 0], [0, 0], [tail, tip], color='red', linewidth=shaft_lw)

            # build triangular pyramid (apex at tip, triangular base perpendicular to axis)
            apex = None
            base_center = None
            if axis == 'x':
                apex = np.array([tip, 0.0, 0.0])
                base_center = np.array([tip - tip_length, 0.0, 0.0])
                # base triangle in YZ plane
                base_pts = [base_center + np.array([0.0, tip_radius * np.cos(a), tip_radius * np.sin(a)]) for a in (0, 2*np.pi/3, 4*np.pi/3)]
            elif axis == 'y':
                apex = np.array([0.0, tip, 0.0])
                base_center = np.array([0.0, tip - tip_length, 0.0])
                # base triangle in XZ plane
                base_pts = [base_center + np.array([tip_radius * np.cos(a), 0.0, tip_radius * np.sin(a)]) for a in (0, 2*np.pi/3, 4*np.pi/3)]
            else:
                apex = np.array([0.0, 0.0, tip])
                base_center = np.array([0.0, 0.0, tip - tip_length])
                # base triangle in XY plane
                base_pts = [base_center + np.array([tip_radius * np.cos(a), tip_radius * np.sin(a), 0.0]) for a in (0, 2*np.pi/3, 4*np.pi/3)]

            # faces: three triangles (apex, base_i, base_{i+1}) and base face
            faces = []
            for i in range(3):
                p1 = base_pts[i]
                p2 = base_pts[(i+1) % 3]
                faces.append([apex.tolist(), p1.tolist(), p2.tolist()])
            # base face (optional, helps with appearance)
            faces.append([base_pts[0].tolist(), base_pts[1].tolist(), base_pts[2].tolist()])

            poly = Poly3DCollection(faces, facecolors='red', edgecolors='k', linewidths=0.5, alpha=1.0)
            ax.add_collection3d(poly)

        # X axis
        if xlim[1] > 0 and x_span_pos > 0:
            tail_x = max(0.0, xlim[1] - arrow_x)
            tip_x = xlim[1]
        else:
            tail_x = 0.0
            tip_x = tail_x + max(arrow_x, 0.08)
        draw_shaft_and_tip(tail_x, tip_x, 'x')

        # Y axis
        if ylim[1] > 0 and y_span_pos > 0:
            tail_y = max(0.0, ylim[1] - arrow_y)
            tip_y = ylim[1]
        else:
            tail_y = 0.0
            tip_y = tail_y + max(arrow_y, 0.08)
        draw_shaft_and_tip(tail_y, tip_y, 'y')

        # Z axis
        if zlim[1] > 0 and z_span_pos > 0:
            tail_z = max(0.0, zlim[1] - arrow_z)
            tip_z = zlim[1]
        else:
            tail_z = 0.0
            tip_z = tail_z + max(arrow_z, 0.08)
        draw_shaft_and_tip(tail_z, tip_z, 'z')
    except Exception:
        pass

    # Create a small legend for perturbation types
    if 'perturbation_type' in df.columns:
        # create proxy artists
        from matplotlib.lines import Line2D
        proxies = [Line2D([0], [0], marker='o', color='w', label=cat,
                          markerfacecolor=palette[cat], markersize=8, markeredgecolor='k')
                   for cat in unique_cats]
        ax.legend(handles=proxies, title='perturbation_type', loc='upper left')

    # Set the desired viewpoint (elevation, azimuth)
    try:
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    except Exception:
        pass

    plt.tight_layout()

    out_path = concept_dir / f"{CONCEPT_NAME}_specificities_3d.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved 3D specificity plot to: {out_path}")

if __name__ == '__main__':
    main()
