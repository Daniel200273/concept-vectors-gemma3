# existing file (was empty)
"""Create summary tables and plots from a single validation JSON file.

This script reads a JSON file containing an array of validation result objects
and builds a compact table and three grouped bar plots showing, for each
configuration, the concept vs unrelated scores for BLEU, ROUGE and SBERT.

Usage:
	python plot_validation_results.py -i "The Lord of the Rings_validation_results.json" -o outdir

Outputs saved to the output directory:
  - summary_table.csv             (tabular CSV of extracted scores)
  - bleu_concept_unrelated.png    (grouped bar plot)
  - rouge_concept_unrelated.png
  - sbert_concept_unrelated.png

The script is defensive about missing fields and will fill missing scores with NaN.
"""

from pathlib import Path
import argparse
import json
import math
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def safe_get(obj, key):
	"""Return obj[key] if present, else None."""
	return obj.get(key) if isinstance(obj, dict) else None


def make_label(obj):
	"""Create a short human-friendly label for a configuration object.

	Prefer `combination_name` if present. Otherwise build from perturbation
	info and vector counts.
	"""
	cn = safe_get(obj, "combination_name")
	if cn:
		return cn
	pt = safe_get(obj, "perturbation_type") or "UNK"
	param = safe_get(obj, "perturbation_param")
	nv = safe_get(obj, "num_vectors_ablated") or safe_get(obj, "actual_vector_count") or safe_get(obj, "vector_count")
	parts = [str(pt).upper()]
	if param is not None:
		parts.append(str(param))
	if nv is not None:
		parts.append(f"n={nv}")
	return "_".join(parts)


def extract_rows(results):
	"""Turn list of result dicts into list of row dicts for a DataFrame."""
	rows = []
	for i, r in enumerate(results):
		label = make_label(r)
		row = {
			"index": i,
			"label": label,
			"concept_bleu": safe_get(r, "concept_bleu_score"),
			"unrelated_bleu": safe_get(r, "unrelated_bleu_score"),
			"concept_rouge": safe_get(r, "concept_rouge_score"),
			"unrelated_rouge": safe_get(r, "unrelated_rouge_score"),
			"concept_sbert": safe_get(r, "concept_sbert_score"),
			"unrelated_sbert": safe_get(r, "unrelated_sbert_score"),
			"bleu_specificity": safe_get(r, "bleu_specificity"),
			"rouge_specificity": safe_get(r, "rouge_specificity"),
			"sbert_specificity": safe_get(r, "sbert_specificity"),
			"combined_specificity": safe_get(r, "combined_specificity"),
			"perturbation_type": safe_get(r, "perturbation_type"),
			"combination_name": safe_get(r, "combination_name"),
			"num_vectors_ablated": safe_get(r, "num_vectors_ablated"),
			"actual_vector_count": safe_get(r, "actual_vector_count"),
			"vector_count": safe_get(r, "vector_count"),
		}
		rows.append(row)
	return rows


def plot_grouped(df, metric_concept, metric_unrelated, metric_name, outpath, figsize=(12, 6), concept_name=None):
	"""Create a grouped bar plot for concept vs unrelated for the named metric.

	df should contain columns 'label', metric_concept and metric_unrelated.
	"""
	plot_df = df[["label", metric_concept, metric_unrelated]].copy()
	# convert to long format for seaborn
	plot_df = plot_df.set_index("label")
	melted = plot_df.reset_index().melt(id_vars=["label"], var_name="kind", value_name="score")

	sns.set(style="whitegrid")
	# Vertical grouped barplot: configuration labels on x, scores on y.
	# Scale width with number of labels but cap it to avoid extremely wide images.
	num_labels = len(plot_df)
	width_per_label = 0.35
	width = max(6, min(20, int(num_labels * width_per_label)))
	height = max(4, figsize[1])
	plt.figure(figsize=(width, height))
	ax = sns.barplot(data=melted, x="label", y="score", hue="kind")
	title = f"{metric_name} — concept vs unrelated"
	if concept_name:
		title = f"{title} — {concept_name}"
	ax.set_title(title)
	ax.set_xlabel("")
	ax.set_ylabel(metric_name)
	# Decide label font size based on number of labels (continuous decrease):
	# - Larger number of labels -> smaller font size.
	# - Caps keep sizes in a readable range.
	max_labelsize = 12
	min_labelsize = 5
	# scale factor controls how quickly size decreases as labels increase
	scale_factor = 0.15
	x_labelsize = int(max(min_labelsize, min(max_labelsize, round(max_labelsize - num_labels * scale_factor))))
	# Rotate x labels if there are many configurations
	if num_labels > 8:
		plt.xticks(rotation=45, ha="right")
	ax.tick_params(axis='x', labelsize=x_labelsize)
	plt.tight_layout()
	plt.savefig(outpath, dpi=150)
	plt.close()


def plot_combined_specificity(df, outpath, figsize=(10, 8), concept_name=None):
	"""Create a single combined specificity plot with BLEU/ROUGE/SBERT per config.

	The plot is horizontal (configs on y, specificity on x) with a hue for the
	specificity type. Rows where all specificity metrics are NaN are dropped.
	"""
	spec_cols = ["bleu_specificity", "rouge_specificity", "sbert_specificity"]
	for c in spec_cols:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors="coerce")

	# Keep only rows where at least one specificity is present
	if not any(c in df.columns for c in spec_cols):
		return
	plot_df = df[["label"] + [c for c in spec_cols if c in df.columns]].copy()
	# drop rows where all spec values are NaN
	plot_df = plot_df.dropna(how="all", subset=[c for c in spec_cols if c in plot_df.columns])
	if plot_df.empty:
		return

	melted = plot_df.melt(id_vars=["label"], var_name="metric", value_name="specificity")
	# drop NaNs
	melted = melted.dropna(subset=["specificity"])
	if melted.empty:
		return

	sns.set(style="whitegrid")
	num_labels = plot_df.shape[0]
	height_per_label = 0.35
	height = max(3, num_labels * height_per_label)
	width = max(6, figsize[0])
	plt.figure(figsize=(width, height))
	ax = sns.barplot(data=melted, y="label", x="specificity", hue="metric")
	title = "Specificity by configuration (BLEU / ROUGE / SBERT)"
	if concept_name:
		title = f"{title} — {concept_name}"
	ax.set_title(title)
	ax.set_ylabel("")
	ax.set_xlabel("Specificity")
	# Adjust y-axis label font size (labels are the configuration names).
	# Use the same continuous decrease strategy as above but with a slightly
	# gentler scaling for long lists.
	num_labels = plot_df.shape[0]
	max_labelsize = 12
	min_labelsize = 5
	scale_factor = 0.12
	y_labelsize = int(max(min_labelsize, min(max_labelsize, round(max_labelsize - num_labels * scale_factor))))
	ax.tick_params(axis='y', labelsize=y_labelsize)
	plt.legend(title="metric")
	plt.tight_layout()
	plt.savefig(outpath, dpi=150)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Plot validation results panorama (BLEU/ROUGE/SBERT concept vs unrelated)")
	parser.add_argument("-i", "--input", required=True, help="Path to a validation JSON file (array of result objects)")
	parser.add_argument("-o", "--outdir", default="plots", help="Output directory to save CSV and images")
	parser.add_argument("--top", type=int, default=0, help="If >0, keep only the top N configurations (by index order) for plotting")
	args = parser.parse_args()

	inp = Path(args.input)
	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	with inp.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list):
		raise SystemExit("Expected top-level JSON array of result objects")

	rows = extract_rows(data)
	df = pd.DataFrame(rows)

	# If label duplicates exist, make them unique by appending index
	if df["label"].duplicated().any():
		df["label"] = df.apply(lambda r: f"{r['label']} (#{r['index']})" if (df['label'] == r['label']).sum() > 1 else r['label'], axis=1)

	# Optionally limit number of configs
	if args.top and args.top > 0:
		df = df.head(args.top)

	# Save a CSV with the numeric summary
	csv_path = outdir / "summary_table.csv"
	df.to_csv(csv_path, index=False)

	# Replace missing values with NaN for plotting
	numeric_cols = ["concept_bleu", "unrelated_bleu", "concept_rouge", "unrelated_rouge", "concept_sbert", "unrelated_sbert"]
	for c in numeric_cols:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors="coerce")

	# Convert specificity columns to numeric (if present)
	spec_cols = ["bleu_specificity", "rouge_specificity", "sbert_specificity", "combined_specificity"]
	for c in spec_cols:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors="coerce")

	# If all values are NaN for a metric, skip plotting that metric
	metrics = [
		("concept_bleu", "unrelated_bleu", "BLEU"),
		("concept_rouge", "unrelated_rouge", "ROUGE"),
		("concept_sbert", "unrelated_sbert", "SBERT"),
	]

	concept_label = inp.stem.replace('_validation_results', '')
	for a, b, name in metrics:
		if df[[a, b]].notna().any().any():
			outpath = outdir / f"{name.lower()}_concept_unrelated.png"
			# Keep width reasonable: cap at 12 inches and scale height instead
			width = 12
			plot_grouped(df, a, b, name, outpath, figsize=(width, 6), concept_name=concept_label)

	# Single combined specificity plot (BLEU/ROUGE/SBERT)
	spec_out = outdir / "combined_specificity.png"
	plot_combined_specificity(df, spec_out, figsize=(10, max(4, int(df.shape[0] * 0.35))), concept_name=concept_label)

	print(textwrap.dedent(f"""
	Done.
	Input: {inp}
	Summary CSV: {csv_path}
	Plots saved to: {outdir}
	"""))


if __name__ == "__main__":
	main()

