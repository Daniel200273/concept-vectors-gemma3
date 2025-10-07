#!/usr/bin/env python3
"""Batch-run plot generation for all validation JSON files.

This script finds files in the `validation-results/` folder that end with
`_validation_results.json`, creates a matching output folder under
`val-plots/<Concept Name>/` and invokes `plot_validation_results.py` for each
file. It handles filenames with spaces correctly.

Examples:
  python batch_generate_plots.py --limit 5 --top 15
  python batch_generate_plots.py --input-dir validation-results --output-root val-plots
"""
import os
import sys
import subprocess
import argparse
import re


def find_json_files(input_dir):
    files = []
    # Accept names like: <Concept>_validation_results.json OR
    # <Concept>_validation_results-v5.json (optional -v<digits> suffix)
    pattern = re.compile(r'_validation_results(?:-v\d+)?\.json$', re.IGNORECASE)
    for name in os.listdir(input_dir):
        if pattern.search(name) and os.path.isfile(os.path.join(input_dir, name)):
            files.append(name)
    return sorted(files)


def concept_name_from_filename(fn):
    # Remove extension, optional -v<number> suffix, and the trailing
    # '_validation_results' marker to yield a clean concept name.
    base = os.path.splitext(fn)[0]
    # strip version suffix like -v5 or -v12
    base = re.sub(r'-v\d+$', '', base, flags=re.IGNORECASE)
    suffix = '_validation_results'
    if base.lower().endswith(suffix):
        return base[:-len(suffix)]
    return base


def main():
    parser = argparse.ArgumentParser(description='Batch generate plots for validation JSON files')
    parser.add_argument('-i', '--input-dir', default='validation-results', help='Directory with validation JSON files')
    args = parser.parse_args()

    input_dir = args.input_dir
    out_root = 'val-plots'

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(2)

    files = find_json_files(input_dir)

    if not files:
        print(f"No validation JSON files found in {input_dir}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_script = os.path.join(script_dir, 'plot_validation_results.py')
    if not os.path.isfile(plot_script):
        print(f"plot_validation_results.py not found in {script_dir}", file=sys.stderr)
        sys.exit(3)

    for name in files:
        concept = concept_name_from_filename(name)
        outdir = os.path.join(out_root, concept)
        os.makedirs(outdir, exist_ok=True)

        input_path = os.path.join(input_dir, name)

        # Build command as a list to safely handle spaces in paths and use current Python
        cmd = [sys.executable, plot_script, '-i', input_path, '-o', outdir]

        print(f"\n=== Processing: {name} ===")
        print('Command:', ' '.join([f"'{c}'" if ' ' in c else c for c in cmd]))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {name}: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
