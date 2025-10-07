#!/usr/bin/env python3
"""
Sequential runner for the projection pipeline.

Runs these scripts in order as subprocesses (like code/run_complete_pipeline.py):
1) extract_candidate_vectors.py (if not skipped)
2) extract_token_embeddings.py 
3) project_and_rank_gpu_final.py

Each script runs in its own directory so relative paths work correctly.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_python_script(script_path: Path, args=None, python_exec=None):
    """Run a Python script as subprocess with its directory as cwd."""
    args = args or []
    python_exec = python_exec or sys.executable
    cmd = [python_exec, str(script_path)] + args
    cwd = str(script_path.parent)
    print(f">>> Running: {' '.join(cmd)} (cwd={cwd})")
    try:
        result = subprocess.run(cmd, check=True, cwd=cwd)
        print(f"<<< Completed: {script_path} (exit {result.returncode})")
    except subprocess.CalledProcessError as e:
        print(f"*** ERROR: Command failed with exit code {e.returncode}: {cmd}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Sequential runner for projection pipeline")
    parser.add_argument("--skip", action="store_true",
                        help="If set, skip running candidate vector extraction step")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to use when calling scripts")
    args = parser.parse_args()

    # Get script directory
    here = Path(__file__).resolve().parent
    
    # Step 1: Extract candidate vectors (unless skipped)
    if not args.skip:
        extract_candidates = here / "extract_candidate_vectors.py"
        run_python_script(extract_candidates, python_exec=args.python)

    # Step 2: Extract token embeddings
    extract_tokens = here / "extract_token_embeddings.py"
    run_python_script(extract_tokens, python_exec=args.python)

    # Step 3: Run final GPU projector (layer-wise)
    final_projector = here / "project_and_rank_gpu_final.py"
    run_python_script(final_projector, python_exec=args.python)

    print("\nProjection pipeline finished successfully.")


if __name__ == "__main__":
    main()
