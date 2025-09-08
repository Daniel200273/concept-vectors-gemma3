"""
Run the complete concept-vector generation + validation pipeline.

Steps (in order):
 1. code/token-gen/test_generation.py
 2. code/projection/run_pipeline.py --gpu --top-k N
 3. code/concept_val_test/generate-qa-baseline.py
 4. code/concept_val_test/simple_concept_validation.py

This script cleans up previous generated artifacts before running.

Assumption: This script lives in the project's `code/` directory.
"""

import sys
import subprocess
import shutil
from pathlib import Path
import os
import argparse

# Global default for top-k to pass to the projection pipeline
TOP_K = 20  # <--- reasonable default; change at top if you want a different default

# Paths (this file is expected to be in the repository's `code/` folder)
HERE = Path(__file__).resolve().parent
TOKEN_GEN_DIR = HERE / "token-gen"
PROJECTION_DIR = HERE / "projection"
CONCEPT_VAL_TEST_DIR = HERE / "concept-val-test"

# Files/dirs to clean before running
CLEAN_PATHS = [
    TOKEN_GEN_DIR / "token-results",
    PROJECTION_DIR / "final_concept_vectors",
    PROJECTION_DIR / "value_vector_results_gpu",
    CONCEPT_VAL_TEST_DIR / "qa-generated.json",
    CONCEPT_VAL_TEST_DIR / "simple_validation_results.json",
]


def safe_remove_path(p: Path):
    """Safely remove a file or directory inside the `code/` directory.

    We refuse to delete anything outside of the `code/` tree to avoid accidents.
    """
    try:
        p = p.resolve()
    except Exception:
        return
    code_root = HERE.resolve()
    if not str(p).startswith(str(code_root)):
        print(f"Refusing to remove path outside code/: {p}")
        return

    if p.is_dir():
        print(f"Removing directory: {p}")
        shutil.rmtree(p)
    elif p.is_file():
        print(f"Removing file: {p}")
        p.unlink()
    else:
        # Might be a globbed pattern or missing; ignore silently
        print(f"Not present (skipping): {p}")


def run_python_script(script_path: Path, args=None, env=None, python_exec=None):
    args = args or []
    python_exec = python_exec or sys.executable
    cmd = [python_exec, str(script_path)] + args
    # Run the script with its own directory as the working directory so
    # relative file paths inside the script resolve correctly.
    cwd = str(script_path.parent)
    print("\n>>> Running:", " ".join(cmd), "(cwd=" + cwd + ")")
    try:
        # Stream output to terminal with correct working directory
        result = subprocess.run(cmd, check=True, cwd=cwd, env=env)
        print(f"<<< Completed: {script_path} (exit {result.returncode})")
    except subprocess.CalledProcessError as e:
        print(f"*** ERROR: Command failed with exit code {e.returncode}: {cmd}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run complete concept-vector pipeline")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="top-k to pass to projection/run_pipeline.py")
    parser.add_argument("--no-clean", action="store_true", help="skip cleaning previous outputs")
    parser.add_argument("--python", default=sys.executable, help="python interpreter to use when calling scripts")
    args = parser.parse_args(argv)

    top_k = args.top_k

    # Python executable to use when spawning scripts (can override with --python)
    python_exec = args.python

    if not args.no_clean:
        print("Cleaning previous outputs...")
        for p in CLEAN_PATHS:
            safe_remove_path(p)

    # Step 1: token generation and validation
    test_generation = TOKEN_GEN_DIR / "test_generation.py"
    if not test_generation.exists():
        print(f"Missing: {test_generation}. Aborting.")
        return 2
    run_python_script(test_generation, args=[], env=None, python_exec=python_exec)

    # Step 2: run projection pipeline (GPU) with top-k
    run_pipeline = PROJECTION_DIR / "run_pipeline.py"
    if not run_pipeline.exists():
        print(f"Missing: {run_pipeline}. Aborting.")
        return 3
    run_python_script(run_pipeline, args=["--gpu", "--top-k", str(top_k)], python_exec=python_exec)

    # Step 3: generate QA baseline
    gen_qa = CONCEPT_VAL_TEST_DIR / "generate-qa-baseline.py"
    if not gen_qa.exists():
        print(f"Missing: {gen_qa}. Aborting.")
        return 4
    run_python_script(gen_qa, python_exec=python_exec)

    # Step 4: run validation
    validation = CONCEPT_VAL_TEST_DIR / "simple_concept_validation.py"
    if not validation.exists():
        print(f"Missing: {validation}. Aborting.")
        return 5
    run_python_script(validation, python_exec=python_exec)

    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
        raise
