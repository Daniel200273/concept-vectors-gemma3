"""
Run the complete concept-vector generation + validation pipeline.

Current steps (in order):
 1. `code/token-gen/test_generation.py`            - generate token lists / candidate tokens
 2. `code/projection/run_pipeline.py`              - run complete projection pipeline (extract + project)
 3. `code/concept-val-test/generate-qa-baseline.py` - generate QA pairs for concepts (kept)
 4. `code/concept-val-test/ensemble_concept_validation_layerwise.py` - run layerwise ensemble validation

Behavior:
 - By default this script cleans a set of output folders under `code/` before running.
 - Pass `--no-clean` to skip removing previous outputs and run using existing artifacts.

Assumption: This script lives in the project's `code/` directory and should be run from repository root.
"""

import sys
import subprocess
import shutil
from pathlib import Path
import os
import argparse

# Note: the final projector `project_and_rank_gpu_final.py` is GPU-first
# and has a fixed configuration; no top-k flag is required.

# Paths (this file is expected to be in the repository's `code/` folder)
HERE = Path(__file__).resolve().parent
TOKEN_GEN_DIR = HERE / "token-gen"
PROJECTION_DIR = HERE / "projection"
CONCEPT_VAL_TEST_DIR = HERE / "concept-val-test"

# Files/dirs to clean before running
CLEAN_PATHS = [
    TOKEN_GEN_DIR / "token-results",
    PROJECTION_DIR / "extracted_vectors",
    PROJECTION_DIR / "token_embeddings", 
    PROJECTION_DIR / "value_vector_results_gpu_layerwise",
    CONCEPT_VAL_TEST_DIR / "qa-generated.json",
    CONCEPT_VAL_TEST_DIR / "validation-results",
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
    parser.add_argument("--no-clean", action="store_true", help="skip cleaning previous outputs")
    parser.add_argument("--skip", action="store_true", help="skip candidate vector extraction in projection pipeline")
    parser.add_argument("--python", default=sys.executable, help="python interpreter to use when calling scripts")
    args = parser.parse_args(argv)

    # Note: no top-k argument — the final projector determines selection parameters

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

    # Step 2: run complete projection pipeline (extract candidates + tokens + final projector)
    projection_pipeline = PROJECTION_DIR / "run_pipeline.py"
    if not projection_pipeline.exists():
        print(f"Missing: {projection_pipeline}. Aborting.")
        return 3
    
    # Pass --skip flag to projection pipeline if provided
    projection_args = ["--skip"] if args.skip else []
    run_python_script(projection_pipeline, args=projection_args, python_exec=python_exec)
    
    # Step 3: generate QA baseline (kept)
    gen_qa = CONCEPT_VAL_TEST_DIR / "generate-qa-baseline.py"
    if not gen_qa.exists():
        print(f"Missing: {gen_qa}. Aborting.")
        return 4
    run_python_script(gen_qa, python_exec=python_exec)

    # Step 4: run ensemble layerwise concept validation
    ensemble_validator = CONCEPT_VAL_TEST_DIR / "ensemble_concept_validation_layerwise.py"
    if not ensemble_validator.exists():
        print(f"Missing: {ensemble_validator}. Aborting.")
        return 5
    run_python_script(ensemble_validator, python_exec=python_exec)

    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
        raise
