# Concept Vectors in Gemma-3 1B

Discovering interpretable concept representations in the Gemma-3 1B model by analyzing intermediate layer activations and value vectors.

## Overview

This project investigates concept vector discovery in Google's Gemma-3 1B model, inspired by the methodology from [ConceptVectors](https://github.com/yihuaihong/ConceptVectors). We extract and validate concept representations by:

1. **Extracting candidate vectors** from MLP down-projection layers
2. **Projecting vectors** onto vocabulary embeddings to find concept-specific activations
3. **Validating concepts** through causal intervention experiments using noise injection

The goal is to identify vectors formed by parameters in the model's MLP layers that promote the activation of sets of words representing interpretable concepts like "Programming", "Science Fiction", or "Blockchain".

## Gemma-3 1B Architecture

# Concept Vectors — Gemma-3 (1B)

> Disclaimer: the python executables in this repository configure a local Hugging Face cache and expect a personal Hugging Face token available in the environment (HF_TOKEN). 

![Automated pipeline](images/automated-pipeline.png)

 # Concept Vectors — Gemma-3 (1B)

Concise, runnable README for extracting and validating interpretable concept directions from Gemma-3 intermediate activations.

Quick summary
- Extract column vectors from MLP down-projection layers.
- Project those vectors to the token embedding matrix to find high-activating tokens.
- Validate candidate directions with causal interventions (noise injection) and targeted QA/adversarial tests.

Minimum setup
1. Create a Python environment and install dependencies:
```bash
pip install -r requirements.txt
```
2. Export Hugging Face token (required for model downloads):
```bash
export HF_TOKEN=your_token_here
```

Run the full pipeline
```bash
python code/run_complete_pipeline.py
```
This runs the end-to-end flow (token generation → projection → ranking → validation). Use `--help` on the script for options.

Run individual phases
- Token generation (candidate concepts / keywords):
   - Main: `code/token-gen/generate_keywords.py`
   - Test harness: `code/token-gen/test_generation.py`
   - Example:
      ```bash
      python code/token-gen/generate_keywords.py --out token-results/keywords.json
      ```
- Projection and ranking (project candidates onto embeddings and rank tokens):
   - Main: `code/projection/run_pipeline.py`
   - Utilities: `code/projection/extract_candidate_vectors.py`, `code/projection/project_and_rank_gpu_final.py`
   - Example (GPU recommended for large vocabularies):
      ```bash
      python code/projection/run_pipeline.py --layer 12 --out projection/results_layer12.json
      ```
- Validation (causal interventions, QA-based tests, specificity scoring):
   - Main: `code/concept-val-test/ensemble_concept_validation_layerwise.py`
   - Supporting scripts: `code/concept-val-test/generate-qa-baseline.py`, `code/concept-val-test/advanced_concept_validation.py`
   - Example:
      ```bash
      python code/concept-val-test/ensemble_concept_validation_layerwise.py --candidates token-results/keywords.json
      ```
- Adversarial / jailbreak testing:
   - Main: `code/jailbreak-test/run_jailbreak_test.py`
   - Helpers: `code/jailbreak-test/ask_adhoc_question.py`
   - Example:
      ```bash
      python code/jailbreak-test/run_jailbreak_test.py --input jailbreak-test/crafted-jailbreak.txt
      ```

Plotting and analysis
- Plot utilities live in `code/concept-val-test/` and include `plot_validation_results.py`, `plot_batch.py`, `plot_3d_specificities.py` (each has a `__main__` entry).

Project structure (main executables shown)
```
code/
├─ run_complete_pipeline.py                 # main: full end-to-end runner
├─ projection/
│  ├─ run_pipeline.py                       # main: projection & ranking driver
│  ├─ extract_candidate_vectors.py          # helper: extract column vectors from model weights
│  ├─ project_and_rank_gpu_final.py         # helper: GPU projection & ranking
│  └─ ...
├─ concept-val-test/
│  ├─ ensemble_concept_validation_layerwise.py  # main: validation ensemble & specificity scoring
│  ├─ advanced_concept_validation.py            # helper: advanced validation experiments
│  ├─ generate-qa-baseline.py                   # helper: create QA baselines
│  ├─ plot_validation_results.py                # main: plotting & summary
│  └─ plot_batch.py                              # main: batch plotting utilities
├─ jailbreak-test/
│  ├─ run_jailbreak_test.py                # main: adversarial / jailbreak runner
│  └─ ask_adhoc_question.py                 # helper: single question runner
├─ token-gen/
│  ├─ generate_keywords.py                  # main: token / keyword generation
│  ├─ validate_keywords.py                  # helper: validate generated keywords
│  └─ test_generation.py                    # test: token generation examples
└─ concept-val/                              # QA generation + validation configs

Notes
- Most heavy steps (projection over full vocab) benefit from a GPU and enough RAM/disk for intermediate files (`code/projection/extracted_vectors/`, `code/projection/full_vocabulary_embeddings/`).
- Use `--help` on each script for available flags and paths.

References
- See `code/` for runnable scripts and `code/projection/` for the extraction + projection implementation.

If you'd like, I can also:
- Add short README snippets inside each subfolder showing the typical command for that component.
- Create a small Makefile / top-level CLI wrapper to run phases selectively.

---
Concise README updated to show the pipeline image, corrected runnable commands, and main executables for each component.
