# Concept Vector Extraction Pipeline

This folder contains a complete automated pipeline for extracting concept vectors from Gemma 3 1B using MLP layer analysis and concept token projections.

## What it does

The pipeline automatically:

1. **Extracts candidate vectors** from Gemma 3 1B MLP layers (configurable range, default: layers 14-22)
2. **Extracts token embeddings** for concept-related vocabulary tokens
3. **Analyzes vectors** by computing activation scores for concept tokens (with optional GPU acceleration)
4. **Ranks vectors** by sum of concept token activation scores (simple metric for maximum concept activation strength)
5. **Generates final results** with best concept vectors identified for each concept

## Requirements

### Prerequisites

1. **Generated concept tokens**: Must have concept token mappings
   - Required file: `../token-gen/concept_keyword_ids.json`
   - Run the token generation pipeline first to create this file

2. **HuggingFace token**: Set environment variable
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```

3. **Python dependencies**:
   ```bash
   pip install torch transformers numpy tqdm
   ```

### Input Files

- `../token-gen/test_concepts.json` - List of concepts to analyze
- `../token-gen/token-results/concept_keyword_ids.json` - Token mappings for concepts

## Usage

### Quick Start

```bash
# Run complete pipeline with default settings
python run_pipeline.py

# Use GPU acceleration (recommended)
python run_pipeline.py --gpu

# Analyze top 50 candidates per concept
python run_pipeline.py --top-k 50 --gpu

# Analyze specific concepts only
python run_pipeline.py --concepts "Artificial intelligence,Germany,Buddhism" --gpu

# Force re-extraction of all data
python run_pipeline.py --force-reextract --gpu
```

### Command Line Options

- `--gpu`: Use GPU acceleration for analysis (recommended)
- `--top-k N`: Number of top candidate vectors per concept (default: 100)
- `--concepts "A,B,C"`: Analyze only specific concepts (comma-separated)
- `--force-reextract`: Force re-extraction even if files exist
- `--base-dir PATH`: Base directory for operations (default: current directory)

## Output

### Generated Files

The pipeline creates several output directories:

- `extracted_vectors/` - Candidate vectors from MLP layers
- `token_embeddings/` - Token embeddings for concepts
- `value_vector_results_v2_gpu/` - Analysis results (GPU mode)
- `final_concept_vectors/` - Final processed results

### Final Results

After completion, check `final_concept_vectors/`:

- `final_concept_vectors.json` - Complete results with best vectors per concept
- `concept_vectors_summary.txt` - Human-readable summary report

### Example Result Structure

```json
{
  "concept_vectors": {
    "Artificial intelligence": {
      "best_candidate": {
        "vector_key": "L18_C2347",
        "layer": 18,
        "neuron": 2347,
        "concept_activation_strength": 1.7904
      },
      "concept_info": {
        "num_tokens": 156,
        "concept_tokens": [{"token": "intelligence", "token_id": 78306}, ...]
      },
      "alternative_candidates": [...]
    }
  }
}
```

## Runtime

- **Step 1** (Candidate extraction): ~10-15 minutes
- **Step 2** (Token embeddings): ~5-10 minutes  
- **Step 3** (GPU analysis): ~20-40 minutes
- **Step 4** (Final processing): ~2-5 minutes
- **Total pipeline**: ~40-70 minutes

The pipeline automatically skips completed steps, so you can resume if interrupted.

## Configuration

### Layer Range

Edit `project_and_rank_gpu.py` to change target layers:

```python
# Global configuration for target MLP layers
TARGET_LAYER_START = 14  # First layer to include
TARGET_LAYER_END = 22    # Last layer to include
```

### Ranking Metric

The pipeline uses a simple sum of activation scores across all concept tokens, prioritizing vectors with maximum overall concept activation strength (regardless of specificity).

## Troubleshooting

**CUDA out of memory**: Use smaller batch sizes or reduce `--top-k`

**Missing concept file**: Run the token generation pipeline first:
```bash
cd ../token-gen
python validate_keywords.py
```

**Long runtime**: Start with a concept subset for testing:
```bash
python run_pipeline.py --concepts "Artificial intelligence" --top-k 20 --gpu
```
