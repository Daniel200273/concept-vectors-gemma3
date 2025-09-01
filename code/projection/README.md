# Concept Vector Projection Pipeline

This folder contains the complete pipeline for extracting concept vectors from Gemma 3 1B by projecting MLP candidate vectors onto concept token embedding subspaces.

## Overview

The pipeline implements the concept vector extraction methodology:

1. **Extract Candidate Vectors**: Extract weight vectors from middle MLP layers (6-20) of Gemma 3 1B
2. **Extract Token Embeddings**: Get embeddings for all vocabulary tokens associated with each concept  
3. **Project and Rank**: Project candidate vectors onto concept token embedding subspaces and rank by explained variance
4. **Select Best Vectors**: Choose the top-k best-fitting vectors for each concept

## Files

### Core Scripts

- `extract_candidate_vectors.py` - Extracts candidate vectors from Gemma 3 1B MLP layers
- `extract_token_embeddings.py` - Extracts token embeddings for concept-related vocabulary tokens
- `project_and_rank.py` - Projects candidates onto concept subspaces and ranks by fit quality
- `run_pipeline.py` - Complete orchestrated pipeline script

### Architecture Details

**Gemma 3 1B Architecture:**
- 26 transformer layers
- Hidden dimension: 1,152
- MLP intermediate dimension: 6,912  
- Vocabulary size: 262,144 tokens

**Target Extraction:**
- Layers: 6-20 (middle portion, 15 layers)
- Candidate vectors per layer: 1,152 (columns of MLP up_proj weights)  
- Total candidates: 15 × 1,152 = 17,280 vectors
- Vector dimension: 6,912 (each column of the weight matrix)

## Usage

### Quick Start (Complete Pipeline)

```bash
# Run complete pipeline with default settings
python run_pipeline.py

# Analyze top 50 candidates per concept
python run_pipeline.py --top-k 50

# Analyze specific concepts only
python run_pipeline.py --concepts "Computer programming,Machine learning,Cryptography"

# Force re-extraction of all data
python run_pipeline.py --force-reextract
```

### Individual Steps

#### 1. Extract Candidate Vectors

```bash
python extract_candidate_vectors.py
```

**Output:**
- `extracted_vectors/candidate_vectors.npy` - NumPy array (17,280 × 6,912)
- `extracted_vectors/candidate_vectors_metadata.json` - Extraction metadata
- `extracted_vectors/vector_index_mapping.json` - Vector ID to layer/column mapping

#### 2. Extract Token Embeddings

```bash
python extract_token_embeddings.py
```

**Prerequisites:** Requires `../token-gen/concept_keyword_ids.json`

**Output:**
- `token_embeddings/token_embeddings.npy` - Token embedding array
- `token_embeddings/token_embeddings_metadata.json` - Concept mappings and metadata
- `token_embeddings/token_id_to_index.json` - Token ID to array index mapping

#### 3. Project and Rank

```bash
python project_and_rank.py
```

**Output:**
- `projection_results/projection_analysis_results.json` - Complete analysis results
- `projection_results/projection_summary.json` - Summary statistics

## Mathematical Approach

### Candidate Vector Extraction

For each MLP layer ℓ in the target range (6-20):
- Extract up_proj weight matrix W^ℓ_V ∈ ℝ^(6912×1152)  
- Each **column** W^ℓ_V[:,i] is a candidate concept vector v^ℓ_i ∈ ℝ^6912
- Total candidates: 15 layers × 1,152 columns = 17,280 vectors

### Concept Vector Identification

For each concept C with associated tokens {T₁, T₂, ..., Tₙ}:

1. **Token Embedding Matrix**: E = [e₁, e₂, ..., eₙ] ∈ ℝⁿˣᵈ
2. **Candidate Vector**: v ∈ ℝᵈ (where d=6912)
3. **Projection**: v_proj = Eᵀ(EEᵀ)⁻¹Ev
4. **Quality Metric**: Explained Variance Ratio = ||v_proj||² / ||v||²

The best concept vectors are those with highest explained variance when projected onto their concept's token embedding subspace.

## Output Structure

### Final Results

After pipeline completion, check `final_concept_vectors/`:

- `final_concept_vectors.json` - Complete results with best vectors per concept
- `concept_vectors_summary.txt` - Human-readable summary report

### Example Final Result

```json
{
  "concept_vectors": {
    "Computer programming": {
      "best_candidate": {
        "layer": 18,
        "column": 3456,  # Column index in up_proj weight matrix
        "explained_variance_ratio": 0.8234,
        "vector_key": "L18_C3456"  # C for column index
      },
      "concept_info": {
        "num_tokens": 15,
        "concept_tokens": [{"token": "programming", "token_id": 12345}, ...]
      }
    }
  }
}
```

## Performance Notes

### Memory Requirements

- **Candidate vectors**: ~477 MB (17,280 vectors × 6,912 dim × 4 bytes)
- **Token embeddings**: ~50-100 MB (depends on unique tokens)
- **GPU memory**: ~8-12 GB for model loading

### Runtime Estimates

- **Step 1** (Candidate extraction): ~10-15 minutes
- **Step 2** (Token embeddings): ~5-10 minutes  
- **Step 3** (Projection analysis): ~30-60 minutes (depends on concepts)
- **Total pipeline**: ~45-90 minutes

### Optimization Tips

1. **Subset analysis**: Use `--concepts` for faster testing
2. **Lower top-k**: Use `--top-k 20` for quicker analysis
3. **GPU memory**: Ensure sufficient GPU memory for model loading
4. **Batch processing**: Steps are already optimized with batching

## Prerequisites

1. **Generated concept tokens**: Run keyword generation pipeline first
   - File: `../token-gen/concept_keyword_ids.json`
   - Contains vocabulary tokens for each concept

2. **Python dependencies**:
   ```bash
   pip install torch transformers numpy scikit-learn matplotlib seaborn pandas tqdm
   ```

3. **Hardware**: GPU recommended (8+ GB VRAM), 16+ GB RAM

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: 
   - Use smaller batch sizes in extraction scripts
   - Clear GPU cache between steps

2. **Missing concept file**:
   ```bash
   cd ../token-gen
   python validate_keywords.py
   ```

3. **Long runtime**: 
   - Start with concept subset for testing
   - Monitor GPU/CPU utilization

### Error Recovery

Pipeline supports resuming from any step. If a step fails:
1. Fix the issue
2. Re-run `run_pipeline.py` (will skip completed steps)
3. Use `--force-reextract` only if you need to regenerate data

## Results Interpretation

### Quality Metrics

- **Explained Variance Ratio**: Higher = better concept alignment (0.0-1.0)
- **Typical good values**: 0.3-0.8+ depending on concept complexity
- **Layer distribution**: Middle-upper layers often perform better

### Analysis

- Compare performance across concepts
- Examine layer/neuron distributions for best vectors
- Look for patterns in high-performing concepts

## Next Steps

After extraction:
1. **Validation**: Test concept vectors on downstream tasks
2. **Visualization**: Create t-SNE/PCA plots of concept vectors
3. **Interpretation**: Analyze what linguistic features the vectors capture
4. **Application**: Use vectors for concept-based model analysis
