# Concept Vectors in Gemma-3 1B

Discovering interpretable concept representations in the Gemma-3 1B model by analyzing intermediate layer activations and value vectors.

## Overview

This project investigates concept vector discovery in Google's Gemma-3 1B model, inspired by the methodology from [ConceptVectors](https://github.com/yihuaihong/ConceptVectors). We extract and validate concept representations by:

1. **Extracting candidate vectors** from MLP down-projection layers
2. **Projecting vectors** onto vocabulary embeddings to find concept-specific activations
3. **Validating concepts** through causal intervention experiments using noise injection

The goal is to identify vectors formed by parameters in the model's MLP layers that promote the activation of sets of words representing interpretable concepts like "Programming", "Science Fiction", or "Blockchain".

## Gemma-3 1B Architecture

### Key Characteristics for Concept Discovery

| Component | Specification | Relevance |
|-----------|---------------|-----------|
| **Layers** | 26 transformer blocks | Concept vectors emerge at different abstraction levels |
| **Hidden Size** | 1152 dimensions | Vector space dimensionality for concept directions |
| **MLP Dimension** | 6912 (6×hidden) | Intermediate representations in feed-forward networks |
| **Attention Heads** | 8 heads | Multi-aspect attention but MLPs are our focus |
| **Vocabulary Size** | ~256k tokens | Large vocabulary for diverse concept expression |
| **Parameter Count** | ~1B parameters | Compact model allowing efficient experimentation |

### Target Architecture Components

- **MLP Down-Projection Layers**: `model.layers[i].mlp.down_proj.weight` (6912 → 1152)
  - These layers compress high-dimensional MLP activations back to residual stream
  - Hypothesized to contain concept-specific directions as column vectors
  - Total candidate vectors: 26 layers × 6912 vectors = 179,712 candidates
- **Token Embeddings**: `model.embed_tokens.weight` (256k × 1152)
  - Used for projecting concept vectors to measure vocabulary activation patterns

## Methodology

### 1. Vector Extraction
- Extract column vectors from MLP down-projection weights across all 26 layers
- Each column represents a potential concept direction in the 1152-dimensional space

### 2. Concept Projection
- Project extracted vectors onto the full vocabulary embedding matrix
- Identify tokens that activate most strongly for each candidate vector
- Filter candidates based on concept-specific activation patterns

### 3. Validation
- Apply Gaussian noise to promising concept vectors
- Measure performance degradation on concept-related vs. unrelated tasks
- Concept-specific vectors should show selective degradation

## Key Results

- **Concept Specificity**: Validated vectors show 10-30% higher degradation on concept-related tasks
- **Layer Distribution**: Most effective concept vectors found in layers 10-20 (middle-to-late layers)
- **Activation Patterns**: Strong concept vectors activate 50-200 highly relevant vocabulary tokens

## Project Structure

```
code/
├── projection/              # Vector extraction and ranking
│   ├── extract_candidate_vectors.py
│   ├── project_and_rank_v2.py
│   └── run_pipeline.py
├── concept_val_test/        # Validation experiments
│   └── advanced_concept_validation.py
├── token-gen/              # Concept definitions and vocabularies
└── concept-val/            # QA generation and testing
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set HuggingFace token**:
   ```bash
   export HF_TOKEN=your_token_here
   ```
   See [SETUP.md](SETUP.md) for detailed instructions.

3. **Run the pipeline**:
   ```bash
   cd code/projection
   python run_pipeline.py
   ```

## Validation Results

The validation framework tests concept vectors by injecting Gaussian noise and measuring:
- **BLEU scores** between original and perturbed outputs
- **ROUGE-L scores** for semantic similarity
- **Concept specificity** (higher degradation on concept-related vs. unrelated questions)

Example results show concept vectors achieving 60-80% specificity rates with clear causal effects on model behavior.

## References

- [ConceptVectors: Human-Interpretable Concept Directions](https://github.com/yihuaihong/ConceptVectors)
- [Gemma-3 Model Documentation](https://huggingface.co/google/gemma-3-1b-it)
- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
