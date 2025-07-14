# Concept Vector Finding in Gemma 3 1B

This repository contains the implementation for finding and validating concept vectors in Google's Gemma 3 1B model using mechanistic interpretability techniques.

## Overview

This project implements a three-stage pipeline for discovering concept vectors within transformer language models:

1. **Stage 1: Candidate Vector Extraction** - Extract MLP intermediate activations and compute vocabulary projections
2. **Stage 2: Automated Scoring** - Score candidates using token pattern analysis (no external LLM required)
3. **Stage 3: Causal Validation** - Validate concept vectors through vector damage testing

## Features

- 🚀 **CUDA-optimized implementation** for GPU acceleration
- 🧠 **Real model analysis** using actual Gemma 3 1B weights
- 📊 **Comprehensive visualization** and analysis tools
- 🔬 **Causal validation** through vector damage experiments
- 💾 **Memory-efficient processing** with batch operations

## Notebooks

### `concept-vectors-gemma3.ipynb`

- **Platform**: macOS with Apple Silicon (MPS)
- **Purpose**: Local development and experimentation
- **Optimizations**: Apple Metal Performance Shaders

### `concept-vectors-gemma3-cuda.ipynb`

- **Platform**: CUDA-enabled GPUs (Google Colab, laboratory clusters)
- **Purpose**: Production-scale experiments
- **Optimizations**: CUDA acceleration with mixed precision

## Setup

### Local Development (macOS)

```bash
# Clone the repository
git clone https://github.com/yourusername/concept-vectors-gemma3.git
cd concept-vectors-gemma3

# Create conda environment
conda env create -f environment.yml
conda activate concept-vectors

# Install additional requirements
pip install -r requirements.txt
```

### CUDA/Colab Setup

```bash
# In Google Colab or CUDA environment
!pip install transformers torch accelerate tqdm matplotlib seaborn

# For laboratory clusters, use the provided setup script
# (Note: setup scripts are not included in this repository)
```

## Usage

### Quick Start

1. **Choose your platform**:

   - For macOS: Use `concept-vectors-gemma3.ipynb`
   - For CUDA/Colab: Use `concept-vectors-gemma3-cuda.ipynb`

2. **Run the notebook**:

   - Execute cells sequentially
   - Adjust `max_vectors` parameter for experiment scale
   - Monitor GPU/memory usage throughout execution

3. **Analyze results**:
   - View concept distribution plots
   - Examine top validated concept vectors
   - Export results to JSON for further analysis

### Configuration Options

```python
# Experiment scale (adjust based on available compute)
max_vectors = 1000  # Vectors per layer (None for full scale)
batch_size = 200    # Batch size for GPU processing
score_threshold = 0.75  # Minimum score for concept candidates
```

### Expected Outputs

- **Concept vectors**: Validated vectors with semantic meanings
- **Performance metrics**: Success rates and computational complexity
- **Visualizations**: Score distributions, concept categories, validation results
- **Results file**: `concept_vector_results_cuda.json` with comprehensive data

## Technical Details

### Model Architecture

- **Model**: Google Gemma 3 1B Instruct
- **Layers**: 18 transformer layers
- **Hidden Dimension**: 2048
- **MLP Dimension**: 8192
- **Vocabulary**: ~32,768 tokens

### Computational Complexity

- **Full scale**: ~4.8 billion parameters to analyze
- **Estimated FLOPs**: ~100 TFLOPs for complete analysis
- **Memory requirements**: 8-12 GB GPU memory

### Concept Categories

The system identifies vectors for concepts including:

- Animals, Colors, Numbers, Emotions
- Technology, Food, Travel, Science
- Language, Time, Space, Body parts

## Research Context

This implementation is based on mechanistic interpretability research for understanding how large language models represent and process concepts internally. The methodology follows established practices in the field while being adapted specifically for the Gemma 3 architecture.

## Results Format

Results are saved in JSON format with the following structure:

```json
{
  "pipeline_stats": {
    "total_candidates": 18000,
    "validated_vectors": 245,
    "success_rate": 0.0136
  },
  "concept_distribution": {
    "animals": 45,
    "technology": 38,
    "emotions": 29
  },
  "validation_results": [...]
}
```

## Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support
- **Transformers**: 4.35+
- **Memory**: 16GB+ RAM, 8GB+ GPU memory
- **Storage**: 5GB+ for model and results

## Contributing

This repository is part of academic research. For questions or collaboration opportunities, please open an issue or contact the maintainer.

## License

This project is for academic and research purposes. Please cite appropriately if used in academic work.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{concept-vectors-gemma3,
  title={Concept Vector Finding in Gemma 3 1B},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/concept-vectors-gemma3}
}
```
