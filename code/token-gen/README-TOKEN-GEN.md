# Token Generation Workflow

## Overview
This workflow generates keywords for concepts and maps them to vocabulary token IDs for concept vector analysis.

## Folder Structure
```
token-gen/
├── README-TOKEN-GEN.md                    # This documentation
├── generate_keywords.py                   # Main keyword generation script
├── validate_keywords.py                   # Keyword validation and mapping
├── test_generation.py                     # Complete pipeline runner
├── keyword_generation_prompt.txt          # LLM prompt template
├── gemma3_vocabulary.json                 # Gemma-3 vocabulary (262k tokens)
├── test_concepts.json                     # Test concepts for development
├── concept-list/                          # Concept collections
│   ├── concepts.json                      # Full concept list
│   └── concepts_categorized.json          # Categorized concepts
└── token-results/                         # Generated outputs (created automatically)
    ├── generated_keywords.json            # Raw keywords from LLM
    ├── generated_keywords_with_descriptions.json  # Keywords with descriptions
    ├── tokenized_keywords.json            # Keywords with token IDs
    ├── generation_summary.json            # Generation statistics
    ├── concept_keyword_ids.json           # Validated token-ID mappings
    ├── concept_keyword_ids_summary.txt    # Human-readable summary
    └── validation_report.json             # Detailed validation statistics
```

## Prerequisites

### 1. Environment Setup
Activate your conda environment:
```
conda activate your_environment_name
```
Inside the environ, HF_TOKEN variable is set. If not, set it as follows:
```
conda env config vars set HF_TOKEN=your_huggingface_token_here
```

### 2. Required Files
- `test_concepts.json` or `concept-list/concepts.json` - List of concepts to process
- `gemma3_vocabulary.json` - Gemma-3 vocabulary (262k tokens)
- `keyword_generation_prompt.txt` - LLM prompt template

## Workflow Steps

### Option 1: Run Complete Pipeline (Recommended)
```bash
cd code/token-gen
python test_generation.py
```
This runs both generation and validation automatically, creating all outputs in `token-results/`.

### Option 2: Run Steps Individually

#### Step 1: Generate Keywords
```bash
python generate_keywords.py
```

**What it does:**
- Loads Gemma-3-12B-IT model for keyword generation
- Processes each concept from `test_concepts.json`
- Uses the prompt template to generate keywords per concept
- Saves results to `token-results/` directory
- No intermediate checkpoints (runs continuously)

**Requirements:**
- GPU with 20GB+ VRAM (for Gemma-3-12B)
- HuggingFace token set as environment variable
- Approximately 30-60 minutes for test concepts

#### Step 2: Validate and Map Keywords
```bash
python validate_keywords.py
```

**What it does:**
- Validates generated keywords against Gemma-3 vocabulary
- Finds all vocabulary variants for each keyword (case variations, prefixes, etc.)
- Maps valid tokens to their token IDs
- Creates comprehensive (token, token_id) tuples
- Saves results to `token-results/` directory

## Configuration Options

### Model Selection (in generate_keywords.py)
```python
model_name = "google/gemma-3-12b-it"  # Default (requires 20GB+ VRAM)
# model_name = "google/gemma-2-2b-it"  # Alternative for smaller GPUs
```

### GPU Selection
```python
gpu_id = 1  # Change to available GPU ID
```

### Input Concepts
- Uses `test_concepts.json` for development/testing
- Switch to `concept-list/concepts.json` for full processing

## Expected Outputs (in token-results/)

### Generation Outputs:
1. **generated_keywords.json**: Raw keywords from LLM (backward compatibility)
2. **generated_keywords_with_descriptions.json**: Keywords with concept descriptions
3. **tokenized_keywords.json**: Keywords with Gemma tokenization
4. **generation_summary.json**: Generation statistics and metadata

### Validation Outputs:
1. **concept_keyword_ids.json**: Validated (vocabulary_token, token_id) mappings
2. **concept_keyword_ids_summary.txt**: Human-readable validation summary
3. **validation_report.json**: Detailed validation analysis and statistics

## Troubleshooting

- **GPU Memory Error**: Switch to smaller model (Gemma-2-2B)
- **HuggingFace Auth**: Ensure `HF_TOKEN` environment variable is set
- **Missing Files**: Check that vocabulary and concept files exist
- **Validation Failures**: Check vocabulary file integrity
- **Output Issues**: `token-results/` directory is created automatically

## Usage in Pipeline
The generated `token-results/concept_keyword_ids.json` is used by the projection pipeline for:
- Filtering concept-relevant tokens
- Scoring candidate vectors
- Concept-specific analysis
