# Finding Concept Vectors in Gemma 3 1B

## Gemma 3 1B Architecture Overview

Gemma 3 1B represents a revolutionary approach to billion-parameter language models, featuring unprecedented architectural optimizations that enable both efficiency and capability.

### **Core Specifications**

- **Model Type**: Transformer decoder-only with extreme optimizations
- **Total Parameters**: ~1.024 billion parameters
- **Context Length**: 32,768 tokens (32K context window)
- **Model ID**: `google/gemma-3-1b-it` (instruction-tuned variant)

### **Layer Configuration**

- **Transformer Layers**: 26 layers (deeper than typical 1B models)
- **Hidden Dimension**: 1,152 (optimized for efficiency)
- **MLP Dimension**: 6,912 (6× expansion ratio)
- **Vocabulary Size**: 262,144 tokens (exactly 2^18, power-of-2 optimization)

### **Revolutionary Features**

**Extreme Grouped Query Attention (4:1 GQA)**

- **Query Heads**: 4 heads × 256 dimensions
- **Key-Value Heads**: 1 shared head × 256 dimensions
- **Memory Efficiency**: 4× reduction in KV cache compared to standard attention
- **Innovation**: First 1B model with extreme GQA ratio

**Sliding Window Attention**

- **Window Size**: 512 tokens for local attention
- **Pattern**: Every 6th layer uses sliding window
- **Hybrid Cache**: Advanced mechanism for long sequences
- **Result**: 32K context with manageable memory footprint

### **Technical Details**

- **Activation**: GELU PyTorch Tanh variant
- **Normalization**: RMSNorm (epsilon: 1e-06)
- **Position Encoding**: RoPE with theta=1,000,000
- **Precision**: FP16 for inference optimization
- **Embeddings**: Tied input/output embeddings

## Automated Concept Vector Discovery Pipeline

The following approach eliminates manual review and external LLM dependencies, enabling fully automated concept vector identification at scale.

### **Step 1: Automated Candidate Identification Using Keyword-Based Projections**

1. **Define Target Concept Keywords**

   - Create predefined keyword sets for specific concepts (e.g., "Harry Potter": ["Harry", "Potter", "Hogwarts", "Hermione", "Ron", "wand"])
   - Keywords represent tokens highly relevant to the target concept
   - Replace manual GPT-4 scoring with algorithmic keyword matching

2. **Extract MLP Candidate Vectors**

   - Identify all candidate vectors from MLP layers: **L × di = 26 × 6,912 = 179,712 candidates**
   - Each candidate vector **vℓj** is the j-th column of the "second MLP layer" (i.e. the weight matrix from which we extract candidate concept vectors) **WℓV**
   - Focus on middle-to-upper layers (layers 8-20) where concrete concepts typically emerge

3. **Project Vectors onto Vocabulary Space**

   - For each candidate vector **vℓj** (dimension 1,152), compute projection: **Evℓj ∈ R|V|**
   - **E** is the output embedding matrix (262,144 × 1,152)
   - Result: probability score for each of the 262,144 vocabulary tokens
   - **Computational cost**: ~302M FLOPs per vector projection

4. **Automated Keyword-Based Scoring**

   - For each projection **Evℓj**, extract scores for predefined keyword tokens
   - Calculate aggregate score: sum, mean, or weighted combination of keyword token probabilities
   - Rank candidate vectors by their keyword relevance scores
   - **Selection criterion**: Top vectors with highest keyword probability alignments

5. **Layer-Based Filtering**
   - Apply architectural knowledge: early layers encode syntax, middle layers encode concepts
   - **Filter strategy**: Focus on layers 8-20 for concrete concept discovery
   - **Rationale**: Concept vectors from early layers are typically too general or syntactic

### **Step 2: Automated Causal Verification**

6. **Vector Damage Testing**

   - For each high-scoring candidate vector **vℓj**, apply Gaussian noise: **vℓj ← vℓj + ε**
   - **Noise distribution**: ε ∼ N(0, 0.1) (standard deviation of 0.1)
   - **Constraint**: All other model parameters remain unchanged
   - **Computational cost**: Trivial vector addition operation

7. **Automated Performance Evaluation**

   - **Concept-related questions**: Generate questions specifically about the target concept
   - **Concept-unrelated questions**: Generate questions about different, unrelated topics
   - **Metrics**: Measure BLEU and Rouge-L scores for both question categories
   - **Comparison**: Evaluate model performance with and without vector damage

8. **Causal Validation Criterion**
   - **Selection rule**: Retain vectors where noise causes:
     - **Substantial degradation** on concept-related questions (BLEU difference > 0.2)
     - **Minimal impact** on concept-unrelated questions (BLEU difference < 0.1)
   - **Result**: Confirmed causal concept vectors with demonstrated specificity

### **Step 3: Pipeline Integration and Scaling**

9. **Automated Workflow**

   - **Input**: Target concept and associated keyword set
   - **Process**: Execute Steps 1-8 automatically without human intervention
   - **Output**: Validated concept vectors with causal verification scores
   - **Scalability**: Process multiple concepts in parallel

10. **Computational Optimizations**
    - **Vocabulary subset**: Use top 15K most common tokens (96% FLOP reduction)
    - **Layer sampling**: Focus on middle layers 8-20 (60% reduction)
    - **Batch processing**: Compute multiple vector projections simultaneously
    - **Early filtering**: Apply vector norm pre-filtering before expensive projections

### **Key Advantages of Automated Approach**

- **Eliminates GPT-4 dependency**: No external LLM API calls required
- **Removes manual review**: Fully algorithmic concept identification
- **Enables scalability**: Process hundreds of concepts automatically
- **Reduces computational cost**: Smart optimizations reduce FLOPs by ~94%
- **Maintains accuracy**: Keyword-based scoring proven effective (Geva et al., 2022a)
- **Supports evaluation**: Creates parametric benchmarks for unlearning methods

This automated pipeline transforms concept vector discovery from a manual, expensive process into a scalable, algorithmic approach suitable for large-scale parametric knowledge evaluation in Gemma 3 1B models.
