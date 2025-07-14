The procedure to find concept vectors involves several computationally intensive steps that would apply to Gemma 3 1B, given its decoder-only transformer architecture, similar to the LLaMA 2 7B and OLMo 7B models on which the CONCEPTVECTORS benchmark was constructed. While the sources provide a general methodology for transformer-based LLMs, it's important to note that **there are no specific concept vector findings or evaluations detailed for Gemma 3 1B within the provided sources**. The CONCEPTVECTORS benchmark itself was built using LLaMA 2 7B and OLMo 7B.

Here's a breakdown of the procedural complexities, tailored to the architectural specifications of Gemma 3 1B:

### **1. Locating Concept Vectors in MLP Layers (Candidate Identification & Initial Filtering)**

- **Core Idea**: Concept vectors are specific parameter vectors within the **Multi-Layer Perceptron (MLP) layers** of the LLM that encode concrete concepts. The outputs from the MLP layers can be viewed as a linear combination of these parameter vectors in the second MLP layer, each promoting a concept in the vocabulary space.
- **Gemma 3 1B Specifics**:
  - **Number of Layers (L)**: 18 transformer layers.
  - **Intermediate MLP Dimension (di)**: 8,192 (which is 4x the hidden dimension) [Query].
  - **Total Candidate Vectors**: For Gemma 3 1B, there are **L _ di = 18 _ 8,192 = 147,456 candidate vectors** to inspect. This is a substantial number, making manual exploration infeasible.
- **Methodology & Complexity**:
  - **Initial Sorting (Vocabulary Projections)**: For each candidate vector `vℓj` (the `j`-th column of `WℓV`, the second MLP layer's weight matrix), it is projected onto the model's vocabulary space to get a score for each token. The projection `Evℓj` is a vector of dimension `|V|`, where `E` is the output embedding matrix (`|V| × d`) and `vℓj` has dimension `d`.
    - **Gemma 3 1B Specifics**:
      - **Vocabulary Size (|V|)**: 32,768 tokens [Query, 143, 145].
      - **Hidden Dimension (d)**: 2,048 [Query].
    - **Complexity**: For each candidate vector `vℓj` (dimension 2,048), this involves a matrix-vector multiplication with `E` (32,768 x 2,048). This operation requires approximately `|V| * d` floating-point multiplications and `|V| * (d-1)` additions [10, *self-correction*].
      - For one vector: `32,768 * 2,048 ≈ 67 million floating-point operations (FLOPs)`.
      - This calculation is repeated for each of the **147,456 candidate vectors**.
      - **Total projection complexity (approx.)**: `147,456 * 67 million FLOPs`, which is in the order of **billions of FLOPs** (approximately 9.88 \* 10^12 FLOPs just for multiplications).
  - **Exclusion**: Based on this average logit value score, approximately **30% of candidate vectors per layer are excluded** to narrow down the search space.

### **2. Automated Scoring and Manual Review**

- **Methodology**: For the remaining candidate vectors (around 70%), an external LLM (like GPT-4) is used to score the top `k` tokens (e.g., `k=200`) from each vector's vocabulary projection. This score (0-1) indicates the clarity and prominence of the concept. A score above **0.85** is used for strong correlation.
- **Complexity**: This step involves **API calls to an external large language model (e.g., GPT-4)**, which adds practical costs in terms of latency, monetary expense, and reliance on external services, rather than direct internal matrix computations on the Gemma 3 1B model itself. The prompt provided to GPT-4 includes the top-K tokens from the vocabulary projection.
- **Manual Review**: Finally, human reviewers manually verify the top-scoring vectors to ensure they represent **clear, concrete, and specific concepts**. This is a qualitative, human-intensive step.

### **3. Causal Validation**

- **Purpose**: To confirm that the identified concept vectors genuinely influence the model's ability to generate information about the target concept, and not unrelated concepts.
- **Methodology**:
  - **Vector Damage**: For a concept vector `vℓj` associated with concept `c`, the vector is **"damaged" by adding Gaussian noise**: `vℓj ← vℓj + ε`, where `ε ∼ N (0, 0.1)` (Gaussian noise with a standard deviation of 0.1). All other model parameters remain unchanged.
  - **Complexity**: This is a **simple vector addition operation** involving a vector of dimension `d` (2,048 for Gemma 3 1B). This operation is computationally trivial compared to the projection step.
  - **Behavioral Evaluation**: The model's performance is then evaluated on concept-related questions and unrelated questions. This involves running inferences with the modified model, measuring metrics like BLEU and Rouge-L scores.
  - **Validation Criterion**: Only concept vectors where the noise leads to a **substantial decrease in performance for concept-related questions** (e.g., BLEU score difference > 0.2) and minimal impact on unrelated questions are retained. This step helps ensure the causal importance and specificity of the identified vectors.

In summary, the most computationally intensive part for Gemma 3 1B would be the **vocabulary projection of hundreds of thousands of candidate MLP vectors**, requiring billions of floating-point operations. The process also relies heavily on external LLM inference (e.g., GPT-4) for automated scoring and significant manual human effort for final verification. The causal validation step, while crucial for specificity, involves comparatively minor computational overhead in modifying the vectors.

## Summary: Gemma3 1B Architecture Details

Based on the official Gemma3 technical documentation and architecture analysis, here are the key architectural specifications:

### 🏗️ **Core Architecture**

- **Model Type**: Transformer decoder-only
- **Total Parameters**: ~1 billion
- **Context Length**: 8,192 tokens

### 🧠 **Layer Configuration**

- **Number of Layers**: **18 transformer layers**
- **Hidden Dimension**: **2,048**
- **MLP Dimension**: **8,192** (4x hidden dimension)
- **Attention Heads**: **16**
- **Head Dimension**: **128** (hidden_dim / num_heads)

### 🔧 **Technical Details**

- **Activation Function**: GELU
- **Normalization**: RMSNorm (Root Mean Square Layer Normalization)
- **Position Encoding**: RoPE (Rotary Position Embedding)
- **Vocabulary Size**: 32,768 tokens
- **Attention Type**: Multi-head self-attention

### 📊 **Parameter Distribution**

- **Embedding Parameters**: ~67M (vocabulary × hidden_dim)
- **Attention Parameters per Layer**: ~16.8M
- **MLP Parameters per Layer**: ~33.6M
- **Total Transformer Parameters**: ~908M
- **Estimated Total**: ~975M parameters

This architecture follows the standard transformer decoder pattern with relatively compact dimensions optimized for 1B parameter efficiency.
