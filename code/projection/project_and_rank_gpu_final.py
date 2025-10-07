#!/usr/bin/env python3
"""
GPU-Enhanced Layer-wise Concept Value Vector Analysis (Configurable Layer Range)

This script provides a GPU-accelerated layer-wise analysis of concept vectors.
Uses CUDA/GPU acceleration for the most computationally intensive parts.

Configuration:
- TARGET_LAYER_START: First MLP layer to analyze (default: 1)
- TARGET_LAYER_END: Last MLP layer to analyze (default: 26)

Key Features:
1. Layer-wise vector evaluation: Each candidate vector scored individually
2. Best layer selection: Analyzes top 100 vectors per layer, finds 5 best layers by mean activation
3. Final selection: 100 vectors × 5 best layers = 500 vectors per concept
4. GPU acceleration for batch matrix multiplication and statistical computations
5. Efficient memory management with CUDA streams
6. Detailed layer analysis with comprehensive statistics

Process:
1. Load candidate vectors (value vectors from MLP down_proj columns, configurable layers) 
2. Load token embeddings (for concept tokens)
3. Score ALL candidate vectors using GPU-accelerated batch computation
4. Organize vectors by layer and find top 100 per layer
5. Rank layers by mean activation score of their top 100 vectors
6. Select 5 best layers and their top 100 vectors each (500 total per concept)
7. Save comprehensive results with layer-wise analysis

Mathematical approach (vectorized):
- Concept token embeddings: E_C = [e1, e2, ..., en] (n_concept_tokens x d)
- Batch candidate vectors: V_batch = [v1, v2, ..., vk] (k x d)
- Batch scores: R_batch = normalize(E_C) @ normalize(V_batch)^T (n_concept_tokens x k)
- Layer ranking: mean(top_100_cosine_scores) per layer
- Final selection: top 100 vectors from each of 5 best layers
"""

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import re
import gc
from typing import Dict, List, Tuple
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME

HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from transformers import AutoTokenizer
from tqdm import tqdm


# Global configuration for target MLP layers
TARGET_LAYER_START = 13  # First layer to include (inclusive)
TARGET_LAYER_END = 26    # Last layer to include (inclusive)
TARGET_LAYERS = list(range(TARGET_LAYER_START, TARGET_LAYER_END + 1))

class ConceptVectorProjectorGPU:
    """GPU-accelerated layer-wise analysis of value vectors by computing token activation scores for concepts
    
    This class implements a sophisticated layer-wise analysis approach:
    1. Scores every candidate vector individually for each concept
    2. Groups vectors by layer and finds top performers per layer
    3. Ranks layers by mean activation score of their top 100 vectors
    4. Selects 5 best layers and their top 100 vectors (500 total per concept)
    
    The analysis provides comprehensive statistics and detailed layer comparisons.
    """
    
    def __init__(self, candidate_vectors_dir: str, token_embeddings_dir: str, device: str = "cuda:1"):
        """
        Initialize the GPU projector
        
        Args:
            candidate_vectors_dir: Directory with candidate vector files
            token_embeddings_dir: Directory with token embedding files
            device: CUDA device to use (default: cuda:1)
        """
        self.candidate_vectors_dir = candidate_vectors_dir
        self.token_embeddings_dir = token_embeddings_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Data storage
        self.candidate_vectors = None
        self.vector_index_mapping = None
        self.token_embeddings = None
        self.token_id_to_string = None
        self.concept_token_mapping = None
        
        print(f"🚀 GPU Projector initialized on device: {self.device}")
        
        # Set CUDA device
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            print(f"🎯 CUDA device set to: {torch.cuda.current_device()}")
    
    def load_candidate_vectors(self):
        """Load candidate vectors and filter for target layers only"""
        layer_range_str = f"{TARGET_LAYER_START}-{TARGET_LAYER_END}"
        print(f"📂 Loading candidate vectors (layers {layer_range_str})...")
        
        # Load vector index mapping
        mapping_path = os.path.join(self.candidate_vectors_dir, "vector_index_mapping.json")
        with open(mapping_path, 'r') as f:
            self.vector_index_mapping = json.load(f)
        
        # Load NumPy array
        vectors_path = os.path.join(self.candidate_vectors_dir, "candidate_vectors.npy")
        
        print(f"  📁 Loading from: {vectors_path}")
        all_candidate_vectors = np.load(vectors_path)
        
        # Filter for target layers only
        target_layers = TARGET_LAYERS
        layer_pattern = re.compile(r"L(\d+)_C\d+")
        
        filtered_indices = []
        filtered_mapping = {}
        new_index = 0
        
        for old_index_str, vector_key in self.vector_index_mapping.items():
            old_index = int(old_index_str)
            match = layer_pattern.match(vector_key)
            if match:
                layer = int(match.group(1))
                if layer in target_layers:
                    filtered_indices.append(old_index)
                    filtered_mapping[str(new_index)] = vector_key
                    new_index += 1
        
        # Filter vectors
        self.candidate_vectors = all_candidate_vectors[filtered_indices]
        self.vector_index_mapping = filtered_mapping
        
        layer_range_str = f"{TARGET_LAYER_START}-{TARGET_LAYER_END}"
        print(f"  ✅ Loaded {self.candidate_vectors.shape[0]:,} vectors from layers {layer_range_str}")
        print(f"  📊 Vector shape: {self.candidate_vectors.shape}")
        
        # Convert to GPU tensor for efficient computation
        # Convert to GPU tensor for efficient computation without forcing dtype
        self.candidate_vectors_dtype = self.candidate_vectors.dtype
        self.candidate_vectors_gpu = torch.from_numpy(self.candidate_vectors).to(self.device)
        print(f"  🚀 Moved vectors to {self.device} (dtype={self.candidate_vectors_gpu.dtype})")
        # Diagnostics: vector norms and dtype
        try:
            vecs_cpu = self.candidate_vectors
            norms = np.linalg.norm(vecs_cpu, axis=1)
            print(f"  🔎 Candidate vectors: shape={vecs_cpu.shape}, dtype={vecs_cpu.dtype}")
            print(f"    mean_norm={float(np.mean(norms)):.6f}, std_norm={float(np.std(norms)):.6f}, max_norm={float(np.max(norms)):.6f}")
        except Exception:
            pass
    
    def load_token_embeddings(self):
        """Load token embeddings for all concepts"""
        print("📂 Loading token embeddings...")
        
        # Load NumPy array
        embeddings_path = os.path.join(self.token_embeddings_dir, "token_embeddings.npy")
        
        print(f"  📁 Loading from: {embeddings_path}")
        self.token_embeddings = np.load(embeddings_path)
        
        # Load metadata and concept mappings (same as CPU version)
        metadata_path = os.path.join(self.token_embeddings_dir, "token_embeddings_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Token embeddings metadata not found: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.token_metadata = json.load(f)
        
        self.concept_mappings = self.token_metadata["concept_mappings"]
        
        # Load token ID to index mapping
        id_to_index_path = os.path.join(self.token_embeddings_dir, "token_id_to_index.json")
        if not os.path.exists(id_to_index_path):
            raise FileNotFoundError(f"Token ID to index mapping not found: {id_to_index_path}")
        with open(id_to_index_path, 'r') as f:
            self.token_id_to_index = {int(k): v for k, v in json.load(f).items()}
        
        # Load token ID to string mapping
        id_to_string_path = os.path.join(self.token_embeddings_dir, "token_id_to_string.json")
        if not os.path.exists(id_to_string_path):
            raise FileNotFoundError(f"Token ID to string mapping not found: {id_to_string_path}")
        with open(id_to_string_path, 'r', encoding='utf-8') as f:
            self.token_id_to_string = {int(k): v for k, v in json.load(f).items()}
        
        # Load validation report for existing token groupings (same as CPU version)
        validation_report_path = os.path.join("..", "token-gen", "token-results", "validation_report.json")
        if os.path.exists(validation_report_path):
            with open(validation_report_path, 'r', encoding='utf-8') as f:
                self.validation_report = json.load(f)
            print("  ✅ Loaded validation report with existing token groupings")
        else:
            print("  ⚠️ Validation report not found, will use basic grouping")
            self.validation_report = None
        
        print(f"  ✅ Loaded {self.token_embeddings.shape[0]:,} token embeddings")
        print(f"  📊 Embedding shape: {self.token_embeddings.shape}")
        print(f"  🎯 Found {len(self.concept_mappings)} concepts")
        # Diagnostics: token embedding norms
        try:
            emb_norms = np.linalg.norm(self.token_embeddings, axis=1)
            print(f"  🔎 Token embeddings: dtype={self.token_embeddings.dtype}")
            print(f"    mean_norm={float(np.mean(emb_norms)):.6f}, std_norm={float(np.std(emb_norms)):.6f}, max_norm={float(np.max(emb_norms)):.6f}")
        except Exception:
            pass
    
    def create_token_groups(self, concept_tokens: List[Tuple[str, int]]) -> Tuple[Dict[str, List[int]], List[str]]:
        """
        Group concept tokens by their base form (same as CPU version)
        
        Args:
            concept_tokens: List of (token_string, token_id) tuples
            
        Returns:
            Tuple of (group_map, group_keys) where:
            - group_map: Maps group key to list of token indices
            - group_keys: List of all group keys
        """
        token_groups = {}
        
        for i, (token, token_id) in enumerate(concept_tokens):
            # Group tokens by their lowercase, cleaned form
            base_form = token.lower().strip()
            base_form = re.sub(r'^▁+', '', base_form)  # Remove leading space markers
            base_form = re.sub(r'[^\w\s]', '', base_form)  # Remove punctuation
            base_form = base_form.strip()
            
            if not base_form:
                base_form = f"token_{i}"  # Fallback for empty tokens
            
            if base_form not in token_groups:
                token_groups[base_form] = []
            token_groups[base_form].append(i)
        
        group_keys = sorted(token_groups.keys())
        return token_groups, group_keys
    
    def compute_concept_token_scores_gpu(self, concept_embeddings_gpu: torch.Tensor, 
                                        batch_vectors_gpu: torch.Tensor,
                                        normalize: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        GPU-accelerated computation of concept token scores using direct cosine similarity
        
        Uses the same approach as project_and_rank_gpu.py: direct normalized dot product
        between concept embeddings and candidate vectors for robust ranking.
        
        Args:
            concept_embeddings_gpu: Concept token embeddings (n_tokens x d) on GPU  
            batch_vectors_gpu: Batch of concept vectors (batch_size x d) on GPU
            normalize: Whether to L2-normalize for cosine similarity (default: True)
            
        Returns:
            Tuple of (batch_scores_gpu, batch_stats) where:
            - batch_scores_gpu: Cosine similarity scores (n_tokens x batch_size) on GPU
            - batch_stats: Dictionary with batch statistics
        """
        # L2-normalize embeddings for cosine similarity (removes norm effects)
            # Capture dtype for metadata
        self.token_embeddings_dtype = self.token_embeddings.dtype
        print(f"  🔎 Token embeddings: dtype={self.token_embeddings_dtype}")
        concept_embeddings_gpu = F.normalize(concept_embeddings_gpu, p=2, dim=1)
        batch_vectors_gpu = F.normalize(batch_vectors_gpu, p=2, dim=1)

        # Direct batch matrix multiplication: E_C @ V_batch^T
        # Shape: concept_embeddings_gpu (n_concept_tokens x d) @ batch_vectors_gpu^T (d x batch_size)
        # Result: (n_concept_tokens x batch_size)
        batch_scores = torch.mm(concept_embeddings_gpu, batch_vectors_gpu.t())
        
        # Compute batch statistics on GPU  
        scores_mean = torch.mean(batch_scores, dim=0)  # (batch_size,)
        scores_std = torch.std(batch_scores, dim=0)    # (batch_size,)
        scores_max = torch.max(batch_scores, dim=0)[0] # (batch_size,)
        scores_min = torch.min(batch_scores, dim=0)[0] # (batch_size,)
        scores_range = scores_max - scores_min         # (batch_size,)
        
        # Vector norms (before normalization if applied)
        vector_norms = torch.norm(batch_vectors_gpu, dim=1)  # (batch_size,)
        
        batch_stats = {
            "scores_mean": scores_mean,      # GPU tensor (batch_size,)
            "scores_std": scores_std,        # GPU tensor (batch_size,)
            "scores_max": scores_max,        # GPU tensor (batch_size,)
            "scores_min": scores_min,        # GPU tensor (batch_size,)
            "scores_range": scores_range,    # GPU tensor (batch_size,)
            "value_vector_norm": vector_norms, # GPU tensor (batch_size,)
            "method": "direct_cosine_similarity"  # Track the method used
        }
        
        return batch_scores, batch_stats
    
    def compute_group_scores_gpu(self, batch_scores_gpu: torch.Tensor, group_map: Dict[str, List[int]], 
                                group_keys: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-accelerated computation of group scores for a batch
        
        Args:
            batch_scores_gpu: Token scores (n_tokens x batch_size) on GPU
            group_map: Maps group key to list of token indices
            group_keys: List of all group keys
            
        Returns:
            Tuple of (group_scores_gpu, best_token_indices_gpu) where:
            - group_scores_gpu: Group scores (num_groups x batch_size) on GPU
            - best_token_indices_gpu: Best token indices per group (num_groups x batch_size) on GPU
        """
        num_groups = len(group_keys)
        batch_size = batch_scores_gpu.shape[1]
        
        # Pre-allocate tensors on GPU. Match score dtype to incoming batch_scores dtype
        score_dtype = batch_scores_gpu.dtype
        group_scores = torch.empty(num_groups, batch_size, device=self.device, dtype=score_dtype)
        best_token_indices = torch.empty(num_groups, batch_size, device=self.device, dtype=torch.long)
        
        # Compute group scores (sum over group members for each vector in batch)
        for gi, group_key in enumerate(group_keys):
            member_indices = group_map[group_key]
            
            if len(member_indices) == 1:
                # Single token in group
                idx = member_indices[0]
                group_scores[gi] = batch_scores_gpu[idx]
                best_token_indices[gi] = idx
            else:
                # Multiple tokens in group - sum all member scores
                member_scores = batch_scores_gpu[member_indices]  # (n_members x batch_size)
                sum_scores = torch.sum(member_scores, dim=0)  # (batch_size,)
                group_scores[gi] = sum_scores
                # For best_token_indices, still track which token had the highest individual score
                max_scores, max_indices = torch.max(member_scores, dim=0)  # (batch_size,)
                best_token_indices[gi] = torch.tensor(member_indices, device=self.device)[max_indices]
        
        return group_scores, best_token_indices
    
    def analyze_concept_value_vectors_gpu(self, concept_name: str, top_k: int = 100) -> Dict:
        """
        GPU-accelerated layer-wise analysis of value vectors for a specific concept
        
        Args:
            concept_name: Name of the concept to analyze
            top_k: Number of top candidates per layer (default: 100)
            
        Returns:
            Dictionary with layer-wise analysis results and best 5 layers
        """
        print(f"  🔍 Analyzing concept: {concept_name}")
        
        # Validate concept exists
        if concept_name not in self.concept_mappings:
            return {"error": f"Concept '{concept_name}' not found in concept mappings"}
        
        # Get concept token embeddings (same as CPU version)
        concept_info = self.concept_mappings[concept_name]
        token_ids = [token_info["token_id"] for token_info in concept_info["tokens"]]
        
        # Get embedding indices for concept tokens
        concept_embedding_indices = [self.token_id_to_index[tid] for tid in token_ids if tid in self.token_id_to_index]
        if not concept_embedding_indices:
            return {"error": f"No valid token embeddings found for concept '{concept_name}'"}
        
        concept_embeddings = self.token_embeddings[concept_embedding_indices]  # Shape: (n_concept_tokens, dim)
        
        # Create concept tokens list for grouping (same format as CPU version)
        # Only include tokens that have valid embeddings
        valid_token_ids = [tid for tid in token_ids if tid in self.token_id_to_index and tid in self.token_id_to_string]
        concept_tokens = [(self.token_id_to_string[tid], tid) for tid in valid_token_ids]
        
        if not concept_tokens:
            return {"error": f"No valid token strings found for concept '{concept_name}' token IDs"}
        
        # Get concept embeddings and move to GPU (preserve original dtype)
        concept_embeddings_gpu = torch.from_numpy(concept_embeddings).to(self.device)
        
        # Create token groups
        group_map, group_keys = self.create_token_groups(concept_tokens)
        num_groups = len(group_keys)
        
        print(f"    📊 {len(concept_tokens)} concept tokens in {num_groups} groups")
        
        # Layer-wise analysis: organize vectors by layer
        layer_to_vector_indices = {}
        layer_pattern = re.compile(r"L(\d+)_C\d+")
        
        for vector_idx_str, vector_key in self.vector_index_mapping.items():
            match = layer_pattern.match(vector_key)
            if match:
                layer = int(match.group(1))
                if layer not in layer_to_vector_indices:
                    layer_to_vector_indices[layer] = []
                layer_to_vector_indices[layer].append(int(vector_idx_str))
        
        print(f"    🔍 Found vectors across {len(layer_to_vector_indices)} layers: {sorted(layer_to_vector_indices.keys())}")
        
        # GPU batch processing
        batch_size = 4000  # Larger batch size for GPU
        n_candidates = self.candidate_vectors_gpu.shape[0]
        
        # Sanity checks: dimension agreement between concept embeddings and candidate vectors
        if concept_embeddings.shape[1] != self.candidate_vectors_gpu.shape[1]:
            raise RuntimeError(f"Dimension mismatch: concept embeddings dim={concept_embeddings.shape[1]} vs candidate vectors dim={self.candidate_vectors_gpu.shape[1]}")

        # Quick sample diagnostics before heavy processing
        try:
            sample_token = concept_embeddings[0]
            sample_vector = self.candidate_vectors_gpu[0].cpu().numpy()
            sample_dot = float(np.dot(sample_token, sample_vector))
            # Compute a small dot distribution between first 100 tokens and first 100 vectors (if available)
            nt = min(100, concept_embeddings.shape[0])
            nv = min(100, self.candidate_vectors_gpu.shape[0])
            small_emb = concept_embeddings[:nt]
            small_vecs = self.candidate_vectors_gpu[:nv].cpu().numpy()
            dots = small_emb @ small_vecs.T
            print(f"  🔎 Sample dot stats (first {nt} tokens x first {nv} vectors): mean={float(np.mean(dots)):.6f}, std={float(np.std(dots)):.6f}, max={float(np.max(dots)):.6f}, min={float(np.min(dots)):.6f}")
            print(f"  🔎 Example single token · vector = {sample_dot:.6f}")
        except Exception:
            pass
            
        # Step 1: Compute scores for ALL vectors
        all_vector_results = []
        
        print(f"    🚀 Processing {n_candidates:,} candidates in batches of {batch_size}")
        
        for i in tqdm(range(0, n_candidates, batch_size), desc=f"    Computing scores"):
            batch_end = min(i + batch_size, n_candidates)
            current_batch_size = batch_end - i
            
            # Get batch of vectors (already on GPU)
            batch_vectors_gpu = self.candidate_vectors_gpu[i:batch_end]  # (current_batch_size x d)
            
            # Compute concept token scores for entire batch
            batch_scores_gpu, batch_stats = self.compute_concept_token_scores_gpu(
                concept_embeddings_gpu, batch_vectors_gpu
            )  # batch_scores_gpu: (n_tokens x current_batch_size)
            
            # Compute group scores for entire batch
            group_scores_gpu, best_token_indices_gpu = self.compute_group_scores_gpu(
                batch_scores_gpu, group_map, group_keys
            )  # group_scores_gpu: (num_groups x current_batch_size)
            
            # Simple ranking metric: sum of all concept token activation scores
            concept_activation_strength = torch.sum(group_scores_gpu, dim=0)  # (current_batch_size,)
            
            # Move results back to CPU for processing
            concept_activation_strength_cpu = concept_activation_strength.cpu().numpy()
            
            # Store basic results for each vector
            for j in range(current_batch_size):
                vector_idx = i + j
                vector_key = self.vector_index_mapping[str(vector_idx)]
                
                all_vector_results.append({
                    "vector_index": vector_idx,
                    "vector_key": vector_key,
                    "concept_activation_strength": float(concept_activation_strength_cpu[j])
                })
        
        # Step 2: Layer-wise analysis - find top vectors per layer and compute layer scores
        layer_analyses = {}
        layer_scores = {}
        
        print(f"    📊 Analyzing layers individually...")
        
        for layer, vector_indices in layer_to_vector_indices.items():
            print(f"      🔍 Analyzing layer {layer} ({len(vector_indices)} vectors)...")
            
            # Get results for this layer's vectors
            layer_results = [r for r in all_vector_results if r["vector_index"] in vector_indices]
            
            # Sort by activation strength
            layer_results.sort(key=lambda x: x["concept_activation_strength"], reverse=True)
            
            # Take top 100 (or all if less than 100)
            top_layer_results = layer_results[:min(top_k, len(layer_results))]
            
            # Calculate mean activation score for this layer's top vectors
            if top_layer_results:
                layer_mean_activation = np.mean([r["concept_activation_strength"] for r in top_layer_results])
                layer_scores[layer] = layer_mean_activation
                
                layer_analyses[layer] = {
                    "layer": layer,
                    "total_vectors": len(layer_results),
                    "top_vectors_analyzed": len(top_layer_results),
                    "mean_activation_score": float(layer_mean_activation),
                    "max_activation_score": float(top_layer_results[0]["concept_activation_strength"]),
                    "min_activation_score": float(top_layer_results[-1]["concept_activation_strength"]),
                    "top_vectors": top_layer_results
                }
                
                print(f"        📈 Layer {layer}: mean_activation={layer_mean_activation:.6f}, max={top_layer_results[0]['concept_activation_strength']:.6f}")
            else:
                layer_scores[layer] = 0.0
                layer_analyses[layer] = {
                    "layer": layer,
                    "total_vectors": 0,
                    "top_vectors_analyzed": 0,
                    "mean_activation_score": 0.0,
                    "top_vectors": []
                }
        
        # Step 3: Select best 5 layers based on mean activation scores
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        best_5_layers = sorted_layers[:5]
        best_layer_numbers = [layer for layer, score in best_5_layers]
        
        print(f"    🏆 Best 5 layers selected:")
        for i, (layer, score) in enumerate(best_5_layers):
            print(f"      {i+1}. Layer {layer}: mean_activation={score:.6f}")
        
        # Step 4: Collect detailed results for best 5 layers (500 total vectors)
        best_layers_detailed_results = []
        total_selected_vectors = 0
        
        for layer_num in best_layer_numbers:
            layer_analysis = layer_analyses[layer_num]
            layer_vectors = layer_analysis["top_vectors"]
            
            # Get detailed information for each vector in this layer
            for vector_result in layer_vectors:
                vector_idx = vector_result["vector_index"]
                
                # Recompute detailed scores for this specific vector
                vector_gpu = self.candidate_vectors_gpu[vector_idx:vector_idx+1]  # (1 x d)
                
                batch_scores_gpu, batch_stats = self.compute_concept_token_scores_gpu(
                    concept_embeddings_gpu, vector_gpu
                )
                
                group_scores_gpu, best_token_indices_gpu = self.compute_group_scores_gpu(
                    batch_scores_gpu, group_map, group_keys
                )
                
                # Convert to CPU for detailed processing
                group_scores_cpu = group_scores_gpu.cpu().numpy()[:, 0]  # (num_groups,)
                best_token_indices_cpu = best_token_indices_gpu.cpu().numpy()[:, 0]
                batch_stats_cpu = {key: (tensor.cpu().numpy()[0] if torch.is_tensor(tensor) else tensor) 
                                  for key, tensor in batch_stats.items()}
                
                # Sort groups by score
                group_sort_indices = np.argsort(group_scores_cpu)[::-1]
                
                # Create detailed result
                detailed_result = {
                    "vector_index": vector_idx,
                    "vector_key": vector_result["vector_key"],
                    "layer": layer_num,
                    "rank_in_layer": layer_vectors.index(vector_result) + 1,
                    "concept_activation_strength": vector_result["concept_activation_strength"],
                    "all_group_mean": float(np.mean(group_scores_cpu)),
                    "all_group_max": float(np.max(group_scores_cpu)),
                    "all_group_std": float(np.std(group_scores_cpu)),
                    "scoring_info": {
                        "scores_mean": float(batch_stats_cpu["scores_mean"]),
                        "scores_std": float(batch_stats_cpu["scores_std"]),
                        "scores_max": float(batch_stats_cpu["scores_max"]),
                        "scores_min": float(batch_stats_cpu["scores_min"]),
                        "scores_range": float(batch_stats_cpu["scores_range"]),
                        "value_vector_norm": float(batch_stats_cpu["value_vector_norm"]),
                        "num_concept_tokens": len(concept_tokens)
                    },
                    "top_groups": [
                        {
                            "group_key": group_keys[gidx],
                            "group_size": len(group_map[group_keys[gidx]]),
                            "best_concept_token_index": int(best_token_indices_cpu[gidx]),
                            "token_id": concept_tokens[int(best_token_indices_cpu[gidx])][1],
                            "token": concept_tokens[int(best_token_indices_cpu[gidx])][0],
                            "score": float(group_scores_cpu[gidx])
                        }
                        for gidx in group_sort_indices[:min(10, len(group_sort_indices))]
                    ]
                }
                
                best_layers_detailed_results.append(detailed_result)
                total_selected_vectors += 1
        
        print(f"    ✅ Selected {total_selected_vectors} vectors from best 5 layers")
        
        # Compute summary statistics for the layer-wise analysis
        all_selected_activations = [r["concept_activation_strength"] for r in best_layers_detailed_results]
        layer_summary_stats = {}
        
        for layer_num in best_layer_numbers:
            layer_results = [r for r in best_layers_detailed_results if r["layer"] == layer_num]
            layer_activations = [r["concept_activation_strength"] for r in layer_results]
            
            layer_summary_stats[str(layer_num)] = {
                "layer": layer_num,
                "num_vectors": len(layer_results),
                "mean_activation": float(np.mean(layer_activations)),
                "std_activation": float(np.std(layer_activations)),
                "max_activation": float(np.max(layer_activations)),
                "min_activation": float(np.min(layer_activations))
            }
        
        analysis_summary = {
            "concept_name": concept_name,
            "num_concept_tokens": len(valid_token_ids),
            "concept_tokens": [
                {"token": self.token_id_to_string.get(tid, f"<UNK:{tid}>"), "token_id": tid}
                for tid in valid_token_ids
            ],
            "analysis_method": "layer_wise_top_k_selection",
            "total_candidates_evaluated": len(all_vector_results),
            "layers_analyzed": len(layer_analyses),
            "best_layers_selected": len(best_layer_numbers),
            "vectors_per_layer": top_k,
            "total_selected_vectors": total_selected_vectors,
            "grouping": {"enabled": True, "method": "variant_max", "num_groups": num_groups},
            "best_layers": [
                {
                    "rank": i + 1,
                    "layer": layer,
                    "mean_activation_score": float(score),
                    "num_vectors": len([r for r in best_layers_detailed_results if r["layer"] == layer])
                }
                for i, (layer, score) in enumerate(best_5_layers)
            ],
            "layer_analyses": {
                str(layer): {
                    "layer": analysis["layer"],
                    "total_vectors_in_layer": analysis["total_vectors"],
                    "top_vectors_selected": analysis["top_vectors_analyzed"],
                    "mean_activation_score": analysis["mean_activation_score"],
                    "max_activation_score": analysis.get("max_activation_score", 0.0),
                    "min_activation_score": analysis.get("min_activation_score", 0.0),
                    "selected_for_final": layer in best_layer_numbers
                }
                for layer, analysis in layer_analyses.items()
            },
            "statistics": {
                "overall_max_activation_strength": float(np.max(all_selected_activations)) if all_selected_activations else 0.0,
                "overall_mean_activation_strength": float(np.mean(all_selected_activations)) if all_selected_activations else 0.0,
                "overall_std_activation_strength": float(np.std(all_selected_activations)) if all_selected_activations else 0.0,
                "best_layer_mean_scores": [float(score) for _, score in best_5_layers],
                "layer_summary": layer_summary_stats,
                "ranking_method": "layer_wise_concept_activation_sum_cosine_gpu"
            },
            "selected_vectors": best_layers_detailed_results,
            "concept_embedding_info": {
                "dimension": concept_embeddings.shape[1],
                "num_concept_tokens": concept_embeddings.shape[0],
                "mean_norm": float(np.mean(np.linalg.norm(concept_embeddings, axis=1)))
            }
        }
        
        return analysis_summary
    
    def analyze_all_concepts(self, top_k: int = 100, concept_subset: List[str] = None) -> Dict:
        """
        Analyze value vectors for all concepts with layer-wise selection (GPU-accelerated)
        
        Args:
            top_k: Number of top candidates per layer (default: 100)
            concept_subset: List of specific concepts to analyze (None = all)
            
        Returns:
            Dictionary with all concept analyses (500 vectors per concept from 5 best layers)
        """
        print("🎯 Starting GPU-accelerated concept analysis...")
        
        # Determine which concepts to analyze
        if concept_subset:
            concepts_to_analyze = [c for c in concept_subset if c in self.concept_mappings]
            missing_concepts = [c for c in concept_subset if c not in self.concept_mappings]
            if missing_concepts:
                print(f"⚠️  Missing concepts (skipped): {missing_concepts}")
        else:
            concepts_to_analyze = list(self.concept_mappings.keys())
        
        print(f"📊 Analyzing {len(concepts_to_analyze)} concepts...")
        
        # Analyze each concept
        concept_analyses = {}
        successful_analyses = []
        
        for concept in tqdm(concepts_to_analyze, desc="Concepts"):
            try:
                analysis = self.analyze_concept_value_vectors_gpu(concept, top_k)
                concept_analyses[concept] = analysis
                
                if "error" not in analysis:
                    successful_analyses.append(analysis)
                else:
                    print(f"⚠️  Analysis failed for '{concept}': {analysis['error']}")
                    
            except Exception as e:
                print(f"❌ Error analyzing '{concept}': {e}")
                concept_analyses[concept] = {"error": str(e)}
        
        # Compute global statistics
        if successful_analyses:
            # Collect statistics from layer-wise analyses
            all_max_activations = [a["statistics"]["overall_max_activation_strength"] for a in successful_analyses]
            all_mean_activations = [a["statistics"]["overall_mean_activation_strength"] for a in successful_analyses]
            all_selected_vector_counts = [a["total_selected_vectors"] for a in successful_analyses]
            
            # Find best performing concept based on mean activation
            best_concept_idx = np.argmax(all_mean_activations)
            best_concept = concepts_to_analyze[best_concept_idx]
            
            # Collect layer information across all concepts
            all_selected_layers = []
            layer_selection_frequency = {}
            
            for analysis in successful_analyses:
                for layer_info in analysis["best_layers"]:
                    layer_num = layer_info["layer"]
                    all_selected_layers.append(layer_num)
                    layer_selection_frequency[layer_num] = layer_selection_frequency.get(layer_num, 0) + 1
            
            # Most frequently selected layers across all concepts
            most_common_layers = sorted(layer_selection_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            global_statistics = {
                "total_concepts_analyzed": len(concepts_to_analyze),
                "successful_concepts": len(successful_analyses),
                "failed_concepts": len(concepts_to_analyze) - len(successful_analyses),
                "analysis_method": "layer_wise_top_k_selection",
                "vectors_per_concept": sum(all_selected_vector_counts) // len(all_selected_vector_counts) if all_selected_vector_counts else 0,
                "target_vectors_per_concept": 500,  # 5 layers × 100 vectors
                "layers_per_concept": 5,
                "vectors_per_layer": top_k,
                "overall_max_activation_strength": float(np.max(all_max_activations)),
                "overall_mean_activation_strength": float(np.mean(all_mean_activations)),
                "best_concept": best_concept,
                "best_concept_mean_activation": float(all_mean_activations[best_concept_idx]),
                "most_selected_layers": [
                    {"layer": layer, "selection_frequency": freq, "percentage": freq/len(successful_analyses)*100}
                    for layer, freq in most_common_layers
                ],
                "distribution_stats": {
                    "max_activations_mean": float(np.mean(all_max_activations)),
                    "max_activations_std": float(np.std(all_max_activations)),
                    "mean_activations_mean": float(np.mean(all_mean_activations)),
                    "mean_activations_std": float(np.std(all_mean_activations)),
                    "selected_vectors_mean": float(np.mean(all_selected_vector_counts)),
                    "selected_vectors_std": float(np.std(all_selected_vector_counts))
                },
                "gpu_acceleration": True,
                "device_used": str(self.device)
            }
        else:
            global_statistics = {
                "total_concepts_analyzed": len(concepts_to_analyze),
                "successful_concepts": 0,
                "failed_concepts": len(concepts_to_analyze),
                "analysis_method": "layer_wise_top_k_selection",
                "gpu_acceleration": True,
                "device_used": str(self.device)
            }
        
        # Clean up GPU memory
        if hasattr(self, 'candidate_vectors_gpu'):
            del self.candidate_vectors_gpu
        torch.cuda.empty_cache()
        
        return {
            "metadata": {
                "analysis_type": "layer_wise_concept_value_vector_analysis_gpu",
                "target_layers": f"{TARGET_LAYER_START}-{TARGET_LAYER_END}",
                "ranking_method": "layer_wise_concept_activation_sum_cosine_gpu",
                "vectors_per_layer": top_k,
                "layers_selected_per_concept": 5,
                "total_vectors_per_concept": 500,  # 5 layers × 100 vectors
                "gpu_accelerated": True,
                "device": str(self.device),
                "total_concepts_available": len(self.concept_mappings),
                "concepts_analyzed": len(concepts_to_analyze)
            },
            "global_statistics": global_statistics,
            "concept_analyses": concept_analyses
        }
    
    def save_results(self, results: Dict, output_dir: str) -> Dict:
        """Save layer-wise analysis results to files"""
        print(f"💾 Saving GPU layer-wise analysis results to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Main results file
        results_file = os.path.join(output_dir, "layer_wise_projection_gpu_analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Summary file
        summary_file = os.path.join(output_dir, "layer_wise_projection_gpu_summary.json")
        summary_data = {
            "metadata": results["metadata"],
            "global_statistics": results["global_statistics"],
            "concept_summary": {
                concept: {
                    "success": "error" not in analysis,
                    "total_selected_vectors": analysis.get("total_selected_vectors", 0),
                    "best_layers": [layer_info["layer"] for layer_info in analysis.get("best_layers", [])],
                    "overall_mean_activation": analysis.get("statistics", {}).get("overall_mean_activation_strength", 0),
                    "layers_analyzed": analysis.get("layers_analyzed", 0),
                    "num_concept_tokens": analysis.get("num_concept_tokens", 0)
                }
                for concept, analysis in results["concept_analyses"].items()
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
        # Save individual concept vector files for easy access
        concept_vectors_dir = os.path.join(output_dir, "concept_vectors")
        os.makedirs(concept_vectors_dir, exist_ok=True)
        
        concept_files_saved = 0
        for concept, analysis in results["concept_analyses"].items():
            if "error" not in analysis and "selected_vectors" in analysis:
                concept_file = os.path.join(concept_vectors_dir, f"{concept}_vectors.json")
                concept_data = {
                    "concept_name": concept,
                    "metadata": {
                        "total_vectors": analysis["total_selected_vectors"],
                        "layers_selected": len(analysis["best_layers"]),
                        "vectors_per_layer": results["metadata"]["vectors_per_layer"],
                        "analysis_method": "layer_wise_selection"
                    },
                    "best_layers": analysis["best_layers"],
                    "selected_vectors": analysis["selected_vectors"]
                }
                
                with open(concept_file, 'w', encoding='utf-8') as f:
                    json.dump(concept_data, f, indent=2, ensure_ascii=False)
                concept_files_saved += 1
        
        print(f"  ✅ Main results saved to: {results_file}")
        print(f"  ✅ Summary saved to: {summary_file}")
        print(f"  ✅ Individual concept files saved: {concept_files_saved} files in {concept_vectors_dir}")
        
        return {
            "results_file": results_file,
            "summary_file": summary_file,
            "concept_vectors_dir": concept_vectors_dir,
            "output_directory": output_dir,
            "concept_files_saved": concept_files_saved
        }
    
    def run_analysis(self, top_k: int = 100, concept_subset: List[str] = None, 
                    output_dir: str = ".") -> Dict:
        """
        Complete GPU-accelerated layer-wise value vector analysis pipeline
        
        Args:
            top_k: Number of top candidates per layer (default: 100)
            concept_subset: List of specific concepts (None = all)
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths and analysis summary
        """
        print("🚀 Starting GPU-Accelerated Layer-wise Concept Value Vector Analysis")
        print("=" * 80)
        print(f"🎯 Device: {self.device}")
        print(f"📊 Analysis method: Layer-wise selection (5 best layers × {top_k} vectors = 500 per concept)")
        
        # Step 1: Load data
        self.load_candidate_vectors()
        self.load_token_embeddings()
        
        # Step 2: Run GPU analysis
        results = self.analyze_all_concepts(top_k, concept_subset)
        
        # Step 3: Save results
        file_info = self.save_results(results, output_dir)
        
        print("\n" + "=" * 80)
        layer_range_str = f"{TARGET_LAYER_START}-{TARGET_LAYER_END}"
        print(f"✅ GPU LAYER-WISE VALUE VECTOR ANALYSIS COMPLETE (LAYERS {layer_range_str})!")
        print("=" * 80)
        print(f"🚀 GPU device used: {self.device}")
        print(f"📊 Results saved to: {output_dir}")
        print(f"🎯 Method: Selected 5 best layers × {top_k} vectors = {5 * top_k} vectors per concept")
        
        return file_info

def main():
    """Main GPU layer-wise analysis function"""
    # Configuration
    candidate_vectors_dir = "extracted_vectors"
    token_embeddings_dir = "token_embeddings"
    output_dir = "value_vector_results_gpu_layerwise"
    top_k = 100  # Top vectors per layer
    
    print("🚀 GPU Layer-wise Concept Vector Analysis")
    print(f"📊 Configuration: {top_k} vectors × 5 best layers = 500 vectors per concept")
    
    # Load concepts from test_concepts.json
    test_concepts_path = os.path.join("..", "token-gen", "test_concepts.json")
    concept_subset = None
    if os.path.exists(test_concepts_path):
        with open(test_concepts_path, 'r') as f:
            concept_subset = json.load(f)
        print(f"📋 Loaded {len(concept_subset)} concepts from test_concepts.json:")
        for concept in concept_subset:
            print(f"  - {concept}")
    
    # Create GPU projector
    projector = ConceptVectorProjectorGPU(candidate_vectors_dir, token_embeddings_dir)
    
    # Run analysis pipeline
    file_info = projector.run_analysis(top_k, concept_subset, output_dir)
    
    print(f"\n🎉 GPU layer-wise value vector analysis completed!")
    print(f"📁 Check the '{output_dir}' folder for results")
    print(f"📄 Individual concept vector files saved in '{output_dir}/concept_vectors/'")
    if concept_subset:
        print(f"🎯 Analyzed concepts: {', '.join(concept_subset)}")
    print(f"📊 Each concept now has 500 vectors (100 from each of 5 best layers)")

if __name__ == "__main__":
    main()
