#!/usr/bin/env python3
"""
GPU-Enhanced Concept Value Vector Analysis (Configurable Layer Range)

This script provides a GPU-accelerated version of concept vector analysis.
Uses CUDA/GPU acceleration for the most computationally intensive parts.

Configuration:
- TARGET_LAYER_START: First MLP layer to analyze (default: 14)
- TARGET_LAYER_END: Last MLP layer to analyze (default: 22)

Key optimizations:
1. Batch matrix multiplication on GPU (E_C @ V_batch)
2. GPU tensor operations for statistical computations
3. Efficient memory management with CUDA streams
4. Vectorized group score computations

Process:
1. Load candidate vectors (value vectors from MLP down_proj columns, configurable layers) 
2. Load token embeddings (for concept tokens)
3. GPU-accelerated batch computation of token activation scores
4. Simple ranking by sum of all concept token activation scores
5. Select top-k candidates for each concept
6. Save results with detailed analysis

Mathematical approach (vectorized):
- Concept token embeddings: E_C = [e1, e2, ..., en] (n_concept_tokens x d)
- Batch candidate vectors: V_batch = [v1, v2, ..., vk] (k x d)
- Batch scores: R_batch = E_C @ V_batch^T (n_concept_tokens x k)
- Ranking score: sum(concept_group_scores) for maximum concept activation strength
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
TARGET_LAYER_START = 1  # First layer to include (inclusive)
TARGET_LAYER_END = 26    # Last layer to include (inclusive)
TARGET_LAYERS = list(range(TARGET_LAYER_START, TARGET_LAYER_END + 1))

class ConceptVectorProjectorGPU:
    """GPU-accelerated analyze value vectors by computing token activation scores for concepts"""
    
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
        self.candidate_vectors_gpu = torch.from_numpy(self.candidate_vectors).float().to(self.device)
        print(f"  🚀 Moved vectors to {self.device}")
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
        GPU-accelerated computation of concept token scores for a batch of vectors
        
        Args:
            concept_embeddings_gpu: Concept token embeddings (n_tokens x d) on GPU
            batch_vectors_gpu: Batch of value vectors (batch_size x d) on GPU
            
        Returns:
            Tuple of (batch_scores_gpu, batch_stats) where:
            - batch_scores_gpu: Token scores (n_tokens x batch_size) on GPU
            - batch_stats: Dictionary with batch statistics
        """
        # Optional normalization to remove norm effects and focus on angular similarity
        if normalize:
            concept_embeddings_gpu = F.normalize(concept_embeddings_gpu, p=2, dim=1)
            batch_vectors_gpu = F.normalize(batch_vectors_gpu, p=2, dim=1)

        # Batch matrix multiplication: E_C @ V_batch^T
        # Shape: (n_concept_tokens, batch_size)
        batch_scores = torch.mm(concept_embeddings_gpu, batch_vectors_gpu.t())
        
        # Compute batch statistics on GPU
        scores_mean = torch.mean(batch_scores, dim=0)  # (batch_size,)
        scores_std = torch.std(batch_scores, dim=0)    # (batch_size,)
        scores_max = torch.max(batch_scores, dim=0)[0] # (batch_size,)
        scores_min = torch.min(batch_scores, dim=0)[0] # (batch_size,)
        scores_range = scores_max - scores_min         # (batch_size,)
        
        # Vector norms
        vector_norms = torch.norm(batch_vectors_gpu, dim=1)  # (batch_size,)
        
        batch_stats = {
            "scores_mean": scores_mean,      # GPU tensor (batch_size,)
            "scores_std": scores_std,        # GPU tensor (batch_size,)
            "scores_max": scores_max,        # GPU tensor (batch_size,)
            "scores_min": scores_min,        # GPU tensor (batch_size,)
            "scores_range": scores_range,    # GPU tensor (batch_size,)
            "value_vector_norm": vector_norms # GPU tensor (batch_size,)
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
        
        # Pre-allocate tensors on GPU
        group_scores = torch.empty(num_groups, batch_size, device=self.device, dtype=torch.float32)
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
    
    def analyze_concept_value_vectors_gpu(self, concept_name: str, top_k: int = 50) -> Dict:
        """
        GPU-accelerated analysis of value vectors for a specific concept
        
        Args:
            concept_name: Name of the concept to analyze
            top_k: Number of top candidates to return
            
        Returns:
            Dictionary with analysis results
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
        
        # Get concept embeddings and move to GPU
        concept_embeddings_gpu = torch.from_numpy(concept_embeddings).float().to(self.device)
        
        # Create token groups
        group_map, group_keys = self.create_token_groups(concept_tokens)
        num_groups = len(group_keys)
        
        print(f"    📊 {len(concept_tokens)} concept tokens in {num_groups} groups")
        
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
        analysis_results = []
        
        print(f"    🚀 Processing {n_candidates:,} candidates in batches of {batch_size}")
        
        for i in tqdm(range(0, n_candidates, batch_size), desc=f"    GPU batches"):
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
            
            # Keep other metrics for analysis but don't use in ranking
            all_group_mean = torch.mean(group_scores_gpu, dim=0)  # (current_batch_size,)
            all_group_max = torch.max(group_scores_gpu, dim=0)[0]  # (current_batch_size,)
            all_group_std = torch.std(group_scores_gpu, dim=0)    # (current_batch_size,)
            
            # Selectivity metrics (for analysis only)
            full_mean = batch_stats["scores_mean"]  # (current_batch_size,)
            selectivity_ratio = all_group_max / torch.clamp(torch.abs(full_mean), min=0.01)
            
            # Move results back to CPU for processing
            group_scores_cpu = group_scores_gpu.cpu().numpy()  # (num_groups x current_batch_size)
            best_token_indices_cpu = best_token_indices_gpu.cpu().numpy()
            concept_activation_strength_cpu = concept_activation_strength.cpu().numpy()
            all_group_mean_cpu = all_group_mean.cpu().numpy()
            all_group_max_cpu = all_group_max.cpu().numpy()
            all_group_std_cpu = all_group_std.cpu().numpy()
            selectivity_ratio_cpu = selectivity_ratio.cpu().numpy()
            
            # Convert batch stats to CPU
            batch_stats_cpu = {
                key: tensor.cpu().numpy() for key, tensor in batch_stats.items()
            }
            
            # Process each vector in the batch
            for j in range(current_batch_size):
                vector_idx = i + j
                vector_key = self.vector_index_mapping[str(vector_idx)]
                
                # Get scores and indices for this vector
                vector_group_scores = group_scores_cpu[:, j]  # (num_groups,)
                vector_best_indices = best_token_indices_cpu[:, j]  # (num_groups,)
                
                # Sort groups by score for this vector
                group_sort_indices = np.argsort(vector_group_scores)[::-1]
                
                # Create scoring info for this vector
                scoring_info = {
                    "scores_mean": float(batch_stats_cpu["scores_mean"][j]),
                    "scores_std": float(batch_stats_cpu["scores_std"][j]),
                    "scores_max": float(batch_stats_cpu["scores_max"][j]),
                    "scores_min": float(batch_stats_cpu["scores_min"][j]),
                    "scores_range": float(batch_stats_cpu["scores_range"][j]),
                    "value_vector_norm": float(batch_stats_cpu["value_vector_norm"][j]),
                    "concept_embedding_matrix_shape": list(concept_embeddings.shape),
                    "num_concept_tokens": len(concept_tokens)
                }
                
                # Store result
                result = {
                    "vector_index": vector_idx,
                    "vector_key": vector_key,
                    "concept_activation_strength": float(concept_activation_strength_cpu[j]),
                    "selectivity_ratio": float(selectivity_ratio_cpu[j]),
                    "concept_specificity": float(all_group_mean_cpu[j] / max(batch_stats_cpu["scores_std"][j], 0.01)),
                    "all_group_mean": float(all_group_mean_cpu[j]),
                    "all_group_max": float(all_group_max_cpu[j]),
                    "all_group_std": float(all_group_std_cpu[j]),
                    "full_mean": float(batch_stats_cpu["scores_mean"][j]),
                    "total_groups": num_groups,
                    "grouping": {"enabled": True, "method": "variant_max", "num_groups": num_groups},
                    "scoring_info": scoring_info,
                    # Top groups for this vector
                    "top_groups": [
                        {
                            "group_key": group_keys[gidx],
                            "group_size": len(group_map[group_keys[gidx]]),
                            "best_concept_token_index": int(vector_best_indices[gidx]),
                            "token_id": concept_tokens[int(vector_best_indices[gidx])][1],  # Get token_id from concept_tokens
                            "token": concept_tokens[int(vector_best_indices[gidx])][0],    # Get token string from concept_tokens
                            "score": float(vector_group_scores[gidx])
                        }
                        for gidx in group_sort_indices[:min(10, len(group_sort_indices))]
                    ],
                    # Back-compat: expose the same top items as tokens list
                    "top_concept_tokens": [
                        {
                            "concept_token_index": int(vector_best_indices[gidx]),
                            "token_id": concept_tokens[int(vector_best_indices[gidx])][1],  # Get token_id from concept_tokens
                            "token": concept_tokens[int(vector_best_indices[gidx])][0],    # Get token string from concept_tokens
                            "score": float(vector_group_scores[gidx])
                        }
                        for gidx in group_sort_indices[:min(10, len(group_sort_indices))]
                    ]
                }
                
                analysis_results.append(result)
        
        # Sort by concept activation strength (highest first)
        analysis_results.sort(key=lambda x: x["concept_activation_strength"], reverse=True)
        
        # Select top-k results
        top_results = analysis_results[:top_k]
        
        # Compute summary statistics (same as CPU version)
        all_group_means = [r["all_group_mean"] for r in analysis_results]
        all_group_maxes = [r["all_group_max"] for r in analysis_results]
        all_group_stds = [r["all_group_std"] for r in analysis_results]
        all_full_means = [r["full_mean"] for r in analysis_results]
        all_ranges = [r["scoring_info"]["scores_range"] for r in analysis_results]
        
        analysis_summary = {
            "concept_name": concept_name,
            "num_concept_tokens": len(valid_token_ids),
            "concept_tokens": [
                {"token": self.token_id_to_string.get(tid, f"<UNK:{tid}>"), "token_id": tid}
                for tid in valid_token_ids
            ],
            "total_candidates_tested": len(analysis_results),
            "top_k": top_k,
            "grouping": {"enabled": True, "method": "variant_max", "num_groups": num_groups},
            "statistics": {
                "max_activation_strength": float(np.max(all_group_means)),
                "mean_activation_strength": float(np.mean(all_group_means)),
                "median_activation_strength": float(np.median(all_group_means)),
                "std_activation_strength": float(np.std(all_group_means)),
                "max_concept_group_max": float(np.max(all_group_maxes)),
                "mean_concept_group_std": float(np.mean(all_group_stds)),
                "max_concept_full_mean": float(np.max(all_full_means)),
                "mean_concept_score_range": float(np.mean(all_ranges)),
                "top_k_activation_mean": float(np.mean([r["concept_activation_strength"] for r in top_results])),
                "ranking_method": "concept_activation_sum_gpu"
            },
            "top_candidates": top_results,
            "concept_embedding_info": {
                "dimension": concept_embeddings.shape[1],
                "num_concept_tokens": concept_embeddings.shape[0],
                "mean_norm": float(np.mean(np.linalg.norm(concept_embeddings, axis=1)))
            }
        }
        
        return analysis_summary
    
    def analyze_all_concepts(self, top_k: int = 50, concept_subset: List[str] = None) -> Dict:
        """
        Analyze value vectors for all concepts (GPU-accelerated)
        
        Args:
            top_k: Number of top candidates per concept
            concept_subset: List of specific concepts to analyze (None = all)
            
        Returns:
            Dictionary with all concept analyses
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
            all_max_activations = [a["statistics"]["max_activation_strength"] for a in successful_analyses]
            all_mean_activations = [a["statistics"]["mean_activation_strength"] for a in successful_analyses]
            
            # Find best performing concept
            best_concept_idx = np.argmax(all_max_activations)
            best_concept = concepts_to_analyze[best_concept_idx]
            
            global_statistics = {
                "total_concepts_analyzed": len(concepts_to_analyze),
                "successful_concepts": len(successful_analyses),
                "failed_concepts": len(concepts_to_analyze) - len(successful_analyses),
                "overall_max_activation_strength": float(np.max(all_max_activations)),
                "overall_mean_activation_strength": float(np.mean(all_mean_activations)),
                "best_concept": best_concept,
                "best_concept_max_activation": float(all_max_activations[best_concept_idx]),
                "distribution_stats": {
                    "max_activations_mean": float(np.mean(all_max_activations)),
                    "max_activations_std": float(np.std(all_max_activations)),
                    "mean_activations_mean": float(np.mean(all_mean_activations)),
                    "mean_activations_std": float(np.std(all_mean_activations)),
                },
                "gpu_acceleration": True,
                "device_used": str(self.device)
            }
        else:
            global_statistics = {
                "total_concepts_analyzed": len(concepts_to_analyze),
                "successful_concepts": 0,
                "failed_concepts": len(concepts_to_analyze),
                "gpu_acceleration": True,
                "device_used": str(self.device)
            }
        
        # Clean up GPU memory
        if hasattr(self, 'candidate_vectors_gpu'):
            del self.candidate_vectors_gpu
        torch.cuda.empty_cache()
        
        return {
            "metadata": {
                "analysis_type": "concept_value_vector_analysis_gpu",
                "target_layers": f"{TARGET_LAYER_START}-{TARGET_LAYER_END}",
                "ranking_method": "concept_activation_sum_gpu",
                "gpu_accelerated": True,
                "device": str(self.device),
                "top_k_per_concept": top_k,
                "total_concepts_available": len(self.concept_mappings),
                "concepts_analyzed": len(concepts_to_analyze)
            },
            "global_statistics": global_statistics,
            "concept_analyses": concept_analyses
        }
    
    def save_results(self, results: Dict, output_dir: str) -> Dict:
        """Save analysis results to files (same as CPU version)"""
        print(f"💾 Saving GPU analysis results to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Main results file
        results_file = os.path.join(output_dir, "projection_gpu_analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Summary file
        summary_file = os.path.join(output_dir, "projection_gpu_summary.json")
        summary_data = {
            "metadata": results["metadata"],
            "global_statistics": results["global_statistics"],
            "concept_summary": {
                concept: {
                    "success": "error" not in analysis,
                    "max_activation_strength": analysis.get("statistics", {}).get("max_activation_strength", 0),
                    "total_candidates": analysis.get("total_candidates_tested", 0),
                    "num_concept_tokens": analysis.get("num_concept_tokens", 0)
                }
                for concept, analysis in results["concept_analyses"].items()
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Results saved to: {results_file}")
        print(f"  ✅ Summary saved to: {summary_file}")
        
        return {
            "results_file": results_file,
            "summary_file": summary_file,
            "output_directory": output_dir
        }
    
    def run_analysis(self, top_k: int = 50, concept_subset: List[str] = None, 
                    output_dir: str = ".") -> Dict:
        """
        Complete GPU-accelerated value vector analysis pipeline
        
        Args:
            top_k: Number of top candidates per concept
            concept_subset: List of specific concepts (None = all)
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths
        """
        print("🚀 Starting GPU-Accelerated Concept Value Vector Analysis")
        print("=" * 70)
        print(f"🎯 Device: {self.device}")
        
        # Step 1: Load data
        self.load_candidate_vectors()
        self.load_token_embeddings()
        
        # Step 2: Run GPU analysis
        results = self.analyze_all_concepts(top_k, concept_subset)
        
        # Step 3: Save results
        file_info = self.save_results(results, output_dir)
        
        print("\n" + "=" * 80)
        layer_range_str = f"{TARGET_LAYER_START}-{TARGET_LAYER_END}"
        print(f"✅ GPU VALUE VECTOR ANALYSIS COMPLETE (LAYERS {layer_range_str})!")
        print("=" * 80)
        print(f"🚀 GPU device used: {self.device}")
        print(f"📊 Results saved to: {output_dir}")
        
        return file_info

def main():
    """Main GPU analysis function"""
    # Configuration
    candidate_vectors_dir = "extracted_vectors"
    token_embeddings_dir = "token_embeddings"
    output_dir = "value_vector_results_gpu"
    top_k = 100
    
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
    
    print(f"\n🎉 GPU value vector analysis completed!")
    print(f"📁 Check the '{output_dir}' folder for results")
    if concept_subset:
        print(f"🎯 Analyzed concepts: {', '.join(concept_subset)}")

if __name__ == "__main__":
    main()
