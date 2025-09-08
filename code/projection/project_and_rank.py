#!/usr/bin/env python3
"""
Concept Value Vector Analysis (Layers 14-22 Only)

This script analyzes value vectors by computing concept-specific token activation scores.
Only considers MLP layers 14-22 for more focused projection analysis.

Process:
1. Load candidate vectors (value vectors from MLP down_proj columns, layers 14-22)
2. Load token embeddings (for concept tokens)
3. For each concept, compute token activation scores using E_C @ vℓj
4. Rank candidates by concept activation strength using ALL concept groups
5. Select top-k candidates for each concept (ranking by mean of ALL token group scores)
6. Save results with detailed analysis

Mathematical approach:
- For concept C with tokens T1, T2, ..., Tn (concept-specific tokens)
- Concept token embeddings: E_C = [e1, e2, ..., en] (matrix of shape n_concept_tokens x d)
- Candidate value vector: vℓj (shape d) - column from down_proj matrix (layers 14-22 only)
- Compute concept token scores: rℓj = E_C @ vℓj (shape n_concept_tokens)
- Measure concept alignment: mean of ALL group scores to use full concept representation
"""

import numpy as np
import json
import os
import re
from typing import Dict, List, Tuple
# Ensure private Hugging Face cache is set before importing transformers
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME

# Export HF_TOKEN if provided externally
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from transformers import AutoTokenizer
    


class ConceptVectorProjector:
    """Analyze value vectors by computing token activation scores for concepts"""
    
    def __init__(self, candidate_vectors_dir: str, token_embeddings_dir: str):
        """
        Initialize the projector
        
        Args:
            candidate_vectors_dir: Directory with candidate vector files
            token_embeddings_dir: Directory with token embedding files
        """
        self.candidate_vectors_dir = candidate_vectors_dir
        self.token_embeddings_dir = token_embeddings_dir
        
        # Data storage
        self.candidate_vectors = None
        self.candidate_metadata = None
        self.token_embeddings = None
        self.token_metadata = None
        self.concept_mappings = None
    
    def load_candidate_vectors(self):
        """Load candidate vectors from extraction results, filtering for layers 14-22 only"""
        print("📊 Loading candidate vectors (layers 14-22 only)...")
        
        # Load NumPy array
        vectors_path = os.path.join(self.candidate_vectors_dir, "candidate_vectors.npy")
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Candidate vectors file not found: {vectors_path}")
        all_candidate_vectors = np.load(vectors_path)
        
        # Load metadata
        metadata_path = os.path.join(self.candidate_vectors_dir, "candidate_vectors_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Candidate vectors metadata not found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.candidate_metadata = json.load(f)
        
        # Load index mapping
        mapping_path = os.path.join(self.candidate_vectors_dir, "vector_index_mapping.json")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Vector index mapping not found: {mapping_path}")
        with open(mapping_path, 'r') as f:
            all_vector_index_mapping = json.load(f)
        
        # Filter for layers 14-22 only
        target_layers = list(range(14, 23))  # 14 to 22 inclusive
        filtered_indices = []
        filtered_mapping = {}
        
        print(f"🎯 Filtering for layers: {target_layers}")
        
        for str_idx, vector_key in all_vector_index_mapping.items():
            # Vector keys are formatted as "L{layer:02d}_C{col:04d}"
            if vector_key.startswith('L'):
                layer_str = vector_key[1:3]  # Extract layer number (2 digits)
                try:
                    layer_num = int(layer_str)
                    if layer_num in target_layers:
                        old_idx = int(str_idx)
                        new_idx = len(filtered_indices)
                        filtered_indices.append(old_idx)
                        filtered_mapping[str(new_idx)] = vector_key
                except ValueError:
                    continue
        
        if not filtered_indices:
            raise ValueError(f"No vectors found for target layers {target_layers}")
        
        # Extract the filtered vectors
        self.candidate_vectors = all_candidate_vectors[filtered_indices]
        self.vector_index_mapping = filtered_mapping
        
        print(f"✅ Loaded {len(filtered_indices):,} candidate vectors from layers 14-22")
        print(f"📏 Vector dimension: {self.candidate_vectors.shape[1]}")
        print(f"🔍 Original total vectors: {all_candidate_vectors.shape[0]:,}")
        print(f"🎯 Filtered vectors: {self.candidate_vectors.shape[0]:,}")
    
    def load_token_embeddings(self):
        """Load token embeddings from extraction results"""
        print("📚 Loading token embeddings...")
        
        # Load NumPy array
        embeddings_path = os.path.join(self.token_embeddings_dir, "token_embeddings.npy")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Token embeddings file not found: {embeddings_path}")
        self.token_embeddings = np.load(embeddings_path)
        
        # Load metadata and concept mappings
        metadata_path = os.path.join(self.token_embeddings_dir, "token_embeddings_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Token embeddings metadata not found: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.token_metadata = json.load(f)
        
        self.concept_mappings = self.token_metadata["concept_mappings"]
        
        # Load ID mappings
        id_to_index_path = os.path.join(self.token_embeddings_dir, "token_id_to_index.json")
        if not os.path.exists(id_to_index_path):
            raise FileNotFoundError(f"Token ID to index mapping not found: {id_to_index_path}")
        with open(id_to_index_path, 'r') as f:
            self.token_id_to_index = {int(k): v for k, v in json.load(f).items()}
        
        id_to_string_path = os.path.join(self.token_embeddings_dir, "token_id_to_string.json")
        if not os.path.exists(id_to_string_path):
            raise FileNotFoundError(f"Token ID to string mapping not found: {id_to_string_path}")
        with open(id_to_string_path, 'r', encoding='utf-8') as f:
            self.token_id_to_string = {int(k): v for k, v in json.load(f).items()}
        
        # Load validation report for existing token groupings
        validation_report_path = os.path.join("..", "token-gen", "token-results", "validation_report.json")
        if os.path.exists(validation_report_path):
            with open(validation_report_path, 'r', encoding='utf-8') as f:
                self.validation_report = json.load(f)
            print("✅ Loaded validation report with existing token groupings")
        else:
            print("⚠️ Validation report not found, will use basic grouping")
            self.validation_report = None
        
        print(f"✅ Loaded {self.token_embeddings.shape[0]:,} token embeddings")
        print(f"📏 Embedding dimension: {self.token_embeddings.shape[1]}")
        print(f"🎯 Found {len(self.concept_mappings)} concepts")
    
    def get_concept_token_groups(self, concept_name: str) -> Dict:
        """
        Get token groups for a concept using existing validation report data
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            Dictionary with grouped tokens and their indices
        """
        if self.validation_report and concept_name in self.validation_report.get("concept_details", {}):
            # Use existing groupings from validation report
            concept_details = self.validation_report["concept_details"][concept_name]
            valid_tuples = concept_details.get("valid_tuples", [])
            
            token_groups = {}
            for original_keyword, token_variants in valid_tuples:
                # Group by original keyword (already grouped in validation)
                group_name = original_keyword.lower()
                
                # Get indices for all token variants of this keyword
                indices = []
                for vocab_token, token_id in token_variants:
                    if token_id in self.token_id_to_index:
                        indices.append(self.token_id_to_index[token_id])
                
                if indices:
                    token_groups[group_name] = {
                        "indices": indices,
                        "tokens": [vocab_token for vocab_token, token_id in token_variants],
                        "original_keyword": original_keyword
                    }
            
            return token_groups
        else:
            # Fallback: basic grouping by token string
            if concept_name not in self.concept_mappings:
                raise ValueError(f"Concept '{concept_name}' not found in concept mappings")
                
            concept_info = self.concept_mappings[concept_name]
            concept_tokens = concept_info["tokens"]
            
            token_groups = {}
            for token_info in concept_tokens:
                token_id = token_info["token_id"]
                token_string = token_info["token"]
                
                if token_id in self.token_id_to_index:
                    group_name = token_string.lower()
                    if group_name not in token_groups:
                        token_groups[group_name] = {
                            "indices": [],
                            "tokens": [],
                            "original_keyword": token_string
                        }
                    
                    token_groups[group_name]["indices"].append(self.token_id_to_index[token_id])
                    token_groups[group_name]["tokens"].append(token_string)
            
            return token_groups
    
    def compute_concept_token_scores(self, value_vector: np.ndarray, concept_embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Compute token scores for concept-specific tokens by multiplying concept embedding matrix with value vector
        
        Args:
            value_vector: Value vector vℓj from down_proj matrix layers 14-22 (shape: d)
            concept_embeddings: Concept token embedding matrix E_C (shape: n_concept_tokens x d)
            
        Returns:
            Tuple of (concept_token_scores, scoring_info)
        """
        # Compute scores: rℓj = E_C @ vℓj
        concept_token_scores = concept_embeddings @ value_vector  # Shape: (n_concept_tokens,)
        
        # Compute scoring metrics
        scores_mean = float(np.mean(concept_token_scores))
        scores_std = float(np.std(concept_token_scores))
        scores_max = float(np.max(concept_token_scores))
        scores_min = float(np.min(concept_token_scores))
        
        # Additional metrics
        scoring_info = {
            "scores_mean": scores_mean,
            "scores_std": scores_std,
            "scores_max": scores_max,
            "scores_min": scores_min,
            "scores_range": scores_max - scores_min,
            "value_vector_norm": float(np.linalg.norm(value_vector)),
            "concept_embedding_matrix_shape": list(concept_embeddings.shape),
            "num_concept_tokens": int(concept_embeddings.shape[0])
        }
        
        return concept_token_scores, scoring_info

    # Helper method to normalize token strings into variant groups
    def _normalize_token_for_grouping(self, s: str) -> str:
        """Normalize a token string to a base form for grouping variants.
        - Lowercase
        - Remove SentencePiece underline markers (▁) and spaces
        - Strip leading/trailing punctuation/underscores/dashes
        """
        if not isinstance(s, str):
            return str(s)
        base = s.lower()
        base = base.replace('▁', ' ').strip()
        # strip leading/trailing punctuation and connectors
        base = re.sub(r'^[\s\-_\.,;:!?\"\'\(\)\[\]\{\}]+', '', base)
        base = re.sub(r'[\s\-_\.,;:!?\"\'\(\)\[\]\{\}]+$', '', base)
        # remove internal spaces for grouping (treat "harry potter" as one key if tokens include space)
        base = base.replace(' ', '')
        return base or s.lower().strip()

    def analyze_concept_value_vectors(self, concept_name: str, top_k: int = 50) -> Dict:
        """
        Analyze value vectors for a single concept by computing token activation scores.
        Variant grouping enabled: tokens are grouped by normalized form; group score = max token score.
        Uses ALL concept groups for ranking since tokens are pre-selected to be concept-specific.
        
        Args:
            concept_name: Name of the concept
            top_k: Number of top candidates to return
            
        Returns:
            Dictionary with value vector analysis results
        """
        print(f"🔍 Analyzing value vectors for concept: {concept_name}")
        
        # Validate concept exists
        if concept_name not in self.concept_mappings:
            raise ValueError(f"Concept '{concept_name}' not found in concept mappings")
        
        # Get concept token embeddings
        concept_info = self.concept_mappings[concept_name]
        token_ids = [token_info["token_id"] for token_info in concept_info["tokens"]]
        
        # Get embedding indices for concept tokens
        concept_embedding_indices = [self.token_id_to_index[tid] for tid in token_ids if tid in self.token_id_to_index]
        if not concept_embedding_indices:
            raise ValueError(f"No valid token embeddings found for concept '{concept_name}'")
            
        concept_embeddings = self.token_embeddings[concept_embedding_indices]  # Shape: (n_concept_tokens, dim)
        
        print(f"  📊 Concept has {len(token_ids)} tokens")
        print(f"  📏 Concept embedding matrix shape: {concept_embeddings.shape}")
        
        # Build variant groups once per concept
        token_strs = [self.token_id_to_string[tid] for tid in token_ids if tid in self.token_id_to_string]
        group_map: Dict[str, List[int]] = {}
        for idx, tok in enumerate(token_strs):
            key = self._normalize_token_for_grouping(tok)
            group_map.setdefault(key, []).append(idx)
        group_keys = list(group_map.keys())
        num_groups = len(group_keys)
        if num_groups == 0:
            raise ValueError(f"No groups formed for concept {concept_name}")
        
        # Analyze all candidate value vectors 
        analysis_results = []
        
        batch_size = 1000  # Process in batches to manage memory
        n_candidates = self.candidate_vectors.shape[0]
        
        for i in tqdm(range(0, n_candidates, batch_size), desc=f"  Computing group scores"):
            batch_end = min(i + batch_size, n_candidates)
            batch_vectors = self.candidate_vectors[i:batch_end]
            
            for j, value_vector in enumerate(batch_vectors):
                vector_idx = i + j
                vector_key = self.vector_index_mapping[str(vector_idx)]
                
                # Compute scores for concept-specific tokens: rℓj = E_C @ vℓj
                concept_token_scores, scoring_info = self.compute_concept_token_scores(
                    value_vector, concept_embeddings
                )
                
                # Compute per-group scores (max over member token scores)
                group_scores = np.empty(num_groups, dtype=np.float32)
                best_token_idx_per_group = np.empty(num_groups, dtype=np.int32)
                for gi, gk in enumerate(group_keys):
                    member_idx = group_map[gk]
                    member_scores = concept_token_scores[member_idx]
                    bi = int(np.argmax(member_scores))
                    group_scores[gi] = float(member_scores[bi])
                    best_token_idx_per_group[gi] = member_idx[bi]
                
                # Use ALL concept groups for ranking since tokens are pre-selected to be concept-specific
                all_group_mean = float(np.mean(group_scores))
                all_group_max = float(np.max(group_scores))
                all_group_std = float(np.std(group_scores))
                
                # Use ALL token groups (no subset limitation)
                # Since tokens are pre-selected to be concept-specific, use all groups
                all_g_indices = np.argsort(group_scores)[::-1]  # All groups, sorted by score
                
                # Also keep full-token mean for reference
                full_mean = scoring_info["scores_mean"]
                
                # Ranking metric options using ALL concept groups:
                # Option 1: Mean of all concept groups (since all tokens are concept-specific)
                concept_activation_strength_mean = all_group_mean
                
                # Option 2: Max score (favors sharp selectivity)
                concept_activation_strength_max = all_group_max
                
                # Option 3: Weighted combination (balance broad + sharp activation across all groups)
                concept_activation_strength_weighted = 0.7 * all_group_max + 0.3 * all_group_mean
                
                # Option 4: Ratio-based (high max relative to std suggests good selectivity within concept)
                if all_group_std > 0:
                    concept_activation_strength_ratio = all_group_max / all_group_std
                else:
                    concept_activation_strength_ratio = all_group_max
                
                # Selectivity metrics for better ranking
                selectivity_ratio = all_group_max / max(abs(full_mean), 0.01) if full_mean != 0 else all_group_max
                activation_spread = scoring_info["scores_std"]
                concept_specificity = all_group_mean / max(activation_spread, 0.01)
                
                # Use mean of all concept groups as primary ranking metric with selectivity bonus
                selectivity_bonus = min(selectivity_ratio * 0.1, 1.0)  # Cap bonus at 1.0
                concept_activation_strength = concept_activation_strength_mean + selectivity_bonus
                
                # Store result
                result = {
                    "vector_index": vector_idx,
                    "vector_key": vector_key,
                    "concept_activation_strength": concept_activation_strength,
                    "selectivity_ratio": selectivity_ratio,
                    "concept_specificity": concept_specificity,
                    "all_group_mean": all_group_mean,  # Mean of all concept groups
                    "all_group_max": all_group_max,    # Max of all concept groups
                    "all_group_std": all_group_std,    # Std of all concept groups
                    "full_mean": full_mean,
                    "total_groups": num_groups,        # Total number of concept groups
                    "grouping": {"enabled": True, "method": "variant_max", "num_groups": num_groups},
                    "scoring_info": scoring_info,
                    # For interpretability: all groups (up to 10 for display) with their best token
                    "top_groups": [
                        {
                            "group_key": group_keys[gidx],
                            "group_size": len(group_map[group_keys[gidx]]),
                            "best_concept_token_index": int(best_token_idx_per_group[gidx]),
                            "token_id": token_ids[int(best_token_idx_per_group[gidx])],
                            "token": self.token_id_to_string[token_ids[int(best_token_idx_per_group[gidx])]],
                            "score": float(group_scores[gidx])
                        }
                        for gidx in all_g_indices[:min(10, len(all_g_indices))]
                    ],
                    # Back-compat: expose the same top items as tokens list (best token per group, all groups)
                    "top_concept_tokens": [
                        {
                            "concept_token_index": int(best_token_idx_per_group[gidx]),
                            "token_id": token_ids[int(best_token_idx_per_group[gidx])],
                            "token": self.token_id_to_string[token_ids[int(best_token_idx_per_group[gidx])]],
                            "score": float(group_scores[gidx])
                        }
                        for gidx in all_g_indices[:min(10, len(all_g_indices))]
                    ]
                }
                
                analysis_results.append(result)
        
        # Sort by concept activation strength (highest top-k mean first)
        analysis_results.sort(key=lambda x: x["concept_activation_strength"], reverse=True)
        
        # Select top-k results
        top_results = analysis_results[:top_k]
        
        # Compute summary statistics
        all_group_means = [r["all_group_mean"] for r in analysis_results]
        all_group_maxes = [r["all_group_max"] for r in analysis_results]
        all_group_stds = [r["all_group_std"] for r in analysis_results]
        all_full_means = [r["full_mean"] for r in analysis_results]
        all_ranges = [r["scoring_info"]["scores_range"] for r in analysis_results]
        
        analysis_summary = {
            "concept_name": concept_name,
            "num_concept_tokens": len(token_ids),
            "concept_tokens": [
                {"token": self.token_id_to_string[tid], "token_id": tid}
                for tid in token_ids
            ],
            "total_candidates_tested": len(analysis_results),
            "top_k": top_k,
            "grouping": {"enabled": True, "method": "variant_max", "num_groups": num_groups},
            "statistics": {
                "max_activation_strength": float(np.max(all_group_means)),     # Based on all concept groups
                "mean_activation_strength": float(np.mean(all_group_means)),   # Based on all concept groups
                "median_activation_strength": float(np.median(all_group_means)), # Based on all concept groups
                "std_activation_strength": float(np.std(all_group_means)),     # Based on all concept groups
                "max_concept_group_max": float(np.max(all_group_maxes)),       # Max across all concept groups
                "mean_concept_group_std": float(np.mean(all_group_stds)),      # Mean std within concept groups
                "max_concept_full_mean": float(np.max(all_full_means)),        # Keep for reference
                "mean_concept_score_range": float(np.mean(all_ranges)),
                "top_k_activation_mean": float(np.mean([r["concept_activation_strength"] for r in top_results])),
                "ranking_method": "all_concept_groups_mean_with_selectivity_bonus"  # Document the ranking method used
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
        Analyze value vectors for all concepts (grouped by variants)
        Uses ALL concept groups for ranking since tokens are pre-selected to be concept-specific.
        
        Args:
            top_k: Number of top candidates per concept
            concept_subset: List of specific concepts to analyze (None = all)
            
        Returns:
            Dictionary with all concept analyses
        """
        print(f"🎯 Analyzing value vectors for all concepts (layers 14-22, top-{top_k} candidates, using ALL concept groups)")
        print("=" * 80)
        
        # Determine which concepts to analyze
        concepts_to_analyze = concept_subset if concept_subset else list(self.concept_mappings.keys())
        
        # Validate that requested concepts exist in concept mappings
        if concept_subset:
            available_concepts = set(self.concept_mappings.keys())
            missing_concepts = [c for c in concept_subset if c not in available_concepts]
            if missing_concepts:
                print(f"⚠️ Warning: The following concepts are not available in concept mappings:")
                for missing in missing_concepts:
                    print(f"    ❌ '{missing}'")
                print(f"\n📋 Available concepts: {sorted(available_concepts)}")
                
                # Filter to only include available concepts
                concepts_to_analyze = [c for c in concept_subset if c in available_concepts]
                if not concepts_to_analyze:
                    print(f"❌ No valid concepts found! Analysis cannot proceed.")
                    return {}
                print(f"✅ Proceeding with {len(concepts_to_analyze)} valid concepts")
        
        print(f"📊 Will analyze {len(concepts_to_analyze)} concepts:")
        for i, concept in enumerate(concepts_to_analyze, 1):
            print(f"    {i:2d}. {concept}")
        
        all_results = {
            "metadata": {
                "analysis_date": "2025-09-06",
                "method": "value_vector_token_scoring_all_groups_layers_14_22",
                "top_k": top_k,
                "ranking_method": "all_concept_groups_mean",
                "layer_range": "14-22",
                "total_concepts": len(concepts_to_analyze),
                "total_candidates": self.candidate_vectors.shape[0],
                "embedding_dimension": self.candidate_vectors.shape[1]
            },
            "concept_analyses": {}
        }
        
        # Analyze each concept
        for i, concept_name in enumerate(concepts_to_analyze):
            print(f"\n[{i+1}/{len(concepts_to_analyze)}] Analyzing: {concept_name}")
            
            try:
                concept_analysis = self.analyze_concept_value_vectors(concept_name, top_k)
                all_results["concept_analyses"][concept_name] = concept_analysis
                
                # Show quick summary
                max_activation = concept_analysis["statistics"]["max_activation_strength"]
                mean_activation = concept_analysis["statistics"]["mean_activation_strength"]
                max_group_score = concept_analysis["statistics"]["max_concept_group_max"]
                print(f"  ✅ Max activation strength: {max_activation:.4f}")
                print(f"  📊 Mean activation strength: {mean_activation:.4f}")
                print(f"  🎯 Max group score: {max_group_score:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {concept_name}: {str(e)}")
                all_results["concept_analyses"][concept_name] = {"error": str(e)}
        
        # Compute global statistics
        successful_analyses = [
            analysis for analysis in all_results["concept_analyses"].values()
            if "error" not in analysis
        ]
        
        if successful_analyses:
            global_all_group_means = [a["statistics"]["max_activation_strength"] for a in successful_analyses]
            global_mean_activations = [a["statistics"]["mean_activation_strength"] for a in successful_analyses]
            global_max_group_maxes = [a["statistics"]["max_concept_group_max"] for a in successful_analyses]
            
            all_results["global_statistics"] = {
                "successful_concepts": len(successful_analyses),
                "failed_concepts": len(concepts_to_analyze) - len(successful_analyses),
                "overall_max_activation_strength": float(np.max(global_all_group_means)),
                "overall_mean_activation_strength": float(np.mean(global_mean_activations)),
                "overall_max_concept_score": float(np.max(global_max_group_maxes)),
                "best_concept": max(successful_analyses, key=lambda x: x["statistics"]["max_activation_strength"])["concept_name"],
                "ranking_method": "all_concept_groups_mean",
                "distribution_stats": {
                    "max_activations_mean": float(np.mean(global_all_group_means)),
                    "max_activations_std": float(np.std(global_all_group_means)),
                    "mean_activations_mean": float(np.mean(global_mean_activations)),
                    "mean_activations_std": float(np.std(global_mean_activations)),
                    "max_scores_mean": float(np.mean(global_max_group_maxes)),
                    "max_scores_std": float(np.std(global_max_group_maxes))
                }
            }
        
        return all_results
    
    def save_results(self, results: Dict, output_dir: str = "."):
        """Save value vector analysis results"""
        print(f"\n💾 Saving value vector analysis results to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        results_path = os.path.join(output_dir, "projection_analysis_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary statistics
        summary_path = None
        if "global_statistics" in results:
            summary_path = os.path.join(output_dir, "projection_summary.json")
            summary = {
                "metadata": results["metadata"],
                "global_statistics": results["global_statistics"],
                "concept_summaries": {
                    concept: {
                        "max_activation_strength": analysis["statistics"]["max_activation_strength"],
                        "mean_activation_strength": analysis["statistics"]["mean_activation_strength"],
                        "max_concept_score": analysis["statistics"]["max_concept_group_max"],
                        "num_tokens": analysis["num_concept_tokens"],
                        "top_candidate": analysis["top_candidates"][0] if analysis["top_candidates"] else None
                    }
                    for concept, analysis in results["concept_analyses"].items()
                    if "error" not in analysis
                }
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved:")
        print(f"    📄 Full results: {results_path}")
        if summary_path:
            print(f"    📊 Summary: {summary_path}")
        
        return {"results_path": results_path, "summary_path": summary_path}
    
    def run_analysis(self, top_k: int = 50, concept_subset: List[str] = None, 
                    output_dir: str = ".") -> Dict:
        """
        Complete value vector analysis pipeline using all concept groups (layers 14-22 only)
        
        Args:
            top_k: Number of top candidates per concept
            concept_subset: List of specific concepts (None = all)
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths
        """
        print("🎯 Starting Concept Value Vector Analysis")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_candidate_vectors()
        self.load_token_embeddings()
        
        # Step 2: Run analysis
        results = self.analyze_all_concepts(top_k, concept_subset)
        
        # Step 3: Save results
        file_info = self.save_results(results, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ VALUE VECTOR ANALYSIS COMPLETE (LAYERS 14-22)!")
        print("=" * 80)
        
        if "global_statistics" in results:
            stats = results["global_statistics"]
            print(f"📊 Analyzed {stats['successful_concepts']} concepts successfully")
            print(f"🏆 Best overall activation (top-k mean): {stats['overall_max_activation_strength']:.4f}")
            print(f"📈 Avg of max activations: {stats['distribution_stats']['max_activations_mean']:.4f}")
            print(f"🎯 Best concept: {stats['best_concept']}")
        
        print(f"📁 Results saved to: {output_dir}")
        
        return file_info


def main():
    """Main value vector analysis function - analyzes layers 14-22 only using test concepts"""
    # Configuration
    candidate_vectors_dir = "extracted_vectors"
    token_embeddings_dir = "token_embeddings"
    output_dir = "value_vector_results_layers_14_22"
    top_k = 100  # Top-k candidate vectors per concept
    
    # Check if required directories exist
    if not os.path.exists(candidate_vectors_dir):
        print(f"❌ Candidate vectors directory not found: {candidate_vectors_dir}")
        print("Please run extract_candidate_vectors.py first")
        return
    
    if not os.path.exists(token_embeddings_dir):
        print(f"❌ Token embeddings directory not found: {token_embeddings_dir}")
        print("Please run extract_token_embeddings.py first")
        return
    
    # Create analyzer  
    analyzer = ConceptVectorProjector(candidate_vectors_dir, token_embeddings_dir)
    
    # Load concepts from test_concepts.json
    test_concepts_path = os.path.join("..", "token-gen", "test_concepts.json")
    if os.path.exists(test_concepts_path):
        with open(test_concepts_path, 'r', encoding='utf-8') as f:
            concept_subset = json.load(f)
        print(f"📋 Loaded {len(concept_subset)} concepts from test_concepts.json:")
        for i, concept in enumerate(concept_subset, 1):
            print(f"  {i:2d}. {concept}")
    else:
        print(f"⚠️ Test concepts file not found: {test_concepts_path}")
        print("📋 Using default concept: Harry Potter")
        concept_subset = ["Harry Potter"]
    
    # Run analysis
    file_info = analyzer.run_analysis(top_k, concept_subset, output_dir)
    
    print(f"\n🎉 Value vector analysis completed!")
    print(f"📁 Check the '{output_dir}' folder for results")
    if concept_subset:
        print(f"🎯 Analyzed concepts: {', '.join(concept_subset)}")

if __name__ == "__main__":
    main()
