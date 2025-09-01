#!/usr/bin/env python3
"""
Concept Value Vector Analysis

This script analyzes value vectors by computing concept-specific token activation scores.

Process:
1. Load candidate vectors (value vectors from MLP down_proj columns)
2. Load token embeddings (for concept tokens)
3. For each concept, compute token activation scores using E_C @ vℓj
4. Rank candidates by concept activation strength
5. Select top-k candidates for each concept (ranking by mean of top-k token scores)
6. Save results with detailed analysis

Mathematical approach:
- For concept C with tokens T1, T2, ..., Tn (concept-specific tokens)
- Concept token embeddings: E_C = [e1, e2, ..., en] (matrix of shape n_concept_tokens x d)
- Candidate value vector: vℓj (shape d) - column from down_proj matrix
- Compute concept token scores: rℓj = E_C @ vℓj (shape n_concept_tokens)
- Measure concept alignment: mean of top-k scores in rℓj (k ≪ n) to focus on strongest matches
"""

import numpy as np
import json
import os
import re
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Configure environment for HuggingFace
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"] = "/media/hdd/usr/martinelli/.cache/huggingface"

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
        """Load candidate vectors from extraction results"""
        print("📊 Loading candidate vectors...")
        
        # Load NumPy array
        vectors_path = os.path.join(self.candidate_vectors_dir, "candidate_vectors.npy")
        self.candidate_vectors = np.load(vectors_path)
        
        # Load metadata
        metadata_path = os.path.join(self.candidate_vectors_dir, "candidate_vectors_metadata.json")
        with open(metadata_path, 'r') as f:
            self.candidate_metadata = json.load(f)
        
        # Load index mapping
        mapping_path = os.path.join(self.candidate_vectors_dir, "vector_index_mapping.json")
        with open(mapping_path, 'r') as f:
            self.vector_index_mapping = json.load(f)
        
        print(f"✅ Loaded {self.candidate_vectors.shape[0]:,} candidate vectors")
        print(f"📏 Vector dimension: {self.candidate_vectors.shape[1]}")
    
    def load_token_embeddings(self):
        """Load token embeddings from extraction results"""
        print("📚 Loading token embeddings...")
        
        # Load NumPy array
        embeddings_path = os.path.join(self.token_embeddings_dir, "token_embeddings.npy")
        self.token_embeddings = np.load(embeddings_path)
        
        # Load metadata and concept mappings
        metadata_path = os.path.join(self.token_embeddings_dir, "token_embeddings_metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.token_metadata = json.load(f)
        
        self.concept_mappings = self.token_metadata["concept_mappings"]
        
        # Load ID mappings
        id_to_index_path = os.path.join(self.token_embeddings_dir, "token_id_to_index.json")
        with open(id_to_index_path, 'r') as f:
            self.token_id_to_index = {int(k): v for k, v in json.load(f).items()}
        
        id_to_string_path = os.path.join(self.token_embeddings_dir, "token_id_to_string.json")
        with open(id_to_string_path, 'r', encoding='utf-8') as f:
            self.token_id_to_string = {int(k): v for k, v in json.load(f).items()}
        
        print(f"✅ Loaded {self.token_embeddings.shape[0]:,} token embeddings")
        print(f"📏 Embedding dimension: {self.token_embeddings.shape[1]}")
        print(f"🎯 Found {len(self.concept_mappings)} concepts")
    
    def compute_concept_token_scores(self, value_vector: np.ndarray, concept_embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Compute token scores for concept-specific tokens by multiplying concept embedding matrix with value vector
        
        Args:
            value_vector: Value vector vℓj from down_proj matrix (shape: d)
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

    # NEW: helper to normalize token strings into variant groups
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

    def analyze_concept_value_vectors(self, concept_name: str, top_k: int = 50, top_tokens_k: int = 100) -> Dict:
        """
        Analyze value vectors for a single concept by computing token activation scores.
        Variant grouping enabled: tokens are grouped by normalized form; group score = max token score.
        Ranking now uses ALL concept groups (since tokens are pre-selected to be concept-specific).
        
        Args:
            concept_name: Name of the concept
            top_k: Number of top candidates to return
            top_tokens_k: Number of top GROUP scores to compute for comparison (ranking uses all groups)
            
        Returns:
            Dictionary with value vector analysis results
        """
        print(f"🔍 Analyzing value vectors for concept: {concept_name}")
        
        # Get concept token embeddings
        concept_info = self.concept_mappings[concept_name]
        token_ids = [token_info["token_id"] for token_info in concept_info["tokens"]]
        
        # Get embedding indices for concept tokens
        concept_embedding_indices = [self.token_id_to_index[tid] for tid in token_ids]
        concept_embeddings = self.token_embeddings[concept_embedding_indices]  # Shape: (n_concept_tokens, dim)
        
        print(f"  📊 Concept has {len(token_ids)} tokens")
        print(f"  📏 Concept embedding matrix shape: {concept_embeddings.shape}")
        
        # Build variant groups once per concept
        token_strs = [self.token_id_to_string[tid] for tid in token_ids]
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
                
                # Also compute top-k subset for comparison/analysis
                k = min(top_tokens_k, num_groups)
                top_g_indices = np.argsort(group_scores)[-k:][::-1]
                topk_group_scores = group_scores[top_g_indices]
                topk_mean = float(np.mean(topk_group_scores))
                topk_max = float(np.max(topk_group_scores))
                
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
                
                # Use mean of all concept groups as primary ranking metric
                concept_activation_strength = concept_activation_strength_mean
                
                # Store result
                result = {
                    "vector_index": vector_idx,
                    "vector_key": vector_key,
                    "concept_activation_strength": concept_activation_strength,
                    "selectivity_ratio": selectivity_ratio,
                    "concept_specificity": concept_specificity,
                    "all_group_mean": all_group_mean,  # New: mean of all concept groups
                    "all_group_max": all_group_max,    # New: max of all concept groups
                    "all_group_std": all_group_std,    # New: std of all concept groups
                    "topk_mean": topk_mean,            # Keep for comparison
                    "topk_max": topk_max,              # Keep for comparison
                    "full_mean": full_mean,
                    "top_tokens_k": k,  # interpreted as groups now
                    "total_groups": num_groups,        # New: total number of concept groups
                    "grouping": {"enabled": True, "method": "variant_max", "num_groups": num_groups},
                    "scoring_info": scoring_info,
                    # For interpretability: top groups (up to 10) with their best token
                    "top_groups": [
                        {
                            "group_key": group_keys[gidx],
                            "group_size": len(group_map[group_keys[gidx]]),
                            "best_concept_token_index": int(best_token_idx_per_group[gidx]),
                            "token_id": token_ids[int(best_token_idx_per_group[gidx])],
                            "token": self.token_id_to_string[token_ids[int(best_token_idx_per_group[gidx])]],
                            "score": float(group_scores[gidx])
                        }
                        for gidx in top_g_indices[:min(10, len(top_g_indices))]
                    ],
                    # Back-compat: expose the same top items as tokens list (best token per top group)
                    "top_concept_tokens": [
                        {
                            "concept_token_index": int(best_token_idx_per_group[gidx]),
                            "token_id": token_ids[int(best_token_idx_per_group[gidx])],
                            "token": self.token_id_to_string[token_ids[int(best_token_idx_per_group[gidx])]],
                            "score": float(group_scores[gidx])
                        }
                        for gidx in top_g_indices[:min(10, len(top_g_indices))]
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
        all_topk_means = [r["topk_mean"] for r in analysis_results]
        all_full_means = [r["full_mean"] for r in analysis_results]
        all_topk_maxes = [r["topk_max"] for r in analysis_results]
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
            "top_tokens_k": top_tokens_k,
            "grouping": {"enabled": True, "method": "variant_max", "num_groups": num_groups},
            "statistics": {
                "max_activation_strength": float(np.max(all_group_means)),     # Now based on all concept groups
                "mean_activation_strength": float(np.mean(all_group_means)),   # Now based on all concept groups
                "median_activation_strength": float(np.median(all_group_means)), # Now based on all concept groups
                "std_activation_strength": float(np.std(all_group_means)),     # Now based on all concept groups
                "max_concept_group_max": float(np.max(all_group_maxes)),       # Max across all concept groups
                "mean_concept_group_std": float(np.mean(all_group_stds)),      # Mean std within concept groups
                "max_concept_full_mean": float(np.max(all_full_means)),        # Keep for reference
                "max_concept_topk_max": float(np.max(all_topk_maxes)),         # Keep for comparison
                "mean_concept_score_range": float(np.mean(all_ranges)),
                "top_k_activation_mean": float(np.mean([r["concept_activation_strength"] for r in top_results])),
                "ranking_method": "all_concept_groups_mean"  # Document the ranking method used
            },
            "top_candidates": top_results,
            "concept_embedding_info": {
                "dimension": concept_embeddings.shape[1],
                "num_concept_tokens": concept_embeddings.shape[0],
                "mean_norm": float(np.mean(np.linalg.norm(concept_embeddings, axis=1)))
            }
        }
        
        return analysis_summary
    
    def analyze_all_concepts(self, top_k: int = 50, concept_subset: Optional[List[str]] = None, top_tokens_k: int = 100) -> Dict:
        """
        Analyze value vectors for all concepts (grouped by variants)
        Now uses ALL concept groups for ranking since tokens are pre-selected to be concept-specific.
        
        Args:
            top_k: Number of top candidates per concept
            concept_subset: List of specific concepts to analyze (None = all)
            top_tokens_k: Number of top group scores to compute for comparison (ranking uses all groups)
            
        Returns:
            Dictionary with all concept analyses
        """
        print(f"🎯 Analyzing value vectors for all concepts (top-{top_k} candidates, using ALL concept groups for ranking)")
        print("=" * 60)
        
        # Determine which concepts to analyze
        concepts_to_analyze = concept_subset if concept_subset else list(self.concept_mappings.keys())
        
        print(f"📊 Will analyze {len(concepts_to_analyze)} concepts")
        
        all_results = {
            "metadata": {
                "analysis_date": "2025-08-06",
                "method": "value_vector_token_scoring_all_groups",  # Updated method name
                "top_k": top_k,
                "top_tokens_k": top_tokens_k,
                "ranking_method": "all_concept_groups_mean",  # Document ranking approach
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
                concept_analysis = self.analyze_concept_value_vectors(concept_name, top_k, top_tokens_k)
                all_results["concept_analyses"][concept_name] = concept_analysis
                
                # Show quick summary
                max_activation = concept_analysis["statistics"]["max_activation_strength"]
                mean_activation = concept_analysis["statistics"]["mean_activation_strength"]
                max_topk = concept_analysis["statistics"]["max_concept_topk_max"]
                print(f"  ✅ Max activation strength: {max_activation:.4f}")
                print(f"  📊 Mean activation strength: {mean_activation:.4f}")
                print(f"  🎯 Max top-k group score: {max_topk:.4f}")
                
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
                        "max_concept_score": analysis["statistics"]["max_concept_topk_max"],
                        "num_tokens": analysis["num_concept_tokens"],
                        "top_tokens_k": analysis.get("top_tokens_k", results["metadata"].get("top_tokens_k")),
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
    
    def run_analysis(self, top_k: int = 50, concept_subset: Optional[List[str]] = None, 
                    output_dir: str = ".", top_tokens_k: int = 100) -> Dict:
        """
        Complete value vector analysis pipeline
        
        Args:
            top_k: Number of top candidates per concept
            concept_subset: List of specific concepts (None = all)
            output_dir: Directory to save results
            top_tokens_k: Number of top token scores to aggregate per concept
            
        Returns:
            Dictionary with file paths
        """
        print("🎯 Starting Concept Value Vector Analysis")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_candidate_vectors()
        self.load_token_embeddings()
        
        # Step 2: Run analysis
        results = self.analyze_all_concepts(top_k, concept_subset, top_tokens_k)
        
        # Step 3: Save results
        file_info = self.save_results(results, output_dir)
        
        print("\n" + "=" * 60)
        print("✅ VALUE VECTOR ANALYSIS COMPLETE!")
        print("=" * 60)
        
        if "global_statistics" in results:
            stats = results["global_statistics"]
            print(f"📊 Analyzed {stats['successful_concepts']} concepts successfully")
            print(f"🏆 Best overall activation (top-k mean): {stats['overall_max_activation_strength']:.4f}")
            print(f"📈 Avg of max activations: {stats['distribution_stats']['max_activations_mean']:.4f}")
            print(f"🎯 Best concept: {stats['best_concept']}")
        
        print(f"📁 Results saved to: {output_dir}")
        
        return file_info


def main():
    """Main value vector analysis function"""
    # Configuration
    candidate_vectors_dir = "extracted_vectors"
    token_embeddings_dir = "token_embeddings"
    output_dir = "value_vector_results"
    top_k = 100
    top_tokens_k = 100  # mean over top-100 concept token groups
    
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
    
    # Test with the 'Harry Potter' concept as requested
    concept_subset = ["Harry Potter"]
    
    # Run analysis
    file_info = analyzer.run_analysis(top_k, concept_subset, output_dir, top_tokens_k)
    
    print(f"\n🎉 Value vector analysis completed!")
    print(f"📁 Check the '{output_dir}' folder for results")

if __name__ == "__main__":
    main()
