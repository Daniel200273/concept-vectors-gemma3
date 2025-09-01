#!/usr/bin/env python3
"""
Concept Value Vector Analysis - Version 2

This script implements a new approach for analyzing value vectors by projecting them onto the entire vocabulary.

New Approach:
1. Load candidate vectors (value vectors from MLP down_proj columns)
2. Load token embeddings (for entire vocabulary)
3. For each concept, project vectors onto entire vocabulary: E_vocab @ v_j
4. FIRST CUT: Drop 30% of vectors with lowest average logit scores
5. SECOND CUT: Find top-k tokens for each vector and compare with concept keywords
   - Keep vectors with strong activation for concept-specific keywords
   - Keep vectors with low activation for non-concept tokens
6. End up with 20 candidate vectors per concept

Mathematical approach:
- Full vocabulary embeddings: E_vocab (shape: vocab_size x d)
- Candidate value vector: v_j (shape: d) - column from down_proj matrix
- Compute full vocabulary scores: scores = E_vocab @ v_j (shape: vocab_size)
- Apply filtering based on concept keyword activation patterns
"""

import numpy as np
import json
import os
import re
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from collections import defaultdict

# Configure environment for HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your HuggingFace token")
    
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = "/media/hdd/usr/martinelli/.cache/huggingface"

class ConceptVectorProjectorV2:
    """Analyze value vectors by projecting onto entire vocabulary and filtering by concept activation patterns"""
    
    def __init__(self, candidate_vectors_dir: str, full_vocab_embeddings_dir: str, concept_embeddings_dir: str):
        """
        Initialize the projector
        
        Args:
            candidate_vectors_dir: Directory with candidate vector files
            full_vocab_embeddings_dir: Directory with FULL vocabulary embeddings (for projection)
            concept_embeddings_dir: Directory with concept-specific embeddings (for concept mappings)
        """
        self.candidate_vectors_dir = candidate_vectors_dir
        self.full_vocab_embeddings_dir = full_vocab_embeddings_dir
        self.concept_embeddings_dir = concept_embeddings_dir
        
        # Data storage
        self.candidate_vectors = None
        self.candidate_metadata = None
        self.full_vocab_embeddings = None
        self.full_vocab_metadata = None
        self.concept_mappings = None
        self.vocab_size = None
        
    def load_candidate_vectors(self):
        """Load candidate vectors from extraction results"""
        print("📊 Loading candidate vectors...")
        
        try:
            # Load NumPy array
            vectors_path = os.path.join(self.candidate_vectors_dir, "candidate_vectors.npy")
            if not os.path.exists(vectors_path):
                raise FileNotFoundError(f"Candidate vectors file not found: {vectors_path}")
            
            self.candidate_vectors = np.load(vectors_path)
            
            # Load metadata
            metadata_path = os.path.join(self.candidate_vectors_dir, "candidate_vectors_metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Candidate vectors metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                self.candidate_metadata = json.load(f)
            
            # Load index mapping
            mapping_path = os.path.join(self.candidate_vectors_dir, "vector_index_mapping.json")
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Vector index mapping file not found: {mapping_path}")
            
            with open(mapping_path, 'r') as f:
                self.vector_index_mapping = json.load(f)
            
            print(f"✅ Loaded {self.candidate_vectors.shape[0]:,} candidate vectors")
            print(f"📏 Vector dimension: {self.candidate_vectors.shape[1]}")
            
        except Exception as e:
            print(f"❌ Error loading candidate vectors: {str(e)}")
            raise
    
    def load_full_vocabulary_embeddings(self):
        """Load FULL vocabulary embeddings for projection"""
        print("📚 Loading FULL vocabulary embeddings...")
        
        try:
            # Load NumPy array
            embeddings_path = os.path.join(self.full_vocab_embeddings_dir, "full_vocabulary_embeddings.npy")
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Full vocabulary embeddings file not found: {embeddings_path}")
            
            self.full_vocab_embeddings = np.load(embeddings_path)
            
            # Load metadata
            metadata_path = os.path.join(self.full_vocab_embeddings_dir, "full_vocabulary_metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Full vocabulary metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.full_vocab_metadata = json.load(f)
            
            self.vocab_size = self.full_vocab_embeddings.shape[0]
            
            # Load ID mappings
            id_to_index_path = os.path.join(self.full_vocab_embeddings_dir, "full_vocab_token_id_to_index.json")
            if not os.path.exists(id_to_index_path):
                raise FileNotFoundError(f"Token ID to index mapping file not found: {id_to_index_path}")
            
            with open(id_to_index_path, 'r') as f:
                self.token_id_to_index = {int(k): v for k, v in json.load(f).items()}
            
            id_to_string_path = os.path.join(self.full_vocab_embeddings_dir, "full_vocab_token_id_to_string.json")
            if not os.path.exists(id_to_string_path):
                raise FileNotFoundError(f"Token ID to string mapping file not found: {id_to_string_path}")
            
            with open(id_to_string_path, 'r', encoding='utf-8') as f:
                self.token_id_to_string = {int(k): v for k, v in json.load(f).items()}
            
            # Create reverse mapping from index to token info
            self.index_to_token_info = {}
            for token_id, index in self.token_id_to_index.items():
                self.index_to_token_info[index] = {
                    "token_id": token_id,
                    "token_string": self.token_id_to_string[token_id]
                }
            
            print(f"✅ Loaded {self.full_vocab_embeddings.shape[0]:,} FULL vocabulary embeddings")
            print(f"📏 Embedding dimension: {self.full_vocab_embeddings.shape[1]}")
            print(f"📚 Full vocabulary size: {self.vocab_size:,}")
            
        except Exception as e:
            print(f"❌ Error loading full vocabulary embeddings: {str(e)}")
            raise
    
    def load_concept_mappings(self):
        """Load concept mappings from concept embeddings directory"""
        print("🎯 Loading concept mappings...")
        
        try:
            # Load concept mappings
            metadata_path = os.path.join(self.concept_embeddings_dir, "token_embeddings_metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Concept embeddings metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                concept_metadata = json.load(f)
            
            if "concept_mappings" not in concept_metadata:
                raise KeyError("No 'concept_mappings' found in concept metadata")
            
            self.concept_mappings = concept_metadata["concept_mappings"]
            
            print(f"🎯 Found {len(self.concept_mappings)} concepts")
            
        except Exception as e:
            print(f"❌ Error loading concept mappings: {str(e)}")
            raise
    
    def compute_full_vocabulary_scores(self, value_vector: np.ndarray) -> np.ndarray:
        """
        Compute scores for entire vocabulary by multiplying vocabulary embedding matrix with value vector
        
        Args:
            value_vector: Value vector v_j from down_proj matrix (shape: d)
            
        Returns:
            Full vocabulary scores: E_vocab @ v_j (shape: vocab_size)
        """
        # Compute scores: scores = E_vocab @ v_j
        scores = self.full_vocab_embeddings @ value_vector  # Shape: (vocab_size,)
        return scores
    
    def _validate_dimensions(self):
        """Validate that candidate vectors and vocabulary embeddings have compatible dimensions"""
        print("🔍 Validating dimensions...")
        
        candidate_dim = self.candidate_vectors.shape[1]
        vocab_dim = self.full_vocab_embeddings.shape[1]
        
        if candidate_dim != vocab_dim:
            raise ValueError(
                f"Dimension mismatch! Candidate vectors: {candidate_dim}, "
                f"Vocabulary embeddings: {vocab_dim}. These must be equal for matrix multiplication."
            )
        
        print(f"✅ Dimensions validated: {candidate_dim} = {vocab_dim}")
    
    def get_concept_keyword_indices(self, concept_name: str) -> Tuple[Set[int], Set[int]]:
        """
        Get concept-specific keyword indices and non-concept token indices
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            Tuple of (concept_keyword_indices, non_concept_indices)
        """
        # Get concept token embeddings
        concept_info = self.concept_mappings[concept_name]
        concept_token_ids = [token_info["token_id"] for token_info in concept_info["tokens"]]
        
        # Get embedding indices for concept tokens
        concept_indices = set()
        missing_tokens = []
        
        for tid in concept_token_ids:
            if tid in self.token_id_to_index:
                concept_indices.add(self.token_id_to_index[tid])
            else:
                missing_tokens.append(tid)
        
        if missing_tokens:
            print(f"⚠️  Warning: {len(missing_tokens)} concept tokens not found in full vocabulary")
            print(f"   Missing token IDs: {missing_tokens[:10]}{'...' if len(missing_tokens) > 10 else ''}")
        
        if not concept_indices:
            raise ValueError(f"No concept tokens found in full vocabulary for concept: {concept_name}")
        
        # Non-concept indices are all other vocabulary indices
        non_concept_indices = set(range(self.vocab_size)) - concept_indices
        
        print(f"   📊 Concept '{concept_name}': {len(concept_indices)} tokens found, {len(non_concept_indices)} non-concept tokens")
        
        return concept_indices, non_concept_indices
    
    def apply_first_cut(self, concept_name: str, drop_percentage: float = 0.3) -> List[int]:
        """
        FIRST CUT: Drop vectors with lowest average logit scores
        
        Args:
            concept_name: Name of the concept
            drop_percentage: Percentage of vectors to drop (default: 0.3 = 30%)
            
        Returns:
            List of vector indices that passed the first cut
        """
        print(f"🔪 Applying FIRST CUT for concept: {concept_name}")
        print(f"   Dropping {drop_percentage*100:.0f}% of vectors with lowest average scores...")
        
        concept_indices, _ = self.get_concept_keyword_indices(concept_name)
        
        # Compute average scores for concept keywords for each vector
        vector_scores = []
        
        batch_size = 100  # Process in batches to manage memory
        n_candidates = self.candidate_vectors.shape[0]
        
        for i in tqdm(range(0, n_candidates, batch_size), desc="  Computing concept scores"):
            batch_end = min(i + batch_size, n_candidates)
            batch_vectors = self.candidate_vectors[i:batch_end]
            
            for j, value_vector in enumerate(batch_vectors):
                vector_idx = i + j
                
                # Compute full vocabulary scores
                full_scores = self.compute_full_vocabulary_scores(value_vector)
                
                # Get scores for concept keywords only
                concept_scores = full_scores[list(concept_indices)]
                
                # Compute average score for concept keywords
                avg_concept_score = float(np.mean(concept_scores))
                
                vector_scores.append((vector_idx, avg_concept_score))
        
        # Sort by average concept score (descending)
        vector_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many to keep
        n_to_keep = int(len(vector_scores) * (1 - drop_percentage))
        kept_vectors = [idx for idx, _ in vector_scores[:n_to_keep]]
        
        print(f"   ✅ Kept {len(kept_vectors):,} vectors out of {len(vector_scores):,}")
        print(f"   📊 Score range: {vector_scores[0][1]:.4f} to {vector_scores[-1][1]:.4f}")
        
        return kept_vectors
    
    def apply_second_cut(self, concept_name: str, candidate_vectors: List[int], 
                        top_k: int = 200, final_candidates: int = 20) -> List[Dict]:
        """
        SECOND CUT: Find top-k tokens for each vector and filter based on concept keyword activation patterns
        
        Args:
            concept_name: Name of the concept
            candidate_vectors: List of vector indices from first cut
            top_k: Number of top tokens to consider for each vector
            final_candidates: Number of final candidate vectors to return
            
        Returns:
            List of final candidate vector results
        """
        print(f"🔪 Applying SECOND CUT for concept: {concept_name}")
        print(f"   Finding top-{top_k} tokens per vector and filtering by concept activation patterns...")
        
        concept_indices, non_concept_indices = self.get_concept_keyword_indices(concept_name)
        
        # Analyze each candidate vector
        vector_analyses = []
        
        for vector_idx in tqdm(candidate_vectors, desc="  Analyzing candidate vectors"):
            value_vector = self.candidate_vectors[vector_idx]
            vector_key = self.vector_index_mapping[str(vector_idx)]
            
            # Compute full vocabulary scores
            full_scores = self.compute_full_vocabulary_scores(value_vector)
            
            # Get top-k token indices
            top_k_indices = np.argsort(full_scores)[-top_k:][::-1]
            top_k_scores = full_scores[top_k_indices]
            
            # Analyze concept vs non-concept activation patterns
            concept_activations = []
            non_concept_activations = []
            
            for idx, score in zip(top_k_indices, top_k_scores):
                if idx in concept_indices:
                    concept_activations.append(float(score))
                else:
                    non_concept_activations.append(float(score))
            
            # Compute metrics
            if concept_activations:
                concept_mean = np.mean(concept_activations)
                concept_max = np.max(concept_activations)
                concept_count = len(concept_activations)
            else:
                concept_mean = concept_max = 0.0
                concept_count = 0
            
            if non_concept_activations:
                non_concept_mean = np.mean(non_concept_activations)
                non_concept_max = np.max(non_concept_activations)
                non_concept_count = len(non_concept_activations)
            else:
                non_concept_mean = non_concept_max = 0.0
                non_concept_count = 0
            
            # Compute selectivity score (higher is better)
            # We want high concept activation and low non-concept activation
            if non_concept_mean > 0:
                selectivity_score = concept_mean / non_concept_mean
            else:
                selectivity_score = concept_mean * 100  # Bonus if no non-concept activation
            
            # Additional metrics
            concept_coverage = concept_count / len(concept_indices)  # How many concept keywords are in top-k
            activation_ratio = concept_mean / max(abs(non_concept_mean), 0.01)
            
            # Store analysis
            analysis = {
                "vector_index": vector_idx,
                "vector_key": vector_key,
                "concept_mean_activation": concept_mean,
                "concept_max_activation": concept_max,
                "concept_activation_count": concept_count,
                "non_concept_mean_activation": non_concept_mean,
                "non_concept_max_activation": non_concept_max,
                "non_concept_activation_count": non_concept_count,
                "selectivity_score": selectivity_score,
                "concept_coverage": concept_coverage,
                "activation_ratio": activation_ratio,
                "top_k_tokens": [
                    {
                        "token_index": int(idx),
                        "token_id": self.index_to_token_info[idx]["token_id"],
                        "token_string": self.index_to_token_info[idx]["token_string"],
                        "score": float(score),
                        "is_concept_keyword": idx in concept_indices
                    }
                    for idx, score in zip(top_k_indices[:20], top_k_scores[:20])  # Show top 20 for analysis
                ]
            }
            
            vector_analyses.append(analysis)
        
        # Sort by selectivity score (descending)
        vector_analyses.sort(key=lambda x: x["selectivity_score"], reverse=True)
        
        # Select final candidates
        final_candidates_list = vector_analyses[:final_candidates]
        
        print(f"   ✅ Selected {len(final_candidates_list)} final candidate vectors")
        print(f"   📊 Selectivity score range: {final_candidates_list[0]['selectivity_score']:.4f} to {final_candidates_list[-1]['selectivity_score']:.4f}")
        
        return final_candidates_list
    
    def analyze_concept_value_vectors(self, concept_name: str, 
                                   drop_percentage: float = 0.3,
                                   top_k: int = 200, 
                                   final_candidates: int = 20) -> Dict:
        """
        Analyze value vectors for a single concept using the new two-stage filtering approach
        
        Args:
            concept_name: Name of the concept
            drop_percentage: Percentage of vectors to drop in first cut
            top_k: Number of top tokens to consider for each vector
            final_candidates: Number of final candidate vectors to return
            
        Returns:
            Dictionary with value vector analysis results
        """
        print(f"🔍 Analyzing value vectors for concept: {concept_name}")
        print(f"   Using new approach: project onto full vocabulary + two-stage filtering")
        
        # Get concept info
        concept_info = self.concept_mappings[concept_name]
        concept_token_ids = [token_info["token_id"] for token_info in concept_info["tokens"]]
        
        print(f"  📊 Concept has {len(concept_token_ids)} keywords")
        print(f"  📚 Full vocabulary size: {self.vocab_size:,}")
        
        # Step 1: Apply first cut
        first_cut_vectors = self.apply_first_cut(concept_name, drop_percentage)
        
        # Step 2: Apply second cut
        final_candidates_list = self.apply_second_cut(
            concept_name, first_cut_vectors, top_k, final_candidates
        )
        
        # Compute summary statistics
        if final_candidates_list:
            concept_means = [c["concept_mean_activation"] for c in final_candidates_list]
            non_concept_means = [c["non_concept_mean_activation"] for c in final_candidates_list]
            selectivity_scores = [c["selectivity_score"] for c in final_candidates_list]
            concept_coverages = [c["concept_coverage"] for c in final_candidates_list]
            
            analysis_summary = {
                "concept_name": concept_name,
                "num_concept_keywords": len(concept_token_ids),
                "concept_keywords": [
                    {"token": self.token_id_to_string[tid], "token_id": tid}
                    for tid in concept_token_ids
                ],
                "vocabulary_size": self.vocab_size,
                "total_candidates_tested": self.candidate_vectors.shape[0],
                "first_cut_kept": len(first_cut_vectors),
                "final_candidates": len(final_candidates_list),
                "drop_percentage": drop_percentage,
                "top_k_tokens": top_k,
                "method": "full_vocabulary_projection_two_stage_filtering",
                "statistics": {
                    "max_concept_activation": float(np.max(concept_means)),
                    "mean_concept_activation": float(np.mean(concept_means)),
                    "min_concept_activation": float(np.min(concept_means)),
                    "max_non_concept_activation": float(np.max(non_concept_means)),
                    "mean_non_concept_activation": float(np.mean(non_concept_means)),
                    "min_non_concept_activation": float(np.min(non_concept_means)),
                    "max_selectivity_score": float(np.max(selectivity_scores)),
                    "mean_selectivity_score": float(np.mean(selectivity_scores)),
                    "min_selectivity_score": float(np.min(selectivity_scores)),
                    "mean_concept_coverage": float(np.mean(concept_coverages)),
                    "min_concept_coverage": float(np.min(concept_coverages))
                },
                "final_candidates": final_candidates_list
            }
        else:
            analysis_summary = {
                "concept_name": concept_name,
                "error": "No candidates passed filtering",
                "method": "full_vocabulary_projection_two_stage_filtering"
            }
        
        return analysis_summary
    
    def analyze_all_concepts(self, drop_percentage: float = 0.3, 
                           top_k: int = 200, final_candidates: int = 20,
                           concept_subset: Optional[List[str]] = None) -> Dict:
        """
        Analyze value vectors for all concepts using the new approach
        
        Args:
            drop_percentage: Percentage of vectors to drop in first cut
            top_k: Number of top tokens to consider for each vector
            final_candidates: Number of final candidate vectors per concept
            concept_subset: List of specific concepts to analyze (None = all)
            
        Returns:
            Dictionary with all concept analyses
        """
        print(f"🎯 Analyzing value vectors for all concepts using new approach")
        print(f"   Method: Full vocabulary projection + two-stage filtering")
        print(f"   First cut: Drop {drop_percentage*100:.0f}% lowest scoring vectors")
        print(f"   Second cut: Select top-{final_candidates} based on concept activation patterns")
        print("=" * 80)
        
        # Determine which concepts to analyze
        concepts_to_analyze = concept_subset if concept_subset else list(self.concept_mappings.keys())
        
        print(f"📊 Will analyze {len(concepts_to_analyze)} concepts")
        
        all_results = {
            "metadata": {
                "analysis_date": "2025-01-27",
                "method": "full_vocabulary_projection_two_stage_filtering",
                "drop_percentage": drop_percentage,
                "top_k": top_k,
                "final_candidates": final_candidates,
                "total_concepts": len(concepts_to_analyze),
                "total_candidates": self.candidate_vectors.shape[0],
                "vocabulary_size": self.vocab_size,
                "embedding_dimension": self.candidate_vectors.shape[1]
            },
            "concept_analyses": {}
        }
        
        # Analyze each concept
        for i, concept_name in enumerate(concepts_to_analyze):
            print(f"\n[{i+1}/{len(concepts_to_analyze)}] Analyzing: {concept_name}")
            
            try:
                concept_analysis = self.analyze_concept_value_vectors(
                    concept_name, drop_percentage, top_k, final_candidates
                )
                all_results["concept_analyses"][concept_name] = concept_analysis
                
                # Show quick summary
                if "error" not in concept_analysis:
                    max_selectivity = concept_analysis["statistics"]["max_selectivity_score"]
                    mean_concept_activation = concept_analysis["statistics"]["mean_concept_activation"]
                    mean_coverage = concept_analysis["statistics"]["mean_concept_coverage"]
                    print(f"  ✅ Max selectivity score: {max_selectivity:.4f}")
                    print(f"  📊 Mean concept activation: {mean_concept_activation:.4f}")
                    print(f"  🎯 Mean concept coverage: {mean_coverage:.4f}")
                else:
                    print(f"  ❌ Error: {concept_analysis['error']}")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {concept_name}: {str(e)}")
                all_results["concept_analyses"][concept_name] = {"error": str(e)}
        
        # Compute global statistics
        successful_analyses = [
            analysis for analysis in all_results["concept_analyses"].values()
            if "error" not in analysis
        ]
        
        if successful_analyses:
            global_selectivity_scores = [a["statistics"]["max_selectivity_score"] for a in successful_analyses]
            global_concept_activations = [a["statistics"]["mean_concept_activation"] for a in successful_analyses]
            global_coverages = [a["statistics"]["mean_concept_coverage"] for a in successful_analyses]
            
            all_results["global_statistics"] = {
                "successful_concepts": len(successful_analyses),
                "failed_concepts": len(concepts_to_analyze) - len(successful_analyses),
                "overall_max_selectivity": float(np.max(global_selectivity_scores)),
                "overall_mean_selectivity": float(np.mean(global_selectivity_scores)),
                "overall_max_concept_activation": float(np.max(global_concept_activations)),
                "overall_mean_concept_activation": float(np.mean(global_concept_activations)),
                "overall_mean_coverage": float(np.mean(global_coverages)),
                "best_concept": max(successful_analyses, key=lambda x: x["statistics"]["max_selectivity_score"])["concept_name"],
                "method": "full_vocabulary_projection_two_stage_filtering"
            }
        
        return all_results
    
    def save_results(self, results: Dict, output_dir: str = "."):
        """Save value vector analysis results"""
        print(f"\n💾 Saving value vector analysis results to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        results_path = os.path.join(output_dir, "projection_v2_analysis_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary statistics
        summary_path = None
        if "global_statistics" in results:
            summary_path = os.path.join(output_dir, "projection_v2_summary.json")
            summary = {
                "metadata": results["metadata"],
                "global_statistics": results["global_statistics"],
                "concept_summaries": {
                    concept: {
                        "max_selectivity_score": analysis["statistics"]["max_selectivity_score"],
                        "mean_concept_activation": analysis["statistics"]["mean_concept_activation"],
                        "mean_concept_coverage": analysis["statistics"]["mean_concept_coverage"],
                        "num_keywords": analysis["num_concept_keywords"],
                        "final_candidates": analysis["final_candidates"]
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
    
    def run_analysis(self, drop_percentage: float = 0.3, top_k: int = 200, 
                    final_candidates: int = 20, concept_subset: Optional[List[str]] = None,
                    output_dir: str = ".") -> Dict:
        """
        Complete value vector analysis pipeline using new approach
        
        Args:
            drop_percentage: Percentage of vectors to drop in first cut
            top_k: Number of top tokens to consider for each vector
            final_candidates: Number of final candidate vectors per concept
            concept_subset: List of specific concepts (None = all)
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths
        """
        print("🎯 Starting Concept Value Vector Analysis - Version 2")
        print("=" * 80)
        
        # Step 1: Load data
        self.load_candidate_vectors()
        self.load_full_vocabulary_embeddings()
        self.load_concept_mappings()
        
        # Step 1.5: Validate dimensions
        self._validate_dimensions()
        
        # Step 2: Run analysis
        results = self.analyze_all_concepts(drop_percentage, top_k, final_candidates, concept_subset)
        
        # Step 3: Save results
        file_info = self.save_results(results, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ VALUE VECTOR ANALYSIS V2 COMPLETE!")
        print("=" * 80)
        
        if "global_statistics" in results:
            stats = results["global_statistics"]
            print(f"📊 Analyzed {stats['successful_concepts']} concepts successfully")
            print(f"🏆 Best overall selectivity score: {stats['overall_max_selectivity']:.4f}")
            print(f"📈 Mean selectivity score: {stats['overall_mean_selectivity']:.4f}")
            print(f"🎯 Best concept: {stats['best_concept']}")
        
        print(f"📁 Results saved to: {output_dir}")
        
        return file_info


def main():
    """Main value vector analysis function"""
    # Configuration
    candidate_vectors_dir = "extracted_vectors"
    full_vocab_embeddings_dir = "full_vocabulary_embeddings"  # NEW: Full vocabulary embeddings
    token_embeddings_dir = "token_embeddings"  # Concept-specific embeddings for mappings
    output_dir = "value_vector_results_v2"
    drop_percentage = 0.3  # Drop 30% in first cut
    top_k = 200  # Consider top-200 tokens per vector
    final_candidates = 20  # End up with 20 vectors per concept
    
    # Check if required directories exist
    if not os.path.exists(candidate_vectors_dir):
        print(f"❌ Candidate vectors directory not found: {candidate_vectors_dir}")
        print("Please run extract_candidate_vectors.py first")
        return
    
    if not os.path.exists(full_vocab_embeddings_dir):
        print(f"❌ Full vocabulary embeddings directory not found: {full_vocab_embeddings_dir}")
        print("Please run extract_full_vocabulary_embeddings.py first")
        return
    
    if not os.path.exists(token_embeddings_dir):
        print(f"❌ Token embeddings directory not found: {token_embeddings_dir}")
        print("Please run extract_token_embeddings.py first")
        return
    
    # Create analyzer  
    analyzer = ConceptVectorProjectorV2(candidate_vectors_dir, full_vocab_embeddings_dir, token_embeddings_dir)
    
    # Test with a subset of concepts
    concept_subset = ["Harry Potter", "Amazon Alexa"]  # Test with 2 concepts
    
    # Run analysis
    file_info = analyzer.run_analysis(
        drop_percentage=drop_percentage,
        top_k=top_k,
        final_candidates=final_candidates,
        concept_subset=concept_subset,
        output_dir=output_dir
    )
    
    print(f"\n🎉 Value vector analysis V2 completed!")
    print(f"📁 Check the '{output_dir}' folder for results")

if __name__ == "__main__":
    main()
