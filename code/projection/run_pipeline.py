#!/usr/bin/env python3
"""
Concept Vector Extraction Pipeline

This script orchestrates the complete pipeline for extracting concept vectors from Gemma 3 1B:

1. Extract candidate vectors from MLP layers
2. Extract token embeddings for concept tokens  
3. Analyze candidate vectors using concept-specific token activation scores
4. Generate final results

Usage:
    python run_pipeline.py
"""

import argparse
import os
import sys
import time
from typing import List, Optional
import json

# Configure environment for HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME

# Import our modules
from extract_candidate_vectors import GemmaCandidateVectorExtractor
from extract_token_embeddings import GemmaTokenEmbeddingExtractor  
from project_and_rank import ConceptVectorProjector
from project_and_rank_gpu import ConceptVectorProjectorGPU

class ConceptVectorPipeline:
    """Complete pipeline for concept vector extraction"""
    
    def __init__(self, base_dir: str = ".", use_gpu: bool = False):
        """
        Initialize pipeline
        
        Args:
            base_dir: Base directory for all operations
            use_gpu: Whether to use GPU acceleration for analysis
        """
        self.base_dir = base_dir
        self.use_gpu = use_gpu
        self.concept_file = "../token-gen/test_concepts.json"
        self.token_mapping_file = "../token-gen/token-results/concept_keyword_ids.json"
        
        # Output directories
        self.candidate_vectors_dir = "extracted_vectors"
        self.token_embeddings_dir = "token_embeddings"
        self.analysis_results_dir = "value_vector_results_gpu" if use_gpu else "value_vector_results"
        self.final_results_dir = "final_concept_vectors"
        
        self.pipeline_start_time = None
        
        print(f"🚀 Pipeline initialized with {'GPU acceleration' if use_gpu else 'CPU processing'}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisite files exist"""
        print("🔍 Checking prerequisites...")
        
        # Check concept file
        if not os.path.exists(self.concept_file):
            print(f"❌ Concept file not found: {self.concept_file}")
            return False
            
        # Check token mapping file  
        if not os.path.exists(self.token_mapping_file):
            print(f"❌ Token mapping file not found: {self.token_mapping_file}")
            print("💡 Run token generation scripts first to create concept_keyword_ids.json")
            return False
        
        print("✅ Prerequisites check passed")
        return True
    
    def step1_extract_candidate_vectors(self, force_reextract: bool = False) -> bool:
        """Extract candidate vectors from Gemma MLP layers"""
        print("\n" + "="*60)
        print("STEP 1: EXTRACTING CANDIDATE VECTORS")
        print("="*60)
        
        # Check if already extracted
        vectors_file = os.path.join(self.candidate_vectors_dir, "candidate_vectors.npy")
        if os.path.exists(vectors_file) and not force_reextract:
            print(f"✅ Candidate vectors already extracted: {vectors_file}")
            print("Use --force-reextract to regenerate")
            return True
        
        try:
            extractor = GemmaCandidateVectorExtractor()
            file_info = extractor.extract_and_save(self.candidate_vectors_dir)
            
            print(f"✅ Step 1 completed successfully")
            print(f"📁 Vectors saved to: {self.candidate_vectors_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error in step 1: {e}")
            return False
    
    def step2_extract_token_embeddings(self, force_reextract: bool = False) -> bool:
        """Extract token embeddings for concept tokens"""
        print("\n" + "="*60)
        print("STEP 2: EXTRACTING TOKEN EMBEDDINGS")
        print("="*60)
        
        # Check if already extracted
        embeddings_file = os.path.join(self.token_embeddings_dir, "token_embeddings.npy")
        if os.path.exists(embeddings_file) and not force_reextract:
            print(f"✅ Token embeddings already extracted: {embeddings_file}")
            print("Use --force-reextract to regenerate")
            return True
        
        try:
            extractor = GemmaTokenEmbeddingExtractor()
            # Use the token mapping file that extract_token_embeddings expects
            file_info = extractor.extract_and_save(self.token_mapping_file, self.token_embeddings_dir)
            
            print(f"✅ Step 2 completed successfully") 
            print(f"📁 Embeddings saved to: {self.token_embeddings_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error in step 2: {e}")
            return False
    
    def step3_analyze_value_vectors(self, top_k: int = 100, concept_subset: Optional[List[str]] = None) -> bool:
        """Analyze candidate vectors using concept-specific token activation scores"""
        print("\n" + "="*60)
        if self.use_gpu:
            print("STEP 3: GPU-ACCELERATED VALUE VECTOR ANALYSIS")
        else:
            print("STEP 3: VALUE VECTOR ANALYSIS")
        print("="*60)
        
        try:
            if self.use_gpu:
                print("🚀 Using GPU acceleration...")
                projector = ConceptVectorProjectorGPU(self.candidate_vectors_dir, self.token_embeddings_dir)
            else:
                print("🖥️  Using CPU processing...")
                projector = ConceptVectorProjector(self.candidate_vectors_dir, self.token_embeddings_dir)
                
            file_info = projector.run_analysis(top_k, concept_subset, self.analysis_results_dir)
            
            print(f"✅ Step 3 completed successfully")
            print(f"📁 Analysis saved to: {self.analysis_results_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error in step 3: {e}")
            return False
    
    def step4_generate_final_results(self) -> bool:
        """Generate final concept vector results with analysis"""
        print("\n" + "="*60)
        print("STEP 4: GENERATING FINAL RESULTS")
        print("="*60)
        
        try:
            os.makedirs(self.final_results_dir, exist_ok=True)
            
            # Load analysis results (different filename for GPU version)
            if self.use_gpu:
                results_file = os.path.join(self.analysis_results_dir, "projection_gpu_analysis_results.json")
            else:
                results_file = os.path.join(self.analysis_results_dir, "projection_analysis_results.json")
                
            with open(results_file, 'r', encoding='utf-8') as f:
                analysis_results = json.load(f)
            
            # Load candidate vector metadata
            candidate_metadata_file = os.path.join(self.candidate_vectors_dir, "candidate_vectors_metadata.json")
            with open(candidate_metadata_file, 'r') as f:
                candidate_metadata = json.load(f)
            
            # Create final concept vector database
            final_results = {
                "metadata": {
                    "pipeline_version": "1.0",
                    "generation_date": "2025-09-06",
                    "model_name": "google/gemma-3-1b-it",
                    "extraction_method": "value_vector_analysis_gpu" if self.use_gpu else "value_vector_analysis",
                    "gpu_accelerated": self.use_gpu,
                    "total_concepts": len(analysis_results["concept_analyses"]),
                    "analysis_method": analysis_results["metadata"]
                },
                "concept_vectors": {},
                "global_analysis": analysis_results.get("global_statistics", {})
            }
            
            # Process each concept
            for concept_name, analysis in analysis_results["concept_analyses"].items():
                if "error" in analysis:
                    print(f"⚠️  Skipping {concept_name} due to analysis error")
                    continue
                
                # Get top candidate
                if not analysis["top_candidates"]:
                    print(f"⚠️  No candidates found for {concept_name}")
                    continue
                
                top_candidate = analysis["top_candidates"][0]
                
                # Parse vector key to get layer and neuron info
                vector_key = top_candidate["vector_key"]
                layer_num = int(vector_key.split('_')[0][1:])  # Extract from "L13_C0001"
                neuron_num = int(vector_key.split('_')[1][1:])
                
                final_results["concept_vectors"][concept_name] = {
                    "best_candidate": {
                        "vector_key": vector_key,
                        "layer": layer_num,
                        "neuron": neuron_num,
                        "concept_activation_strength": top_candidate["concept_activation_strength"],
                        "concept_analysis": top_candidate["scoring_info"]
                    },
                    "concept_info": {
                        "num_tokens": analysis["num_concept_tokens"],
                        "concept_tokens": analysis["concept_tokens"]
                    },
                    "analysis_stats": analysis["statistics"],
                    "alternative_candidates": analysis["top_candidates"][1:6] if len(analysis["top_candidates"]) > 1 else []
                }
            
            # Save final results
            final_results_file = os.path.join(self.final_results_dir, "final_concept_vectors.json")
            with open(final_results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            # Generate summary report
            self.generate_summary_report(final_results)
            
            print(f"✅ Step 4 completed successfully")
            print(f"📁 Final results saved to: {self.final_results_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error in step 4: {e}")
            return False
    
    def generate_summary_report(self, final_results: dict):
        """Generate a human-readable summary report"""
        report_path = os.path.join(self.final_results_dir, "concept_vectors_summary.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("GEMMA 3 1B CONCEPT VECTOR EXTRACTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Pipeline info
            metadata = final_results["metadata"]
            f.write(f"Model: {metadata['model_name']}\n")
            f.write(f"Generation Date: {metadata['generation_date']}\n")
            f.write(f"Total Concepts: {metadata['total_concepts']}\n")
            f.write(f"GPU Accelerated: {metadata.get('gpu_accelerated', False)}\n")
            f.write(f"Extraction Method: {metadata['extraction_method']}\n\n")
            
            # Global statistics
            if "global_analysis" in final_results and final_results["global_analysis"]:
                stats = final_results["global_analysis"]
                f.write("GLOBAL STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Successful Concepts: {stats['successful_concepts']}\n")
                f.write(f"Best Overall Activation Strength: {stats['overall_max_activation_strength']:.4f}\n")
                f.write(f"Average Max Activation Strength: {stats['distribution_stats']['max_activations_mean']:.4f}\n")
                f.write(f"Best Performing Concept: {stats['best_concept']}\n\n")
            
            # Top concepts by activation strength
            concept_performances = [
                (concept, data["best_candidate"]["concept_activation_strength"])
                for concept, data in final_results["concept_vectors"].items()
            ]
            concept_performances.sort(key=lambda x: x[1], reverse=True)
            
            f.write("TOP 20 CONCEPTS BY ACTIVATION STRENGTH\n")
            f.write("-" * 50 + "\n")
            for i, (concept, activation_strength) in enumerate(concept_performances[:20]):
                data = final_results["concept_vectors"][concept]
                layer = data["best_candidate"]["layer"]
                neuron = data["best_candidate"]["neuron"]
                num_tokens = data["concept_info"]["num_tokens"]
                f.write(f"{i+1:2d}. {concept:<30} | Act: {activation_strength:.4f} | L{layer:2d}N{neuron:4d} | {num_tokens:2d} tokens\n")
            
            f.write(f"\n\nDETAILED CONCEPT ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            # Detailed info for each concept
            for concept, data in sorted(final_results["concept_vectors"].items()):
                best = data["best_candidate"]
                f.write(f"Concept: {concept}\n")
                f.write(f"  Best Vector: Layer {best['layer']}, Neuron {best['neuron']}\n")
                f.write(f"  Activation Strength: {best['concept_activation_strength']:.4f}\n")
                f.write(f"  Concept Tokens: {data['concept_info']['num_tokens']}\n")
                
                # Show a few example tokens
                example_tokens = data["concept_info"]["concept_tokens"][:5]
                token_strs = [f"'{t['token']}'" for t in example_tokens]
                f.write(f"  Example Tokens: {', '.join(token_strs)}\n")
                if len(data["concept_info"]["concept_tokens"]) > 5:
                    f.write(f"  (and {len(data['concept_info']['concept_tokens']) - 5} more...)\n")
                f.write("\n")
        
        print(f"📊 Summary report saved to: {report_path}")
    
    def run_complete_pipeline(self, top_k: int = 100, concept_subset: Optional[List[str]] = None,
                            force_reextract: bool = False) -> bool:
        """Run the complete concept vector extraction pipeline"""
        self.pipeline_start_time = time.time()
        
        print("🚀 STARTING CONCEPT VECTOR EXTRACTION PIPELINE")
        print("=" * 60)
        print(f"📊 Target: Extract concept vectors for Gemma 3 1B")
        print(f"🎯 Top-k candidates per concept: {top_k}")
        print(f"⚡ Processing mode: {'GPU-accelerated' if self.use_gpu else 'CPU'}")
        if concept_subset:
            print(f"🔍 Analyzing subset: {len(concept_subset)} concepts")
        print("")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Step 1: Extract candidate vectors
        if not self.step1_extract_candidate_vectors(force_reextract):
            return False
        
        # Step 2: Extract token embeddings  
        if not self.step2_extract_token_embeddings(force_reextract):
            return False
        
        # Step 3: Analyze value vectors
        if not self.step3_analyze_value_vectors(top_k, concept_subset):
            return False
        
        # Step 4: Generate final results
        if not self.step4_generate_final_results():
            return False
        
        # Pipeline complete
        total_time = time.time() - self.pipeline_start_time
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📁 Final results in: {self.final_results_dir}")
        print(f"📊 Summary report: {os.path.join(self.final_results_dir, 'concept_vectors_summary.txt')}")
        
        return True

def main():
    """Main pipeline function with command line arguments"""
    parser = argparse.ArgumentParser(description="Concept Vector Extraction Pipeline for Gemma 3 1B")
    parser.add_argument("--top-k", type=int, default=100, 
                       help="Number of top candidate vectors to analyze per concept (default: 100)")
    parser.add_argument("--concepts", type=str, 
                       help="Comma-separated list of specific concepts to analyze (default: all)")
    parser.add_argument("--force-reextract", action="store_true",
                       help="Force re-extraction even if files exist")
    parser.add_argument("--base-dir", type=str, default=".",
                       help="Base directory for pipeline operations (default: current directory)")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU acceleration for value vector analysis (requires CUDA)")
    
    args = parser.parse_args()
    
    # Parse concept subset
    concept_subset = None
    if args.concepts:
        concept_subset = [c.strip() for c in args.concepts.split(",")]
        print(f"🎯 Will analyze specific concepts: {concept_subset}")
    
    # Create and run pipeline
    pipeline = ConceptVectorPipeline(args.base_dir, use_gpu=args.gpu)
    
    success = pipeline.run_complete_pipeline(
        top_k=args.top_k,
        concept_subset=concept_subset,
        force_reextract=args.force_reextract
    )
    
    if success:
        print(f"\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
