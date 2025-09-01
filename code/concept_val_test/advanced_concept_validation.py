"""
Advanced Concept Vector Validation
Inspired by ConceptVectors repository methodology
https://github.com/yihuaihong/ConceptVectors
"""

import torch
import numpy as np
import json
import random
import statistics
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import copy
from typing import List, Dict, Tuple, Any

# Disable Triton compiler for older GPUs
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# HuggingFace token (add your token here)
HF_TOKEN = "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"  # Replace with your actual token

# Set seeds for reproducibility
random.seed(999)
np.random.seed(999)
torch.manual_seed(999)

class ConceptVectorValidator:
    def __init__(self, model_name: str = "google/gemma-3-1b-it", device: str = "cuda"):
        """
        Initialize the concept vector validator
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            token=HF_TOKEN
        ).eval()
        
        print(f"✅ Model loaded on {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Store original parameters for restoration
        self.original_params = None
        self.current_params = None
        
    def backup_original_params(self):
        """Backup original model parameters"""
        self.original_params = copy.deepcopy(self.model.state_dict())
        
    def restore_original_params(self):
        """Restore original model parameters"""
        if self.original_params is not None:
            self.model.load_state_dict(self.original_params)
            
    def inject_noise_to_vector(self, location: Tuple[int, int], noise_scale: float = 0.1):
        """
        Inject Gaussian noise to specific concept vector location
        
        Args:
            location: (layer, dimension) tuple
            noise_scale: Standard deviation of Gaussian noise
        """
        layer, dimension = location
        
        # Determine the correct parameter name based on model architecture
        if 'gemma' in self.model_name.lower():
            param_name = f'model.layers.{layer}.mlp.down_proj.weight'
        elif 'llama' in self.model_name.lower():
            param_name = f'model.layers.{layer}.mlp.down_proj.weight'
        elif 'olmo' in self.model_name.lower():
            param_name = f'model.transformer.blocks.{layer}.ff_out.weight'
        else:
            # Default to Gemma/LLaMA structure
            param_name = f'model.layers.{layer}.mlp.down_proj.weight'
        
        # Create Gaussian noise
        hidden_size = self.model.config.hidden_size
        noise = torch.normal(0, noise_scale, size=(hidden_size,)).to(self.device)
        
        # Apply noise to the concept vector
        current_state = self.model.state_dict()
        current_state[param_name][:, dimension] += noise
        self.model.load_state_dict(current_state)
        
    def generate_answers(self, questions: List[str], max_new_tokens: int = 100) -> List[str]:
        """
        Generate answers for a list of questions
        
        Args:
            questions: List of questions to answer
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of generated answers
        """
        # Format questions properly
        formatted_questions = [f"Question: {q}\nAnswer:" for q in questions]
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_questions, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False
        ).to(self.device)
        
        # Generate responses
        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                temperature=None,
                top_p=None,
                top_k=None
            )
        
        # Decode only the newly generated tokens
        answers = self.tokenizer.batch_decode(
            generation_output[:, -max_new_tokens:], 
            skip_special_tokens=True
        )
        
        return answers
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate"""
        try:
            smoothing = SmoothingFunction().method1
            reference_tokens = reference.strip().split()
            candidate_tokens = candidate.strip().split()
            
            if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
                return 0.0
                
            score = sentence_bleu(
                [reference_tokens], 
                candidate_tokens, 
                smoothing_function=smoothing
            )
            return score
        except:
            return 0.0
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score between reference and candidate"""
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference.strip(), candidate.strip())
            return scores['rougeL'].fmeasure
        except:
            return 0.0
    
    def random_select_except(self, concept_list: List[Dict], n: int, exclude_index: int) -> List[Dict]:
        """
        Randomly select n concepts excluding the one at exclude_index
        
        Args:
            concept_list: List of concept dictionaries
            n: Number of concepts to select
            exclude_index: Index to exclude
            
        Returns:
            List of selected concepts
        """
        candidates = [concept for i, concept in enumerate(concept_list) if i != exclude_index]
        if len(candidates) < n:
            return candidates
        return random.sample(candidates, n)
    
    def validate_concept_vector(
        self, 
        concept_data: Dict, 
        concept_list: List[Dict],
        concept_index: int,
        location: Tuple[int, int], 
        noise_scale: float = 0.1,
        n_unrelated_concepts: int = 5
    ) -> Dict[str, Any]:
        """
        Validate a single concept vector using noise injection methodology
        
        Args:
            concept_data: Dictionary containing concept information and QA pairs
            concept_list: List of all concepts for selecting unrelated questions
            concept_index: Index of current concept in concept_list
            location: (layer, dimension) tuple for concept vector location
            noise_scale: Standard deviation of Gaussian noise
            n_unrelated_concepts: Number of unrelated concepts to sample
            
        Returns:
            Dictionary containing validation results
        """
        concept_name = concept_data.get('concept', 'Unknown')  # Changed from 'Concept' to 'concept'
        qa_pairs = concept_data.get('qa', [])  # Changed from 'QA' to 'qa'
        questions = [pair.get('q', '') for pair in qa_pairs if pair.get('q')]  # Extract questions from qa pairs
        
        if not questions:
            print(f"Warning: No questions found for concept {concept_name}")
            return None
        
        print(f"Validating concept: {concept_name}")
        print(f"Location: Layer {location[0]}, Dimension {location[1]}")
        print(f"Number of concept questions: {len(questions)}")
        
        # Select unrelated questions from other concepts
        unrelated_questions = []
        random_selection = self.random_select_except(concept_list, n_unrelated_concepts, concept_index)
        
        for selection in random_selection:
            if 'qa' in selection:  # Changed from 'QA' to 'qa'
                qa_pairs = selection['qa']
                selection_questions = [pair.get('q', '') for pair in qa_pairs if pair.get('q')]
                unrelated_questions.extend(selection_questions)
        
        # Limit unrelated questions to match concept questions count
        if len(unrelated_questions) > len(questions):
            unrelated_questions = random.sample(unrelated_questions, len(questions))
        
        print(f"Number of unrelated questions: {len(unrelated_questions)}")
        
        # Backup original parameters
        self.backup_original_params()
        
        try:
            # Generate original answers (without noise)
            print("Generating original answers...")
            original_answers = self.generate_answers(questions)
            original_unrelated_answers = self.generate_answers(unrelated_questions)
            
            # Inject noise and generate perturbed answers
            print(f"Injecting noise (σ={noise_scale}) and generating perturbed answers...")
            self.inject_noise_to_vector(location, noise_scale)
            perturbed_answers = self.generate_answers(questions)
            perturbed_unrelated_answers = self.generate_answers(unrelated_questions)
            
            # Calculate metrics
            bleu_scores = []
            rouge_l_scores = []
            unrelated_bleu_scores = []
            unrelated_rouge_l_scores = []
            
            # Calculate scores for concept-specific questions
            for perturbed, original in zip(perturbed_answers, original_answers):
                bleu_scores.append(self.calculate_bleu(original, perturbed))
                rouge_l_scores.append(self.calculate_rouge_l(original, perturbed))
            
            # Calculate scores for unrelated questions
            for perturbed, original in zip(perturbed_unrelated_answers, original_unrelated_answers):
                unrelated_bleu_scores.append(self.calculate_bleu(original, perturbed))
                unrelated_rouge_l_scores.append(self.calculate_rouge_l(original, perturbed))
            
            # Aggregate scores
            avg_bleu = statistics.mean(bleu_scores) if bleu_scores else 0.0
            avg_rouge_l = statistics.mean(rouge_l_scores) if rouge_l_scores else 0.0
            avg_unrelated_bleu = statistics.mean(unrelated_bleu_scores) if unrelated_bleu_scores else 0.0
            avg_unrelated_rouge_l = statistics.mean(unrelated_rouge_l_scores) if unrelated_rouge_l_scores else 0.0
            
            # Calculate degradation differences
            bleu_drop_target = 1.0 - avg_bleu
            bleu_drop_unrelated = 1.0 - avg_unrelated_bleu
            rouge_drop_target = 1.0 - avg_rouge_l
            rouge_drop_unrelated = 1.0 - avg_unrelated_rouge_l
            
            # Concept-specificity indicators
            bleu_specificity = bleu_drop_target - bleu_drop_unrelated
            rouge_specificity = rouge_drop_target - rouge_drop_unrelated
            
            results = {
                'concept': concept_name,
                'location': location,
                'noise_scale': noise_scale,
                'n_questions': len(questions),
                'n_unrelated_questions': len(unrelated_questions),
                
                # Target concept scores
                'bleu_score': avg_bleu,
                'rouge_l_score': avg_rouge_l,
                'bleu_drop_target': bleu_drop_target,
                'rouge_drop_target': rouge_drop_target,
                
                # Unrelated concept scores
                'unrelated_bleu_score': avg_unrelated_bleu,
                'unrelated_rouge_l_score': avg_unrelated_rouge_l,
                'bleu_drop_unrelated': bleu_drop_unrelated,
                'rouge_drop_unrelated': rouge_drop_unrelated,
                
                # Concept-specificity measures
                'bleu_specificity': bleu_specificity,
                'rouge_specificity': rouge_specificity,
                
                # Individual scores for analysis
                'individual_bleu_scores': bleu_scores,
                'individual_rouge_scores': rouge_l_scores,
                'individual_unrelated_bleu_scores': unrelated_bleu_scores,
                'individual_unrelated_rouge_scores': unrelated_rouge_l_scores,
                
                # Sample answers for inspection
                'sample_original_answers': original_answers[:3],
                'sample_perturbed_answers': perturbed_answers[:3],
                'sample_unrelated_original': original_unrelated_answers[:3],
                'sample_unrelated_perturbed': perturbed_unrelated_answers[:3]
            }
            
            # Print validation results
            print(f"Results for {concept_name}:")
            print(f"  Target BLEU: {avg_bleu:.4f}, Target ROUGE-L: {avg_rouge_l:.4f}")
            print(f"  Unrelated BLEU: {avg_unrelated_bleu:.4f}, Unrelated ROUGE-L: {avg_unrelated_rouge_l:.4f}")
            print(f"  BLEU Specificity: {bleu_specificity:.4f}, ROUGE Specificity: {rouge_specificity:.4f}")
            
            # Check for concept-specificity (following ConceptVectors criteria)
            specificity_threshold = 0.1  # Adjustable threshold
            is_concept_specific = (bleu_specificity > specificity_threshold or 
                                 rouge_specificity > specificity_threshold)
            
            results['is_concept_specific'] = is_concept_specific
            results['specificity_threshold'] = specificity_threshold
            
            if is_concept_specific:
                print(f"  ✓ Concept vector shows specificity (threshold: {specificity_threshold})")
            else:
                print(f"  ✗ Concept vector does not show clear specificity")
            
            return results
            
        finally:
            # Always restore original parameters
            self.restore_original_params()
    
    def batch_validate_concepts(
        self, 
        concept_list: List[Dict], 
        concept_vectors: Dict,
        noise_scales: List[float] = [0.1, 0.2, 0.3],
        output_file: str = None
    ) -> List[Dict]:
        """
        Validate multiple concepts with different noise scales
        
        Args:
            concept_list: List of concept dictionaries with QA pairs
            concept_vectors: Dictionary mapping concept names to (layer, dimension)
            noise_scales: List of noise scales to test
            output_file: Optional file to save results
            
        Returns:
            List of validation results
        """
        all_results = []
        
        for noise_scale in noise_scales:
            print(f"\n{'='*50}")
            print(f"Testing with noise scale σ = {noise_scale}")
            print(f"{'='*50}")
            
            scale_results = []
            
            for idx, concept_data in enumerate(concept_list):
                concept_name = concept_data.get('concept', f'concept_{idx}')  # Changed from 'Concept' to 'concept'
                
                # Find concept vector location
                if concept_name in concept_vectors:
                    location = concept_vectors[concept_name]
                else:
                    print(f"Warning: No vector location found for concept {concept_name}")
                    continue
                
                # Validate this concept
                try:
                    result = self.validate_concept_vector(
                        concept_data, 
                        concept_list, 
                        idx, 
                        location, 
                        noise_scale
                    )
                    
                    if result:
                        scale_results.append(result)
                        
                except Exception as e:
                    print(f"Error validating concept {concept_name}: {str(e)}")
                    continue
            
            all_results.extend(scale_results)
            
            # Summary for this noise scale
            concept_specific_count = sum(1 for r in scale_results if r.get('is_concept_specific', False))
            print(f"\nSummary for σ = {noise_scale}:")
            print(f"  Concepts tested: {len(scale_results)}")
            print(f"  Concept-specific vectors: {concept_specific_count}")
            print(f"  Specificity rate: {concept_specific_count/len(scale_results)*100:.1f}%" if scale_results else "N/A")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return all_results


def load_concept_vectors_from_test_results(results_file: str) -> Dict[str, Tuple[int, int]]:
    """
    Load concept vector locations from the selective test results
    
    Args:
        results_file: Path to selective_test_results.json
        
    Returns:
        Dictionary mapping concept names to (layer, dimension) tuples
    """
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    concept_vectors = {}
    
    # Extract from concept analyses - use the top vector for each concept
    if 'concept_analyses' in results_data:
        for concept_name, analysis in results_data['concept_analyses'].items():
            if 'top_candidates' in analysis and analysis['top_candidates']:
                # Get the top vector (first in the list)
                top_vector = analysis['top_candidates'][0]
                vector_key = top_vector.get('vector_key')
                
                if vector_key:
                    # Parse vector key like "L22_C4787" -> layer=22, dimension=4787
                    try:
                        parts = vector_key.split('_')
                        layer = int(parts[0][1:])  # Remove 'L' prefix
                        dimension = int(parts[1][1:])  # Remove 'C' prefix
                        concept_vectors[concept_name] = (layer, dimension)
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse vector key {vector_key} for concept {concept_name}")
    
    return concept_vectors


def main():
    """Main function to run concept vector validation"""
    
    # Additional compatibility settings for older GPUs
    try:
        torch._dynamo.config.suppress_errors = True
        torch.backends.cuda.enable_flash_sdp(False)
    except:
        pass  # Ignore if these settings are not available
    
    # Download required NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
    except:
        print("Warning: Could not download NLTK data")
    
    # Paths
    base_path = Path("/media/hdd/usr/martinelli/concept-vectors-gemma3")
    qa_file = base_path / "code/concept-val/qa.json"
    summary_file = base_path / "code/projection/test_results/selective_test_results.json"  # Use existing results instead
    output_file = base_path / "code/concept_val_test/advanced_validation_results.json"
    
    # Load QA data
    print("Loading QA data...")
    with open(qa_file, 'r') as f:
        concept_list = json.load(f)
    
    print(f"Loaded {len(concept_list)} concepts")
    
    # Load concept vector locations
    print("Loading concept vector locations...")
    concept_vectors = load_concept_vectors_from_test_results(summary_file)
    print(f"Loaded locations for {len(concept_vectors)} concepts")
    
    # Filter concepts to only those that have both QA data and test results
    available_concept_names = set(concept_vectors.keys())
    qa_concept_names = set(item.get('concept') for item in concept_list)
    
    # Find intersection
    valid_concepts = available_concept_names & qa_concept_names
    print(f"Concepts with both QA data and test results: {valid_concepts}")
    
    if not valid_concepts:
        print("❌ No concepts found with both QA data and test results!")
        print(f"QA concepts: {qa_concept_names}")
        print(f"Test result concepts: {available_concept_names}")
        return
    
    # Filter concept_list to only valid concepts
    filtered_concept_list = [item for item in concept_list if item.get('concept') in valid_concepts]
    filtered_concept_vectors = {k: v for k, v in concept_vectors.items() if k in valid_concepts}
    
    print(f"Running validation on {len(filtered_concept_list)} concepts: {list(valid_concepts)}")
    
    # Initialize validator
    print("Initializing validator...")
    validator = ConceptVectorValidator(
        model_name="google/gemma-3-1b-it",
        device="cuda"
    )
    
    # Run validation on filtered concepts (remove test subset limit)
    print(f"Running validation on {len(filtered_concept_list)} concepts")
    
    # Run validation
    results = validator.batch_validate_concepts(
        concept_list=filtered_concept_list,
        concept_vectors=filtered_concept_vectors,
        noise_scales=[0.1, 0.2, 0.3],
        output_file=str(output_file)
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    specific_count = sum(1 for r in results if r.get('is_concept_specific', False))
    
    print(f"Total validation tests: {total_tests}")
    print(f"Concept-specific vectors found: {specific_count}")
    print(f"Overall specificity rate: {specific_count/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
    
    if specific_count > 0:
        print(f"\n✓ Found {specific_count} concept vectors showing clear specificity!")
        print("These vectors demonstrate causal concept representation.")
    else:
        print(f"\n✗ No concept vectors showed clear specificity.")
        print("Consider:")
        print("  - Different noise scales")
        print("  - Different vector selection criteria")
        print("  - Alternative validation methodologies")


if __name__ == "__main__":
    main()
