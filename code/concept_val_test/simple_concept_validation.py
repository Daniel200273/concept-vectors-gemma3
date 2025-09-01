"""
Simple Concept Vector Validation
Using existing test results from selective_test_results.json
"""

import torch
import numpy as np
import json
import random
import statistics
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import copy
from typing import List, Dict, Tuple, Any

# Set seeds for reproducibility
random.seed(999)
np.random.seed(999)
torch.manual_seed(999)

class SimpleConceptValidator:
    def __init__(self, model_name: str = "google/gemma-1.1-1b", device: str = "cuda"):
        """
        Initialize the simple concept validator
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Store original parameters for restoration
        self.original_params = None
        
    def backup_original_params(self):
        """Backup original model parameters"""
        self.original_params = copy.deepcopy(self.model.state_dict())
        
    def restore_original_params(self):
        """Restore original model parameters"""
        if self.original_params is not None:
            self.model.load_state_dict(self.original_params)
            
    def inject_noise_to_vector(self, layer: int, dimension: int, noise_scale: float = 0.1):
        """
        Inject Gaussian noise to specific concept vector location
        
        Args:
            layer: Layer number
            dimension: Dimension in the layer
            noise_scale: Standard deviation of Gaussian noise
        """
        # For Gemma, use model.layers.{layer}.mlp.down_proj.weight
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
                pad_token_id=self.tokenizer.eos_token_id
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
            rouge = Rouge()
            scores = rouge.get_scores(candidate.strip(), reference.strip())
            return scores[0]['rouge-l']['f']
        except:
            return 0.0
    
    def validate_concept_vector_from_key(
        self, 
        vector_key: str,
        concept_qa: List[str],
        unrelated_qa: List[str],
        noise_scale: float = 0.2
    ) -> Dict[str, Any]:
        """
        Validate a concept vector using its vector key
        
        Args:
            vector_key: Vector key in format "L{layer}_C{dimension}"
            concept_qa: List of concept-specific questions
            unrelated_qa: List of unrelated questions
            noise_scale: Standard deviation of Gaussian noise
            
        Returns:
            Dictionary containing validation results
        """
        # Parse vector key to get layer and dimension
        try:
            parts = vector_key.split('_')
            layer = int(parts[0][1:])  # Remove 'L' prefix
            dimension = int(parts[1][1:])  # Remove 'C' prefix
        except:
            raise ValueError(f"Invalid vector key format: {vector_key}")
        
        print(f"Validating vector {vector_key}: Layer {layer}, Dimension {dimension}")
        print(f"Concept questions: {len(concept_qa)}")
        print(f"Unrelated questions: {len(unrelated_qa)}")
        
        # Backup original parameters
        self.backup_original_params()
        
        try:
            # Generate original answers (without noise)
            print("Generating original answers...")
            original_concept_answers = self.generate_answers(concept_qa)
            original_unrelated_answers = self.generate_answers(unrelated_qa)
            
            # Inject noise and generate perturbed answers
            print(f"Injecting noise (σ={noise_scale}) and generating perturbed answers...")
            self.inject_noise_to_vector(layer, dimension, noise_scale)
            perturbed_concept_answers = self.generate_answers(concept_qa)
            perturbed_unrelated_answers = self.generate_answers(unrelated_qa)
            
            # Calculate metrics
            concept_bleu_scores = []
            concept_rouge_scores = []
            unrelated_bleu_scores = []
            unrelated_rouge_scores = []
            
            # Calculate scores for concept-specific questions
            for perturbed, original in zip(perturbed_concept_answers, original_concept_answers):
                concept_bleu_scores.append(self.calculate_bleu(original, perturbed))
                concept_rouge_scores.append(self.calculate_rouge_l(original, perturbed))
            
            # Calculate scores for unrelated questions
            for perturbed, original in zip(perturbed_unrelated_answers, original_unrelated_answers):
                unrelated_bleu_scores.append(self.calculate_bleu(original, perturbed))
                unrelated_rouge_scores.append(self.calculate_rouge_l(original, perturbed))
            
            # Aggregate scores
            avg_concept_bleu = statistics.mean(concept_bleu_scores) if concept_bleu_scores else 0.0
            avg_concept_rouge = statistics.mean(concept_rouge_scores) if concept_rouge_scores else 0.0
            avg_unrelated_bleu = statistics.mean(unrelated_bleu_scores) if unrelated_bleu_scores else 0.0
            avg_unrelated_rouge = statistics.mean(unrelated_rouge_scores) if unrelated_rouge_scores else 0.0
            
            # Calculate degradation measures
            concept_bleu_degradation = 1.0 - avg_concept_bleu
            concept_rouge_degradation = 1.0 - avg_concept_rouge
            unrelated_bleu_degradation = 1.0 - avg_unrelated_bleu
            unrelated_rouge_degradation = 1.0 - avg_unrelated_rouge
            
            # Concept-specificity indicators (positive means more degradation on concept vs unrelated)
            bleu_specificity = concept_bleu_degradation - unrelated_bleu_degradation
            rouge_specificity = concept_rouge_degradation - unrelated_rouge_degradation
            
            results = {
                'vector_key': vector_key,
                'layer': layer,
                'dimension': dimension,
                'noise_scale': noise_scale,
                'n_concept_questions': len(concept_qa),
                'n_unrelated_questions': len(unrelated_qa),
                
                # Concept scores
                'concept_bleu_score': avg_concept_bleu,
                'concept_rouge_score': avg_concept_rouge,
                'concept_bleu_degradation': concept_bleu_degradation,
                'concept_rouge_degradation': concept_rouge_degradation,
                
                # Unrelated scores
                'unrelated_bleu_score': avg_unrelated_bleu,
                'unrelated_rouge_score': avg_unrelated_rouge,
                'unrelated_bleu_degradation': unrelated_bleu_degradation,
                'unrelated_rouge_degradation': unrelated_rouge_degradation,
                
                # Concept-specificity measures
                'bleu_specificity': bleu_specificity,
                'rouge_specificity': rouge_specificity,
                
                # Individual scores for analysis
                'individual_concept_bleu': concept_bleu_scores,
                'individual_concept_rouge': concept_rouge_scores,
                'individual_unrelated_bleu': unrelated_bleu_scores,
                'individual_unrelated_rouge': unrelated_rouge_scores,
                
                # Sample answers for inspection
                'sample_original_concept': original_concept_answers[:2],
                'sample_perturbed_concept': perturbed_concept_answers[:2],
                'sample_original_unrelated': original_unrelated_answers[:2],
                'sample_perturbed_unrelated': perturbed_unrelated_answers[:2]
            }
            
            # Print validation results
            print(f"Results for {vector_key}:")
            print(f"  Concept BLEU: {avg_concept_bleu:.4f}, Concept ROUGE-L: {avg_concept_rouge:.4f}")
            print(f"  Unrelated BLEU: {avg_unrelated_bleu:.4f}, Unrelated ROUGE-L: {avg_unrelated_rouge:.4f}")
            print(f"  BLEU Specificity: {bleu_specificity:.4f}, ROUGE Specificity: {rouge_specificity:.4f}")
            
            # Check for concept-specificity
            specificity_threshold = 0.1  # Adjustable threshold
            is_concept_specific = (bleu_specificity > specificity_threshold or 
                                 rouge_specificity > specificity_threshold)
            
            results['is_concept_specific'] = is_concept_specific
            results['specificity_threshold'] = specificity_threshold
            
            if is_concept_specific:
                print(f"  ✓ Vector shows concept specificity (threshold: {specificity_threshold})")
            else:
                print(f"  ✗ Vector does not show clear concept specificity")
            
            return results
            
        finally:
            # Always restore original parameters
            self.restore_original_params()


def load_concept_vectors_from_results(results_file: str) -> Dict[str, Dict]:
    """
    Load concept vectors from the selective test results
    
    Args:
        results_file: Path to selective_test_results.json
        
    Returns:
        Dictionary mapping concept names to their top vectors
    """
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    concept_vectors = {}
    
    if 'concept_analyses' in results_data:
        for concept_name, analysis in results_data['concept_analyses'].items():
            if 'top_candidates' in analysis:
                concept_vectors[concept_name] = analysis['top_candidates']
    
    return concept_vectors


def main():
    """Main function to run simple concept vector validation"""
    
    # Paths
    base_path = Path("/media/hdd/usr/martinelli/concept-vectors-gemma3")
    qa_file = base_path / "code/concept-val/qa.json"
    results_file = base_path / "code/projection/test_results/selective_test_results.json"
    output_file = base_path / "code/concept_val_test/simple_validation_results.json"
    
    # Load QA data
    print("Loading QA data...")
    with open(qa_file, 'r') as f:
        qa_data = json.load(f)
    
    # Create concept name to QA mapping
    concept_qa_map = {}
    for item in qa_data:
        concept_name = item.get('concept')  # Changed from 'Concept' to 'concept'
        if concept_name:
            qa_pairs = item.get('qa', [])  # Changed from 'QA' to 'qa'
            # Extract just the questions from qa pairs
            questions = [pair.get('q', '') for pair in qa_pairs if pair.get('q')]
            concept_qa_map[concept_name] = questions
    
    print(f"Loaded QA data for {len(concept_qa_map)} concepts")
    
    # Load concept vector results
    print("Loading concept vector results...")
    concept_vectors = load_concept_vectors_from_results(results_file)
    print(f"Loaded vector results for {len(concept_vectors)} concepts")
    
    # Initialize validator
    print("Initializing validator...")
    validator = SimpleConceptValidator(
        model_name="google/gemma-1.1-1b",
        device="cuda"
    )
    
    # Validation parameters
    noise_scales = [0.1, 0.2, 0.3]
    max_vectors_per_concept = 3  # Test top 3 vectors per concept
    
    all_results = []
    
    for noise_scale in noise_scales:
        print(f"\n{'='*50}")
        print(f"Testing with noise scale σ = {noise_scale}")
        print(f"{'='*50}")
        
        scale_results = []
        
        for concept_name, vectors in concept_vectors.items():
            if concept_name not in concept_qa_map:
                print(f"Warning: No QA data found for concept {concept_name}")
                continue
            
            concept_qa = concept_qa_map[concept_name]
            if not concept_qa:
                print(f"Warning: Empty QA list for concept {concept_name}")
                continue
            
            # Get unrelated questions (from other concepts)
            unrelated_qa = []
            for other_concept, other_qa in concept_qa_map.items():
                if other_concept != concept_name and other_qa:
                    unrelated_qa.extend(other_qa)
            
            # Limit unrelated questions
            if len(unrelated_qa) > len(concept_qa):
                unrelated_qa = random.sample(unrelated_qa, len(concept_qa))
            
            print(f"\nTesting concept: {concept_name}")
            print(f"Concept questions: {len(concept_qa)}")
            print(f"Unrelated questions: {len(unrelated_qa)}")
            
            # Test top vectors for this concept
            for i, vector_info in enumerate(vectors[:max_vectors_per_concept]):
                vector_key = vector_info.get('vector_key')
                if not vector_key:
                    continue
                
                print(f"\n  Testing vector {i+1}/{max_vectors_per_concept}: {vector_key}")
                
                try:
                    result = validator.validate_concept_vector_from_key(
                        vector_key=vector_key,
                        concept_qa=concept_qa,
                        unrelated_qa=unrelated_qa,
                        noise_scale=noise_scale
                    )
                    
                    result['concept_name'] = concept_name
                    result['vector_rank'] = i + 1
                    result['vector_info'] = vector_info
                    
                    scale_results.append(result)
                    
                except Exception as e:
                    print(f"    Error testing vector {vector_key}: {str(e)}")
                    continue
        
        all_results.extend(scale_results)
        
        # Summary for this noise scale
        specific_count = sum(1 for r in scale_results if r.get('is_concept_specific', False))
        print(f"\nSummary for σ = {noise_scale}:")
        print(f"  Vectors tested: {len(scale_results)}")
        print(f"  Concept-specific vectors: {specific_count}")
        print(f"  Specificity rate: {specific_count/len(scale_results)*100:.1f}%" if scale_results else "N/A")
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(all_results)
    specific_count = sum(1 for r in all_results if r.get('is_concept_specific', False))
    
    print(f"Total validation tests: {total_tests}")
    print(f"Concept-specific vectors found: {specific_count}")
    print(f"Overall specificity rate: {specific_count/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
    
    # Analysis by concept
    concept_analysis = {}
    for result in all_results:
        concept = result.get('concept_name')
        if concept not in concept_analysis:
            concept_analysis[concept] = {'total': 0, 'specific': 0}
        concept_analysis[concept]['total'] += 1
        if result.get('is_concept_specific', False):
            concept_analysis[concept]['specific'] += 1
    
    print(f"\nPer-concept analysis:")
    for concept, stats in concept_analysis.items():
        rate = stats['specific'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {concept}: {stats['specific']}/{stats['total']} ({rate:.1f}%)")
    
    if specific_count > 0:
        print(f"\n✓ Found {specific_count} concept vectors showing clear specificity!")
        print("These vectors demonstrate causal concept representation.")
        
        # Show best examples
        specific_results = [r for r in all_results if r.get('is_concept_specific', False)]
        specific_results.sort(key=lambda x: max(x.get('bleu_specificity', 0), x.get('rouge_specificity', 0)), reverse=True)
        
        print(f"\nTop 5 most specific vectors:")
        for i, result in enumerate(specific_results[:5]):
            bleu_spec = result.get('bleu_specificity', 0)
            rouge_spec = result.get('rouge_specificity', 0)
            print(f"  {i+1}. {result.get('concept_name')} - {result.get('vector_key')}")
            print(f"     BLEU specificity: {bleu_spec:.4f}, ROUGE specificity: {rouge_spec:.4f}")
    else:
        print(f"\n✗ No concept vectors showed clear specificity.")
        print("Consider:")
        print("  - Different noise scales")
        print("  - Different vector selection criteria")
        print("  - Alternative validation methodologies")
        print("  - Lower specificity threshold")


if __name__ == "__main__":
    main()
