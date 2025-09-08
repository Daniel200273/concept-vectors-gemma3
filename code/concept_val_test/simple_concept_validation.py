"""
Simple Concept Vector Validation
Using existing test results from selective_test_results.json
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

# ========================
# GLOBAL CONFIGURATION
# ========================

# Model and HuggingFace Configuration
MODEL_ID = os.environ.get("GEMMA_MODEL", "google/gemma-3-1b-it")
DEVICE = "cuda:1"

# HuggingFace setup
os.environ.setdefault("HF_HOME", "/media/hdd/usr/martinelli/.cache/huggingface")
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Ablation Configuration
ABLATION = True  # When True, ablate vectors; when False, add noise
ABLATION_VALUE = -100.0  # Value to set ablated vectors to (should be far from normal range)
NOISE_SCALE = 0.3  # Standard deviation for Gaussian noise (when not ablating)

# Validation Configuration  
MAX_VECTORS_PER_CONCEPT = 1  # Maximum number of vectors to test per concept
SPECIFICITY_THRESHOLD = 0.1  # Threshold for considering a result "concept-specific"

# Generation Configuration
MAX_NEW_TOKENS = 100  # Maximum tokens to generate per answer

# Disable PyTorch optimizations that require newer CUDA capabilities
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_AUTOGRAD_CACHE'] = '1'
torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention
torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable memory efficient attention

# Set seeds for reproducibility
random.seed(999)
np.random.seed(999)
torch.manual_seed(999)

class SimpleConceptValidator:
    def __init__(self, model_name: str = MODEL_ID, device: str = DEVICE):
        """
        Initialize the simple concept validator
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda:1' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        print(f"Loading model: {model_name} on device: {device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use full precision to avoid quantization
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid compilation issues
            # Explicitly disable quantization
            load_in_8bit=False,
            load_in_4bit=False,
            quantization_config=None,
            token=HF_TOKEN
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Store original parameters for restoration (only backup once)
        self.original_params = copy.deepcopy(self.model.state_dict())
        
    def restore_original_params(self):
        """Restore original model parameters"""
        self.model.load_state_dict(self.original_params)
            
    def inject_noise_to_vector(self, layer: int, dimension: int, noise_scale: float = NOISE_SCALE):
        """
        Inject Gaussian noise to specific concept vector location
        
        Args:
            layer: Layer number
            dimension: Dimension in the layer
            noise_scale: Standard deviation of Gaussian noise
        """
        # For Gemma, use model.layers.{layer}.mlp.down_proj.weight
        param_name = f'model.layers.{layer}.mlp.down_proj.weight'
        
    print(f"Attempting to modify parameter: {param_name} (layer {layer}, dim {dimension})")
        
        def _modify_tensor(tensor, tensor_name, use_state_dict=False):
            """Helper function to modify tensor and print debug info"""
            if dimension < 0 or dimension >= tensor.shape[1]:
                raise IndexError(f"Dimension index {dimension} out of range for tensor with shape {tensor.shape}")

            # Store original values for verification
            original_values = tensor[:, dimension].clone()
            orig_min = original_values.min().item()
            orig_max = original_values.max().item()
            orig_mean = float(original_values.mean().item())
            orig_std = float(original_values.std().item())

            with torch.no_grad():
                if ABLATION:
                    tensor[:, dimension] = ABLATION_VALUE
                else:
                    hidden_size = self.model.config.hidden_size
                    noise = torch.normal(0, noise_scale, size=(hidden_size,)).to(tensor.device if not use_state_dict else self.device)
                    tensor[:, dimension] += noise

            # Compute new stats for quick verification
            new_values = tensor[:, dimension]
            new_min = new_values.min().item()
            new_max = new_values.max().item()
            new_mean = float(new_values.mean().item())
            new_std = float(new_values.std().item())

            action = 'ablation' if ABLATION else f'noise(σ={noise_scale})'
            print(f"  -> Performed {action} on {tensor_name}[ :, {dimension} ]")
            print(f"     before: min={orig_min:.6f}, max={orig_max:.6f}, mean={orig_mean:.6f}, std={orig_std:.6f}")
            print(f"     after : min={new_min:.6f}, max={new_max:.6f}, mean={new_mean:.6f}, std={new_std:.6f}")
            
            return original_values
        
        # Try to find the actual parameter object via named_parameters() for in-place edit
        found = False
        for name, param in self.model.named_parameters(recurse=True):
            if name == param_name:
                found = True
                # Sanity check shape and device
                shape = tuple(param.data.shape)
                dev = param.device
                print(f"Found parameter '{name}' shape={shape} device={dev}; modifying in-place")

                original_values = _modify_tensor(param.data, name, use_state_dict=False)
                break

        if found:
            # Clear any cached computations
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
            return

    print(f"WARNING: Parameter {param_name} not found via named_parameters(); falling back to state_dict")
        # Fallback: operate on state_dict and reload (slower but robust)
        state = self.model.state_dict()
        if param_name not in state:
            raise KeyError(f"Parameter not found in model state: {param_name}")

        # Use helper function for state_dict path too
        tensor = state[param_name]
    original_values = _modify_tensor(tensor, param_name, use_state_dict=True)

    # Reload the model with modified state
    self.model.load_state_dict(state)
        
        # Clear any cached computations
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
        
    def generate_answers(self, questions: List[str], max_new_tokens: int = MAX_NEW_TOKENS, context: str = "") -> List[str]:
        """
        Generate answers for a list of questions using proper Gemma 3 chat format
        
        Args:
            questions: List of questions to answer
            max_new_tokens: Maximum number of new tokens to generate
            context: Context string for progress display (e.g., "baseline concept", "perturbed unrelated")
            
        Returns:
            List of generated answers
        """
        answers = []
        total_questions = len(questions)
        
    print(f"    Generating {context} answers ({total_questions})...")
        
        # Process each question individually using proper chat template
    for i, question in enumerate(questions, 1):
            
            # Create proper message format for Gemma 3
            messages = [
                {
                    "role": "user", 
                    "content": question
                }
            ]
            
            # Apply chat template and move tensors to the model's device (use same pattern as direct load snippet)
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            )

            # Move inputs to the device where the model parameters live
            model_device = next(self.model.parameters()).device
            try:
                inputs = inputs.to(model_device)
            except Exception:
                # Fall back to manual move if BatchEncoding doesn't support .to()
                inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    output_scores=False,
                    return_dict_in_generate=False
                )

            # Decode only the newly generated tokens and append
            new_tokens = generation_output[0][inputs["input_ids"].shape[-1]:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            answers.append(answer)
        
    print(f"    Completed {context} generation ({total_questions} answers)")
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
            if not reference.strip() or not candidate.strip():
                return 0.0
                
            scores = self.rouge_scorer.score(reference.strip(), candidate.strip())
            return scores['rougeL'].fmeasure
        except:
            return 0.0
    
    def validate_concept_vector_from_key(
        self, 
        vector_key: str,
        concept_qa: Dict[str, List[str]],
        unrelated_qa: Dict[str, List[str]],
        noise_scale: float = NOISE_SCALE
    ) -> Dict[str, Any]:
        """
        Validate a concept vector using its vector key
        
        Args:
            vector_key: Vector key in format "L{layer}_C{dimension}"
            concept_qa: Dict with 'questions' and 'answers' lists for concept-specific Q&A
            unrelated_qa: Dict with 'questions' and 'answers' lists for unrelated Q&A
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
        
        concept_questions = concept_qa['questions']
        concept_baseline_answers = concept_qa['answers']
        unrelated_questions = unrelated_qa['questions']
        unrelated_baseline_answers = unrelated_qa['answers']
        
    print(f"Validating {vector_key} (L{layer},C{dimension}) — concept_q={len(concept_questions)}, unrelated_q={len(unrelated_questions)}")
        
        try:
            # Inject noise/ablation and generate perturbed answers
            if ABLATION:
                print(f"  Applying ablation (value={ABLATION_VALUE})")
                self.inject_noise_to_vector(layer, dimension, 0.0)  # noise_scale irrelevant for ablation
            else:
                print(f"  Applying noise σ={noise_scale}")
                self.inject_noise_to_vector(layer, dimension, noise_scale)
            
            # Generate perturbed answers with context labels
            perturbed_concept_answers = self.generate_answers(
                concept_questions, 
                context="perturbed concept"
            )
            perturbed_unrelated_answers = self.generate_answers(
                unrelated_questions, 
                context="perturbed unrelated"
            )
            
            # Restore model immediately after perturbation test
            self.restore_original_params()

            # Calculate metrics (compute quietly, print summary only)
            concept_bleu_scores = []
            concept_rouge_scores = []
            unrelated_bleu_scores = []
            unrelated_rouge_scores = []

            for perturbed, baseline in zip(perturbed_concept_answers, concept_baseline_answers):
                concept_bleu_scores.append(self.calculate_bleu(baseline, perturbed))
                concept_rouge_scores.append(self.calculate_rouge_l(baseline, perturbed))

            for perturbed, baseline in zip(perturbed_unrelated_answers, unrelated_baseline_answers):
                unrelated_bleu_scores.append(self.calculate_bleu(baseline, perturbed))
                unrelated_rouge_scores.append(self.calculate_rouge_l(baseline, perturbed))
            
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
                'perturbation_type': 'ablation' if ABLATION else 'noise',
                'noise_scale': 0.0 if ABLATION else noise_scale,
                'n_concept_questions': len(concept_questions),
                'n_unrelated_questions': len(unrelated_questions),
                
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
                
                # Sample answers for inspection (baseline from qa.json, perturbed generated)
                'sample_baseline_concept': concept_baseline_answers[:2],
                'sample_perturbed_concept': perturbed_concept_answers[:2],
                'sample_baseline_unrelated': unrelated_baseline_answers[:2],
                'sample_perturbed_unrelated': perturbed_unrelated_answers[:2]
            }
            
            # Print validation results
            print(f"  Results — concept BLEU={avg_concept_bleu:.4f}, ROUGE-L={avg_concept_rouge:.4f}; unrelated BLEU={avg_unrelated_bleu:.4f}, ROUGE-L={avg_unrelated_rouge:.4f}")
            print(f"  Specificity — BLEU={bleu_specificity:.4f}, ROUGE={rouge_specificity:.4f}")
            
            # Check for concept-specificity
            specificity_threshold = SPECIFICITY_THRESHOLD  # Adjustable threshold
            is_concept_specific = (bleu_specificity > specificity_threshold or 
                                 rouge_specificity > specificity_threshold)
            
            results['is_concept_specific'] = is_concept_specific
            results['specificity_threshold'] = specificity_threshold
            
            print(f"  Concept-specific: {is_concept_specific} (threshold={specificity_threshold})")
            
            return results
            
        except Exception as e:
            # Always restore original parameters on error
            self.restore_original_params()
            raise e


def load_concept_vectors_from_results(results_file: str) -> Dict[str, Dict]:
    """
    Load concept vectors from the final concept vectors results
    
    Args:
        results_file: Path to final_concept_vectors.json
        
    Returns:
        Dictionary mapping concept names to their best candidate vectors
    """
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    concept_vectors = {}

    # Extract from final_concept_vectors.json format
    # Collect best_candidate and any alternative_candidates so we can test multiple ranks
    if 'concept_vectors' in results_data:
        for concept_name, concept_data in results_data['concept_vectors'].items():
            candidates = []
            if 'best_candidate' in concept_data and concept_data['best_candidate']:
                candidates.append(concept_data['best_candidate'])
            # alternative_candidates may be present as a list
            alts = concept_data.get('alternative_candidates', []) or []
            for alt in alts:
                # Ensure alt looks like a candidate dict before appending
                if isinstance(alt, dict):
                    candidates.append(alt)

            if candidates:
                concept_vectors[concept_name] = candidates
                print(f"Loaded {len(candidates)} candidate(s) for {concept_name}: {[c.get('vector_key','?') for c in candidates]}")
    
    print(f"Total concepts loaded: {len(concept_vectors)}")
    return concept_vectors


def main():
    """Main function to run simple concept vector validation"""
    
        # Paths
    base_path = Path("/media/hdd/usr/martinelli/concept-vectors-gemma3")
    qa_file = base_path / "code/concept_val_test/qa-generated.json"
    results_file = base_path / "code/projection/final_concept_vectors/final_concept_vectors.json"
    output_file = base_path / "code/concept_val_test/simple_validation_results.json"
    
    # Load QA data (use generated QA pairs)
    print(f"Loading QA data from: {qa_file}")
    with open(qa_file, 'r') as f:
        qa_data = json.load(f)
    
    # Create concept name to QA mapping
    concept_qa_map = {}
    for item in qa_data:
        concept_name = item.get('concept', item.get('Concept'))  # Try both 'concept' and 'Concept'
        if concept_name:
            qa_pairs = item.get('qa', item.get('QA', []))  # Try both 'qa' and 'QA'
            # Extract questions and answers from qa pairs
            questions = []
            answers = []
            for pair in qa_pairs:
                q = pair.get('q', pair.get('Q', ''))
                a = pair.get('a', pair.get('A', ''))
                if q and a:  # Only add if both question and answer exist
                    questions.append(q)
                    answers.append(a)
            if questions and answers:  # Only add if there are valid Q&A pairs
                concept_qa_map[concept_name] = {'questions': questions, 'answers': answers}
    
    print(f"Loaded QA data for {len(concept_qa_map)} concepts")
    
    # Load concept vector results
    print("Loading concept vector results...")
    concept_vectors = load_concept_vectors_from_results(results_file)
    print(f"Loaded vector results for {len(concept_vectors)} concepts")
    
    # Check for concept overlap
    qa_concepts = set(concept_qa_map.keys())
    vector_concepts = set(concept_vectors.keys())
    overlap = qa_concepts.intersection(vector_concepts)
    print(f"Concepts with both QA and vectors: {len(overlap)}")
    if not overlap:
        print("ERROR: No concepts have both QA data and vector data!")
        return
    
    # Validation parameters: use only the strongest noise level and test top N candidates
    noise_scale = NOISE_SCALE  # strongest noise to test
    max_vectors_per_concept = MAX_VECTORS_PER_CONCEPT  # test top N candidates per concept

    # Initialize validator and backup parameters once
    print("Initializing validator...")
    validator = SimpleConceptValidator(
        model_name=MODEL_ID,
        device=DEVICE
    )

    # The backup is already done in __init__ method

    all_results = []

    print(f"\n{'='*60}")
    if ABLATION:
        print(f"🎛️  Testing with ablation (zeroing), up to {max_vectors_per_concept} candidates per concept")
    else:
        print(f"🎛️  Testing with single noise scale σ = {noise_scale}, up to {max_vectors_per_concept} candidates per concept")
    print(f"{'='*60}\n")

    concept_count = 0
    total_concepts = len([c for c in concept_vectors.keys() if c in concept_qa_map])

    for concept_name, vectors in concept_vectors.items():
            if concept_name not in concept_qa_map:
                print(f"⚠️ Warning: No QA data found for concept {concept_name}")
                continue
                
            concept_count += 1
            print(f"\n{'🧪' * 3} CONCEPT {concept_count}/{total_concepts}: {concept_name.upper()} {'🧪' * 3}")
            
            concept_qa = concept_qa_map[concept_name]
            if not concept_qa['questions'] or not concept_qa['answers']:
                print(f"⚠️ Warning: Empty QA list for concept {concept_name}")
                continue
            
            # Get unrelated questions and answers (from other concepts)
            unrelated_questions = []
            unrelated_answers = []
            for other_concept, other_qa in concept_qa_map.items():
                if other_concept != concept_name and other_qa['questions'] and other_qa['answers']:
                    unrelated_questions.extend(other_qa['questions'])
                    unrelated_answers.extend(other_qa['answers'])
            
            # Limit unrelated questions to match concept questions length
            if len(unrelated_questions) > len(concept_qa['questions']):
                indices = random.sample(range(len(unrelated_questions)), len(concept_qa['questions']))
                unrelated_questions = [unrelated_questions[i] for i in indices]
                unrelated_answers = [unrelated_answers[i] for i in indices]
            
            unrelated_qa = {'questions': unrelated_questions, 'answers': unrelated_answers}
            
            print(f"\n🎯 Testing concept: {concept_name}")
            print(f"   📝 Concept questions: {len(concept_qa['questions'])}")
            print(f"   📝 Unrelated questions: {len(unrelated_questions)}")
            
            # Show sample baseline answers for this concept
            print(f"   Sample baseline (first 2 Q&A shown)")
            for i, (q, a) in enumerate(zip(concept_qa['questions'][:2], concept_qa['answers'][:2])):
                print(f"      Q{i+1}: {q[:60]}{'...' if len(q) > 60 else ''}")
                print(f"      A{i+1}: {a[:80]}{'...' if len(a) > 80 else ''}")
            
            # Test up to `max_vectors_per_concept` candidates (best first)
            for rank, candidate in enumerate(vectors[:max_vectors_per_concept], start=1):
                best_vector = candidate
                # Handle the vector_key field
                vector_key = best_vector.get('vector_key') or best_vector.get('key') or best_vector.get('id')
                if not vector_key:
                    print(f"    ⚠️ Warning: No vector_key found in vector_info for {concept_name} (rank {rank})")
                    continue

                print(f"\n  🧬 Testing candidate rank {rank}: {vector_key}")
                print(f"     📍 Location: Layer {best_vector.get('layer', '?')}, Neuron {best_vector.get('neuron', '?')}")
                print(f"     💪 Activation strength: {best_vector.get('concept_activation_strength', '?')}")

                try:
                    result = validator.validate_concept_vector_from_key(
                        vector_key=vector_key,
                        concept_qa=concept_qa,
                        unrelated_qa=unrelated_qa,
                        noise_scale=noise_scale
                    )

                    result['concept_name'] = concept_name
                    result['vector_rank'] = rank
                    result['vector_info'] = best_vector

                    all_results.append(result)

                except Exception as e:
                    print(f"    Error testing vector {vector_key}: {str(e)}")
                    continue
        
    # Single-run summary printed below
    
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
