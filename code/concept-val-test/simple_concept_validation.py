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

# HuggingFace setup - force private cache locations so this script always uses
# the user's private cache regardless of external environment variables.
# This must be set before importing transformers/huggingface so the
# libraries pick up the correct cache paths at import time.
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME

HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
import logging as py_logging
import warnings
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

# Ablation Configuration
ABLATION = True  # When True, ablate vectors; when False, add noise
# Value to set ablated vectors to. This can be 0.0 (zeroing) or a sentinel
# out-of-distribution value (e.g. -100.0). Verification logic below compares
# the modified column against this configured value rather than assuming zero.
ABLATION_VALUE = 0.0
NOISE_SCALE = 0.3  # Standard deviation for Gaussian noise (when not ablating)

# Validation Configuration  
MAX_VECTORS_PER_CONCEPT = 5  # Maximum number of vectors to test per concept
SPECIFICITY_THRESHOLD = 0.1  # Threshold for considering a result "concept-specific"

# Generation Configuration
MAX_NEW_TOKENS = 64  # Maximum tokens to generate per answer

# Disable PyTorch optimizations that require newer CUDA capabilities
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_AUTOGRAD_CACHE'] = '1'
# Suppress transformers warnings about generation flags
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# Also set HuggingFace logging to error and silence known generation-flag warnings
hf_logging.set_verbosity_error()
py_logging.getLogger("transformers.generation").setLevel(py_logging.ERROR)
warnings.filterwarnings("ignore", message="The following generation flags are not valid and may be ignored")
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
            device_map=None,  # Ensure single-device loading (no sharding)
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid compilation issues
            # Explicitly disable quantization
            load_in_8bit=False,
            load_in_4bit=False,
            quantization_config=None,
            token=HF_TOKEN
        )
        
        # Move entire model to specified device and set to eval mode
        self.model.to(device)
        self.model.eval()  # Ensure dropout is off for consistent results
        print(f"✅ Model loaded on {device} in eval mode")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Store original parameters for restoration (only backup once)
        self.original_params = copy.deepcopy(self.model.state_dict())
        
        # Run validation test
        self._test_direct_parameter_modification()
        
    def _debug_parameter_paths(self):
        """Debug: List all down_proj parameters to verify correct paths"""
        print("🔍 Debugging parameter paths...")
        down_proj_params = []
        for name, param in self.model.named_parameters():
            if 'down_proj' in name:
                down_proj_params.append((name, param.shape, param.device, param.data_ptr()))
                print(f"  {name}: shape={param.shape}, device={param.device}, data_ptr={param.data_ptr()}")
        
        if not down_proj_params:
            print("  ❌ No down_proj parameters found!")
        else:
            print(f"  ✅ Found {len(down_proj_params)} down_proj parameters")
        print()
        
    def _test_direct_parameter_modification(self):
        """Test that direct parameter modification works properly using the same approach as validation"""
        print("🧪 Testing direct parameter modification...")
        
        # First debug parameter paths
        self._debug_parameter_paths()
        
        # Test question
        test_question = "What is the plot of Star Wars? Give me few lines."
        
        # Generate baseline answer using the same method as validation
        print("  Generating baseline answer...")
        baseline_answer = self.generate_answers([test_question], context="baseline test")[0]
        print(f"  Baseline: {baseline_answer}")
        
        # Test parameter modification using the same approach as validation
        test_layer = 6
        print(f"  Ablating entire layer {test_layer} MLP down projection using validation method...")
        
        # Get the weight tensor using the same approach as inject_noise_to_vector
        try:
            param = self.model.model.layers[test_layer].mlp.down_proj.weight
            print(f"  Original weight shape: {param.shape}")
            
            # COMPREHENSIVE DEBUGGING - Check if we're editing the live parameter
            print("🔬 Parameter identity checks:")
            print(f"  param device: {param.device}")
            print(f"  param id/data_ptr before: {id(param)}, {param.data_ptr()}, requires_grad={param.requires_grad}")
            print(f"  param norm before: {param.norm().item():.6f}")
            print(f"  slice before: {param.flatten()[:6].tolist()}")
            
            # Store original for restoration (same pattern as inject_noise_to_vector)
            with torch.no_grad():
                # Ablate the entire matrix (all dimensions) to 0.0 - same as ABLATION_VALUE
                param.fill_(ABLATION_VALUE)
                print(f"  ✂️  Ablated entire down_proj matrix (set to {ABLATION_VALUE})")
            
            # Check parameter after modification
            print("🔬 Parameter checks after modification:")
            print(f"  slice after: {param.flatten()[:6].tolist()}")
            print(f"  param norm after: {param.norm().item():.6f}")
            print(f"  param id/data_ptr after: {id(param)}, {param.data_ptr()}")

            # Verify ablation against configured ABLATION_VALUE
            if ABLATION:
                target_full = torch.full_like(param, fill_value=ABLATION_VALUE)
                if torch.allclose(param, target_full, atol=1e-6, rtol=1e-4):
                    print(f"  ✅ Layer {test_layer} down_proj successfully set to {ABLATION_VALUE}")
                else:
                    maxdiff = (param - target_full).abs().max().item()
                    print(f"  ❌ Ablation mismatch for layer {test_layer} - max abs diff={maxdiff:.6e} (expected={ABLATION_VALUE})")
            
            
            
            # Clear cache if available (same as inject_noise_to_vector)
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
                
        except (IndexError, AttributeError) as e:
            print(f"  ❌ Error accessing layer {test_layer}: {e}")
            return
        
        # Generate modified answer using the same method as validation
        print("  Generating modified answer...")
        modified_answer = self.generate_answers([test_question], context="modified test")[0]
        print(f"  Modified: {modified_answer}")
        
        # Restore original parameters using the same method as validation
        print("  Restoring original parameters...")
        self.restore_original_params()
        
        # Verify restoration worked
        restored_param = self.model.model.layers[test_layer].mlp.down_proj.weight
        print(f"  Restored param norm: {restored_param.norm().item():.6f}")
        
        # Check if answers are different
        if baseline_answer != modified_answer:
            print("  ✅ Direct parameter modification working correctly!")
        else:
            print("  ⚠️  Warning: Answers identical - this suggests model robustness or modification not taking effect")
            print("      This could indicate:")
            print("      - Model has redundant pathways for this knowledge")
            print("      - Other layers compensate for the ablated layer")
            print("      - Knowledge is distributed across multiple layers")
        print()
        
    def restore_original_params(self):
        """Restore original model parameters"""
        self.model.load_state_dict(self.original_params)
            
    def inject_noise_to_vector(self, layer: int, dimension: int, noise_scale: float = NOISE_SCALE):
        """
        Inject Gaussian noise to specific concept vector location using direct module access
        Following Gemma 3B validation methodology
        
        Args:
            layer: Layer number (L)
            dimension: Dimension in the layer (C - concept dimension)  
            noise_scale: Standard deviation of Gaussian noise
        """
        print(f"Modifying layer {layer}, dimension {dimension}")
        
        # Access the parameter directly through the module hierarchy
        # In Gemma's architecture: model.model.layers[L].mlp.down_proj.weight
        try:
            weight = self.model.model.layers[layer].mlp.down_proj.weight  # shape [hidden, hidden]
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Cannot access layer {layer} down_proj weight: {e}")
        
        # Validate dimension bounds
        if dimension < 0 or dimension >= weight.shape[1]:
            raise IndexError(f"Dimension {dimension} out of range for tensor shape {weight.shape}")
        
        # Direct weight modification and verification
        with torch.no_grad():
            # Store original column for verification
            orig_col = weight[:, dimension].clone()
            orig_norm = orig_col.norm().item()
            
            print(f"  🔬 Column {dimension} before: norm={orig_norm:.6f}")
            print(f"      Sample values: {orig_col[:3].tolist()}")
            
            if ABLATION:
                # Ablate: set column C to the configured ablation value
                # (may be 0.0 for zeroing or an extreme sentinel like -100.0)
                weight[:, dimension] = ABLATION_VALUE
                print(f"  ✂️  Ablated column {dimension} (set to {ABLATION_VALUE})")
            else:
                # Inject noise: add Gaussian noise to column C
                noise = torch.normal(0, noise_scale, size=(weight.shape[0],), device=weight.device)
                weight[:, dimension] += noise
                print(f"  🎲 Added noise σ={noise_scale} to column {dimension}")
            
            # Verify the modification took effect
            new_col = weight[:, dimension]
            new_norm = new_col.norm().item()
            print(f"  🔬 Column {dimension} after: norm={new_norm:.6f}")
            print(f"     - Sample values: {new_col[:5].tolist()}")
            
        
        # Clear any cached computations to force fresh forward pass
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
        
    def generate_answers(self, questions: List[str], max_new_tokens: int = MAX_NEW_TOKENS, context: str = "") -> List[str]:
        """
        Generate answers for a list of questions using proper Gemma 3 chat format
        
        Args:
            questions: List of questions to answerq
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
            # Ensure inputs are a plain dict so we can sanitize generation kwargs
            try:
                inputs = dict(inputs)
            except Exception:
                pass

            # Remove unsupported generation flags that some tokenizers may include
            for _bad in ('top_p', 'top_k', 'cache_implementation'):
                if _bad in inputs:
                    inputs.pop(_bad, None)

            try:
                inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            except Exception:
                # Fall back: try BatchEncoding .to() if available
                try:
                    inputs = inputs.to(model_device)
                except Exception:
                    pass
            
            # Generate response using Gemma 3B methodology
            with torch.no_grad():
                # Greedy response generation with proper parameters
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Pure greedy decoding (no sampling)
                    pad_token_id=self.tokenizer.eos_token_id,  # Proper termination
                    use_cache=False,  # Force fresh forward pass (no stale cache)
                    output_scores=True,  # Get raw logits for analysis
                    return_dict_in_generate=True  # Return structured output
                )

            # Decode only the newly generated tokens and append
            new_tokens = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            answers.append(answer)
            
            # Print each question and its generated answer for debugging
            if "perturbed" in context.lower():
                print(f"      Q{i}: {question[:80]}{'...' if len(question) > 80 else ''}")
                print(f"      A{i}: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        
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
