"""
Ensemble Concept Vector Validation
Ablates all candidate vectors for a concept simultaneously to test ensemble effects
"""

import torch
import numpy as np
import json
import random
import statistics
import os
import warnings
from pathlib import Path

# Suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Ensure private Hugging Face cache is set before importing transformers
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME

# Respect an externally set HF_TOKEN and export it for HuggingFace libraries
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your HuggingFace token")
os.environ["HF_TOKEN"] = HF_TOKEN

from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import copy
from typing import List, Dict, Tuple, Any

# Optional semantic similarity (SBERT)
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:
    SentenceTransformer = None
    sbert_util = None

# ========================
# GLOBAL CONFIGURATION
# ========================

# Model and HuggingFace Configuration
MODEL_ID = os.environ.get("GEMMA_MODEL", "google/gemma-3-1b-it")
DEVICE = "cuda:1"

# Perturbation Configuration - Test all three types
GAUSSIAN_STD = 0.1     # Standard deviation for Gaussian noise perturbation
SCALE_VAL = -2        # Scale value for weight inversion (negative to invert)
ABLATION_VAL = -0.5       # Value to set ablated vectors to (zero ablation)

# Validation Configuration
UNRELATED_QUESTIONS_COUNT = 9  # Number of unrelated questions to use for comparison
SPECIFICITY_THRESHOLD = 0.2    # Threshold for considering a result "concept-specific"

# Generation Configuration
MAX_NEW_TOKENS = 64  # Maximum tokens to generate per answer

# Disable PyTorch optimizations that require newer CUDA capabilities
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_AUTOGRAD_CACHE'] = '1'
torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention
torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable memory efficient attention


class EnsembleConceptValidator:
    def __init__(self, model_name: str = MODEL_ID, device: str = DEVICE, force_download: bool = True):
        """
        Initialize the ensemble concept validator
        
        Args:
            model_name: HuggingFace model name (uses global MODEL_ID by default)
            device: Device to run on (uses global DEVICE by default)
        """
        self.device = device
        self.torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Force clean reload from HuggingFace to avoid corrupted cache
        self.force_download = force_download
        if self.force_download:
            print(f"Force reloading model from HuggingFace: {model_name} on device: {device}")
            print("  🔄 Clearing any cached model to ensure clean state...")
        else:
            print(f"Loading model from HuggingFace (no forced download): {model_name} on device: {device}")
        
        # Load model with force_download to bypass cache
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # Use native bfloat16 when available (allows hardware bfloat16 support); do not force full float32
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid compilation issues
            token=os.environ.get("HF_TOKEN") or None,
            # Explicitly disable quantization
            load_in_8bit=False,
            load_in_4bit=False,
            quantization_config=None,
            # Force download to ensure clean model
            force_download=self.force_download,
            local_files_only=False
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN") or None,
            # Force download for tokenizer too
            force_download=self.force_download,
            local_files_only=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Store original parameters for restoration (save CPU copy to avoid device issues)
        print("  💾 Backing up original model parameters to CPU...")
        self.original_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        print(f"  ✅ Model loaded successfully on {next(self.model.parameters()).device}")

        # Load SBERT for semantic similarity if available
        self.sbert = None
        if SentenceTransformer is not None:
            try:
                sbert_device = str(self.torch_device) if torch.cuda.is_available() else 'cpu'
                # use small, fast model by default
                self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=sbert_device)
                print(f"  ✅ Loaded SBERT model on {sbert_device}")
            except Exception as e:
                print(f"  ⚠️ Could not load SBERT model: {e}")
                self.sbert = None
        else:
            print("  ⚠️ sentence-transformers not installed; SBERT similarity disabled")
        
    def restore_original_params(self):
        """Restore original model parameters"""
        print("  🔄 Restoring original model parameters...")
        try:
            # Move CPU-stored originals to model device before loading
            model_device = next(self.model.parameters()).device
            device_mapped = {k: v.to(model_device) for k, v in self.original_params.items()}
            self.model.load_state_dict(device_mapped)
            print("  ✅ Original parameters restored successfully")
        except Exception as e:
            print(f"  ⚠️ Fallback: trying direct CPU load - {e}")
            # Fallback: try loading CPU tensors directly (PyTorch will remap)
            self.model.load_state_dict(self.original_params)
        
        # Ensure model is on expected device
        try:
            self.model.to(self.torch_device)
        except Exception:
            pass
    
    def debug_check_vector_values(self, vector_keys: List[str], context: str = ""):
        """
        Debug function to check actual values of concept vectors in MLP weights
        Shows debug info for first vector only to reduce noise
        
        Args:
            vector_keys: List of vector keys to check
            context: Context for the debug message
        """
        if not vector_keys:
            return
            
        print(f"\n🔍 DEBUG: Checking vector values{' (' + context + ')' if context else ''}...")
        print(f"    Showing debug for first vector only (total: {len(vector_keys)} vectors)")
        
        # Only debug the first vector to reduce log noise
        vector_key = vector_keys[0]
        try:
            # Parse vector key
            parts = vector_key.split('_')
            layer = int(parts[0][1:])  # Remove 'L' prefix  
            dimension = int(parts[1][1:])  # Remove 'C' prefix
            
            # Get the parameter
            param_name = f'model.layers.{layer}.mlp.down_proj.weight'
            
            # Find the parameter in the model
            found = False
            for name, param in self.model.named_parameters():
                if name == param_name:
                    found = True
                    values = param.data[:, dimension].clone()
                    
                    # Calculate statistics
                    min_val = values.min().item()
                    max_val = values.max().item()
                    mean_val = values.mean().item()
                    std_val = values.std().item()
                    
                    # Basic statistics for debugging
                    print(f"  {vector_key} (L{layer}_C{dimension}):")
                    print(f"    Min: {min_val:.6f}, Max: {max_val:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                    
                    break
            
            if not found:
                print(f"  ❌ Parameter {param_name} not found for {vector_key}")
                
        except Exception as e:
            print(f"  ❌ Error checking {vector_key}: {str(e)}")
              
        print(f"🔍 DEBUG: Vector check complete\n")
            
    def inject_noise_to_vector(self, layer: int, dimension: int, perturbation_param: float = 0.1, perturbation_type: str = 'gaussian'):
        """
        Apply perturbation to specific concept vector location
        
        Args:
            layer: Layer number
            dimension: Dimension in the layer
            perturbation_param: Parameter for perturbation (std for gaussian, scale for scale, value for ablation)
            perturbation_type: Type of perturbation ('gaussian', 'scale', 'ablation')
        """
        # For Gemma, use model.layers.{layer}.mlp.down_proj.weight
        param_name = f'model.layers.{layer}.mlp.down_proj.weight'
        
        def _modify_tensor_inplace(param_tensor, tensor_name):
            """Helper function to modify tensor in-place with debug info"""
            if dimension < 0 or dimension >= param_tensor.shape[1]:
                raise IndexError(f"Dimension index {dimension} out of range for tensor with shape {param_tensor.shape}")

            # Store original values for verification
            original_values = param_tensor[:, dimension].clone()

            with torch.no_grad():
                if perturbation_type == 'ablation':
                    param_tensor[:, dimension] = perturbation_param
                elif perturbation_type == 'scale':
                    param_tensor[:, dimension] *= perturbation_param
                elif perturbation_type == 'gaussian':
                    hidden_size = self.model.config.hidden_size
                    noise = torch.normal(0, perturbation_param, size=(hidden_size,)).to(param_tensor.device)
                    param_tensor[:, dimension] += noise
                else:
                    raise ValueError(f"Unknown perturbation type: {perturbation_type}")
            
            # Verify change
            new_values = param_tensor[:, dimension]
            changed = not torch.equal(original_values, new_values)
            return changed, original_values, new_values
        
        # Try to find the actual parameter object via named_parameters() for in-place edit
        found = False
        for name, param in self.model.named_parameters(recurse=True):
            if name == param_name:
                found = True
                changed, orig, new = _modify_tensor_inplace(param.data, name)
                if not changed:
                    print(f"  ⚠️ WARNING: In-place edit did not change parameter {name}")
                break

        if found:
            # Clear any cached computations
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
            return

        # Fallback: operate on state_dict and reload (slower but robust)
        print(f"  ⚠️ Parameter {param_name} not found via named_parameters(), using state_dict fallback")
        state = self.model.state_dict()
        if param_name not in state:
            raise KeyError(f"Parameter not found in model: {param_name}")

        # Apply modification to state_dict tensor
        tensor = state[param_name]
        model_device = next(self.model.parameters()).device
        
        if tensor.device != model_device:
            print(f"  🔧 Moving tensor from {tensor.device} to model device {model_device}")
            tensor = tensor.to(model_device)
            state[param_name] = tensor
        
        changed, orig, new = _modify_tensor_inplace(tensor, param_name)
        
        # Reload the model with modified state
        self.model.load_state_dict(state)
        
        # Verify the change took effect
        verify_state = self.model.state_dict()
        verify_values = verify_state[param_name][:, dimension]
        final_changed = not torch.equal(orig, verify_values)
        
        if not final_changed:
            print(f"  ❌ ERROR: State dict reload did not preserve changes for {param_name}")
        
        # Clear any cached computations
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()

    def ablate_ensemble_vectors(self, vector_keys: List[str], perturbation_param: float = 0.1, perturbation_type: str = 'gaussian'):
        """
        Apply perturbation to multiple concept vectors simultaneously
        
        Args:
            vector_keys: List of vector keys in format "L{layer}_C{dimension}"
            perturbation_param: Parameter for perturbation (std for gaussian, scale for scale, value for ablation)
            perturbation_type: Type of perturbation ('gaussian', 'scale', 'ablation')
        """
        perturbation_names = {
            'gaussian': 'Adding Gaussian noise to',
            'scale': 'Scaling',
            'ablation': 'Ablating'
        }
        print(f"🔧 {perturbation_names.get(perturbation_type, 'Modifying')} {len(vector_keys)} vectors simultaneously:")
        
        # Parse all vector keys first
        vector_locations = []
        layers_used = set()
        for vector_key in vector_keys:
            try:
                parts = vector_key.split('_')
                layer = int(parts[0][1:])  # Remove 'L' prefix
                dimension = int(parts[1][1:])  # Remove 'C' prefix
                vector_locations.append((layer, dimension, vector_key))
                layers_used.add(layer)
            except:
                print(f"  ⚠️ Warning: Invalid vector key format: {vector_key}")
                continue
        
        if not vector_locations:
            raise ValueError("No valid vector keys provided")
        
        print(f"     Targeting {len(vector_locations)} vectors in layers {sorted(layers_used)}")
        
        # Apply modifications to all vectors (minimal output)
        for layer, dimension, vector_key in vector_locations:
            self.inject_noise_to_vector(layer, dimension, perturbation_param, perturbation_type)
        
        print(f"     ✅ Applied {perturbation_type} to all vectors")
        
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
        
        print(f"    🤖 Generating {context} answers for {total_questions} questions...")
        
        # Process each question individually using proper chat template
        for i, question in enumerate(questions, 1):
            print(f"      [{i}/{total_questions}] Q: {question}")
            
            # Create proper message format for Gemma 3
            # Prepend system instruction so model answers directly without hedging
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions directly, without hedging or prefacing your response."
                },
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
                    use_cache=False,
                    output_scores=False,
                    return_dict_in_generate=False
                )

            # Decode only the newly generated tokens
            new_tokens = generation_output[0][inputs["input_ids"].shape[-1]:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            answers.append(answer)
            
            # Show the generated answer (full text)
            print(f"      [{i}/{total_questions}] A: {answer}")
            print()  # Empty line for readability
        
        print(f"    ✅ Completed {context} generation ({total_questions} answers)")
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

    def compute_sbert_similarities(self, references: List[str], candidates: List[str], batch_size: int = 32) -> List[float]:
        """Compute per-pair SBERT cosine similarities for equal-length lists.

        Returns list of floats (one similarity per pair). If SBERT is unavailable,
        returns zeros.
        """
        if not self.sbert or sbert_util is None:
            return [0.0] * len(candidates)

        # Encode separately and compute diagonal similarities
        refs_emb = self.sbert.encode(references, convert_to_tensor=True, batch_size=batch_size, device=self.sbert.device)
        cands_emb = self.sbert.encode(candidates, convert_to_tensor=True, batch_size=batch_size, device=self.sbert.device)

        sims = sbert_util.cos_sim(refs_emb, cands_emb)  # matrix
        diag = sims.diag().cpu().tolist()
        return [float(x) for x in diag]
    
    def test_full_layer_ablation(self, layer: int = 5, test_question: str = "Write a SQL query to select all users from a database table named 'users'.") -> Dict[str, str]:
        """
        Strong test: ablate entire MLP down_proj layer to zero, generate answer before/after, compare
        
        Args:
            layer: Layer number to completely ablate
            test_question: Question to test with (SQL by default)
            
        Returns:
            Dict with 'before', 'after', 'changed', and layer info
        """
        print('\n' + '='*72)
        print(f"💥 FULL LAYER ABLATION TEST: Zeroing entire down_proj layer {layer}")
        print('='*72 + '\n')

        # Generate answer before ablation
        print("  📝 Generating baseline answer...")
        before_answers = self.generate_answers([test_question], max_new_tokens=64, context="baseline")
        before_answer = before_answers[0] if before_answers else ""

        # Get parameter name for this layer
        param_name = f'model.layers.{layer}.mlp.down_proj.weight'

        # Apply full layer inversion (multiply weights by -1)
        print(f"  🛠 Inverting entire layer {layer} down_proj weights (multiply by -1)...")
        found = False
        original_tensor = None

        # Find the parameter and zero it out completely
        for name, param in self.model.named_parameters(recurse=True):
            if name == param_name:
                found = True
                # Store original for verification
                original_tensor = param.data.clone()

                # Invert the entire parameter tensor (multiply by -1)
                with torch.no_grad():
                    param.data.mul_(-1.0)

                # Verify it's actually inverted (mean should flip sign)
                inverted_mean = param.data.mean().item()
                print('\n' + '    ' + '-'*64)
                print(f"    Layer {layer} parameter shape: {param.data.shape}")
                print(f"    Inverted mean: {inverted_mean:.6f} (original mean: {original_tensor.mean().item():.6f})")
                print('    ' + '-'*64 + '\n')
                break

        if not found:
            print(f"  ❌ ERROR: Could not find parameter {param_name}")
            return {'error': f'Parameter {param_name} not found'}

        # Generate answer after ablation
        print("  📝 Generating ablated answer...")
        after_answers = self.generate_answers([test_question], max_new_tokens=64, context="ablated")
        after_answer = after_answers[0] if after_answers else ""

        # Restore original weights
        print('\n' + '  🔄 Restoring original layer weights...')
        for name, param in self.model.named_parameters(recurse=True):
            if name == param_name:
                with torch.no_grad():
                    param.data.copy_(original_tensor)

                # Verify restoration
                restored_mean = param.data.mean().item()
                original_mean = original_tensor.mean().item()
                print(f"    Restored mean: {restored_mean:.6f} (original mean was {original_mean:.6f})")
                break

        # Compare answers
        changed = before_answer.strip() != after_answer.strip()

        result = {
            'layer_ablated': layer,
            'param_name': param_name,
            'test_question': test_question,
            'before': before_answer,
            'after': after_answer,
            'changed': changed,
            'param_shape': str(original_tensor.shape) if original_tensor is not None else 'unknown'
        }

        print('\n' + '='*40 + ' RESULT ' + '='*40)
        print(f"Layer ablated: {layer} ({param_name})")
        print(f"Question: {test_question}")
        print('\n' + '  BEFORE:' )
        print(f"    {before_answer}\n")
        print('  AFTER:')
        print(f"    {after_answer}\n")
        print(f"Changed:  {'✅ YES' if changed else '❌ NO'}")
        print('='*92 + '\n')

        return result
    
    def validate_ensemble_concept_vectors(
        self, 
        concept_name: str,
        vector_keys: List[str],
        concept_qa: Dict[str, List[str]],
        unrelated_qa: Dict[str, List[str]],
        perturbation_type: str = 'gaussian',
        perturbation_param: float = GAUSSIAN_STD
    ) -> Dict[str, Any]:
        """
        Validate concept vectors by perturbing all candidates simultaneously
        
        Args:
            concept_name: Name of the concept being tested
            vector_keys: List of vector keys in format "L{layer}_C{dimension}"
            concept_qa: Dict with 'questions' and 'answers' lists for concept-specific Q&A
            unrelated_qa: Dict with 'questions' and 'answers' lists for unrelated Q&A
            perturbation_type: Type of perturbation ('gaussian', 'scale', 'ablation')
            perturbation_param: Parameter for perturbation (std for gaussian, scale for scale, value for ablation)
            
        Returns:
            Dictionary containing validation results
        """
        concept_questions = concept_qa['questions']
        concept_baseline_answers = concept_qa['answers']
        unrelated_questions = unrelated_qa['questions']
        unrelated_baseline_answers = unrelated_qa['answers']
        
        print(f"Validating ensemble concept '{concept_name}' with {len(vector_keys)} vectors:")
        for vector_key in vector_keys:
            print(f"  - {vector_key}")
        print(f"Concept questions: {len(concept_questions)}")
        print(f"Unrelated questions: {len(unrelated_questions)}")
        
        try:
            # Use existing baseline answers from qa.json (no need to generate)
            print("📋 Using baseline answers from qa.json...")
            
            # Apply perturbation based on type
            print(f"   🔧 Applying {perturbation_type} perturbation...")
            if perturbation_type == 'gaussian':
                self.ablate_ensemble_vectors(vector_keys, perturbation_param, perturbation_type='gaussian')
            elif perturbation_type == 'scale':
                self.ablate_ensemble_vectors(vector_keys, perturbation_param, perturbation_type='scale')
            elif perturbation_type == 'ablation':
                self.ablate_ensemble_vectors(vector_keys, perturbation_param, perturbation_type='ablation')
            
            # Generate perturbed answers
            print(f"   📝 Generating perturbed responses...")
            perturbed_concept_answers = self.generate_answers(
                concept_questions, 
                context="concept"
            )
            perturbed_unrelated_answers = self.generate_answers(
                unrelated_questions, 
                context="unrelated"
            )
            
            # Restore model immediately after perturbation test
            print("   🔄 Restoring model...")
            self.restore_original_params()
            
            # Calculate metrics
            print("📊 Calculating similarity metrics...")
            concept_bleu_scores = []
            concept_rouge_scores = []
            unrelated_bleu_scores = []
            unrelated_rouge_scores = []
            
            print(f"    📈 Computing scores for {len(concept_baseline_answers)} concept Q&A pairs...")
            # Calculate scores for concept-specific questions
            for i, (perturbed, baseline) in enumerate(zip(perturbed_concept_answers, concept_baseline_answers)):
                bleu = self.calculate_bleu(baseline, perturbed)
                rouge = self.calculate_rouge_l(baseline, perturbed)
                concept_bleu_scores.append(bleu)
                concept_rouge_scores.append(rouge)
                print(f"      Concept {i+1}: BLEU={bleu:.3f}, ROUGE={rouge:.3f}")
            
            print(f"    📈 Computing scores for {len(unrelated_baseline_answers)} unrelated Q&A pairs...")
            # Calculate scores for unrelated questions
            for i, (perturbed, baseline) in enumerate(zip(perturbed_unrelated_answers, unrelated_baseline_answers)):
                bleu = self.calculate_bleu(baseline, perturbed)
                rouge = self.calculate_rouge_l(baseline, perturbed)
                unrelated_bleu_scores.append(bleu)
                unrelated_rouge_scores.append(rouge)
                print(f"      Unrelated {i+1}: BLEU={bleu:.3f}, ROUGE={rouge:.3f}")

            # Compute SBERT similarities (semantic)
            concept_sbert_sims = self.compute_sbert_similarities(concept_baseline_answers, perturbed_concept_answers)
            unrelated_sbert_sims = self.compute_sbert_similarities(unrelated_baseline_answers, perturbed_unrelated_answers)

            avg_concept_sbert = statistics.mean(concept_sbert_sims) if concept_sbert_sims else 0.0
            avg_unrelated_sbert = statistics.mean(unrelated_sbert_sims) if unrelated_sbert_sims else 0.0

            concept_sbert_degradation = 1.0 - avg_concept_sbert
            unrelated_sbert_degradation = 1.0 - avg_unrelated_sbert
            sbert_specificity = concept_sbert_degradation - unrelated_sbert_degradation

            print(f"    📈 SBERT: concept_avg={avg_concept_sbert:.3f}, unrelated_avg={avg_unrelated_sbert:.3f}, specificity={sbert_specificity:.3f}")
            
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
                'concept_name': concept_name,
                'vector_keys': vector_keys,
                'num_vectors_ablated': len(vector_keys),
                'perturbation_type': f'ensemble_{perturbation_type}',
                'perturbation_param': perturbation_param,
                'n_concept_questions': len(concept_questions),
                'n_unrelated_questions': len(unrelated_questions),
                
                # Concept scores
                    # Average similarity between perturbed answers and baseline (concept questions)
                    # - 'concept_bleu_score': mean BLEU score over all concept Q&A pairs (higher = perturbed answers
                    #    are more similar to baseline answers)
                    # - 'concept_rouge_score': mean ROUGE-L F1 over all concept Q&A pairs (higher = more similar)
                    # Degradation measures (1 - score) indicate how much the model changed after perturbation
                    'concept_bleu_score': avg_concept_bleu,
                    'concept_rouge_score': avg_concept_rouge,
                    'concept_bleu_degradation': concept_bleu_degradation,  # 1 - concept_bleu_score (higher = more change)
                    'concept_rouge_degradation': concept_rouge_degradation,  # 1 - concept_rouge_score

                    # Unrelated scores
                    # Same metrics computed on unrelated questions (acts as a control set)
                    # - 'unrelated_bleu_score' / 'unrelated_rouge_score': average similarity on unrelated QAs
                    # - 'unrelated_*_degradation': 1 - corresponding score (higher = more change)
                    'unrelated_bleu_score': avg_unrelated_bleu,
                    'unrelated_rouge_score': avg_unrelated_rouge,
                    'unrelated_bleu_degradation': unrelated_bleu_degradation,
                    'unrelated_rouge_degradation': unrelated_rouge_degradation,

                    # Concept-specificity measures
                    # Positive values mean concept questions degraded more than unrelated (i.e., effect is specific)
                    # - 'bleu_specificity': concept_bleu_degradation - unrelated_bleu_degradation
                    # - 'rouge_specificity': concept_rouge_degradation - unrelated_rouge_degradation
                    'bleu_specificity': bleu_specificity,
                    'rouge_specificity': rouge_specificity,

                    # Individual scores for analysis (per-question lists, same order as input questions)
                    # - 'individual_concept_bleu': BLEU per concept question
                    # - 'individual_concept_rouge': ROUGE-L per concept question
                    # - 'individual_unrelated_bleu': BLEU per unrelated question
                    # - 'individual_unrelated_rouge': ROUGE-L per unrelated question
                    'individual_concept_bleu': concept_bleu_scores,
                    'individual_concept_rouge': concept_rouge_scores,
                    'individual_unrelated_bleu': unrelated_bleu_scores,
                    'individual_unrelated_rouge': unrelated_rouge_scores,

                    # SBERT (semantic) scores
                    'concept_sbert_score': avg_concept_sbert,
                    'unrelated_sbert_score': avg_unrelated_sbert,
                    'concept_sbert_degradation': concept_sbert_degradation,
                    'unrelated_sbert_degradation': unrelated_sbert_degradation,
                    'sbert_specificity': sbert_specificity,
                
                # Complete answer sets for full analysis (baseline from qa.json, perturbed generated)
                'all_baseline_concept': concept_baseline_answers,
                'all_perturbed_concept': perturbed_concept_answers,
                'all_baseline_unrelated': unrelated_baseline_answers,
                'all_perturbed_unrelated': perturbed_unrelated_answers
            }
            
            # Print validation results
            print(f"Results for ensemble concept '{concept_name}':")
            print(f"  Vectors ablated: {len(vector_keys)}")
            print(f"  Concept BLEU: {avg_concept_bleu:.4f}, Concept ROUGE-L: {avg_concept_rouge:.4f}")
            print(f"  Unrelated BLEU: {avg_unrelated_bleu:.4f}, Unrelated ROUGE-L: {avg_unrelated_rouge:.4f}")
            print(f"  BLEU Specificity: {bleu_specificity:.4f}, ROUGE Specificity: {rouge_specificity:.4f}")
            
            # Check for concept-specificity using a weighted combination of metrics
            # Give more importance to semantic (SBERT) similarity
            specificity_threshold = SPECIFICITY_THRESHOLD

            # Default weights (sum to 1.0) — SBERT is prioritized
            w_sbert = 0.50
            w_bleu = 0.25
            w_rouge = 0.25

            # Combined specificity (higher = concept degraded more than unrelated)
            combined_specificity = (w_sbert * sbert_specificity +
                                    w_bleu * bleu_specificity +
                                    w_rouge * rouge_specificity)

            # Combined threshold is slightly lower than individual thresholds to be sensitive
            combined_specificity_threshold = 0.30

            # If SBERT is unavailable, fall back to original BLEU/ROUGE OR logic
            if self.sbert is None:
                is_concept_specific = (bleu_specificity > specificity_threshold or 
                                       rouge_specificity > specificity_threshold)
                used_method = 'bleu_or_rouge_fallback'
            else:
                is_concept_specific = (combined_specificity > combined_specificity_threshold)
                used_method = 'weighted_combined'

            # Store decision and debugging fields
            results['is_concept_specific'] = is_concept_specific
            results['specificity_threshold'] = specificity_threshold
            results['combined_specificity'] = combined_specificity
            results['combined_specificity_threshold'] = combined_specificity_threshold
            results['specificity_weights'] = {'sbert': w_sbert, 'bleu': w_bleu, 'rouge': w_rouge}
            results['specificity_method'] = used_method

            # Diagnostic output
            if used_method == 'weighted_combined':
                print(f"  ✓ Using weighted combined specificity (SBERT-heavy): combined={combined_specificity:.4f} "
                      f"(threshold={combined_specificity_threshold})")
                print(f"    Contributions -> SBERT: {sbert_specificity:.4f}, BLEU: {bleu_specificity:.4f}, ROUGE: {rouge_specificity:.4f}")
            else:
                if is_concept_specific:
                    print(f"  ✓ Ensemble shows concept specificity (BLEU/ROUGE fallback, threshold: {specificity_threshold})")
                else:
                    print(f"  ✗ Ensemble does not show clear concept specificity")
            
            return results
            
        except Exception as e:
            # Always restore original parameters on error
            self.restore_original_params()
            raise e


def load_concept_vectors_from_layerwise_results(results_file: str) -> Dict[str, Dict]:
    """
    Load concept vectors from the layer-wise GPU analysis results
    
    Args:
        results_file: Path to layer_wise_projection_gpu_analysis_results.json
        
    Returns:
        Dictionary mapping concept names to their layer-wise vector data
        Format: {concept_name: {'best_layers': [...], 'selected_vectors': [...], 'layer_analyses': {...}}}
    """
    print(f"Loading layer-wise concept vectors from: {results_file}")
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    concept_vectors = {}

    # Extract from layer_wise_projection_gpu_analysis_results.json format
    if 'concept_analyses' in results_data:
        for concept_name, concept_data in results_data['concept_analyses'].items():
            if 'best_layers' in concept_data and 'selected_vectors' in concept_data:
                concept_vectors[concept_name] = {
                    'best_layers': concept_data['best_layers'],
                    'selected_vectors': concept_data['selected_vectors'],
                    'layer_analyses': concept_data.get('layer_analyses', {})
                }
                
                # Count vectors per layer
                layer_vector_counts = {}
                for vector in concept_data['selected_vectors']:
                    layer = vector['layer']
                    if layer not in layer_vector_counts:
                        layer_vector_counts[layer] = 0
                    layer_vector_counts[layer] += 1
                
                print(f"Loaded concept '{concept_name}':")
                print(f"  - {len(concept_data['best_layers'])} best layers: {[layer['layer'] for layer in concept_data['best_layers']]}")
                print(f"  - {len(concept_data['selected_vectors'])} total selected vectors")
                print(f"  - Vectors per layer: {dict(sorted(layer_vector_counts.items()))}")
    
    print(f"Total concepts loaded: {len(concept_vectors)}")
    return concept_vectors


def generate_layer_vector_combinations() -> List[Dict]:
    """
    Generate all combinations of layer counts, vector counts, and perturbation types for testing
    
    Returns:
        List of combination dictionaries with 'layer_count', 'vector_count', and 'perturbation_type' keys
        
    Note: vector_count represents vectors PER LAYER, so total vectors = layer_count × vector_count
    """
    layer_counts = [1, 2, 5]  # Number of best layers to use
    vector_counts = [10, 20, 50]  # Number of vectors PER LAYER
    perturbation_types = [
        {'type': 'gaussian', 'name': 'Gaussian Noise', 'param': GAUSSIAN_STD},
        {'type': 'scale', 'name': 'Scale Inversion', 'param': SCALE_VAL},
        {'type': 'ablation', 'name': 'Ablation', 'param': ABLATION_VAL}
    ]
    
    combinations = []
    for layer_count in layer_counts:
        for vector_count in vector_counts:
            for perturbation in perturbation_types:
                total_vectors = layer_count * vector_count
                combinations.append({
                    'layer_count': layer_count,
                    'vector_count': vector_count,
                    'total_vectors': total_vectors,
                    'perturbation_type': perturbation['type'],
                    'perturbation_name': perturbation['name'],
                    'perturbation_param': perturbation['param'],
                    'combination_name': f"L{layer_count}_V{vector_count}_{perturbation['type'].upper()}"
                })
    
    print(f"Generated {len(combinations)} layer/vector/perturbation combinations:")
    print(f"  - Layer counts: {layer_counts}")
    print(f"  - Vector counts per layer: {vector_counts}")
    print(f"  - Perturbation types: {[p['type'] for p in perturbation_types]}")
    print(f"  - Total combinations: {len(layer_counts)} × {len(vector_counts)} × {len(perturbation_types)} = {len(combinations)}")
    print(f"  - Total vector ranges: {min(layer_counts) * min(vector_counts)} to {max(layer_counts) * max(vector_counts)} vectors per test")
    
    return combinations


def extract_vectors_for_combination(concept_data: Dict, layer_count: int, vector_count: int) -> List[Dict]:
    """
    Extract vectors for a specific layer/vector count combination
    
    Args:
        concept_data: Concept data from layer-wise results
        layer_count: Number of best layers to use (1, 2, or 5)
        vector_count: Number of vectors to extract PER LAYER
        
    Returns:
        List of vector dictionaries to use for this combination
    """
    best_layers = concept_data['best_layers'][:layer_count]  # Take top N layers
    selected_vectors = concept_data['selected_vectors']
    
    # Get vectors from the selected best layers
    layer_numbers = {layer['layer'] for layer in best_layers}
    filtered_vectors = [v for v in selected_vectors if v['layer'] in layer_numbers]
    
    # Group vectors by layer
    vectors_by_layer = {}
    for vector in filtered_vectors:
        layer = vector['layer']
        if layer not in vectors_by_layer:
            vectors_by_layer[layer] = []
        vectors_by_layer[layer].append(vector)
    
    # Sort vectors within each layer by rank_in_layer
    for layer in vectors_by_layer:
        vectors_by_layer[layer].sort(key=lambda x: x['rank_in_layer'])
    
    # Select vector_count vectors from each layer
    selected_vectors_for_combo = []
    for layer in sorted(vectors_by_layer.keys()):
        layer_vectors = vectors_by_layer[layer][:vector_count]  # Take top vector_count from this layer
        selected_vectors_for_combo.extend(layer_vectors)
    
    return selected_vectors_for_combo


def main():
    """Main function to run systematic layerwise concept vector validation"""
    
    # Paths
    base_path = Path("/media/hdd/usr/martinelli/concept-vectors-gemma3")
    qa_file = base_path / "code/concept-val-test/qa-generated.json"
    results_file = base_path / "code/projection/value_vector_results_gpu_layerwise/layer_wise_projection_gpu_analysis_results.json"
    output_dir = base_path / "code/concept-val-test/validation-results"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    print(f"QA concepts: {list(concept_qa_map.keys())}")
    
    # Load concept vector results from layerwise analysis
    print("Loading layerwise concept vector results...")
    concept_vectors = load_concept_vectors_from_layerwise_results(results_file)
    print(f"Loaded vector results for {len(concept_vectors)} concepts")
    print(f"Vector concepts: {list(concept_vectors.keys())}")
    
    # Check for concept overlap
    qa_concepts = set(concept_qa_map.keys())
    vector_concepts = set(concept_vectors.keys())
    overlap = qa_concepts.intersection(vector_concepts)
    print(f"Concepts with both QA and vectors: {list(overlap)}")
    if not overlap:
        print("ERROR: No concepts have both QA data and vector data!")
        return
    
    # Generate all layer/vector combinations to test
    combinations = generate_layer_vector_combinations()
    
    # Initialize validator once
    print("Initializing ensemble validator...")
    validator = EnsembleConceptValidator(
        model_name=MODEL_ID,
        device=DEVICE
    )

    print(f"\n{'='*60}")
    print(f"🎛️  Testing ALL PERTURBATION TYPES: Gaussian Noise, Scale Inversion, and Ablation")
    print(f"Testing {len(combinations)} layer/vector/perturbation combinations per concept")
    print(f"Specificity threshold: {SPECIFICITY_THRESHOLD}")
    print(f"Unrelated questions count: {UNRELATED_QUESTIONS_COUNT}")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    concept_count = 0
    total_concepts = len([c for c in concept_vectors.keys() if c in concept_qa_map])
    overall_results = []

    for concept_name, concept_data in concept_vectors.items():
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
        
        # Limit unrelated questions to the configured count
        if len(unrelated_questions) > UNRELATED_QUESTIONS_COUNT:
            indices = random.sample(range(len(unrelated_questions)), UNRELATED_QUESTIONS_COUNT)
            unrelated_questions = [unrelated_questions[i] for i in indices]
            unrelated_answers = [unrelated_answers[i] for i in indices]
        
        unrelated_qa = {'questions': unrelated_questions, 'answers': unrelated_answers}
        
        print(f"🎯 Testing concept: {concept_name}")
        print(f"   📝 Concept questions: {len(concept_qa['questions'])}")
        print(f"   📝 Unrelated questions: {len(unrelated_questions)}")
        
        # Show sample baseline answers for this concept
        print(f"   📋 Sample baseline answers:")
        for i, (q, a) in enumerate(zip(concept_qa['questions'][:2], concept_qa['answers'][:2])):
            print(f"      Q{i+1}: {q}")
            print(f"      A{i+1}: {a}")
            print()  # Add blank line for readability
        
        # Results for all combinations of this concept
        concept_combination_results = []
        
        # Test all layer/vector combinations for this concept
        for combo_idx, combination in enumerate(combinations, 1):
            layer_count = combination['layer_count']
            vector_count = combination['vector_count']
            combo_name = combination['combination_name']
            
            # Clear test separation
            print(f"\n{'='*80}")
            print(f"🧪 TEST {combo_idx}/{len(combinations)}: {concept_name.upper()} - {combo_name}")
            print(f"   📊 Configuration: {layer_count} layers × {vector_count} vectors/layer = {layer_count * vector_count} total vectors")
            print(f"   🎛️  Perturbation: {combination['perturbation_name']} (param: {combination['perturbation_param']})")
            print(f"{'='*80}")
            
            # Extract vectors for this combination
            try:
                selected_vectors = extract_vectors_for_combination(concept_data, layer_count, vector_count)
                
                if not selected_vectors:
                    print(f"    ⚠️ Warning: No vectors found for combination {combo_name}")
                    continue
                
                vector_keys = [v['vector_key'] for v in selected_vectors]
                selected_layers = sorted(set(v['layer'] for v in selected_vectors))
                
                print(f"   🎯 Target: {len(vector_keys)} vectors across layers {selected_layers}")
                print(f"   📝 Questions: {len(concept_qa['questions'])} concept + {len(unrelated_qa['questions'])} unrelated")
                
                # Run validation for this combination
                result = validator.validate_ensemble_concept_vectors(
                    concept_name=concept_name,
                    vector_keys=vector_keys,
                    concept_qa=concept_qa,
                    unrelated_qa=unrelated_qa,
                    perturbation_type=combination['perturbation_type'],
                    perturbation_param=combination['perturbation_param']
                )
                
                # Add combination metadata
                result['layer_count'] = combination['layer_count']
                result['vector_count'] = combination['vector_count']  # vectors per layer
                result['vector_count_per_layer'] = combination['vector_count']
                result['total_vectors'] = combination['total_vectors']  # layer_count × vector_count
                result['perturbation_type'] = combination['perturbation_type']
                result['perturbation_param'] = combination['perturbation_param']
                result['combination_name'] = combination['combination_name']
                result['selected_layers'] = sorted(set(v['layer'] for v in selected_vectors))
                result['actual_vector_count'] = len(vector_keys)
                
                concept_combination_results.append(result)
                overall_results.append(result)
                
                # Show clear result
                is_specific = result.get('is_concept_specific', False)
                bleu_spec = result.get('bleu_specificity', 0)
                rouge_spec = result.get('rouge_specificity', 0)
                status = "✅ SPECIFIC" if is_specific else "❌ NOT SPECIFIC"
                print(f"   📊 RESULT: {status}")
                print(f"       BLEU Specificity: {bleu_spec:.4f} | ROUGE Specificity: {rouge_spec:.4f}")
                print(f"       Threshold: {SPECIFICITY_THRESHOLD} | {'PASSED' if is_specific else 'FAILED'}")
                print(f"{'='*80}")  # End separator for this test
                
            except Exception as e:
                print(f"    Error testing combination {combo_name} for {concept_name}: {str(e)}")
                continue
        
        # Save results for this concept
        concept_output_file = output_dir / f"{concept_name}_validation_results.json"
        with open(concept_output_file, 'w') as f:
            json.dump(concept_combination_results, f, indent=2)
        print(f"\n  💾 Saved {len(concept_combination_results)} combination results for {concept_name} to: {concept_output_file}")
        
        # Show concept summary
        specific_combos = [r for r in concept_combination_results if r.get('is_concept_specific', False)]
        print(f"  📊 Concept {concept_name} summary: {len(specific_combos)}/{len(concept_combination_results)} combinations are specific")
    
    # Save overall results
    overall_output_file = output_dir / "overall_validation_results.json"
    with open(overall_output_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    print(f"\n💾 Saved overall results to: {overall_output_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL LAYERWISE VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(overall_results)
    specific_count = sum(1 for r in overall_results if r.get('is_concept_specific', False))
    
    print(f"Total validation tests: {total_tests}")
    print(f"Concepts tested: {total_concepts}")
    print(f"Combinations per concept: {len(combinations)}")
    print(f"Concept-specific results found: {specific_count}")
    print(f"Overall specificity rate: {specific_count/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
    
    # Analysis by combination type
    print(f"\nPer-combination analysis:")
    for combination in combinations:
        combo_name = combination['combination_name']
        combo_results = [r for r in overall_results if r.get('combination_name') == combo_name]
        combo_specific = sum(1 for r in combo_results if r.get('is_concept_specific', False))
        print(f"  {combo_name}: {combo_specific}/{len(combo_results)} concepts specific ({combo_specific/len(combo_results)*100:.1f}%)")
    
    # Best performing combinations
    if specific_count > 0:
        print(f"\n✓ Found {specific_count} specific results across all combinations!")
        
        # Show best results
        overall_results.sort(key=lambda x: max(x.get('bleu_specificity', 0), x.get('rouge_specificity', 0)), reverse=True)
        
        print(f"\nTop 10 results:")
        for i, result in enumerate(overall_results[:10]):
            concept = result.get('concept_name')
            combo = result.get('combination_name')
            bleu_spec = result.get('bleu_specificity', 0)
            rouge_spec = result.get('rouge_specificity', 0)
            layers = result.get('selected_layers', [])
            is_specific = result.get('is_concept_specific', False)
            status = "✓" if is_specific else "✗"
            print(f"  {i+1}. {concept} ({combo}) - Layers {layers}")
            print(f"     {status} BLEU: {bleu_spec:.4f}, ROUGE: {rouge_spec:.4f}")
    else:
        print(f"\n✗ No combinations showed clear specificity.")
        print("Consider:")
        print("  - Different layer/vector combinations")
        print("  - Different vector selection criteria")
        print("  - Alternative validation methodologies")
        print("  - Lower specificity threshold")


if __name__ == "__main__":
    import sys
    
    # Check for test mode
    # CLI is order-insensitive: detect 'test' and optional 'no-force' anywhere in args.
    cli_args = [a.lower() for a in sys.argv[1:]]

    # Default: force download unless overridden
    force_flag = True

    # CLI flags take precedence over environment
    if any(a in ('no-force', '--no-force') for a in cli_args):
        force_flag = False
    elif any(a in ('force', '--force') for a in cli_args):
        force_flag = True
    else:
        env_force = os.environ.get('FORCE_DOWNLOAD', None)
        if env_force is not None:
            force_flag = env_force.lower() in ('1', 'true', 'yes')

    if 'test' in cli_args:
        print("🚀 Running layerwise validation with full layer ablation test...")

        validator = EnsembleConceptValidator(force_download=force_flag)
        
        print("\n💥 Testing full layer ablation to verify perturbations work...")
        # SQL question to test knowledge-based responses  
        random_question = "What is the plot of Harry Potter and the Sorcerer's Stone?"
        ablation_result = validator.test_full_layer_ablation(layer=5, test_question=random_question)
        
        if 'error' in ablation_result:
            print(f"❌ Ablation test failed: {ablation_result['error']}")
        else:
            if ablation_result['changed']:
                print("✅ SUCCESS: Full layer ablation changed the model's output - perturbations are working!")
                print("Now you can run actual concept validation with confidence.")
                print("Run without 'test' argument to run full concept validation.")
            else:
                print("⚠️  WARNING: Full layer ablation didn't change output - may need to investigate further")
                print("Possible issues: layer selection, output comparison, model caching")
        
        print("\n✅ Layer ablation test complete!")
        
    else:
        # Run full concept validation
        main()
