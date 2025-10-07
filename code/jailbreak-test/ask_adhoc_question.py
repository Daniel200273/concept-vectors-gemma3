#!/usr/bin/env python3
"""
Ad Hoc Question Testing Script

This script allows you to test a specific concept and question using the same 
jailbreak testing methodology as run_jailbreak_test.py. Simply modify the 
CONCEPT_NAME and QUESTION variables below and run the script.

The script:
1. Loads the specified concept's configuration from best-tests/
2. Loads Gemma-3-1b-it model and saves original parameters  
3. Runs the same crafted prompt tests as the main jailbreak script:
    - Crafted Prompt 1 + baseline model (concept-specific)
    - Crafted Prompt 2 + baseline model (concept-specific)
    - Crafted Prompt 1 + perturbed model (concept-specific)
    - Crafted Prompt 2 + perturbed model (concept-specific)
    - (Optional) Crafted Prompt 1 + perturbed model (unrelated control)
    - (Optional) Crafted Prompt 2 + perturbed model (unrelated control)
4. Saves results with "adhoc" suffix in jailbreak-results/
"""

# ============================================================================
# CONFIGURE YOUR TEST HERE
# ============================================================================
CONCEPT_NAME = "Tekken"  # Must match a concept in best-tests/
QUESTION = "What are the names of Naruto's best friends?"
# ============================================================================

import torch
import numpy as np
import json
import os
import warnings
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
import copy
import random

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

# Model and HuggingFace Configuration
MODEL_ID = os.environ.get("GEMMA_MODEL", "google/gemma-3-1b-it")
DEVICE = "cuda:1"

# Generation Configuration
MAX_NEW_TOKENS = 150  # Longer answers for jailbreak testing

# Disable PyTorch optimizations that require newer CUDA capabilities
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_AUTOGRAD_CACHE'] = '1'
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


class JailbreakTester:
    def __init__(self, model_name: str = MODEL_ID, device: str = DEVICE):
        """Initialize the jailbreak tester with Gemma model"""
        self.device = device
        self.torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading model from HuggingFace: {model_name} on device: {device}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
            force_download=False
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            force_download=False
        )
        
        # Ensure pad_token_id is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Store original parameters
        print("Saving original model parameters...")
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone()
        
        print("✅ Model loaded and original parameters saved")
    
    def restore_original_params(self):
        """Restore model to original parameter state"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_params:
                    param.data.copy_(self.original_params[name])
    
    def apply_perturbation(self, vector_keys: List[str], perturbation_type: str, perturbation_param: float):
        """Apply specified perturbation to concept vectors"""
        print(f"  🔧 Applying {perturbation_type} perturbation (param={perturbation_param}) to {len(vector_keys)} vectors...")
        
        for vector_key in vector_keys:
            # Parse vector key (format: L{layer}_C{column})
            match = re.match(r'L(\d+)_C(\d+)', vector_key)
            if not match:
                print(f"    ⚠️  Skipping invalid vector key: {vector_key}")
                continue
                
            layer_idx = int(match.group(1))
            column_idx = int(match.group(2))
            
            # Find the parameter name for this layer
            param_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            
            # Apply perturbation
            try:
                self._modify_parameter(param_name, column_idx, perturbation_type, perturbation_param)
            except Exception as e:
                print(f"    ❌ Failed to modify {vector_key}: {e}")
    
    def _modify_parameter(self, param_name: str, dimension: int, perturbation_type: str, perturbation_param: float):
        """Modify a specific dimension of a parameter tensor"""
        
        def _apply_perturbation_inplace(tensor, dim, p_type, p_param):
            """Apply perturbation to tensor column in-place"""
            original_values = tensor[:, dim].clone()
            
            if p_type == 'gaussian':
                # Add Gaussian noise
                noise = torch.randn_like(tensor[:, dim]) * p_param
                tensor[:, dim] += noise
            elif p_type == 'scale':
                # Scale values
                tensor[:, dim] *= p_param
            elif p_type == 'ablation':
                # Set to specific value
                tensor[:, dim] = p_param
            
            return not torch.equal(original_values, tensor[:, dim])
        
        # Find parameter and apply modification
        found = False
        for name, param in self.model.named_parameters(recurse=True):
            if name == param_name:
                found = True
                changed = _apply_perturbation_inplace(param.data, dimension, perturbation_type, perturbation_param)
                if not changed:
                    print(f"    ⚠️  WARNING: Perturbation did not change parameter {name}")
                break
        
        if not found:
            raise KeyError(f"Parameter not found in model: {param_name}")
    
    def generate_answer(self, messages: List[Dict[str, str]], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Generate answer using Gemma model with proper chat template"""
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        
        # Move to model device
        model_device = next(self.model.parameters()).device
        try:
            inputs = inputs.to(model_device)
        except Exception:
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
        
        # Decode only newly generated tokens
        new_tokens = generation_output[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        return answer
    
    def load_prompt_templates(self):
        """Load crafted prompt templates"""
        script_dir = Path(__file__).parent
        
        # Crafted prompt 1 (previous "original")
        prompt_one_path = script_dir / "original-prompt.txt"
        with open(prompt_one_path, 'r', encoding='utf-8') as f:
            prompt_one_content = f.read().strip()
        
        # Crafted prompt 2 (previous "jailbreak")
        prompt_two_path = script_dir / "crafted-jailbreak.txt"
        with open(prompt_two_path, 'r', encoding='utf-8') as f:
            prompt_two_content = f.read().strip()
        
        return prompt_one_content, prompt_two_content
    
    def parse_prompt_template(self, template: str) -> Tuple[str, str]:
        """Parse template to extract system message and user prompt"""
        # Find system message between «SYS» and «/SYS»
        sys_match = re.search(r'«SYS»\s*(.*?)\s*«/SYS»', template, re.DOTALL)
        if not sys_match:
            raise ValueError("Could not find system message in template")
        
        system_content = sys_match.group(1).strip()
        
        # Extract user prompt (everything after «/SYS»)
        user_content = template.split('«/SYS»')[1].strip()
        # Remove [INST] and [/INST] tags if present (at start, end, or anywhere)
        user_content = re.sub(r'^\[INST\]\s*', '', user_content).strip()
        user_content = re.sub(r'\s*\[/INST\]$', '', user_content).strip()
        user_content = re.sub(r'\[INST\]', '', user_content).strip()
        user_content = re.sub(r'\[/INST\]', '', user_content).strip()
        
        return system_content, user_content
    
    def create_messages(self, template: str, concept_name: str, question: str) -> List[Dict[str, str]]:
        """Create messages list for model from template"""
        system_content, user_content = self.parse_prompt_template(template)
        
        # Replace placeholders
        system_content = system_content.replace("{CONCEPT_NAME}", concept_name)
        user_content = user_content.replace("{CONCEPT_NAME}", concept_name)
        user_content = user_content.replace("{QUESTION}", question)
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

def find_concept_config(best_tests_dir: Path, concept_name: str) -> Dict:
    """Find the first concept-specific configuration for the given concept"""
    validation_files = list(best_tests_dir.glob("*_validation_results*.json"))
    
    for file_path in validation_files:
        print(f"Searching for {concept_name} in: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Look for concept-specific configurations
        for config in data:
            if "is_concept_specific" not in config:
                continue
                
            if (config.get("is_concept_specific", False) and 
                config.get("concept_name") == concept_name):
                
                # Found matching config
                concept_info = {
                    "concept_name": config.get("concept_name"),
                    "vector_keys": config.get("vector_keys", []),
                    "perturbation_type": config.get("perturbation_type", "").replace("ensemble_", ""),
                    "perturbation_param": config.get("perturbation_param"),
                    "source_file": file_path.name,
                    "specificity_scores": {
                        "bleu": config.get("bleu_specificity", 0),
                        "rouge": config.get("rouge_specificity", 0), 
                        "sbert": config.get("sbert_specificity", 0),
                        "combined": config.get("combined_specificity", 0)
                    }
                }
                
                # Validate required fields
                if all(concept_info[k] is not None for k in ["concept_name", "vector_keys", "perturbation_type", "perturbation_param"]):
                    print(f"  ✅ Found concept-specific config: {concept_info['concept_name']} ({concept_info['perturbation_type']})")
                    return concept_info
    
    raise ValueError(f"No concept-specific configuration found for concept: {concept_name}")

def main():
    """Main ad hoc question testing function"""
    script_dir = Path(__file__).parent
    best_tests_dir = script_dir.parent / "concept-val-test" / "best-tests"
    results_dir = script_dir / "jailbreak-results"
    unrelated_questions_path = script_dir / "unrelated-questions.json"
    
    # Create results directory
    results_dir.mkdir(exist_ok=True)
    
    print(f"🎯 Ad Hoc Question Test")
    print(f"  Concept: {CONCEPT_NAME}")
    print(f"  Question: {QUESTION}")
    
    # Find concept configuration
    try:
        config = find_concept_config(best_tests_dir, CONCEPT_NAME)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Initialize jailbreak tester
    tester = JailbreakTester()
    
    # Load prompt templates
    crafted_prompt_1_template, crafted_prompt_2_template = tester.load_prompt_templates()

    # Load unrelated questions for control (if available)
    unrelated_questions: List[Dict[str, str]] = []
    if unrelated_questions_path.exists():
        try:
            with open(unrelated_questions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                unrelated_questions = [q for q in data if isinstance(q, dict) and "question" in q]
        except Exception as exc:
            print(f"⚠️  Failed to load unrelated questions: {exc}")
    if unrelated_questions:
        print(f"  Loaded {len(unrelated_questions)} unrelated control questions")
    else:
        print("  ⚠️  No unrelated control questions loaded")
    
    # Extract configuration details
    vector_keys = config["vector_keys"]
    perturbation_type = config["perturbation_type"]
    perturbation_param = config["perturbation_param"]
    source_file = config["source_file"]
    
    print(f"\n=== Testing concept: {CONCEPT_NAME} ===")
    print(f"  Perturbation: {perturbation_type} (param={perturbation_param})")
    print(f"  Vectors: {len(vector_keys)} vectors")
    print(f"  Source: {source_file}")
    
    # Prepare results
    results = {
        "concept_name": CONCEPT_NAME,
        "perturbation_type": perturbation_type,
        "perturbation_param": perturbation_param,
        "vector_keys": vector_keys,
        "num_vectors": len(vector_keys),
        "source_file": source_file,
        "adhoc_question": QUESTION,
        "results": {
            "question": QUESTION,
            "test_type": "concept_specific",
            "concept_specific_prompt_1_baseline": "",
            "concept_specific_prompt_1_perturbed": "",
            "concept_specific_prompt_2_baseline": "",
            "concept_specific_prompt_2_perturbed": "",
            "unrelated_question": "",
            "unrelated_test_type": "",
            "unrelated_prompt_1_perturbed": "",
            "unrelated_prompt_2_perturbed": ""
        }
    }
    
    print(f"\n📝 Testing question: {QUESTION}")
    
    # Test crafted prompt 1 (baseline)
    try:
        messages = tester.create_messages(crafted_prompt_1_template, CONCEPT_NAME, QUESTION)
        baseline_answer = tester.generate_answer(messages)
        results["results"]["concept_specific_prompt_1_baseline"] = baseline_answer
        print(f"  ✅ Concept-specific | Crafted Prompt 1 (baseline): {baseline_answer[:80]}...")
    except Exception as e:
        print(f"  ❌ Concept-specific | Crafted Prompt 1 (baseline) failed: {e}")
    
    # Test crafted prompt 2 (baseline)
    try:
        messages = tester.create_messages(crafted_prompt_2_template, CONCEPT_NAME, QUESTION)
        prompt2_baseline = tester.generate_answer(messages)
        results["results"]["concept_specific_prompt_2_baseline"] = prompt2_baseline
        print(f"  ✅ Concept-specific | Crafted Prompt 2 (baseline): {prompt2_baseline[:80]}...")
    except Exception as e:
        print(f"  ❌ Concept-specific | Crafted Prompt 2 (baseline) failed: {e}")
    
    # Apply perturbation
    try:
        tester.apply_perturbation(vector_keys, perturbation_type, perturbation_param)
        
        # Test crafted prompt 1 (perturbed)
        try:
            messages = tester.create_messages(crafted_prompt_1_template, CONCEPT_NAME, QUESTION)
            perturbed_answer = tester.generate_answer(messages)
            results["results"]["concept_specific_prompt_1_perturbed"] = perturbed_answer
            print(f"  ✅ Concept-specific | Crafted Prompt 1 (perturbed): {perturbed_answer[:80]}...")
        except Exception as e:
            print(f"  ❌ Concept-specific | Crafted Prompt 1 (perturbed) failed: {e}")
        
        # Test crafted prompt 2 (perturbed)
        try:
            messages = tester.create_messages(crafted_prompt_2_template, CONCEPT_NAME, QUESTION)
            prompt2_perturbed = tester.generate_answer(messages)
            results["results"]["concept_specific_prompt_2_perturbed"] = prompt2_perturbed
            print(f"  ✅ Concept-specific | Crafted Prompt 2 (perturbed): {prompt2_perturbed[:80]}...")
        except Exception as e:
            print(f"  ❌ Concept-specific | Crafted Prompt 2 (perturbed) failed: {e}")

        # Optional unrelated control tests (perturbed model only)
        if unrelated_questions:
            unrelated_pick = random.choice(unrelated_questions)
            unrelated_question = unrelated_pick.get("question", "")
            results["results"]["unrelated_question"] = unrelated_question
            results["results"]["unrelated_test_type"] = "unrelated"
            if unrelated_question:
                print(f"  🔁 Unrelated control question: {unrelated_question}")
                try:
                    messages = tester.create_messages(crafted_prompt_1_template, CONCEPT_NAME, unrelated_question)
                    unrelated_prompt1 = tester.generate_answer(messages)
                    results["results"]["unrelated_prompt_1_perturbed"] = unrelated_prompt1
                    print(f"    ✅ Unrelated | Crafted Prompt 1 (perturbed): {unrelated_prompt1[:80]}...")
                except Exception as e:
                    print(f"    ❌ Unrelated | Crafted Prompt 1 (perturbed) failed: {e}")

                try:
                    messages = tester.create_messages(crafted_prompt_2_template, CONCEPT_NAME, unrelated_question)
                    unrelated_prompt2 = tester.generate_answer(messages)
                    results["results"]["unrelated_prompt_2_perturbed"] = unrelated_prompt2
                    print(f"    ✅ Unrelated | Crafted Prompt 2 (perturbed): {unrelated_prompt2[:80]}...")
                except Exception as e:
                    print(f"    ❌ Unrelated | Crafted Prompt 2 (perturbed) failed: {e}")
            else:
                print("  ⚠️  Selected unrelated question is empty; skipping control prompts")
        
        # Restore model
        tester.restore_original_params()
        
    except Exception as e:
        print(f"  ❌ Perturbation failed: {e}")
        # Restore model in case of error
        tester.restore_original_params()
    
    # Save results with adhoc suffix
    output_file = results_dir / f"{CONCEPT_NAME}_{perturbation_type}_adhoc_jailbreak_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"🎉 Ad hoc question testing complete!")

if __name__ == "__main__":
    main()