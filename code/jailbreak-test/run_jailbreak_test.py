#!/usr/bin/env python3
"""
Jailbreak Testing Script

This script:
1. Extracts concept-specific configurations from validation results in best-tests/
2. Loads Gemma-3-1b-it model and saves original parameters
3. For each concept-specific config:
   - Applies perturbations to concept vectors
   - Extracts matching questions from QA files
    - Tests both crafted prompt 1 and crafted prompt 2 templates
   - Saves results to jailbreak-results/

The script follows the same model loading and perturbation methods as ensemble_concept_validation_layerwise.py
"""

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

# Set seeds for reproducibility

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
            torch_dtype=torch.bfloat16,
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
        """Load crafted prompt templates used for evaluation"""
        script_dir = Path(__file__).parent
        
        # Crafted prompt 1 uses the previous "original" template
        prompt_one_path = script_dir / "original-prompt.txt"
        with open(prompt_one_path, 'r', encoding='utf-8') as f:
            prompt_one_content = f.read().strip()
        
        # Crafted prompt 2 uses the previous "crafted jailbreak" template
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

def load_validation_results(best_tests_dir: Path) -> List[Dict]:
    """Load validation results and filter for concept-specific configurations"""
    results = []
    
    # Find all validation result files
    validation_files = list(best_tests_dir.glob("*_validation_results*.json"))
    
    for file_path in validation_files:
        print(f"Loading validation results from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter for concept-specific configurations
        for config in data:
            # Check if this config is concept-specific
            # Require explicit is_concept_specific field - no fallbacks
            if "is_concept_specific" not in config:
                raise ValueError(f"Missing 'is_concept_specific' field in config from {file_path.name}")
            
            is_concept_specific = config.get("is_concept_specific", False)
            
            if is_concept_specific:
                # Extract required fields
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
                    results.append(concept_info)
                    print(f"  ✅ Found concept-specific config: {concept_info['concept_name']} ({concept_info['perturbation_type']})")
                else:
                    print(f"  ⚠️  Skipping incomplete config: {concept_info}")
    
    print(f"Found {len(results)} concept-specific configurations")
    return results

def load_questions_for_concept(best_tests_dir: Path, concept_name: str, source_file: str) -> List[Dict]:
    """Load questions for a specific concept from matching QA file"""
    # Extract version from source file (e.g., -v3, -v4)
    version_match = re.search(r'-v(\d+)', source_file)
    if version_match:
        version = f"-v{version_match.group(1)}"
        qa_file = best_tests_dir / f"qa-generated{version}.json"
    else:
        qa_file = best_tests_dir / "qa-generated.json"
    
    if not qa_file.exists():
        print(f"  ⚠️  QA file not found: {qa_file}")
        return []
    
    print(f"  Loading questions from: {qa_file}")
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # Find matching concept
    for concept_data in qa_data:
        if concept_data.get("concept") == concept_name:
            return concept_data.get("qa", [])
    
    print(f"  ⚠️  No questions found for concept: {concept_name}")
    return []


def load_unrelated_questions(data_path: Path) -> List[Dict[str, str]]:
    """Load unrelated questions used for perturbed control checks."""
    if not data_path.exists():
        print(f"⚠️  Unrelated questions file not found: {data_path}")
        return []

    with open(data_path, "r", encoding="utf-8") as f:
        try:
            questions = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"⚠️  Failed to parse unrelated questions file: {exc}")
            return []

    if not isinstance(questions, list):
        print("⚠️  Unexpected format for unrelated questions; expected a list")
        return []

    valid_questions: List[Dict[str, str]] = []
    for item in questions:
        if isinstance(item, dict) and "question" in item:
            valid_questions.append(item)

    if not valid_questions:
        print("⚠️  No valid unrelated questions found in file")

    return valid_questions

def main():
    """Main jailbreak testing function"""
    script_dir = Path(__file__).parent
    best_tests_dir = script_dir.parent / "concept-val-test" / "best-tests"
    results_dir = script_dir / "jailbreak-results"
    unrelated_questions_path = script_dir / "unrelated-questions.json"
    
    # Create results directory
    results_dir.mkdir(exist_ok=True)
    
    # Load validation results
    concept_configs = load_validation_results(best_tests_dir)
    
    if not concept_configs:
        print("No concept-specific configurations found!")
        return

    unrelated_questions = load_unrelated_questions(unrelated_questions_path)
    if unrelated_questions:
        print(f"Loaded {len(unrelated_questions)} unrelated control questions")
    else:
        print("⚠️  Proceeding without unrelated control questions")
    
    # Initialize jailbreak tester
    tester = JailbreakTester()
    
    # Load prompt templates
    crafted_prompt_1_template, crafted_prompt_2_template = tester.load_prompt_templates()
    
    print(f"\n🎯 Testing {len(concept_configs)} concept-specific configurations...")
    
    # Track configs per concept for indexing
    concept_counts = {}
    for config in concept_configs:
        concept_name = config["concept_name"]
        concept_counts[concept_name] = concept_counts.get(concept_name, 0) + 1
    
    # Show concept breakdown
    for concept_name, count in concept_counts.items():
        if count > 1:
            print(f"  📋 {concept_name}: {count} configurations")
        else:
            print(f"  📋 {concept_name}: {count} configuration")
    
    # Track current index per concept
    concept_indices = {}
    
    # Process each concept configuration
    for i, config in enumerate(concept_configs, 1):
        concept_name = config["concept_name"]
        vector_keys = config["vector_keys"]
        perturbation_type = config["perturbation_type"]
        perturbation_param = config["perturbation_param"]
        source_file = config["source_file"]
        
        # Get current index for this concept
        concept_indices[concept_name] = concept_indices.get(concept_name, 0) + 1
        config_index = concept_indices[concept_name]
        
        print(f"\n=== [{i}/{len(concept_configs)}] Testing concept: {concept_name} (config {config_index}) ===")
        print(f"  Perturbation: {perturbation_type} (param={perturbation_param})")
        print(f"  Vectors: {len(vector_keys)} vectors")
        
        # Load questions for this concept
        questions = load_questions_for_concept(best_tests_dir, concept_name, source_file)
        if not questions:
            print(f"  ⚠️  Skipping {concept_name} - no questions found")
            continue
        
        print(f"  Questions: {len(questions)} loaded")
        
        # Prepare results for this concept
        concept_results = {
            "concept_name": concept_name,
            "perturbation_type": perturbation_type,
            "perturbation_param": perturbation_param,
            "vector_keys": vector_keys,
            "num_vectors": len(vector_keys),
            "source_file": source_file,
            "results": []
        }
        
        # Test each question
        for j, qa_pair in enumerate(questions, 1):
            question = qa_pair.get("q", "")
            if not question:
                continue
            
            print(f"    [{j}/{len(questions)}] Testing question: {question}")
            
            question_result = {
                "question": question,
                "concept_specific_prompt_1_baseline": "",
                "concept_specific_prompt_1_perturbed": "",
                "concept_specific_prompt_2_baseline": "",
                "concept_specific_prompt_2_perturbed": "",
                "unrelated_question": "",
                "unrelated_prompt_1_baseline": "",
                "unrelated_prompt_1_perturbed": "",
                "unrelated_prompt_2_baseline": "",
                "unrelated_prompt_2_perturbed": ""
            }
            
            unrelated_question = ""
            if unrelated_questions:
                unrelated_entry = random.choice(unrelated_questions)
                unrelated_question = unrelated_entry.get("question", "")
                if unrelated_question:
                    question_result["unrelated_question"] = unrelated_question
                    print(f"      🔁 Unrelated control question selected: {unrelated_question}")
                else:
                    print("      ⚠️  Selected unrelated question was empty, skipping control prompts")

        # Test crafted prompt 1 (baseline)
            try:
                messages = tester.create_messages(crafted_prompt_1_template, concept_name, question)
                baseline_answer = tester.generate_answer(messages)
                question_result["concept_specific_prompt_1_baseline"] = baseline_answer
                print(f"      ✅ Concept-specific | Crafted Prompt 1 (baseline): {baseline_answer[:50]}...")
            except Exception as e:
                print(f"      ❌ Concept-specific | Crafted Prompt 1 (baseline) failed: {e}")
            
            # Test crafted prompt 2 (baseline)
            try:
                messages = tester.create_messages(crafted_prompt_2_template, concept_name, question)
                prompt2_baseline = tester.generate_answer(messages)
                question_result["concept_specific_prompt_2_baseline"] = prompt2_baseline
                print(f"      ✅ Concept-specific | Crafted Prompt 2 (baseline): {prompt2_baseline[:50]}...")
            except Exception as e:
                print(f"      ❌ Concept-specific | Crafted Prompt 2 (baseline) failed: {e}")

            # Unrelated baseline evaluations (if available)
            if unrelated_question:
                try:
                    messages = tester.create_messages(crafted_prompt_1_template, concept_name, unrelated_question)
                    unrelated_baseline_1 = tester.generate_answer(messages)
                    question_result["unrelated_prompt_1_baseline"] = unrelated_baseline_1
                    print(f"      ✅ Unrelated | Crafted Prompt 1 (baseline): {unrelated_baseline_1[:50]}...")
                except Exception as e:
                    print(f"      ❌ Unrelated | Crafted Prompt 1 (baseline) failed: {e}")

                try:
                    messages = tester.create_messages(crafted_prompt_2_template, concept_name, unrelated_question)
                    unrelated_baseline_2 = tester.generate_answer(messages)
                    question_result["unrelated_prompt_2_baseline"] = unrelated_baseline_2
                    print(f"      ✅ Unrelated | Crafted Prompt 2 (baseline): {unrelated_baseline_2[:50]}...")
                except Exception as e:
                    print(f"      ❌ Unrelated | Crafted Prompt 2 (baseline) failed: {e}")
            
            # Apply perturbation
            try:
                tester.apply_perturbation(vector_keys, perturbation_type, perturbation_param)
                
                # Test crafted prompt 1 (perturbed)
                try:
                    messages = tester.create_messages(crafted_prompt_1_template, concept_name, question)
                    perturbed_answer = tester.generate_answer(messages)
                    question_result["concept_specific_prompt_1_perturbed"] = perturbed_answer
                    print(f"      ✅ Concept-specific | Crafted Prompt 1 (perturbed): {perturbed_answer[:50]}...")
                except Exception as e:
                    print(f"      ❌ Concept-specific | Crafted Prompt 1 (perturbed) failed: {e}")
                
                # Test crafted prompt 2 (perturbed)
                try:
                    messages = tester.create_messages(crafted_prompt_2_template, concept_name, question)
                    prompt2_perturbed = tester.generate_answer(messages)
                    question_result["concept_specific_prompt_2_perturbed"] = prompt2_perturbed
                    print(f"      ✅ Concept-specific | Crafted Prompt 2 (perturbed): {prompt2_perturbed[:50]}...")
                except Exception as e:
                    print(f"      ❌ Concept-specific | Crafted Prompt 2 (perturbed) failed: {e}")

                # Ask a random unrelated question with the perturbed model
                if unrelated_question:
                    try:
                        messages = tester.create_messages(crafted_prompt_1_template, concept_name, unrelated_question)
                        unrelated_prompt1 = tester.generate_answer(messages)
                        question_result["unrelated_prompt_1_perturbed"] = unrelated_prompt1
                        print(f"      ✅ Unrelated | Crafted Prompt 1 (perturbed): {unrelated_prompt1[:50]}...")
                    except Exception as e:
                        print(f"      ❌ Unrelated | Crafted Prompt 1 (perturbed) failed: {e}")

                    try:
                        messages = tester.create_messages(crafted_prompt_2_template, concept_name, unrelated_question)
                        unrelated_prompt2 = tester.generate_answer(messages)
                        question_result["unrelated_prompt_2_perturbed"] = unrelated_prompt2
                        print(f"      ✅ Unrelated | Crafted Prompt 2 (perturbed): {unrelated_prompt2[:50]}...")
                    except Exception as e:
                        print(f"      ❌ Unrelated | Crafted Prompt 2 (perturbed) failed: {e}")
                
                # Restore model
                tester.restore_original_params()
                
            except Exception as e:
                print(f"      ❌ Perturbation failed: {e}")
                # Restore model in case of error
                tester.restore_original_params()
            
            # Add question result
            concept_results["results"].append(question_result)
        
        # Save results for this concept with index to handle multiple configs
        if concept_counts[concept_name] > 1:
            output_file = results_dir / f"{concept_name}_{perturbation_type}_config{config_index:02d}_jailbreak_results.json"
        else:
            output_file = results_dir / f"{concept_name}_{perturbation_type}_jailbreak_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(concept_results, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Results saved to: {output_file}")
    
    print(f"\n🎉 Jailbreak testing complete! Results saved to: {results_dir}")

if __name__ == "__main__":
    main()