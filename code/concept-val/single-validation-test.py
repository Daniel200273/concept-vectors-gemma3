#!/usr/bin/env python3
"""
Single Concept Vector Validation Test

This script validates concept vectors by:
1. Loading Gemma 3 1B model
2. Reading value vector analysis results
3. Applying Gaussian noise (or ablation) to a selected concept vector
4. Testing model responses before and after modification

Goal: observe effect of perturbing a candidate concept vector direction.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

# NEW: metrics imports
try:
    import sacrebleu  # type: ignore
    from rouge_score import rouge_scorer  # type: ignore
except Exception:
    sacrebleu = None
    rouge_scorer = None

# Configure environment for HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your HuggingFace token")
    
os.environ["HF_TOKEN"] = HF_TOKEN
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(PRIVATE_HF_HOME, "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(PRIVATE_HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(PRIVATE_HF_HOME, "datasets")

# Disable PyTorch compilation for compatibility
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

class ConceptVectorValidator:
    """Validate concept vectors by testing model behavior before/after modification"""
    
    def __init__(self, device: str = "cuda:1", results_dir: str = "../projection/value_vector_results"):
        self.device = device
        self.results_dir = results_dir
        self.model = None
        self.tokenizer = None
        self.original_weights = {}
        self._hooks: List = []  # forward hooks for directional ablation
        
        # Results storage
        self.concept_results = None
        self.target_concept_info = None  # Holds selected concept and its analysis
    
    def load_model(self):
        """Load Gemma 3 1B model for causal language modeling"""
        print("🔄 Loading Gemma 3 1B model...")
        model_name = "google/gemma-3-1b-it"
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ["HF_TOKEN"]
        )
        
        # Load model for text generation with full precision
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use full precision
            device_map=self.device,
            trust_remote_code=True,
            token=os.environ["HF_TOKEN"],
            # Explicitly disable quantization
            load_in_8bit=False,
            load_in_4bit=False,
            quantization_config=None
        ).eval()
        
        print(f"✅ Model loaded on {self.device} with full precision")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_concept_results(self, preferred_concept: Optional[str] = None):
        """Load projection results and select a concept to test."""
        print("📚 Loading concept vector analysis results...")
        
        # Try final_concept_vectors.json first, then fall back to projection_analysis_results.json
        final_results_file = os.path.join(self.results_dir, "final_concept_vectors.json")
        analysis_results_file = os.path.join(self.results_dir, "projection_analysis_results.json")
        
        if os.path.exists(final_results_file):
            print("📁 Loading from final_concept_vectors.json")
            with open(final_results_file, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            
            # Convert final format to analysis format for compatibility
            self.concept_results = {
                'metadata': final_data.get('metadata', {}),
                'concept_analyses': {}
            }
            
            for concept_name, concept_data in final_data.get('concept_vectors', {}).items():
                # Convert best_candidate + alternative_candidates to top_candidates list
                top_candidates = []
                
                # Add best candidate
                best = concept_data.get('best_candidate', {})
                if best:
                    best_candidate = {
                        'vector_key': best.get('vector_key'),
                        'concept_activation_strength': best.get('concept_activation_strength'),
                        'vector_index': 0,  # placeholder
                        'scoring_info': best.get('concept_analysis', {})
                    }
                    top_candidates.append(best_candidate)
                
                # Add alternative candidates
                alternatives = concept_data.get('alternative_candidates', [])
                for alt in alternatives:
                    alt_candidate = {
                        'vector_key': alt.get('vector_key'),
                        'concept_activation_strength': alt.get('concept_activation_strength'),
                        'vector_index': alt.get('vector_index', 0),
                        'scoring_info': alt.get('scoring_info', {})
                    }
                    top_candidates.append(alt_candidate)
                
                self.concept_results['concept_analyses'][concept_name] = {
                    'top_candidates': top_candidates,
                    'num_concept_tokens': concept_data.get('concept_info', {}).get('num_tokens', 0),
                    'concept_tokens': concept_data.get('concept_info', {}).get('concept_tokens', [])
                }
                
        elif os.path.exists(analysis_results_file):
            print("📁 Loading from projection_analysis_results.json")
            with open(analysis_results_file, 'r', encoding='utf-8') as f:
                self.concept_results = json.load(f)
        else:
            raise FileNotFoundError(f"Neither final_concept_vectors.json nor projection_analysis_results.json found in {self.results_dir}")
        
        analyses = self.concept_results.get('concept_analyses', {})
        if not analyses:
            raise ValueError("No concept analyses found in results file")
        
        # Select concept
        concept_name = preferred_concept
        if concept_name is None or concept_name not in analyses or 'error' in analyses.get(concept_name, {}):
            # pick best by concept_activation_strength of top candidate
            best_concept = None
            best_score = -float('inf')
            for name, analysis in analyses.items():
                if 'error' in analysis:
                    continue
                if analysis.get('top_candidates'):
                    score = analysis['top_candidates'][0].get('concept_activation_strength', -float('inf'))
                    if score > best_score:
                        best_score = score
                        best_concept = name
            concept_name = best_concept
        
        if concept_name is None:
            raise ValueError("No valid concept found in results")
        
        self.target_concept_info = {
            'name': concept_name,
            'analysis': analyses[concept_name]
        }
        num_candidates = len(self.target_concept_info['analysis'].get('top_candidates', []))
        topk = self.target_concept_info['analysis'].get('top_tokens_k', self.concept_results.get('metadata', {}).get('top_tokens_k'))
        print(f"🎯 Target concept: {concept_name}")
        print(f"📊 Available candidate vectors: {num_candidates}")
        if topk:
            print(f"🔎 Ranking used top-{topk} tokens per candidate")
    
    def get_concept_vector_location(self, candidate_info: dict) -> Tuple[int, int]:
        """Get the layer and neuron indices of a candidate vector from vector_key like 'L07_C0820'"""
        vector_key = candidate_info['vector_key']
        parts = vector_key.split('_')
        layer_idx = int(parts[0][1:])  # after 'L'
        neuron_idx = int(parts[1][1:])  # after 'C'
        return layer_idx, neuron_idx
    
    def backup_original_weights(self, layer_idx: int):
        layer = self.model.model.layers[layer_idx]
        self.original_weights[layer_idx] = layer.mlp.down_proj.weight.data.clone()
        print(f"💾 Backed up original weights for layer {layer_idx}")
    
    def apply_vector_ablation(self, layer_idx: int, neuron_idx: int, mode: str = "zero", scale: float = 0.0):
        """Edit the concept vector (down_proj column).
        mode="zero": set the column to zero
        mode="scale": scale column by (1 - scale)
        mode="gaussian": add Gaussian noise ε ~ N(0, scale) elementwise
        """
        assert mode in ("zero", "scale", "gaussian"), "mode must be 'zero', 'scale', or 'gaussian'"
        mode_desc = f"{mode}{' (scale='+str(scale)+')' if mode in ('scale','gaussian') else ''}"
        print(f"🔧 Applying vector edit ({mode_desc}) to L{layer_idx}, C{neuron_idx}")
        self.backup_original_weights(layer_idx)
        layer = self.model.model.layers[layer_idx]
        down_proj = layer.mlp.down_proj
        with torch.no_grad():
            col = down_proj.weight.data[:, neuron_idx]
            original_norm = torch.norm(col).item()
            if mode == "zero":
                down_proj.weight.data[:, neuron_idx] = 0.0 * col
            elif mode == "scale":
                down_proj.weight.data[:, neuron_idx] = (1.0 - scale) * col
            else:  # gaussian noise
                noise = torch.randn_like(col) * float(scale)
                down_proj.weight.data[:, neuron_idx] = col + noise
            new_norm = torch.norm(down_proj.weight.data[:, neuron_idx]).item()
        print(f"  📊 Column norm: {original_norm:.4f} -> {new_norm:.4f}")
    
    def restore_original_weights(self, layer_idx: int):
        if layer_idx in self.original_weights:
            layer = self.model.model.layers[layer_idx]
            layer.mlp.down_proj.weight.data = self.original_weights[layer_idx]
            print(f"🔄 Restored original weights for layer {layer_idx}")
        else:
            print(f"⚠️ No backup found for layer {layer_idx}")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 150, do_sample: bool = False, temperature: float = 0.0) -> str:
        """Generate response using the chat template. Deterministic by default."""
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=1.0,
                    repetition_penalty=1.05
                )
            response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[Error: {str(e)}]"
    
    def create_concept_questions(self, concept_name: str) -> List[str]:
        """Create a small set of questions tailored to the concept name."""
        lower = concept_name.lower()
        if "programming" in lower:
            return [
                f"What is {concept_name}? Explain the key principles.",
                f"Give me an example of {concept_name} in practice."
            ]
        elif "harry potter" in lower:
            return [
                f"Who is the main character in {concept_name} and what is the setting?",
                f"Name key elements or terms associated with {concept_name}."
            ]
        else:
            return [
                f"What is {concept_name}? Provide a clear explanation.",
                f"Give me an example of {concept_name}."
            ]
    
    def apply_directional_residual_ablation(self, layer_idx: int, neuron_idx: int, scale: float = 1.0):
        """Apply directional ablation on the MLP output at a layer by removing the
        component along the candidate vector v (down_proj column) from activations.
        y' = y - scale * proj_v(y) where proj_v(y) = ((y·v̂) v̂).
        """
        layer = self.model.model.layers[layer_idx]
        with torch.no_grad():
            v = layer.mlp.down_proj.weight.data[:, neuron_idx].to(self.model.device)
            v_hat = v / (torch.norm(v) + 1e-8)
            v_hat = v_hat.to(dtype=self.model.dtype)  # match model dtype (fp16)
        print(f"🔧 Applying directional residual ablation at L{layer_idx} using v from C{neuron_idx} (scale={scale})")

        def hook_fn(module, inputs, output):
            # output shape: (batch, seq, hidden)
            y = output
            # proj scalar per token: (batch, seq)
            s = torch.einsum('bsh,h->bs', y, v_hat)
            # subtract projection
            y_new = y - scale * s[:, :, None] * v_hat[None, None, :]
            return y_new

        handle = layer.mlp.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
        self._hooks.append(handle)
        # No weight backup needed for directional hooks

    def remove_all_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        if self._hooks:
            print(f"🔌 Removed {len(self._hooks)} forward hook(s)")
        self._hooks = []

    def test_single_candidate(self, concept_name: str, candidate_info: dict, candidate_rank: int, ablation_mode: str = "directional", ablation_scale: float = 1.0, num_questions: int = 2) -> dict:
        """Test one candidate before/after ablation.
        ablation_mode: 'directional' (activation projection removal), 'zero', 'scale', or 'gaussian' (weight column noise with std=ablation_scale)
        """
        layer_idx, neuron_idx = self.get_concept_vector_location(candidate_info)
        activation_strength = candidate_info.get('concept_activation_strength', 0.0)
        print(f"\n🧪 Testing concept: {concept_name}")
        print(f"🔢 Candidate rank: #{candidate_rank}")
        print(f"📍 Vector: L{layer_idx}, C{neuron_idx} | strength: {activation_strength:.4f}")
        print(f"🔑 Vector key: {candidate_info['vector_key']}")
        
        # Create questions
        questions = self.create_concept_questions(concept_name)[:num_questions]
        print(f"❓ Number of test questions: {len(questions)}")
        
        # BEFORE (deterministic)
        print("\n" + "="*50)
        print("🟢 TESTING BEFORE MODIFICATION")
        print("="*50)
        original_responses = []
        for i, q in enumerate(questions, 1):
            print(f"\n[Q{i}] {q}")
            a = self.generate_response(q, do_sample=False, temperature=0.0)
            original_responses.append(a)
            print(f"[A{i}] {a}")
        
        # Apply intervention
        print("\n" + "="*50)
        print("🪓 APPLYING VECTOR ABLATION/NOISE")
        print("="*50)
        if ablation_mode == "directional":
            self.apply_directional_residual_ablation(layer_idx, neuron_idx, scale=ablation_scale)
        else:
            self.apply_vector_ablation(layer_idx, neuron_idx, mode=ablation_mode, scale=ablation_scale)
        
        # AFTER (deterministic)
        print("\n" + "="*50)
        print("🔴 TESTING AFTER MODIFICATION")
        print("="*50)
        modified_responses = []
        for i, q in enumerate(questions, 1):
            print(f"\n[Q{i}] {q}")
            a = self.generate_response(q, do_sample=False, temperature=0.0)
            modified_responses.append(a)
            print(f"[A{i}] {a}")
        
        # Restore original weights
        self.restore_original_weights(layer_idx)
        self.remove_all_hooks()
        
        # Collect results
        result = {
            "concept_name": concept_name,
            "candidate_info": candidate_info,
            "candidate_rank": candidate_rank,
            "ablation_mode": ablation_mode,
            "ablation_scale": ablation_scale,
            "questions": questions,
            "original_responses": original_responses,
            "modified_responses": modified_responses,
        }
        return result

# --- CLI ---
def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Single Concept Vector Validation Test")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device or device_map for model (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--results-dir", type=str, default="../projection/final_concept_vectors/", help="Directory with final_concept_vectors.json or projection_analysis_results.json")
    parser.add_argument("--concept", type=str, default=None, help="Preferred concept name to test (optional)")
    parser.add_argument("--candidate-idx", type=int, default=0, help="Index of candidate in top_candidates (0-based)")
    parser.add_argument("--ablation-mode", type=str, default="zero", choices=["directional", "zero", "scale", "gaussian"], help="Intervention mode")
    parser.add_argument("--ablation-scale", type=float, default=1.0, help="Scale for 'directional', 'scale', or 'gaussian' modes")
    parser.add_argument("--num-questions", type=int, default=2, help="Number of questions to ask per concept")
    parser.add_argument("--save", type=str, default=None, help="Path to save JSON result (file or directory). If directory, an auto name is used.")
    args = parser.parse_args(argv)

    validator = ConceptVectorValidator(device=args.device, results_dir=args.results_dir)
    validator.load_model()
    validator.load_concept_results(preferred_concept=args.concept)

    concept_name = validator.target_concept_info["name"]
    analysis = validator.target_concept_info["analysis"]
    top_candidates = analysis.get("top_candidates", [])
    if not top_candidates:
        raise SystemExit("No top_candidates available to test.")

    idx = max(0, min(args.candidate_idx, len(top_candidates) - 1))
    candidate_info = top_candidates[idx]

    result = validator.test_single_candidate(
        concept_name=concept_name,
        candidate_info=candidate_info,
        candidate_rank=idx + 1,
        ablation_mode=args.ablation_mode,
        ablation_scale=args.ablation_scale,
        num_questions=args.num_questions,
    )

    if args.save:
        save_path = Path(args.save)
        if save_path.is_dir() or str(args.save).endswith(os.sep):
            vec_key = str(candidate_info.get("vector_key", "unknown")).replace(os.sep, "-")
            fname = f"single_validation_{concept_name}_{vec_key}.json"
            out_path = save_path / fname
        else:
            out_path = save_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved results to {out_path}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
