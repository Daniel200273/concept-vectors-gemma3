#!/usr/bin/env python3
"""
Causal Validation of Concept Vectors via Gaussian Noise (Gemma 3 1B compatible)

Goal:
- Validate candidate concept vectors by testing if damaging them with Gaussian noise
  reduces model performance on questions about the target concept, but does not
  affect unrelated concepts.

Features:
- Loads a Gemma model (default: "google/gemma-1b" per request; override via --model)
- Identifies concept vectors by layer and neuron index (or vector_key like L07_C0820)
- Injects Gaussian noise into the corresponding down_proj column in-place
- Evaluates before/after on target and unrelated QA sets
- Computes BLEU (sacrebleu) and ROUGE-L (rouge_score)
- Restores original weights to avoid cumulative effects
- Outputs a CSV/JSON summary with selection criterion to an output directory

Expected QA file (use --qa-file only): qa.json grouped per concept
[
  {
    "concept": "Blockchain",
    "category": "Technology",
    "qa": [ {"q": "...", "a": "..."}, ... ]
  },
  ...
]
(Note: answers in qa.json were generated beforehand; the script compares post-perturbation
answers against these references and does not generate a pre-perturbation baseline.)

Selection criteria implemented:
- target_bleu_drop - unrelated_bleu_drop > diff_threshold (default 0.2)
- unrelated_bleu_drop <= max_unrelated_drop (default 0.05)
"""

import os
import json
import math
import argparse
from typing import List, Dict, Tuple, Optional, Any

import torch
import numpy as np

# Require dependencies (no fallbacks)
import sacrebleu  # type: ignore
from rouge_score import rouge_scorer  # type: ignore
import pandas as pd  # type: ignore

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as transformers_logging

# Silence non-actionable generation warnings (e.g. invalid generation flags like 'top_k')
transformers_logging.set_verbosity_error()

# Env defaults to match repo
os.environ.setdefault("HF_HOME", "/media/hdd/usr/martinelli/.cache/huggingface")
os.environ.setdefault("HF_TOKEN", "")
# Disable compilation for older GPUs (CUDA capability < 7.0)
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# -----------------------------
# Utility: generation
# -----------------------------
def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 200, deterministic: bool = True) -> str:
    # Match generation kwargs strategy from generate_qa_from_pool.py exactly
    messages = [{"role": "user", "content": prompt}]
    
    # Build generation kwargs compatible with model's generation config
    gen_cfg = getattr(model, "generation_config", None) or getattr(model, "config", None)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not deterministic,
    }
    # Add optional params only if supported by generation config
    if not deterministic and hasattr(gen_cfg, "temperature"):
        gen_kwargs["temperature"] = 0.7
    if hasattr(gen_cfg, "top_p"):
        gen_kwargs["top_p"] = 1.0
    if hasattr(gen_cfg, "repetition_penalty"):
        gen_kwargs["repetition_penalty"] = 1.05

    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    else:
        raise RuntimeError("Tokenizer must support apply_chat_template (use an Instruct-tuned Gemma tokenizer).")

# -----------------------------
# Metrics
# -----------------------------
def compute_bleu(preds: List[str], refs: List[str]) -> float:
    # sacrebleu expects references as list of lists
    return float(sacrebleu.corpus_bleu(preds, [refs]).score) / 100.0


def compute_rouge_l(preds: List[str], refs: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return float(np.mean(scores)) if scores else 0.0

# -----------------------------
# Noise injection and restore
# -----------------------------
def parse_vector_key(vector_key: str) -> Tuple[int, int]:
    """Parse a vector key like 'L17_C1767' into (layer_idx, neuron_idx)."""
    parts = vector_key.split('_')
    if len(parts) != 2 or not parts[0].startswith('L') or not parts[1].startswith('C'):
        raise ValueError(f"Invalid vector key format: {vector_key}. Expected format: 'L##_C####'")
    
    layer_idx = int(parts[0][1:])  # Remove 'L' prefix
    neuron_idx = int(parts[1][1:])  # Remove 'C' prefix
    return layer_idx, neuron_idx


def vector_from_location(model, layer_idx: int, neuron_idx: int) -> Tuple[torch.Tensor, torch.nn.Module]:
    layer = model.model.layers[layer_idx]
    down_proj = layer.mlp.down_proj
    # Column view (hidden_size x intermediate_size) -> select column
    col = down_proj.weight[:, neuron_idx]
    return col, down_proj


def inject_noise_to_vector(model, layer: int, index: int, sigma: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise N(0, sigma) to down_proj column in-place. Returns a copy of original column."""
    with torch.no_grad():
        col, down_proj = vector_from_location(model, layer, index)
        original = col.data.clone()
        noise = torch.randn_like(col) * float(sigma)
        col.add_(noise)
        return original


def restore_vector(model, layer: int, index: int, original_col: torch.Tensor) -> None:
    with torch.no_grad():
        col, _ = vector_from_location(model, layer, index)
        col.copy_(original_col)

# -----------------------------
# Evaluation helpers
# -----------------------------
# Keep loader that accepts ["q1", ...] or [{"q": "..."}] though pool is preferred

def load_questions(path: Optional[str], default_questions: List[str]) -> List[str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], str):
                return [str(x) for x in data]
            elif data and isinstance(data[0], dict):
                out: List[str] = []
                for item in data:
                    q = item.get("q") or item.get("question")
                    if q:
                        out.append(str(q))
                return out
    return list(default_questions)


def generate_answers_for_questions(model, tokenizer, questions: List[str], deterministic: bool = True) -> List[str]:
    """Generate answers for a list of questions sequentially."""
    answers = []
    for q in questions:
        answer = generate_answer(tokenizer, model, q, deterministic=deterministic)
        answers.append(answer)
    return answers


def load_grouped_qa(path: str) -> List[Dict[str, Any]]:
    """Load qa.json grouped per concept: [{concept, category, qa:[{q,a}, ...]}, ...]."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"qa.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("qa.json must be a JSON list of concept entries")
    # Basic validation
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        qa_list = item.get("qa") or []
        if not isinstance(qa_list, list):
            continue
        cleaned.append(item)
    if not cleaned:
        raise ValueError("qa.json contains no valid concept QA entries")
    return cleaned

# -----------------------------
# Main routine
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Causal validation of concept vectors via Gaussian noise")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it", help="HF model id (default: google/gemma-1b)")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device map or cuda:N/cpu")
    # Concept and vectors
    parser.add_argument("--concept", type=str, default=None, help="Validated concept name; if omitted, uses first concept in qa.json")
    parser.add_argument("--results-json", type=str, default="../projection/value_vector_results/projection_analysis_results.json", help="Path to projection_analysis_results.json for vector selection")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top candidates to test for the concept")
    parser.add_argument("--vector-keys", type=str, nargs="*", default=None, help="Explicit vector keys like L07_C0820 (overrides concept/top-n)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Gaussian noise std dev")
    # QA source
    parser.add_argument("--qa-file", type=str, default="qa.json", help="Path to qa.json grouped per concept with model-generated answers")
    # Output directory (auto filenames)
    parser.add_argument("--out-dir", type=str, default="concept_val_results", help="Directory to write outputs (CSV/JSON)")
    # Thresholds
    parser.add_argument("--diff-threshold", type=float, default=0.2, help="Required (target_drop - unrelated_drop) > threshold")
    parser.add_argument("--max-unrelated-drop", type=float, default=0.05, help="Maximum allowed unrelated BLEU drop")
    args = parser.parse_args()

    # Ensure output directory
    os.makedirs(args.out_dir, exist_ok=True)
    out_base = f"{(args.concept or 'auto').replace(' ', '_')}_sigma{args.sigma}"
    out_csv = os.path.join(args.out_dir, f"{out_base}.csv")
    out_json = os.path.join(args.out_dir, f"{out_base}.json")

    # Load model
    print(f"🔄 Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN") or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") or None,
    ).eval()
    print(f"✅ Loaded on {model.device}")

    # Load QA file and pick concepts: first is target, next five are unrelated
    concepts_qa = load_grouped_qa(args.qa_file)
    if len(concepts_qa) < 6:
        raise SystemExit("qa.json must contain at least 6 concept entries (1 target + 5 unrelated)")

    target_entry = concepts_qa[0]
    unrelated_entries = concepts_qa[1:6]

    target_concept_name = target_entry.get("concept") or (args.concept or "")
    if not target_concept_name:
        raise SystemExit("Target concept name is missing (provide --concept or ensure qa.json has it)")

    # If user provided --concept, prefer it for vector selection; otherwise use first concept name
    selected_concept = args.concept or target_concept_name

    target_questions = [pair.get("q", "") for pair in target_entry.get("qa", []) if isinstance(pair, dict) and pair.get("q")]
    baseline_target_answers = [pair.get("a", "") for pair in target_entry.get("qa", []) if isinstance(pair, dict) and pair.get("a")]

    # Build unrelated by concatenating QA from up to 5 following concepts
    unrel_questions: List[str] = []
    baseline_unrel_answers: List[str] = []
    unrelated_concepts_used: List[str] = []
    for entry in unrelated_entries:
        cname = entry.get("concept")
        if cname:
            unrelated_concepts_used.append(cname)
        qa_list = entry.get("qa", [])
        for pair in qa_list:
            if isinstance(pair, dict) and pair.get("q") and pair.get("a"):
                unrel_questions.append(pair["q"])
                baseline_unrel_answers.append(pair["a"])

    if not target_questions or not baseline_target_answers:
        raise SystemExit("No target Q/A found in qa.json first concept entry")
    if not unrel_questions or not baseline_unrel_answers:
        raise SystemExit("No unrelated Q/A found in qa.json (next concepts)")

    print("\n=" * 30)
    print(f"🟢 Using qa.json baseline answers | target='{selected_concept}' q={len(target_questions)} | unrelated concepts={len(unrelated_concepts_used)} total_q={len(unrel_questions)}")

    # Determine candidate vector keys
    vector_keys: List[str] = []
    if args.vector_keys:
        vector_keys = list(args.vector_keys)
    elif selected_concept and os.path.exists(args.results_json):
        with open(args.results_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        analysis = results.get("concept_analyses", {}).get(selected_concept)
        if not analysis or "top_candidates" not in analysis:
            raise SystemExit(f"No analysis/top_candidates for concept: {selected_concept}")
        top = analysis["top_candidates"][: args.top_n]
        vector_keys = [c["vector_key"] for c in top]
    else:
        raise SystemExit("Provide --vector-keys or (--results-json containing the selected concept)")

    print(f"🎯 Testing {len(vector_keys)} candidate vectors: {vector_keys}")

    rows = []
    detailed = []

    for vk in vector_keys:
        layer_idx, neuron_idx = parse_vector_key(vk)
        print("\n" + "=" * 60)
        print(f"🔧 Injecting noise into {vk} (L{layer_idx}, C{neuron_idx}), sigma={args.sigma}")

        # Inject noise
        original_col = inject_noise_to_vector(model, layer_idx, neuron_idx, sigma=args.sigma)

        # Evaluate after: generate answers under noise
        after_target_answers = generate_answers_for_questions(model, tokenizer, target_questions, deterministic=True)
        after_unrel_answers = generate_answers_for_questions(model, tokenizer, unrel_questions, deterministic=True)

        # Print truncated sample answers for debugging
        def truncate_text(text: str, max_len: int = 80) -> str:
            return text[:max_len] + "..." if len(text) > max_len else text
        
        print(f"📝 Sample target answers (after noise):", flush=True)
        for i, (q, a) in enumerate(zip(target_questions[:2], after_target_answers[:2])):
            baseline_a = baseline_target_answers[i] if i < len(baseline_target_answers) else "N/A"
            print(f"  Q{i+1}: {truncate_text(q, 60)}", flush=True)
            print(f"  A{i+1} (baseline): {truncate_text(baseline_a, 70)}", flush=True)
            print(f"  A{i+1} (noisy):    {truncate_text(a, 70)}", flush=True)
            print(flush=True)
        
        print(f"📝 Sample unrelated answers (after noise):", flush=True)
        for i, (q, a) in enumerate(zip(unrel_questions[:2], after_unrel_answers[:2])):
            baseline_a = baseline_unrel_answers[i] if i < len(baseline_unrel_answers) else "N/A"
            print(f"  Q{i+1}: {truncate_text(q, 60)}", flush=True)
            print(f"  A{i+1} (baseline): {truncate_text(baseline_a, 70)}", flush=True)
            print(f"  A{i+1} (noisy):    {truncate_text(a, 70)}", flush=True)
            print(flush=True)

        # Restore
        restore_vector(model, layer_idx, neuron_idx, original_col)

        # Similarities and drops (1 - similarity) against qa.json references
        target_bleu_sim = compute_bleu(after_target_answers, baseline_target_answers)
        target_rouge_sim = compute_rouge_l(after_target_answers, baseline_target_answers)
        unrel_bleu_sim = compute_bleu(after_unrel_answers, baseline_unrel_answers)
        unrel_rouge_sim = compute_rouge_l(after_unrel_answers, baseline_unrel_answers)

        target_bleu_drop = float(1.0 - target_bleu_sim)
        unrelated_bleu_drop = float(1.0 - unrel_bleu_sim)
        target_rouge_drop = float(1.0 - target_rouge_sim)
        unrelated_rouge_drop = float(1.0 - unrel_rouge_sim)

        selected = (target_bleu_drop - unrelated_bleu_drop) > args.diff_threshold and (unrelated_bleu_drop <= args.max_unrelated_drop)

        row = {
            "vector_id": vk,
            "target_bleu_diff": target_bleu_drop,
            "unrelated_bleu_diff": unrelated_bleu_drop,
            "selected": bool(selected),
        }
        rows.append(row)

        detailed.append({
            "vector_id": vk,
            "layer": layer_idx,
            "neuron": neuron_idx,
            "sigma": args.sigma,
            "refs": {"target_q": len(target_questions), "unrel_q": len(unrel_questions)},
            "sims": {"target": {"bleu": target_bleu_sim, "rougeL": target_rouge_sim}, "unrelated": {"bleu": unrel_bleu_sim, "rougeL": unrel_rouge_sim}},
            "drops": {"target": {"bleu": target_bleu_drop, "rougeL": target_rouge_drop}, "unrelated": {"bleu": unrelated_bleu_drop, "rougeL": unrelated_rouge_drop}},
            "selected": selected,
        })

        print(f"➡️  Target BLEU drop: {target_bleu_drop:.3f} | Unrelated BLEU drop: {unrelated_bleu_drop:.3f} | Selected={selected}")

    # Save outputs (require pandas)
    df = pd.DataFrame(rows, columns=["vector_id", "target_bleu_diff", "unrelated_bleu_diff", "selected"])
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved CSV: {out_csv}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": args.model,
                "sigma": args.sigma,
                "diff_threshold": args.diff_threshold,
                "max_unrelated_drop": args.max_unrelated_drop,
                "vector_keys": vector_keys,
                "qa_file": args.qa_file,
                "target_concept": selected_concept,
                "unrelated_concepts_used": unrelated_concepts_used,
                "out_dir": args.out_dir,
                "out_csv": out_csv,
                "out_json": out_json,
            },
            "results": detailed,
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved JSON: {out_json}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
