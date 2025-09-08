#!/usr/bin/env python3
"""
Generate QA from concept questions pool using Gemma 3 1B

- Loads model: google/gemma-3-1b-it
- Reads questions from questions.json (same folder by default)
  Format:
  [
    {"concept": "...", "category": "...", "questions": ["q1", "q2", ...]},
    ...
  ]
- Generates an answer for every question via chat template
- Writes qa.json grouped per concept:
  {"concept": str, "category": str | None, "qa": [{"q": str, "a": str}, ...]}

Run:
  python generate_qa_from_pool.py

Optional env:
  HF_TOKEN: Hugging Face token if required for gated models
  HF_HOME:  Hugging Face cache dir
"""

import os
import json
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "questions.json")
DEFAULT_OUTPUT = os.path.join(HERE, "qa.json")
MODEL_ID = os.environ.get("GEMMA_MODEL", "google/gemma-3-1b-it")

PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(PRIVATE_HF_HOME, "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(PRIVATE_HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(PRIVATE_HF_HOME, "datasets")

# Get HuggingFace token from environment variable for security
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your HuggingFace token")
os.environ["HF_TOKEN"] = HF_TOKEN

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'


def load_pool(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"questions.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("questions.json must be a JSON list")
    return data


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 200, deterministic: bool = True) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
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
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**enc, **gen_kwargs)
            return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        return f"[Error: {e}]"


def main(in_path: str = DEFAULT_INPUT, out_path: str = DEFAULT_OUTPUT):
    print(f"🔄 Loading model: {MODEL_ID}")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # Use full precision to avoid quantization
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN") or None)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") or None,
        # Explicitly disable quantization
        load_in_8bit=False,
        load_in_4bit=False,
        quantization_config=None
    ).to(device).eval()
    print(f"✅ Loaded on {device} with full precision ({dtype})")

    pool = load_pool(in_path)

    grouped_results: List[Dict[str, Any]] = []
    total = 0
    for item in pool:
        concept: Optional[str] = item.get("concept")
        category: Optional[str] = item.get("category")
        qs: List[str] = []
        if isinstance(item.get("questions"), list):
            qs.extend([str(q) for q in item["questions"]])
        if isinstance(item.get("question"), str):
            qs.append(str(item["question"]))
        if not qs:
            continue
        qa_pairs: List[Dict[str, str]] = []
        for q in qs:
            total += 1
            print(f"[{total}] 📝 {concept or 'Unknown'} | {q[:80]}...")
            a = generate_answer(tokenizer, model, q, deterministic=True)
            qa_pairs.append({"q": q, "a": a})
        grouped_results.append({
            "concept": concept,
            "category": category,
            "qa": qa_pairs,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grouped_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Wrote QA for {len(grouped_results)} concepts ({total} Qs) to {out_path}")


if __name__ == "__main__":
    main()
