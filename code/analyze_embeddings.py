#!/usr/bin/env python3
"""
Analyze differences between input and output token embedding matrices for a model

This script attempts to load the input embeddings and the output (lm) embeddings
from a HuggingFace model (for example: google/gemma-3-1b-it) and computes per-token
L2 differences and cosine similarities between corresponding token vectors. It
prints summary statistics and the top tokens with the largest differences.

It also supports a --dry-run mode that generates small random matrices so you can
run and validate the script without downloading the full model.

Usage examples:
  python code/analyze_embeddings.py --model google/gemma-3-1b-it --device cpu --topk 50
  python code/analyze_embeddings.py --dry-run --vocab-size 5000 --hidden-size 128

Outputs:
  - Summary printed to stdout
  - Optional CSV/JSON saved to --out-dir when specified

"""
from __future__ import annotations
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
# Use the same private Hugging Face cache as other projection scripts
# This ensures models download to the user's private cache directory
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ['HF_HOME'] = PRIVATE_HF_HOME

# Respect HF token if provided externally
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your HuggingFace token")
os.environ["HF_TOKEN"] = HF_TOKEN

try:
    from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
except Exception:
    AutoModelForCausalLM = None  # type: ignore


def get_embeddings_from_model(model_name: str, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a model and return (input_embeddings, output_embeddings) as numpy arrays.

    Tries multiple fallbacks to find an output/head embedding matrix.
    """
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers import failed - ensure transformers is installed")

    # Load config first to inspect architecture without weights if possible
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model '{model_name}' on {device} (this may download large files)...")
    # Prefer the CausalLM class because it usually exposes lm_head
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": device} if device.startswith("cuda") or device == "cpu" else None,
    torch_dtype=torch.float32,
    # Explicitly disable quantized loading to ensure full-precision weights
    load_in_8bit=False,
    load_in_4bit=False,
    quantization_config=None,
    )

    # Input embeddings
    in_emb_module = None
    try:
        in_emb_module = model.get_input_embeddings()
    except Exception:
        # Fallbacks
        in_emb_module = getattr(model, "embed_tokens", None) or getattr(model, "wte", None)

    if in_emb_module is None:
        raise RuntimeError("Could not find input embedding module on the model")

    in_weights = in_emb_module.weight.detach().cpu().numpy()

    # Output embeddings: try get_output_embeddings, lm_head, or other common names
    out_weights = None
    try:
        out_mod = model.get_output_embeddings()
        if out_mod is not None and hasattr(out_mod, "weight"):
            out_weights = out_mod.weight.detach().cpu().numpy()
    except Exception:
        out_weights = None

    if out_weights is None:
        # Try common attribute names
        for name in ("lm_head", "embed_out", "output_projection", "head"):
            mod = getattr(model, name, None)
            if mod is not None and hasattr(mod, "weight"):
                out_weights = mod.weight.detach().cpu().numpy()
                break

    if out_weights is None:
        # As last resort, check for a tied linear layer with transposed shape
        # Some models store a final linear with weight shape (hidden, vocab)
        for name in dir(model):
            attr = getattr(model, name)
            if hasattr(attr, "weight"):
                w = getattr(attr, "weight")
                if isinstance(w, torch.Tensor):
                    arr = w.detach().cpu().numpy()
                    if arr.ndim == 2 and arr.shape[0] == in_weights.shape[1] and arr.shape[1] == in_weights.shape[0]:
                        # transposed linear: (hidden, vocab) -> transpose
                        out_weights = arr.T
                        break

    if out_weights is None:
        raise RuntimeError("Could not locate an output/LM head embedding matrix on the model")

    # Normalize shapes: if out has shape (hidden, vocab) transpose it
    if out_weights.ndim == 2 and out_weights.shape[1] == in_weights.shape[1] and out_weights.shape[0] == in_weights.shape[0]:
        # shapes already match: (vocab, hidden)
        pass
    elif out_weights.ndim == 2 and out_weights.shape[0] == in_weights.shape[1] and out_weights.shape[1] == in_weights.shape[0]:
        out_weights = out_weights.T
    else:
        # If vocab sizes differ, trim to the min common rows
        min_rows = min(in_weights.shape[0], out_weights.shape[0])
        if min_rows == 0:
            raise RuntimeError("Embeddings have incompatible shapes")
        in_weights = in_weights[:min_rows]
        out_weights = out_weights[:min_rows]

    return in_weights.astype(np.float32), out_weights.astype(np.float32)


def analyze_matrices(in_w: np.ndarray, out_w: np.ndarray, token_strings: Optional[dict] = None, topk: int = 50, out_dir: Optional[str] = None):
    """Compute per-token diffs and print/save summary and top-k lists."""
    assert in_w.ndim == 2 and out_w.ndim == 2
    n_tokens, dim = in_w.shape

    diffs = in_w - out_w
    l2_per_token = np.linalg.norm(diffs, axis=1)
    # cosine similarity between corresponding rows
    def cos_sim(a, b):
        an = np.linalg.norm(a, axis=1)
        bn = np.linalg.norm(b, axis=1)
        denom = an * bn
        # avoid division by zero
        denom = np.where(denom == 0, 1e-8, denom)
        sim = np.sum(a * b, axis=1) / denom
        return sim

    cos_per_token = cos_sim(in_w, out_w)

    stats = {
        "n_tokens": int(n_tokens),
        "dim": int(dim),
        "l2_mean": float(np.mean(l2_per_token)),
        "l2_std": float(np.std(l2_per_token)),
        "l2_median": float(np.median(l2_per_token)),
        "l2_max": float(np.max(l2_per_token)),
        "cos_mean": float(np.mean(cos_per_token)),
        "cos_std": float(np.std(cos_per_token)),
        "cos_min": float(np.min(cos_per_token)),
        "cos_max": float(np.max(cos_per_token)),
    }

    print("\nEmbedding comparison summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Top-k by L2 difference
    top_idx_l2 = np.argsort(-l2_per_token)[:topk]
    top_idx_cos_low = np.argsort(cos_per_token)[:topk]

    def token_name(i):
        if token_strings is None:
            return str(i)
        return token_strings.get(str(i), token_strings.get(i, str(i)))

    top_l2 = [(int(i), token_name(i), float(l2_per_token[i]), float(cos_per_token[i])) for i in top_idx_l2]
    top_cos = [(int(i), token_name(i), float(l2_per_token[i]), float(cos_per_token[i])) for i in top_idx_cos_low]

    print(f"\nTop {topk} tokens by L2 difference (token_id, token, L2, cosine):")
    for tid, tok, l2v, cosv in top_l2[:min(20, len(top_l2))]:
        print(f"  {tid}\t{tok}\t{l2v:.6f}\t{cosv:.6f}")

    print(f"\nTop {topk} tokens with lowest cosine similarity (token_id, token, L2, cosine):")
    for tid, tok, l2v, cosv in top_cos[:min(20, len(top_cos))]:
        print(f"  {tid}\t{tok}\t{l2v:.6f}\t{cosv:.6f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Save full per-token table as JSON
        rows = []
        for i in range(n_tokens):
            rows.append({
                "token_id": int(i),
                "token": token_name(i),
                "l2": float(l2_per_token[i]),
                "cos": float(cos_per_token[i])
            })
        json_path = os.path.join(out_dir, "embedding_diffs.json")
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump({"stats": stats, "rows": rows}, fj, indent=2, ensure_ascii=False)
        print(f"\nSaved detailed per-token diffs to: {json_path}")


def load_token_strings(path: str) -> Optional[dict]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"Token strings file not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed loading token strings: {e}")
        return None


def main():
    """Simple non-CLI entrypoint.

    - Tries to load `google/gemma-3-1b-it` on CPU.
    - If loading fails (or transformers not installed), falls back to a small random dry-run.
    - Prints results to console only; does not write files.
    """
    model_name = "google/gemma-3-1b-it"
    device = "cuda:1"
    topk = 100

    print(f"Loading model '{model_name}' on device {device}...")
    # Do not catch exceptions here: if the model or device is unavailable the script
    # will raise and exit as the user requested (no fallback behavior).
    in_w, out_w = get_embeddings_from_model(model_name, device)

    # Print results only (no files written)
    analyze_matrices(in_w, out_w, token_strings=None, topk=topk, out_dir=None)


if __name__ == "__main__":
    main()
