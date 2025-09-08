#!/usr/bin/env python3
"""
Generate QA pairs for concepts using Gemma 3 1B and a prompt template

- Loads model: google/gemma-3-1b-it
- Reads concepts from test_concepts.json
- Uses prompt template from qa-prompt.txt
- Generates QA pairs for each concept
- Writes qa.json in the same format as code/concept-val/qa.json

Run:
  python generate-qa-pairs.py

Optional env:
  HF_TOKEN: Hugging Face token if required for gated models
  HF_HOME:  Hugging Face cache dir
"""

import os
import json
import re
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
CONCEPTS_FILE = os.path.join(os.path.dirname(HERE), "token-gen", "test_concepts.json")
PROMPT_FILE = os.path.join(HERE, "qa-prompt.txt")
DEFAULT_OUTPUT = os.path.join(HERE, "qa-generated.json")
MODEL_ID = os.environ.get("GEMMA_MODEL", "google/gemma-3-1b-it")

os.environ.setdefault("HF_HOME", "/media/hdd/usr/martinelli/.cache/huggingface")

# Get HuggingFace token from environment variable for security
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your HuggingFace token")
os.environ["HF_TOKEN"] = HF_TOKEN

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'


def load_concepts(path: str) -> List[str]:
    """Load the list of concepts from test_concepts.json"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"concepts file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("test_concepts.json must be a JSON list")
    return data


def load_prompt_template(path: str) -> str:
    """Load the prompt template from qa-prompt.txt"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompt file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def generate_questions_for_concept(tokenizer, model, prompt_template: str, concept: str, max_new_tokens: int = 400) -> Optional[Dict[str, Any]]:
    """Generate questions-only JSON for a specific concept using the prompt template.

    The prompt template should request a JSON object with keys: concept, category, qa (list of objects with only 'q').
    """
    # Substitute concept name in the template
    prompt = prompt_template.replace("{CONCEPT_NAME}", concept)

    messages = [{"role": "user", "content": prompt}]
    try:
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }

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
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Try to parse the JSON response
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)

                # Validate the structure: concept, category, qa list (qa entries may only have 'q')
                if (isinstance(parsed_json, dict) and
                    "concept" in parsed_json and
                    "category" in parsed_json and
                    "qa" in parsed_json and
                    isinstance(parsed_json["qa"], list)):
                    return parsed_json
                else:
                    print(f"  ⚠️ Invalid JSON structure for {concept}")
                    return None
            else:
                print(f"  ⚠️ No JSON found in response for {concept}")
                return None

        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON parse error for {concept}: {e}")
            print(f"  Response: {response[:200]}...")
            return None

    except Exception as e:
        print(f"  ❌ Generation error for {concept}: {e}")
        return None


def answer_single_question(tokenizer, model, concept: str, question: str, max_new_tokens: int = 200) -> str:
    """Prompt the model once per question to obtain an answer string.

    Returns the answer as plain text.
    """
    answer_prompt = (
        f"{question}\n"
    )

    messages = [{"role": "user", "content": answer_prompt}]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
    }

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
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    # Clean response
    return response


def main(concepts_path: str = CONCEPTS_FILE, prompt_path: str = PROMPT_FILE, out_path: str = DEFAULT_OUTPUT):
    print(f"🔄 Loading model: {MODEL_ID}")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # Use full precision to avoid quantization issues
    dtype = torch.float32  # Always use full precision

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

    # Load concepts and prompt template
    concepts = load_concepts(concepts_path)
    prompt_template = load_prompt_template(prompt_path)
    
    print(f"📋 Loaded {len(concepts)} concepts from {concepts_path}")
    print(f"📝 Loaded prompt template from {prompt_path}")
    
    # Phase 1: generate questions-only JSON for each concept
    questions_only: List[Dict[str, Any]] = []
    total = len(concepts)
    for i, concept in enumerate(concepts, 1):
        print(f"[{i}/{total}] 🔄 Generating questions for: {concept}")
        q_data = generate_questions_for_concept(tokenizer, model, prompt_template, concept)
        if q_data:
            questions_only.append(q_data)
            print(f"  ✅ Generated {len(q_data.get('qa', []))} questions for {concept}")
        else:
            print(f"  ❌ Failed to generate questions for {concept}")

    # Phase 2: answer each question with one prompt per question
    final_results: List[Dict[str, Any]] = []
    for qobj in questions_only:
        concept = qobj.get("concept")
        category = qobj.get("category")
        qa_list = qobj.get("qa", [])
        answered_qa = []

        print(f"\n📝 Answering questions for concept: {concept} ({len(qa_list)} questions)")
        for qi, qitem in enumerate(qa_list, 1):
            # qa entries may be dicts like {"q": "..."} or plain strings; handle both
            if isinstance(qitem, dict):
                question_text = qitem.get("q") or qitem.get("question") or qitem.get("Q")
            elif isinstance(qitem, str):
                question_text = qitem
            else:
                print(f"  ⚠️ Skipping unsupported QA entry type at index {qi}: {type(qitem)}")
                continue

            if not question_text:
                print(f"  ⚠️ Skipping empty question at index {qi}")
                continue
            print(f"  [{qi}/{len(qa_list)}] Q: {question_text}")
            answer_text = answer_single_question(tokenizer, model, concept, question_text)
            print(f"    A: {answer_text[:120]}{'...' if len(answer_text) > 120 else ''}")
            answered_qa.append({"q": question_text, "a": answer_text})

        final_results.append({
            "concept": concept,
            "category": category,
            "qa": answered_qa
        })

    # Save final results
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    successful = len(final_results)
    failed = total - successful
    print(f"\n💾 Wrote QA for {successful} concepts to {out_path}")
    print(f"📊 Summary: {successful} successful, {failed} failed out of {total} total concepts")

    if failed > 0:
        failed_concepts = [c for c in concepts if not any(r.get("concept") == c for r in final_results)]
        print(f"❌ Failed concepts: {failed_concepts}")


if __name__ == "__main__":
    main()