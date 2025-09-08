import json
import re
import os
import glob
import warnings
# Ensure private Hugging Face cache is set before importing transformers
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME

# Respect HF token if provided externally
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Suppress common HuggingFace warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*slow processor.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")

# Disable PyTorch compilation for compatibility with older GPUs
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Enable memory fragmentation fix for CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def load_model_and_tokenizer():
    """Load the gemma-3-4b-it model and processor with proper chat template support."""
    print("Loading model and processor...")
    model_name = "google/gemma-3-4b-it"  # Large model
    # model_name = "google/gemma-2-2b-it"     # Smaller alternative for testing
    # Uncomment the line above and comment the large model if you encounter issues
    
    # Check available GPU memory and select best GPU
    gpu_id = 1  # Manually set GPU ID - change this as needed
    if torch.cuda.is_available():
        print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s)")
        
        # Check if the specified GPU exists and has enough memory
        min_memory_gb = 20  # Minimum 20GB for Gemma-3-12B
        
        if gpu_id < torch.cuda.device_count():
            gpu_props = torch.cuda.get_device_properties(gpu_id)
            gpu_memory_gb = gpu_props.total_memory / 1e9
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} - {gpu_memory_gb:.1f} GB")
            
            if gpu_memory_gb < min_memory_gb:
                print(f"Error: GPU {gpu_id} only has {gpu_memory_gb:.1f} GB (need {min_memory_gb} GB)")
                print("Please select a different GPU with more memory.")
                raise RuntimeError(f"Insufficient GPU memory: {gpu_memory_gb:.1f} GB < {min_memory_gb} GB required")
        else:
            print(f"Error: GPU {gpu_id} not available. Available GPUs: {torch.cuda.device_count()}")
            raise RuntimeError(f"GPU {gpu_id} not found. Available GPUs: 0-{torch.cuda.device_count()-1}")
    else:
        print("CUDA not available - will use CPU (very slow)")
        gpu_id = None
    
    # Personal cache directories (set via environment variables)
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/media/hdd/usr/martinelli/.cache/huggingface/transformers')
    offload_folder = '/media/hdd/usr/martinelli/.cache/offload'
    
    print(f"Using cache directory: {cache_dir}")
    print(f"Using offload directory: {offload_folder}")
    
    # Ensure directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(offload_folder, exist_ok=True)
    
    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,  # Allow download if not cached
            token=HF_TOKEN,          # Use your personal token
            use_fast=True            # Use fast processor to avoid warnings
        )
        
        print("Loading model...")
        # Load model with optimizations for large models and cluster environment
        device_map = {"": gpu_id} if gpu_id is not None else "auto"
        
        # Use full precision to avoid quantization issues
        print("Loading with full precision (float32)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,  # Use full precision
            device_map=device_map,      # Use selected GPU or auto-select
            low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
            trust_remote_code=True,     # Allow custom code if needed
            local_files_only=False,     # Allow download if not cached
            offload_folder=offload_folder,  # Offload to disk if needed
            token=HF_TOKEN,             # Use your personal token
            attn_implementation="eager",  # Use eager attention for compatibility
            max_memory={gpu_id: "20GB"} if gpu_id is not None else None,  # Adjust for float32
            # Explicitly disable quantization
            load_in_8bit=False,
            load_in_4bit=False,
            quantization_config=None
        ).eval()  # Set to evaluation mode
        print("✓ Successfully loaded model with full precision (float32)")
        
        # Verify model and processor compatibility
        print("Verifying model-processor compatibility...")
        
        # Test with proper message format for Gemma3 processor
        test_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello world"}]
            }
        ]
        
        try:
            test_inputs = processor.apply_chat_template(
                test_messages, 
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            print(f"Test encoding successful -> {test_inputs['input_ids'].shape[1]} tokens")
        except Exception as e:
            print(f"Warning: Processor test failed: {e}")
            print("Continuing anyway...")
        
        # Check vocabulary size match
        try:
            # Try different ways to get vocab size depending on model type
            if hasattr(model.config, 'vocab_size'):
                model_vocab_size = model.config.vocab_size
            elif hasattr(model.config, 'vocabulary_size'):
                model_vocab_size = model.config.vocabulary_size
            elif hasattr(model.config, 'tokenizer_vocab_size'):
                model_vocab_size = model.config.tokenizer_vocab_size
            else:
                print("Cannot determine model vocabulary size from config")
                model_vocab_size = "Unknown"
                
            processor_vocab_size = len(processor.tokenizer) if hasattr(processor, 'tokenizer') else "Unknown"
            print(f"Model vocab size: {model_vocab_size}")
            print(f"Processor vocab size: {processor_vocab_size}")
            
            if model_vocab_size != "Unknown" and processor_vocab_size != "Unknown" and model_vocab_size != processor_vocab_size:
                print("WARNING: Model and processor vocabulary sizes don't match!")
                print("This might cause the CUDA assertion error.")
        except Exception as e:
            print(f"Could not check vocabulary compatibility: {e}")
            print("Proceeding anyway...")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to:")
        print("1. Network issues on the cluster")
        print("2. Concurrent access by other users")
        print("3. Insufficient disk space")
        print("4. Authentication issues")
        raise e
    
    print("Model and processor loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test basic generation immediately after loading
    print("\nTesting model generation capability...")
    if not test_model_generation(model, processor):
        print("ERROR: Model generation test failed!")
        print("This model/processor combination is not working properly.")
        raise RuntimeError("Model generation test failed - cannot proceed with keyword generation")
    
    print("✓ Model generation test passed - ready for keyword generation!\n")
    return model, processor

def load_concepts():
    """Load the list of concepts from test_concepts.json."""
    print("Loading concepts from test_concepts.json...")
    try:
        with open('test_concepts.json', 'r', encoding='utf-8') as f:
            concepts = json.load(f)
        print(f"✅ Loaded {len(concepts)} test concepts")
        return concepts
    except FileNotFoundError:
        print("❌ test_concepts.json not found")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing test_concepts.json: {e}")
        return []

def load_prompt_template():
    """Load the prompt template from keyword_generation_prompt.txt."""
    with open('keyword_generation_prompt.txt', 'r') as f:
        prompt_template = f.read()
    return prompt_template

def test_model_generation(model, processor):
    """Test basic model generation to verify it's working correctly."""
    print("Testing basic generation...")
    try:
        # Create test messages following HuggingFace template
        test_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Say hello in one word."}]
            }
        ]
        
        # Use processor following HuggingFace pattern
        test_input = processor.apply_chat_template(
            test_messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
        
        input_len = test_input["input_ids"].shape[-1]
        print(f"Test input length: {input_len} tokens")
        
        # Generate following HuggingFace pattern
        outputs = model.generate(**test_input, max_new_tokens=5)
        
        # Decode following HuggingFace pattern
        test_response = processor.decode(outputs[0][test_input["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        print(f"Basic generation test result: '{test_response}'")
        
        if test_response:
            print("✓ Generation test PASSED - model is working!")
            return True
        else:
            print("✗ Generation test FAILED - empty response")
            return False
            
    except Exception as e:
        print(f"Basic generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_keywords_with_llm(model, processor, concept, prompt_template):
    """Generate keywords for a concept using the LLM with proper chat template."""
    # Format the prompt with the concept - use the correct placeholder
    formatted_prompt = prompt_template.replace("{CONCEPT_NAME}", concept)
    
    # Create messages following HuggingFace template format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": formatted_prompt}
            ]
        }
    ]
    
    try:
        # Apply chat template following HuggingFace pattern
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        print(f"    ① Input length for '{concept}': {input_len} tokens")
        
        # Clear GPU cache before generation to free up memory
        torch.cuda.empty_cache()
        
        # Generate response with memory-efficient settings
        with torch.no_grad():  # Disable gradient computation to save memory
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128,  # Much smaller - keywords don't need long responses
                do_sample=True,      # Enable sampling for efficiency
                temperature=0.7,     # Add some randomness
                pad_token_id=processor.tokenizer.eos_token_id,  # Set pad token
                use_cache=True,      # Enable KV cache for efficiency
                cache_implementation="static"  # Use static cache to prevent memory growth
            )
        
        # Clear cache again after generation
        torch.cuda.empty_cache()
        
        # Decode only the new tokens (response) following HuggingFace pattern
        response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        
        if not response:
            print(f"WARNING: Empty response for concept '{concept}'")
            return []
        
        # Extract keywords and description from the response
        result = extract_keywords_from_response(response)
        
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error for concept '{concept}': {e}")
        # Clear cache and try with even smaller settings
        torch.cuda.empty_cache()
        
        try:
            print(f"    Retrying with reduced settings...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=64,   # Very small for retry
                    do_sample=False,     # Greedy decoding for efficiency
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=False      # Disable cache to save memory
                )
            
            torch.cuda.empty_cache()
            response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            
            if response:
                return extract_keywords_from_response(response)
            else:
                print(f"    Retry failed: Empty response for '{concept}'")
                return {"description": "", "keywords": []}
                
        except Exception as retry_e:
            print(f"    Retry also failed: {retry_e}")
            torch.cuda.empty_cache()
            return {"description": "", "keywords": []}
        
    except Exception as e:
        print(f"Error generating keywords for concept '{concept}': {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()  # Clear cache on any error
        return {"description": "", "keywords": []}

def extract_keywords_from_response(response):
    """Extract keywords and description from the LLM response."""
    # This function should be consistent with parse_keywords_from_response
    return parse_keywords_from_response(response)

def parse_keywords_from_response(response_text):
    """Parse keywords and description from the LLM response with new format."""
    
    # Remove markdown code blocks if present
    cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
    
    # Initialize results
    concept_description = ""
    keywords = []
    
    try:
        # Look for CONCEPT DESCRIPTION: line
        desc_match = re.search(r'CONCEPT DESCRIPTION:\s*(.+?)(?=\n|KEYWORDS:)', cleaned_text, re.DOTALL)
        if desc_match:
            concept_description = desc_match.group(1).strip()
            print(f"    📝 Description: {concept_description[:100]}...")
        
        # Look for KEYWORDS: [...] section
        keywords_match = re.search(r'KEYWORDS:\s*\[(.*?)\]', cleaned_text, re.DOTALL)
        if keywords_match:
            content = keywords_match.group(1)
            
            # Parse the actual JSON array content
            try:
                # Try to parse as proper JSON array
                json_array_text = '[' + content + ']'
                raw_keywords = json.loads(json_array_text)
                
                # Filter out long sentences and keep only actual keywords
                keywords = []
                for item in raw_keywords:
                    # Skip if it's a long sentence (more than 3 words or contains "CONCEPT DESCRIPTION")
                    if (len(item.split()) <= 3 and 
                        "CONCEPT DESCRIPTION" not in item and
                        not item.startswith("following the") and
                        not item.startswith("and themes") and
                        len(item) < 50):  # Skip very long strings
                        keywords.append(item.strip())
                
            except json.JSONDecodeError:
                # Fallback: split by comma and clean up
                raw_keywords = [token.strip(' "\'') for token in content.split(',')]
                keywords = []
                for item in raw_keywords:
                    if (item.strip() and 
                        len(item.split()) <= 3 and 
                        "CONCEPT DESCRIPTION" not in item and
                        len(item.strip()) < 50):
                        keywords.append(item.strip())
        else:
            # Fallback to old format parsing
            json_match = re.search(r'"[^"]+"\s*:\s*\[(.*?)\]', cleaned_text, re.DOTALL)
            if json_match:
                content = json_match.group(1)
                try:
                    json_array_text = '[' + content + ']'
                    raw_keywords = json.loads(json_array_text)
                    keywords = [kw for kw in raw_keywords if isinstance(kw, str) and len(kw.split()) <= 3 and len(kw) < 50]
                except json.JSONDecodeError:
                    raw_keywords = [token.strip(' "\'') for token in content.split(',')]
                    keywords = [kw.strip() for kw in raw_keywords if kw.strip() and len(kw.split()) <= 3 and len(kw) < 50]
            else:
                keywords = extract_keywords_fallback(cleaned_text)
    except Exception as e:
        print(f"    ⚠️ Parsing error: {e}")
        keywords = extract_keywords_fallback(cleaned_text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    duplicates_found = 0
    
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
        else:
            duplicates_found += 1
    
    # Show a small sample of final keywords
    sample_size = min(10, len(unique_keywords))
    sample_keywords = unique_keywords[:sample_size]
    print(f"    ② Final: {len(unique_keywords)} keywords (duplicates removed: {duplicates_found})")
    print(f"    ③ Sample: {sample_keywords}")
    
    return {
        "description": concept_description,
        "keywords": unique_keywords
    }

def extract_keywords_fallback(text):
    """Fallback method to extract keywords from free text."""
    # Look for comma-separated values
    lines = text.split('\n')
    keywords = []
    for line in lines:
        if ',' in line:
            tokens = [t.strip(' "\'') for t in line.split(',')]
            keywords.extend([t for t in tokens if t and len(t) > 1])
    return keywords



def tokenize_concepts_and_keywords(results, model_name="google/gemma-3-1b-it"):
    """Tokenize concept descriptions and keywords using Gemma tokenizer."""
    from transformers import AutoTokenizer
    
    print(f"\n🔤 Tokenizing descriptions and keywords with {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        
        tokenized_results = {}
        
        for concept, data in results.items():
            print(f"  Tokenizing: {concept}")
            
            if isinstance(data, dict) and 'description' in data and 'keywords' in data:
                description = data['description']
                keywords = data['keywords']
            else:
                # Fallback for old format
                description = f"Concept related to {concept}"
                keywords = data if isinstance(data, list) else []
            
            # Tokenize description
            desc_tokens = tokenizer.encode(description, add_special_tokens=False)
            
            # Tokenize each keyword
            keyword_tokens = {}
            for keyword in keywords:
                kw_tokens = tokenizer.encode(keyword, add_special_tokens=False)
                keyword_tokens[keyword] = kw_tokens
            
            tokenized_results[concept] = {
                "description": description,
                "description_tokens": desc_tokens,
                "keywords": keywords,
                "keyword_tokens": keyword_tokens,
                "description_token_count": len(desc_tokens),
                "keyword_count": len(keywords)
            }
        
        print(f"✅ Tokenization complete for {len(tokenized_results)} concepts")
        return tokenized_results
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return results

def main():
    """Main function to generate keywords for all concepts."""
    print("Starting keyword generation process...")
    
    # Create output directory
    output_dir = "token-results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    
    # Load all required components
    model, processor = load_model_and_tokenizer()
    concepts = load_concepts()
    prompt_template = load_prompt_template()
    
    # Initialize results dictionary
    results = {}
    
    # Process each concept
    for i, concept in enumerate(concepts):
        print(f"\nProcessing concept {i+1}/{len(concepts)}: {concept}")
        
        try:
            concept_data = generate_keywords_with_llm(
                model, processor, concept, prompt_template
            )
            results[concept] = concept_data
                
        except Exception as e:
            print(f"Error processing {concept}: {str(e)}")
            results[concept] = {"description": "", "keywords": []}  # Empty structure as fallback
    
    # Tokenize results
    tokenized_results = tokenize_concepts_and_keywords(results)
    
    # Save results in multiple formats
    print(f"\n💾 Saving results to {output_dir}/...")
    
    # 1. Save raw generated keywords (original format for compatibility)
    keywords_only = {}
    for concept, data in results.items():
        if isinstance(data, dict) and 'keywords' in data:
            keywords_only[concept] = data['keywords']
        else:
            keywords_only[concept] = data if isinstance(data, list) else []
    
    with open(os.path.join(output_dir, 'generated_keywords.json'), 'w') as f:
        json.dump(keywords_only, f, indent=2)
    
    # 2. Save full data with descriptions
    with open(os.path.join(output_dir, 'generated_keywords_with_descriptions.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 3. Save tokenized results
    with open(os.path.join(output_dir, 'tokenized_keywords.json'), 'w') as f:
        json.dump(tokenized_results, f, indent=2)
    
    # 4. Create summary
    summary = {
        "total_concepts": len(results),
        "successful_generations": len([r for r in results.values() if r]),
        "average_keywords_per_concept": sum(len(data.get('keywords', [])) if isinstance(data, dict) else len(data) if isinstance(data, list) else 0 for data in results.values()) / len(results),
        "model_used": "google/gemma-3-4b-it",
        "generation_date": "2025-09-06"
    }
    
    with open(os.path.join(output_dir, 'generation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Completed! Generated keywords for {len(results)} concepts")
    print("📁 Files created:")
    print(f"  - {output_dir}/generated_keywords.json (keywords only, original format)")
    print(f"  - {output_dir}/generated_keywords_with_descriptions.json (with concept descriptions)")
    print(f"  - {output_dir}/tokenized_keywords.json (with token IDs)")
    print(f"  - {output_dir}/generation_summary.json (statistics)")
    
    return tokenized_results

if __name__ == "__main__":
    main()
