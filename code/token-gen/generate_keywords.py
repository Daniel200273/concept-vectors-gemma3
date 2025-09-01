import json
import re
import os
import glob
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

# Disable PyTorch compilation for compatibility with older GPUs
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Set environment variables for personal Hugging Face cache
os.environ['HF_HOME'] = '/media/hdd/usr/martinelli/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/media/hdd/usr/martinelli/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/media/hdd/usr/martinelli/.cache/huggingface/datasets'
os.environ['HF_HUB_CACHE'] = '/media/hdd/usr/martinelli/.cache/huggingface/hub'

# Set your personal HuggingFace token to avoid conflicts with other users
# Set via environment variable for security
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your HuggingFace token")

def load_model_and_tokenizer():
    """Load the Gemma-3-12B-IT model and processor with proper chat template support."""
    print("Loading model and processor...")
    model_name = "google/gemma-3-12b-it"  # Large model
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
            resume_download=True,    # Resume interrupted downloads
            token=HF_TOKEN          # Use your personal token
        )
        
        print("Loading model...")
        # Load model with optimizations for large models and cluster environment
        device_map = {"": gpu_id} if gpu_id is not None else "auto"
        
        # Try bfloat16 first (preferred), fallback to float16 for older GPUs
        try:
            print("Attempting to load with bfloat16...")
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,  # Try bfloat16 first
                device_map=device_map,      # Use selected GPU or auto-select
                low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
                trust_remote_code=True,     # Allow custom code if needed
                local_files_only=False,     # Allow download if not cached
                resume_download=True,       # Resume interrupted downloads
                offload_folder=offload_folder,  # Offload to disk if needed
                token=HF_TOKEN,             # Use your personal token
                attn_implementation="eager"  # Use eager attention for compatibility
            ).eval()  # Set to evaluation mode
            print("✓ Successfully loaded model with bfloat16")
        except Exception as e:
            print(f"bfloat16 failed ({e}), trying float16...")
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,  # Fallback to float16 for older GPU compatibility
                device_map=device_map,      # Use selected GPU or auto-select
                low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
                trust_remote_code=True,     # Allow custom code if needed
                local_files_only=False,     # Allow download if not cached
                resume_download=True,       # Resume interrupted downloads
                offload_folder=offload_folder,  # Offload to disk if needed
                token=HF_TOKEN,             # Use your personal token
                attn_implementation="eager"  # Use eager attention for compatibility
            ).eval()  # Set to evaluation mode
            print("✓ Successfully loaded model with float16")
        
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
    """Load concepts from concepts.json."""
    with open('concepts.json', 'r') as f:
        concepts = json.load(f)
    
    print(f"Loaded {len(concepts)} concepts")
    return concepts

def load_vocabulary():
    """Load the Gemma 3 vocabulary from gemma3_vocabulary.json."""
    with open('gemma3_vocabulary.json', 'r') as f:
        data = json.load(f)
    
    # Handle the nested structure - vocabulary is under 'vocabulary' key
    if 'vocabulary' in data:
        vocab = data['vocabulary']
        print(f"Loaded vocabulary with {len(vocab)} tokens")
        if 'metadata' in data:
            print(f"Model: {data['metadata'].get('model_name', 'Unknown')}")
    else:
        # Fallback if it's a flat structure
        vocab = data
        print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    return vocab

def load_prompt_template():
    """Load the prompt template from keyword_generation_prompt.txt."""
    with open('keyword_generation_prompt.txt', 'r') as f:
        prompt_template = f.read()
    return prompt_template

def create_full_prompt(concept_name, prompt_template, vocabulary):
    """Create the full prompt including the concept name and vocabulary."""
    # Replace the concept name placeholder
    prompt = prompt_template.replace("{CONCEPT_NAME}", concept_name)
    
    # Add vocabulary information to the prompt
    vocab_list = list(vocabulary.keys())[:1000]  # Use first 1000 tokens as example
    vocab_sample = ", ".join(f'"{token}"' for token in vocab_list[:50])  # Show first 50 as sample
    
    vocabulary_section = f"""

AVAILABLE VOCABULARY SAMPLE (first 50 of {len(vocabulary)} tokens):
{vocab_sample}...

IMPORTANT: Your response must ONLY contain tokens that exist in the provided Gemma vocabulary. 
The vocabulary contains {len(vocabulary)} unique tokens. Make sure each of your 200 selected tokens 
is present in this vocabulary.

"""
    
    prompt += vocabulary_section
    return prompt

def test_model_generation(model, processor):
    """Test basic model generation to verify it's working correctly."""
    print("Testing basic generation...")
    try:
        # Create test messages in the exact official format
        test_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Say hello in one word."}]
            }
        ]
        
        # Use processor to apply chat template and tokenize
        test_input = processor.apply_chat_template(
            test_messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device, dtype=next(model.parameters()).dtype)
        
        input_len = test_input["input_ids"].shape[-1]
        print(f"Test input length: {input_len} tokens")
        
        with torch.inference_mode():
            generation = model.generate(
                **test_input,
                max_new_tokens=5,        # Reduced from 10 for cleaner output
                do_sample=False,         # Use deterministic generation for test
                pad_token_id=processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]
        
        test_response = processor.decode(generation, skip_special_tokens=True).strip()
        print(f"Generated {len(generation)} new tokens")
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

def generate_keywords_with_llm(model, processor, concept, prompt_template, vocabulary):
    """Generate keywords for a concept using the LLM with proper chat template."""
    # Format the prompt with the concept - use the correct placeholder
    formatted_prompt = prompt_template.replace("{CONCEPT_NAME}", concept)
    
    # Create messages in the EXACT format from the official Gemma3 template
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful AI assistant specialized in generating relevant keywords from a specific vocabulary for given concepts. Always provide exactly 10 keywords from the provided vocabulary that are most relevant to the concept."}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": formatted_prompt}
            ]
        }
    ]
    
    try:
        # Apply chat template and tokenize using processor - following exact template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=next(model.parameters()).dtype)
        
        input_len = inputs["input_ids"].shape[-1]
        print(f"    ① Input length for '{concept}': {input_len} tokens")
        
        # Generate response using the same pattern as the template
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=1024,      # Reduced from 2048 to save memory
                do_sample=True,
                temperature=0.7,         # Some creativity but focused
                repetition_penalty=1.05
            )
            generation = generation[0][input_len:]
        
        # Decode the new tokens (response)
        response = processor.decode(generation, skip_special_tokens=True).strip()
        
        if not response:
            print(f"WARNING: Empty response for concept '{concept}'")
            return []
        
        # Extract keywords from the response
        keywords = extract_keywords_from_response(response, vocabulary)
        
        return keywords
        
    except Exception as e:
        print(f"Error generating keywords for concept '{concept}': {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return []

def extract_keywords_from_response(response, vocabulary):
    """Extract keywords from the LLM response."""
    # This function should be consistent with parse_keywords_from_response
    return parse_keywords_from_response(response, vocabulary)

def parse_keywords_from_response(response_text, vocabulary):
    """Parse keywords from the LLM response."""
    
    # Remove markdown code blocks if present
    cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
    
    # Try to parse as proper JSON first
    try:
        # Look for the JSON structure: "concept": [...keywords...]
        json_match = re.search(r'"[^"]+"\s*:\s*\[(.*?)\]', cleaned_text, re.DOTALL)
        if json_match:
            content = json_match.group(1)
            
            # Parse the actual JSON array content
            try:
                # Try to parse as proper JSON array
                json_array_text = '[' + content + ']'
                keywords = json.loads(json_array_text)
            except json.JSONDecodeError:
                # Fallback: split by comma and clean up
                keywords = [token.strip(' "\'') for token in content.split(',')]
                keywords = [kw.strip() for kw in keywords if kw.strip()]
        else:
            keywords = extract_keywords_fallback(cleaned_text)
    except Exception as e:
        keywords = extract_keywords_fallback(cleaned_text)
    
    # Remove duplicates while preserving order (no vocabulary validation)
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
    
    return unique_keywords

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

def find_last_checkpoint():
    """Find the last saved checkpoint to resume from."""
    import glob
    
    # Look for intermediate files
    checkpoint_files = glob.glob('intermediate_keywords_*.json')
    if not checkpoint_files:
        print("No checkpoint files found - starting from beginning")
        return {}, 0
    
    # Extract numbers and find the highest
    checkpoint_numbers = []
    for file in checkpoint_files:
        try:
            # Extract number from filename like 'intermediate_keywords_70.json'
            number = int(file.split('_')[-1].split('.')[0])
            checkpoint_numbers.append(number)
        except:
            continue
    
    if not checkpoint_numbers:
        print("No valid checkpoint files found - starting from beginning")
        return {}, 0
    
    last_checkpoint = max(checkpoint_numbers)
    checkpoint_file = f'intermediate_keywords_{last_checkpoint}.json'
    
    print(f"Found checkpoint: {checkpoint_file}")
    
    try:
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} completed concepts from checkpoint")
        print(f"Resuming from concept {last_checkpoint + 1}")
        return results, last_checkpoint
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_file}: {e}")
        print("Starting from beginning")
        return {}, 0

def main():
    """Main function to generate keywords for all concepts."""
    print("Starting keyword generation process...")
    
    # Load all required components
    model, processor = load_model_and_tokenizer()
    concepts = load_concepts()
    vocabulary = load_vocabulary()
    prompt_template = load_prompt_template()
    
    # Check for existing progress and resume if possible
    results, start_index = find_last_checkpoint()
    
    # Process each concept starting from where we left off
    for i, concept in enumerate(concepts[start_index:], start=start_index):
        print(f"\nProcessing concept {i+1}/{len(concepts)}: {concept}")
        
        try:
            keywords = generate_keywords_with_llm(
                model, processor, concept, prompt_template, vocabulary
            )
            results[concept] = keywords
            
            # Save intermediate results periodically
            if (i + 1) % 10 == 0:
                with open(f'intermediate_keywords_{i+1}.json', 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved intermediate results after {i+1} concepts")
                
        except Exception as e:
            print(f"Error processing {concept}: {str(e)}")
            results[concept] = []  # Empty list as fallback
    
    # Save final results
    with open('generated_keywords.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Generated keywords for {len(results)} concepts")
    print("Results saved to 'generated_keywords.json'")

if __name__ == "__main__":
    main()
