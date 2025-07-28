import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("🔥 Running in Google Colab - GPU acceleration available")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally")

# Global variables to cache loaded model and tokenizer
_cached_model = None
_cached_tokenizer = None

def setup_colab_environment():
    """Setup Google Colab environment if needed"""
    if IN_COLAB:
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  No GPU detected. Make sure to enable GPU in Runtime > Change runtime type")
        
        # Install required packages if not available
        try:
            import transformers
            print("✅ Transformers already installed")
        except ImportError:
            print("📦 Installing transformers...")
            os.system("pip install transformers torch accelerate")
    
    return torch.cuda.is_available()

def load_concepts(concepts_file_path=None):
    """Load concept names from concepts.json file
    
    Args:
        concepts_file_path (str, optional): Path to the concepts.json file.
            If None, will look in the same directory as this script.
    
    Returns:
        list: List of all concept names extracted from the file
    """
    # If no path provided, use default path in the same directory
    if concepts_file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        concepts_file_path = os.path.join(script_dir, "concepts.json")
    
    try:
        # Load and parse JSON file
        with open(concepts_file_path, 'r', encoding='utf-8') as f:
            concepts_data = json.load(f)
            
        # Extract all concept names from all categories
        concept_names = []
        for category, concepts in concepts_data.get("categories", {}).items():
            concept_names.extend(concepts)
        
        print(f"✅ Successfully loaded {len(concept_names)} concept names")
        return concept_names
    
    except FileNotFoundError:
        print(f"❌ Error: Concepts file not found at {concepts_file_path}")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {concepts_file_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading concepts: {str(e)}")
        return []

def load_gemma3_model(force_reload=False):
    """Load Gemma 3 model optimized for Colab GPU"""
    global _cached_model, _cached_tokenizer
    
    # Setup environment first
    gpu_available = setup_colab_environment()
    
    # Check if model is already loaded and we don't want to force reload
    if not force_reload and _cached_model is not None and _cached_tokenizer is not None:
        print(f"✅ Using cached model and tokenizer")
        return _cached_model, _cached_tokenizer
    
    # Choose model based on available memory
    if IN_COLAB and gpu_available:
        # Check GPU memory to decide which model to use
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 14:  # T4 has ~15GB, A100 has more
            model_name = "google/gemma-3-4b-it"
            print(f"🚀 Loading larger model {model_name} (GPU memory: {gpu_memory_gb:.1f}GB)")
        else:
            model_name = "google/gemma-3-1b-it"
            print(f"🚀 Loading smaller model {model_name} (GPU memory: {gpu_memory_gb:.1f}GB)")
    else:
        # Default to smaller model for local or CPU-only environments
        model_name = "google/gemma-3-1b-it"
        print(f"🚀 Loading {model_name} (CPU or local environment)")
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure model loading parameters based on environment
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }
    
    if gpu_available:
        model_kwargs["device_map"] = "auto"
        # Enable optimizations for GPU
        if IN_COLAB:
            model_kwargs["use_cache"] = True
    else:
        # CPU-only configuration
        model_kwargs["torch_dtype"] = torch.float32  # CPU works better with float32
        
    print("🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Move to GPU if available and not using device_map
    if gpu_available and "device_map" not in model_kwargs:
        model = model.to("cuda")
        print("🔥 Model moved to GPU")
    
    # Cache the loaded model and tokenizer
    _cached_model = model
    _cached_tokenizer = tokenizer
    
    print(f"✅ Model loaded successfully!")
    print(f"📍 Model device: {next(model.parameters()).device}")
    
    return model, tokenizer

def get_prompt_template():
    """Get the keyword generation prompt template
    
    Returns:
        str: The prompt template with {CONCEPT_NAME} placeholder
    """
    prompt_template = """You are an expert in computational linguistics and concept identification. I need you to generate exactly 200 tokens/keywords that are closely related to the concept "{CONCEPT_NAME}" for use in language model concept vector identification.

**CRITICAL REQUIREMENT:** All tokens must exist as single tokens in the vocabulary. Avoid multi-word phrases that would be split during tokenization.

The keywords should help a neural network identify when this concept is being discussed or referenced in text. Include diverse types of tokens:

**Requirements:**
1. Generate exactly 200 unique tokens
2. Each token must be a single vocabulary item (not split by tokenizer)
3. Include various types of related words
4. Prioritize tokens that would appear in the same context as the concept
5. Consider how the concept might be referenced indirectly
6. Include both common and specific terms
7. Format as a simple comma-separated list

**Token Types to Include:**
- Main names, titles, or direct references
- Core related terms and technical terminology
- Words that commonly appear in the same context
- Associated entities (people, places, organizations)
- Descriptive and emotional language
- Abbreviations and acronyms

**Format your response as:**
"{CONCEPT_NAME}": [
"token1", "token2", "token3", ... (exactly 200 tokens)
]

**Concept to process: "{CONCEPT_NAME}"**

Remember: These tokens will be used to identify concept vectors in neural networks, so think about what words would co-occur with this concept in natural language text."""

    print(f"✅ Prompt template ready")
    return prompt_template

def parse_keywords_from_response(response, concept_name):
    """Parse the model response to extract just the list of keywords
    
    Args:
        response (str): Raw response from the model
        concept_name (str): The concept name for validation
    
    Returns:
        list: List of keywords/tokens, or empty list if parsing fails
    """
    try:
        # Look for the JSON-like structure in the response
        # The response should contain something like: "ConceptName": ["token1", "token2", ...]
        
        # Find the start of the array (after the colon and opening bracket)
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            print(f"⚠️  Could not find token array in response for {concept_name}")
            return []
        
        # Extract the array content
        array_content = response[start_idx+1:end_idx]
        
        # Split by commas and clean up each token
        tokens = []
        for token in array_content.split(','):
            # Remove quotes, whitespace, and other formatting
            clean_token = token.strip().strip('"').strip("'").strip()
            if clean_token and clean_token not in tokens:  # Avoid duplicates
                tokens.append(clean_token)
        
        print(f"✅ Parsed {len(tokens)} tokens for {concept_name}")
        return tokens[:200]  # Ensure we don't exceed 200 tokens
        
    except Exception as e:
        print(f"❌ Error parsing response for {concept_name}: {str(e)}")
        return []

def generate_keywords_for_concept(concept_name, model, tokenizer, prompt_template, max_tokens=1500):
    """Generate keywords for a single concept using the Gemma model
    
    Args:
        concept_name (str): The concept to generate keywords for
        model: The loaded Gemma model
        tokenizer: The loaded tokenizer
        prompt_template (str): The prompt template with {CONCEPT_NAME} placeholder
        max_tokens (int): Maximum tokens to generate (reduced for Colab efficiency)
    
    Returns:
        str: Generated keywords response from the model
    """
    # Replace the placeholder with the actual concept name
    prompt = prompt_template.replace("{CONCEPT_NAME}", concept_name)
    
    print(f"🎯 Generating keywords for: {concept_name}")
    
    # Tokenize the prompt with appropriate settings for Colab
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024,  # Reduced for better memory management
        padding=False
    )
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Configure generation parameters optimized for Colab
    generation_config = {
        "max_new_tokens": max_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,  # Add nucleus sampling for better quality
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
        "use_cache": True,  # Enable KV caching for efficiency
    }
    
    # Add attention mask if available
    if "attention_mask" in inputs:
        generation_config["attention_mask"] = inputs["attention_mask"]
    
    # Generate response with memory optimization
    with torch.no_grad():
        if torch.cuda.is_available():
            # Clear cache before generation for better memory management
            torch.cuda.empty_cache()
        
        outputs = model.generate(
            inputs["input_ids"],
            **generation_config
        )
        
        if torch.cuda.is_available():
            # Clear cache after generation
            torch.cuda.empty_cache()
    
    # Decode the response, removing the input prompt
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    return response
    """Generate keywords for a single concept using the Gemma model
    
    Args:
        concept_name (str): The concept to generate keywords for
        model: The loaded Gemma model
        tokenizer: The loaded tokenizer
        prompt_template (str): The prompt template with {CONCEPT_NAME} placeholder
        max_tokens (int): Maximum tokens to generate
    
    Returns:
        str: Generated keywords response from the model
    """
    # Replace the placeholder with the actual concept name
    prompt = prompt_template.replace("{CONCEPT_NAME}", concept_name)
    
    print(f"🎯 Generating keywords for: {concept_name}")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode the response, removing the input prompt
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    return response

def process_all_concepts(concepts_file_path=None, output_file_path=None):
    """Process all concepts and generate a complete JSON file with keywords
    
    Args:
        concepts_file_path (str, optional): Path to the concepts.json file
        output_file_path (str, optional): Path for the output JSON file.
            If None, will save as 'generated_keywords.json' in the same directory.
    
    Returns:
        dict: Dictionary containing all generated keywords by concept
    """
    print("🚀 Starting automated keyword generation process...")
    
    # Load concepts
    concept_names = load_concepts(concepts_file_path)
    if not concept_names:
        print("❌ No concepts loaded, aborting process")
        return {}
    
    # Load prompt template
    prompt_template = get_prompt_template()
    if not prompt_template:
        print("❌ No prompt template loaded, aborting process")
        return {}
    
    # Load model
    model, tokenizer = load_gemma3_model()
    
    # Prepare output file path
    if output_file_path is None:
        if IN_COLAB:
            output_file_path = "generated_keywords.json"  # Save in current directory for Colab
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_file_path = os.path.join(script_dir, "generated_keywords.json")
    
    # Initialize results dictionary - simplified structure
    model_name = "google/gemma-3-4b-it" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 14 * 1024**3 else "google/gemma-3-1b-it"
    
    results = {
        "metadata": {
            "total_concepts": len(concept_names),
            "generation_date": "2025-07-28",
            "model_used": model_name,
            "prompt_version": "1.0",
            "environment": "Google Colab" if IN_COLAB else "Local",
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        }
    }
    
    # Track success/failure counts
    successful_count = 0
    failed_count = 0
    
    # Add progress tracking for Colab
    total_concepts = len(concept_names)
    
    # Process each concept
    for i, concept_name in enumerate(concept_names, 1):
        progress_percent = (i / total_concepts) * 100
        print(f"\n📝 Processing concept {i}/{total_concepts} ({progress_percent:.1f}%): {concept_name}")
        
        try:
            # Generate keywords for this concept
            response = generate_keywords_for_concept(
                concept_name, model, tokenizer, prompt_template
            )
            
            # Parse the response to extract clean token list
            tokens = parse_keywords_from_response(response, concept_name)
            
            if tokens:
                # Store just the concept name and its tokens
                results[concept_name] = tokens
                successful_count += 1
                print(f"✅ Completed: {concept_name} ({len(tokens)} tokens)")
            else:
                print(f"⚠️  No valid tokens extracted for: {concept_name}")
                failed_count += 1
            
            # Save progress more frequently in Colab (every 5 concepts)
            if IN_COLAB and (i % 5 == 0 or i == total_concepts):
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"💾 Progress saved ({i}/{total_concepts} concepts)")
            elif not IN_COLAB:
                # Save after each concept when running locally
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Error processing {concept_name}: {str(e)}")
            failed_count += 1
            # Continue processing other concepts even if one fails
    
    print(f"\n🎉 Process completed! Results saved to: {output_file_path}")
    print(f"📊 Successfully processed: {successful_count} concepts")
    print(f"❌ Failed: {failed_count} concepts")
    
    return results

def download_file_in_colab(file_path):
    """Download file in Google Colab"""
    if IN_COLAB:
        try:
            from google.colab import files
            files.download(file_path)
            print(f"📥 File {file_path} ready for download!")
        except Exception as e:
            print(f"⚠️  Could not initiate download: {str(e)}")

def upload_concepts_file_in_colab():
    """Upload concepts.json file in Google Colab"""
    if IN_COLAB:
        try:
            from google.colab import files
            print("📤 Please upload your concepts.json file:")
            uploaded = files.upload()
            
            # Find the uploaded concepts.json file
            for filename in uploaded.keys():
                if filename.endswith('.json') and 'concept' in filename.lower():
                    print(f"✅ Found concepts file: {filename}")
                    return filename
            
            # If no concepts file found, check for any JSON file
            if uploaded:
                filename = list(uploaded.keys())[0]
                print(f"📁 Using uploaded file: {filename}")
                return filename
                
        except Exception as e:
            print(f"❌ Error uploading file: {str(e)}")
    
    return None

# Main execution function
def main():
    """Main function to run the keyword generation process"""
    print("🤖 Gemma 3 Keyword Generation Tool")
    print("=" * 50)
    
    # Handle file upload in Colab
    concepts_file = None
    if IN_COLAB:
        print("🔄 Setting up Google Colab environment...")
        
        # Check if concepts.json exists in current directory
        if not os.path.exists("concepts.json"):
            print("📂 concepts.json not found. Please upload it.")
            concepts_file = upload_concepts_file_in_colab()
        else:
            print("✅ Found concepts.json in current directory")
            concepts_file = "concepts.json"
    
    # Process all concepts
    if concepts_file:
        results = process_all_concepts(concepts_file)
    else:
        results = process_all_concepts()
    
    if results:
        print("\n✅ Keyword generation completed successfully!")
        
        # Offer download in Colab
        if IN_COLAB:
            output_file = "generated_keywords.json"
            if os.path.exists(output_file):
                download_file_in_colab(output_file)
    else:
        print("\n❌ Keyword generation failed!")

if __name__ == "__main__":
    main()
