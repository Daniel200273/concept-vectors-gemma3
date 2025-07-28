import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global variables to cache loaded model and tokenizer
_cached_model = None
_cached_tokenizer = None

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
    """Load Gemma 3 4B model and return model, tokenizer, and config"""
    global _cached_model, _cached_tokenizer
    
    # Check if model is already loaded and we don't want to force reload
    if not force_reload and _cached_model is not None and _cached_tokenizer is not None:
        print(f"✅ Using cached model and tokenizer")
        return _cached_model, _cached_tokenizer
    
    model_name = "google/gemma-3-4b-it"
    
    print(f"🚀 Loading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Cache the loaded model and tokenizer
    _cached_model = model
    _cached_tokenizer = tokenizer
    
    print(f"✅ Model loaded successfully!")
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

def generate_keywords_for_concept(concept_name, model, tokenizer, prompt_template, max_tokens=2000):
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

def process_all_concepts(output_file_path=None):
    """Process all concepts and generate a complete JSON file with keywords
    
    Args:
        output_file_path (str, optional): Path for the output JSON file.
            If None, will save as 'generated_keywords.json' in the same directory.
    
    Returns:
        dict: Dictionary containing all generated keywords by concept
    """
    print("🚀 Starting automated keyword generation process...")
    
    # Load concepts
    concept_names = load_concepts()
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file_path = os.path.join(script_dir, "generated_keywords.json")
    
    # Initialize results dictionary - simplified structure
    results = {
        "metadata": {
            "total_concepts": len(concept_names),
            "generation_date": "2025-07-28",
            "model_used": "google/gemma-3-4b-it",
            "prompt_version": "1.0"
        }
    }
    
    # Track success/failure counts
    successful_count = 0
    failed_count = 0
    
    # Process each concept
    for i, concept_name in enumerate(concept_names, 1):
        print(f"\n📝 Processing concept {i}/{len(concept_names)}: {concept_name}")
        
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
            
            # Save progress after each concept (in case of interruption)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Error processing {concept_name}: {str(e)}")
            failed_count += 1
    
    print(f"\n🎉 Process completed! Results saved to: {output_file_path}")
    print(f"📊 Successfully processed: {successful_count} concepts")
    print(f"❌ Failed: {failed_count} concepts")
    
    return results

# Main execution function
def main():
    """Main function to run the keyword generation process"""
    print("🤖 Gemma 3 Keyword Generation Tool")
    print("=" * 50)
    
    # Process all concepts
    results = process_all_concepts()
    
    if results:
        print("\n✅ Keyword generation completed successfully!")
    else:
        print("\n❌ Keyword generation failed!")

if __name__ == "__main__":
    main()
