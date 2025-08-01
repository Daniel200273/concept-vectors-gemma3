import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer():
    # Change to gemma-3-4b-it or other larger model when running on a powerful GPU
    """Load the Gemma-3-12B-IT model and tokenizer."""
    print("Loading model and tokenizer...")
    model_name = "google/gemma-3-12b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

def load_concepts():
    """Load concepts from concepts.json."""
    with open('concepts.json', 'r') as f:
        concepts = json.load(f)
    
    print(f"Loaded {len(concepts)} concepts")
    return concepts

def load_vocabulary():
    """Load the Gemma 3 vocabulary from gemma3_vocabulary.json."""
    with open('gemma3_vocabulary.json', 'r') as f:
        vocab = json.load(f)
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

IMPORTANT: Your response must ONLY contain tokens that exist in the Gemma 3 1B vocabulary. 
The vocabulary contains {len(vocabulary)} unique tokens. Make sure each of your 200 selected tokens 
is present in this vocabulary.

"""
    
    prompt += vocabulary_section
    return prompt

def generate_keywords_for_concept(model, tokenizer, concept_name, prompt_template, vocabulary):
    """Generate keywords for a single concept using the LLM."""
    print(f"Generating keywords for: {concept_name}")
    
    # Create the full prompt
    full_prompt = create_full_prompt(concept_name, prompt_template, vocabulary)
    
    # Tokenize and generate
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (remove the prompt)
    generated_text = response[len(full_prompt):].strip()
    
    # Parse the keywords from the response
    keywords = parse_keywords_from_response(generated_text, vocabulary)
    
    print(f"Generated {len(keywords)} keywords for {concept_name}")
    return keywords

def parse_keywords_from_response(response_text, vocabulary):
    """Parse keywords from the LLM response and validate against vocabulary."""
    # Try to extract JSON-like format first
    json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
    if json_match:
        try:
            # Clean up the matched content and try to parse as JSON
            content = json_match.group(1)
            # Split by comma and clean up quotes
            keywords = [token.strip(' "\'') for token in content.split(',')]
            keywords = [kw.strip() for kw in keywords if kw.strip()]
        except:
            # Fallback to simple parsing
            keywords = extract_keywords_fallback(response_text)
    else:
        keywords = extract_keywords_fallback(response_text)
    
    # Validate keywords against vocabulary
    valid_keywords = []
    for keyword in keywords:
        if keyword in vocabulary:
            valid_keywords.append(keyword)
        else:
            # Try to find similar tokens in vocabulary
            similar = find_similar_token(keyword, vocabulary)
            if similar:
                valid_keywords.append(similar)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in valid_keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords[:200]  # Ensure we don't exceed 200 tokens

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

def find_similar_token(keyword, vocabulary, max_suggestions=1):
    """Find a similar token in vocabulary if exact match not found."""
    keyword_lower = keyword.lower()
    
    # First try exact match with different casing
    for token in vocabulary:
        if token.lower() == keyword_lower:
            return token
    
    # Try substring match
    for token in vocabulary:
        if keyword_lower in token.lower() or token.lower() in keyword_lower:
            return token
    
    return None

def main():
    """Main function to generate keywords for all concepts."""
    print("Starting keyword generation process...")
    
    # Load all required components
    model, tokenizer = load_model_and_tokenizer()
    concepts = load_concepts()
    vocabulary = load_vocabulary()
    prompt_template = load_prompt_template()
    
    # Dictionary to store results
    results = {}
    
    # Process each concept
    for i, concept in enumerate(concepts):
        print(f"\nProcessing concept {i+1}/{len(concepts)}: {concept}")
        
        try:
            keywords = generate_keywords_for_concept(
                model, tokenizer, concept, prompt_template, vocabulary
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
            
        break  # Uncomment to stop after first concept for testing
    
    # Save final results
    with open('generated_keywords.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Generated keywords for {len(results)} concepts")
    print("Results saved to 'generated_keywords.json'")

if __name__ == "__main__":
    main()
