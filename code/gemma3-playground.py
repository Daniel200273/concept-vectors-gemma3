import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global variables to cache loaded model and tokenizer
_cached_model = None
_cached_tokenizer = None

def load_gemma3_model(force_reload=False):
    """Load Gemma 3 1B model and return model, tokenizer, and config"""
    global _cached_model, _cached_tokenizer
    
    # Check if model is already loaded and we don't want to force reload
    if not force_reload and _cached_model is not None and _cached_tokenizer is not None:
        print(f"✅ Using cached model and tokenizer")
        return _cached_model, _cached_tokenizer
    
    model_name = "google/gemma-3-1b-it"
    
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

def get_architectural_details(model, tokenizer):
    """Extract and display architectural details of Gemma 3 1B"""
    config = model.config
    
    print("\n" + "="*60)
    print("🏗️  GEMMA 3 1B ARCHITECTURAL DETAILS")
    print("="*60)
    
    # Core Architecture
    print(f"\n📋 Core Configuration:")
    print(f"  Model Type: {config.model_type}")
    print(f"  Architecture: {config.architectures[0] if hasattr(config, 'architectures') else 'Transformer'}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Layer Configuration
    print(f"\n🏗️  Layer Configuration:")
    print(f"  Number of Layers: {config.num_hidden_layers}")
    print(f"  Hidden Dimension: {config.hidden_size}")
    print(f"  Intermediate Size (MLP): {config.intermediate_size}")
    print(f"  Number of Attention Heads: {config.num_attention_heads}")
    
    # Advanced Features
    print(f"\n⚡ Advanced Features:")
    if hasattr(config, 'num_key_value_heads'):
        print(f"  Key-Value Heads (GQA): {config.num_key_value_heads}")
    if hasattr(config, 'sliding_window'):
        print(f"  Sliding Window: {config.sliding_window}")
    print(f"  Max Position Embeddings: {config.max_position_embeddings}")
    print(f"  RMS Norm Epsilon: {config.rms_norm_eps}")
    
    # Vocabulary
    print(f"\n📚 Vocabulary Configuration:")
    print(f"  Vocabulary Size: {config.vocab_size:,}")
    print(f"  Tokenizer Vocabulary Size: {len(tokenizer):,}")
    print(f"  Padding Token ID: {config.pad_token_id}")
    print(f"  EOS Token ID: {config.eos_token_id}")
    print(f"  BOS Token ID: {config.bos_token_id}")
    
    # Model Memory
    print(f"\n💾 Memory Usage:")
    total_params = sum(p.numel() for p in model.parameters())
    if model.dtype == torch.float16:
        memory_gb = total_params * 2 / 1e9  # 2 bytes per parameter for float16
    else:
        memory_gb = total_params * 4 / 1e9  # 4 bytes per parameter for float32
    print(f"  Model Size: ~{memory_gb:.2f} GB")
    print(f"  Data Type: {model.dtype}")
    
    # MLP Configuration (Important for concept vectors)
    print(f"\n🧠 MLP Configuration (for Concept Vectors):")
    print(f"  MLP Input Dimension: {config.hidden_size}")
    print(f"  MLP Intermediate Dimension: {config.intermediate_size}")
    print(f"  Total Candidate Vectors: {config.num_hidden_layers * config.intermediate_size:,}")
    print(f"  Activation Function: {getattr(config, 'hidden_act', 'Unknown')}")
    
    return config

def explore_vocabulary(tokenizer):
    """Explore vocabulary entries and their format"""
    print("\n" + "="*60)
    print("📖 VOCABULARY EXPLORATION")
    print("="*60)
    
    vocab_size = len(tokenizer)
    print(f"\n📊 Vocabulary Statistics:")
    print(f"  Total Vocabulary Size: {vocab_size:,}")
    print(f"  Tokenizer Type: {type(tokenizer).__name__}")
    
    # Special tokens
    print(f"\n🔧 Special Tokens:")
    special_tokens = {
        'PAD': tokenizer.pad_token,
        'BOS': tokenizer.bos_token,
        'EOS': tokenizer.eos_token,
        'UNK': tokenizer.unk_token,
    }
    
    for name, token in special_tokens.items():
        if token:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {name}: '{token}' (ID: {token_id})")
    
    # Sample vocabulary entries
    print(f"\n📝 Sample Vocabulary Entries (First 50):")
    print("  ID    | Token                | Decoded")
    print("  ------|---------------------|------------------")
    
    for i in range(min(50, vocab_size)):
        try:
            # Get token string
            token = tokenizer.convert_ids_to_tokens(i)
            # Decode the token
            decoded = tokenizer.decode([i])
            # Clean up display
            token_display = str(token)[:20] if token else "None"
            decoded_display = repr(decoded)[:20] if decoded else "None"
            print(f"  {i:5d} | {token_display:19s} | {decoded_display}")
        except:
            print(f"  {i:5d} | ERROR               | ERROR")
    
    # Sample from different ranges
    ranges_to_sample = [
        ("Common tokens", range(1000, 1020)),
        ("Mid-range tokens", range(vocab_size//2, vocab_size//2 + 10)),
        ("High-range tokens", range(vocab_size-20, vocab_size))
    ]
    
    for range_name, token_range in ranges_to_sample:
        print(f"\n📝 {range_name}:")
        print("  ID    | Token                | Decoded")
        print("  ------|---------------------|------------------")
        
        for i in token_range:
            if i < vocab_size:
                try:
                    token = tokenizer.convert_ids_to_tokens(i)
                    decoded = tokenizer.decode([i])
                    token_display = str(token)[:20] if token else "None"
                    decoded_display = repr(decoded)[:20] if decoded else "None"
                    print(f"  {i:5d} | {token_display:19s} | {decoded_display}")
                except:
                    print(f"  {i:5d} | ERROR               | ERROR")
    
    # Test tokenization examples
    print(f"\n🧪 Tokenization Examples:")
    test_strings = [
        "Hello world!",
        "The quick brown fox",
        "Machine learning is amazing",
        "Gemma 3 1B model",
        "🚀 Emoji test 🎯",
        "Harry Potter and the Philosopher's Stone. Hermione Granger is a brilliant witch.",
    ]
    
    for text in test_strings:
        tokens = tokenizer.encode(text)
        token_strings = tokenizer.convert_ids_to_tokens(tokens)
        print(f"  Text: '{text}'")
        print(f"    Tokens: {tokens}")
        print(f"    Token strings: {token_strings}")
        print(f"    Length: {len(tokens)} tokens")
        print()

def main():
    """Main function to load model and explore architecture"""
    print("🔍 Gemma 3 1B Model Explorer")
    print("="*50)
    
    # Load model
    model, tokenizer = load_gemma3_model()
    
    # Get architectural details
    config = get_architectural_details(model, tokenizer)
    
    # Explore vocabulary
    explore_vocabulary(tokenizer)
    
    print(f"\n✅ Model exploration completed!")

if __name__ == "__main__":
    main()