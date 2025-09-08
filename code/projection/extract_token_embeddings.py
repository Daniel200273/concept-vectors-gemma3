#!/usr/bin/env python3
"""
Token Embedding Extraction for Projection

This script extracts token embeddings from Gemma 3 1B for all vocabulary tokens
identified in the concept_keyword_ids.json file. These embeddings will be used
as the target space for projecting candidate vectors.

Process:
1. Load concept_keyword_ids.json (vocabulary tokens for each concept)
2. Extract embeddings for all unique tokens from Gemma 3 1B
3. Save embeddings in structured format for projection analysis
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Set, Tuple
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import gc

# Add your Hugging Face token via environment variable for security
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable with your HuggingFace token")

# Set environment variables for personal Hugging Face cache
os.environ['HF_HOME'] = '/media/hdd/usr/martinelli/.cache/huggingface'

class GemmaTokenEmbeddingExtractor:
    """Extract token embeddings from Gemma 3 1B embedding layer"""
    
    def __init__(self, model_name: str = "google/gemma-3-1b-it", device: str = "cuda:1"):
        """
        Initialize the embedding extractor
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.embedding_layer = None
        
        # Gemma 3 1B specs
        self.hidden_size = 1152
        self.vocab_size = 262144
    
    def load_model(self):
        """Load Gemma 3 1B model and extract embedding layer"""
        print(f"🚀 Loading {self.model_name} for embedding extraction...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with full precision
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use full precision
            device_map=self.device,
            trust_remote_code=True,
            # Explicitly disable quantization
            load_in_8bit=False,
            load_in_4bit=False,
            quantization_config=None
        )
        
        # Extract embedding layer
        self.embedding_layer = self.model.embed_tokens
        
        print(f"✅ Model loaded successfully!")
        print(f"📋 Embedding layer shape: {self.embedding_layer.weight.shape}")
        print(f"📊 Vocabulary size: {len(self.tokenizer):,}")
    
    def load_concept_tokens(self, concept_file: str) -> Dict[str, List[Tuple[str, int]]]:
        """
        Load concept vocabulary tokens from JSON file
        
        Args:
            concept_file: Path to concept_keyword_ids.json
            
        Returns:
            Dict mapping concept names to list of (token, token_id) tuples
        """
        print(f"📚 Loading concept tokens from {concept_file}...")
        
        with open(concept_file, 'r', encoding='utf-8') as f:
            concept_data = json.load(f)
        
        print(f"✅ Loaded {len(concept_data)} concepts")
        
        # Count total tokens and unique tokens
        all_tokens = set()
        total_mappings = 0
        
        for concept, token_list in concept_data.items():
            for token, token_id in token_list:
                all_tokens.add((token, token_id))
                total_mappings += 1
        
        print(f"📊 Total token mappings: {total_mappings:,}")
        print(f"📊 Unique tokens: {len(all_tokens):,}")
        
        return concept_data
    
    def extract_token_embeddings(self, concept_data: Dict[str, List[Tuple[str, int]]]) -> Dict:
        """
        Extract embeddings for all tokens in concept data
        
        Args:
            concept_data: Dict of concept -> [(token, token_id), ...]
            
        Returns:
            Database with token embeddings and metadata
        """
        print(f"\n🔍 Extracting token embeddings...")
        
        # Collect all unique token IDs
        unique_tokens = {}  # token_id -> token_string
        for concept, token_list in concept_data.items():
            for token, token_id in token_list:
                unique_tokens[token_id] = token
        
        print(f"📊 Extracting embeddings for {len(unique_tokens):,} unique tokens")
        
        # Extract embeddings
        embedding_db = {
            "metadata": {
                "model_name": self.model_name,
                "extraction_date": "2025-08-06",
                "embedding_dimension": self.hidden_size,
                "total_unique_tokens": len(unique_tokens),
                "vocabulary_size": self.vocab_size,
                "data_type": "float32"
            },
            "embeddings": {},
            "concept_mappings": {}
        }
        
        # Batch process token IDs for efficiency
        token_ids = list(unique_tokens.keys())
        batch_size = 1000
        
        print(f"🔄 Processing in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(token_ids), batch_size), desc="Extracting embeddings"):
            batch_ids = token_ids[i:i+batch_size]
            
            # Convert to tensor
            ids_tensor = torch.tensor(batch_ids, device=self.embedding_layer.weight.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.embedding_layer(ids_tensor)  # Shape: (batch_size, hidden_size)
                batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float32)
            
            # Store embeddings
            for j, token_id in enumerate(batch_ids):
                token_string = unique_tokens[token_id]
                embedding = batch_embeddings[j]
                
                embedding_db["embeddings"][str(token_id)] = {
                    "token_id": token_id,
                    "token": token_string,
                    "embedding": embedding.tolist(),
                    "norm": float(np.linalg.norm(embedding))
                }
        
        # Create concept mappings
        print(f"🗂️  Creating concept mappings...")
        for concept, token_list in concept_data.items():
            concept_tokens = []
            for token, token_id in token_list:
                concept_tokens.append({
                    "token": token,
                    "token_id": token_id,
                    "embedding_key": str(token_id)
                })
            
            embedding_db["concept_mappings"][concept] = {
                "num_tokens": len(concept_tokens),
                "tokens": concept_tokens
            }
        
        print(f"✅ Extracted embeddings for {len(embedding_db['embeddings']):,} tokens")
        return embedding_db
    
    def save_embeddings(self, embedding_db: Dict, output_dir: str = "."):
        """
        Save token embeddings in multiple formats
        
        Args:
            embedding_db: Database of token embeddings
            output_dir: Directory to save files
        """
        print(f"\n💾 Saving token embeddings to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save metadata and concept mappings only (no need for full database)
        metadata_path = os.path.join(output_dir, "token_embeddings_metadata.json")
        metadata = {
            "metadata": embedding_db["metadata"],
            "concept_mappings": embedding_db["concept_mappings"]
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 2. Save embeddings as NumPy array for efficient computation
        # Use existing token IDs directly from the concept file
        token_ids = sorted([int(k) for k in embedding_db["embeddings"].keys()])
        embeddings_array = np.array([
            embedding_db["embeddings"][str(tid)]["embedding"] 
            for tid in token_ids
        ], dtype=np.float32)
        
        numpy_path = os.path.join(output_dir, "token_embeddings.npy")
        np.save(numpy_path, embeddings_array)
        
        # 3. Save simple token ID to array index mapping (much smaller than before)
        id_to_index = {tid: i for i, tid in enumerate(token_ids)}
        mapping_path = os.path.join(output_dir, "token_id_to_index.json")
        with open(mapping_path, 'w') as f:
            json.dump(id_to_index, f, indent=2)
        
        # 4. Save token strings for reference (reuse existing token strings from concept file)
        token_strings = {tid: embedding_db["embeddings"][str(tid)]["token"] for tid in token_ids}
        strings_path = os.path.join(output_dir, "token_id_to_string.json")
        with open(strings_path, 'w', encoding='utf-8') as f:
            json.dump(token_strings, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved token embeddings in optimized formats:")
        print(f"    📋 Metadata + concept mappings: {metadata_path}")
        print(f"    🔢 NumPy embeddings: {numpy_path} (shape: {embeddings_array.shape})")
        print(f"    🗂️  Token ID → array index: {mapping_path}")
        print(f"    📝 Token ID → string: {strings_path}")
        print(f"    💡 Note: Using existing token IDs from concept_keyword_ids.json")
        
        # File size info
        file_size_mb = os.path.getsize(numpy_path) / (1024 * 1024)
        print(f"    💾 Embedding array size: {file_size_mb:.1f} MB")
        
        return {
            "metadata_path": metadata_path,
            "numpy_path": numpy_path,
            "mapping_path": mapping_path,
            "strings_path": strings_path,
            "embeddings_shape": embeddings_array.shape
        }
    
    def extract_and_save(self, concept_file: str, output_dir: str = ".") -> Dict:
        """
        Complete pipeline: load concept tokens, extract embeddings, save results
        
        Args:
            concept_file: Path to concept_keyword_ids.json
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths and metadata
        """
        print("🎯 Starting Token Embedding Extraction Pipeline")
        print("=" * 60)
        
        # Step 1: Load model
        self.load_model()
        
        # Step 2: Load concept tokens
        concept_data = self.load_concept_tokens(concept_file)
        
        # Step 3: Extract embeddings
        embedding_db = self.extract_token_embeddings(concept_data)
        
        # Step 4: Save results
        file_info = self.save_embeddings(embedding_db, output_dir)
        
        # Clean up memory
        del embedding_db
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 60)
        print("✅ EMBEDDING EXTRACTION COMPLETE!")
        print("=" * 60)
        print(f"📊 Extracted embeddings for {file_info['embeddings_shape'][0]:,} tokens")
        print(f"📁 Saved to: {output_dir}")
        
        return file_info

def main():
    """Main embedding extraction function"""
    # Configuration - updated path to token-results
    concept_file = "../token-gen/token-results/concept_keyword_ids.json"
    output_dir = "token_embeddings"
    
    # Check if concept file exists
    if not os.path.exists(concept_file):
        print(f"❌ Concept file not found: {concept_file}")
        print("Please run the keyword validation script first to generate concept_keyword_ids.json")
        return
    
    # Create extractor
    extractor = GemmaTokenEmbeddingExtractor()
    
    # Run extraction pipeline
    file_info = extractor.extract_and_save(concept_file, output_dir)
    
    print(f"\n🎉 Token embedding extraction completed!")
    print(f"📁 Check the '{output_dir}' folder for results")

if __name__ == "__main__":
    main()
