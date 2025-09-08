#!/usr/bin/env python3
"""
Extract Full Vocabulary Embeddings for Gemma-3

This script extracts embeddings for the ENTIRE vocabulary, not just concept-specific tokens.
This is required for the new projection approach in project_and_rank_v2.py.

The new approach projects vectors onto the full vocabulary: E_vocab @ v_j
"""

import torch
import numpy as np
import json
import os
# Ensure private Hugging Face cache is set before importing transformers
PRIVATE_HF_HOME = "/media/hdd/usr/martinelli/.cache/huggingface"
os.environ["HF_HOME"] = PRIVATE_HF_HOME

# Respect HF token if provided externally
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
from typing import Dict



class FullVocabularyEmbeddingExtractor:
    """Extract embeddings for the entire Gemma-3 vocabulary"""
    
    def __init__(self, model_name: str = "google/gemma-3-1b-it", device: str = "auto"):
        """
        Initialize the extractor
        
        Args:
            model_name: HuggingFace model name
            device: Device to use for extraction
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.embedding_layer = None
        
        # Gemma 3 1B specs
        self.hidden_size = 1152
        self.vocab_size = None  # Will be set from actual model
    
    def load_model(self):
        """Load Gemma 3 1B model and extract embedding layer"""
        print(f"🚀 Loading {self.model_name} for full vocabulary embedding extraction...")
        
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
        print(f"📊 Tokenizer vocabulary size: {len(self.tokenizer):,}")
        print(f"🎯 Model embedding layer size: {self.embedding_layer.weight.shape[0]:,}")
    
    def load_full_vocabulary(self, vocab_file: str) -> Dict[str, int]:
        """
        Load full vocabulary from JSON file
        
        Args:
            vocab_file: Path to gemma3_vocabulary.json
            
        Returns:
            Dict mapping token strings to token IDs
        """
        print(f"📚 Loading full vocabulary from {vocab_file}...")
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Extract vocabulary mapping
        vocabulary = vocab_data.get("vocabulary", {})
        metadata = vocab_data.get("metadata", {})
        
        print(f"✅ Loaded vocabulary with {len(vocabulary):,} tokens")
        print(f"📊 Metadata: {metadata}")
        
        return vocabulary
    
    def extract_full_vocabulary_embeddings(self, vocabulary: Dict[str, int]) -> Dict:
        """
        Extract embeddings for ALL tokens in the vocabulary
        
        Args:
            vocabulary: Dict of token_string -> token_id
            
        Returns:
            Database with full vocabulary embeddings
        """
        print(f"\n🔍 Extracting embeddings for ENTIRE vocabulary...")
        print(f"📊 Total tokens to process: {len(vocabulary):,}")
        
        # Get the actual vocabulary size from the model
        actual_vocab_size = self.embedding_layer.weight.shape[0]
        print(f"🎯 Model's actual vocabulary size: {actual_vocab_size:,}")
        
        # Only use token IDs that are within the model's vocabulary bounds
        valid_token_ids = []
        id_to_string = {}
        
        # First, add all tokens from the vocabulary file that are within bounds
        for token_string, token_id in vocabulary.items():
            if 0 <= token_id < actual_vocab_size:
                valid_token_ids.append(token_id)
                id_to_string[token_id] = token_string
        
        # Then, add any missing token IDs from 0 to actual_vocab_size-1
        for token_id in range(actual_vocab_size):
            if token_id not in id_to_string:
                valid_token_ids.append(token_id)
                id_to_string[token_id] = f"<token_{token_id}>"
        
        # Remove duplicates and sort
        valid_token_ids = sorted(list(set(valid_token_ids)))
        
        print(f"📋 Will extract embeddings for {len(valid_token_ids):,} valid token IDs")
        print(f"📊 Token ID range: {min(valid_token_ids):,} to {max(valid_token_ids):,}")
        
        # Extract embeddings
        embedding_db = {
            "metadata": {
                "model_name": self.model_name,
                "extraction_date": "2025-01-27",
                "embedding_dimension": self.hidden_size,
                "total_vocabulary_size": len(valid_token_ids),
                "actual_model_vocab_size": actual_vocab_size,
                "data_type": "float32",
                "extraction_type": "full_vocabulary"
            },
            "embeddings": {},
            "vocabulary_mapping": id_to_string
        }
        
        # Batch process valid token IDs for efficiency
        batch_size = 1000
        
        print(f"🔄 Processing in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(valid_token_ids), batch_size), desc="Extracting embeddings"):
            batch_ids = valid_token_ids[i:i+batch_size]
            
            # Convert to tensor
            ids_tensor = torch.tensor(batch_ids, device=self.embedding_layer.weight.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.embedding_layer(ids_tensor)  # Shape: (batch_size, hidden_size)
                batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float32)
            
            # Store embeddings
            for j, token_id in enumerate(batch_ids):
                token_string = id_to_string[token_id]
                embedding = batch_embeddings[j]
                
                embedding_db["embeddings"][str(token_id)] = {
                    "token_id": token_id,
                    "token": token_string,
                    "embedding": embedding.tolist(),
                    "norm": float(np.linalg.norm(embedding))
                }
        
        print(f"✅ Extracted embeddings for {len(embedding_db['embeddings']):,} tokens")
        return embedding_db
    
    def save_embeddings(self, embedding_db: Dict, output_dir: str = "."):
        """
        Save full vocabulary embeddings in multiple formats
        
        Args:
            embedding_db: Database of full vocabulary embeddings
            output_dir: Directory to save files
        """
        print(f"\n💾 Saving full vocabulary embeddings to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save full database as JSON (very large file!)
        json_path = os.path.join(output_dir, "full_vocabulary_embeddings.json")
        print(f"  Saving full database to {json_path}...")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_db, f, indent=2, ensure_ascii=False)
        
        # 2. Save metadata and vocabulary mapping only
        metadata_path = os.path.join(output_dir, "full_vocabulary_metadata.json")
        metadata = {
            "metadata": embedding_db["metadata"],
            "vocabulary_mapping": embedding_db["vocabulary_mapping"]
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 3. Save embeddings as NumPy array for efficient computation
        # Create ordered arrays
        token_ids = sorted([int(k) for k in embedding_db["embeddings"].keys()])
        embeddings_array = np.array([
            embedding_db["embeddings"][str(tid)]["embedding"] 
            for tid in token_ids
        ], dtype=np.float32)
        
        numpy_path = os.path.join(output_dir, "full_vocabulary_embeddings.npy")
        np.save(numpy_path, embeddings_array)
        
        # 4. Save token ID to array index mapping
        id_mapping = {tid: i for i, tid in enumerate(token_ids)}
        mapping_path = os.path.join(output_dir, "full_vocab_token_id_to_index.json")
        with open(mapping_path, 'w') as f:
            json.dump(id_mapping, f, indent=2)
        
        # 5. Save token strings for reference
        token_strings = {tid: embedding_db["embeddings"][str(tid)]["token"] for tid in token_ids}
        strings_path = os.path.join(output_dir, "full_vocab_token_id_to_string.json")
        with open(strings_path, 'w', encoding='utf-8') as f:
            json.dump(token_strings, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved full vocabulary embeddings in multiple formats:")
        print(f"    📄 Full database: {json_path}")
        print(f"    📋 Metadata + mappings: {metadata_path}")
        print(f"    🔢 NumPy array: {numpy_path} (shape: {embeddings_array.shape})")
        print(f"    🗂️  ID mappings: {mapping_path}")
        print(f"    📝 Token strings: {strings_path}")
        
        return {
            "json_path": json_path,
            "metadata_path": metadata_path,
            "numpy_path": numpy_path,
            "mapping_path": mapping_path,
            "strings_path": strings_path
        }
    
    def run_extraction(self, vocab_file: str, output_dir: str = "."):
        """
        Complete extraction pipeline
        
        Args:
            vocab_file: Path to gemma3_vocabulary.json
            output_dir: Directory to save extracted embeddings
        """
        print("🎯 Starting Full Vocabulary Embedding Extraction")
        print("=" * 80)
        
        # Step 1: Load model
        self.load_model()
        
        # Step 2: Load vocabulary
        vocabulary = self.load_full_vocabulary(vocab_file)
        
        # Step 3: Extract embeddings
        embedding_db = self.extract_full_vocabulary_embeddings(vocabulary)
        
        # Step 4: Save embeddings
        file_info = self.save_embeddings(embedding_db, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ FULL VOCABULARY EXTRACTION COMPLETE!")
        print("=" * 80)
        print(f"📊 Extracted {len(embedding_db['embeddings']):,} token embeddings")
        print(f"📏 Embedding dimension: {self.hidden_size}")
        print(f"📁 Results saved to: {output_dir}")
        
        return file_info


def main():
    """Main extraction function"""
    parser = argparse.ArgumentParser(description="Extract full vocabulary embeddings for Gemma-3")
    parser.add_argument("--vocab_file", default="../token-gen/gemma3_vocabulary.json", 
                       help="Path to gemma3_vocabulary.json")
    parser.add_argument("--output_dir", default="full_vocabulary_embeddings", 
                       help="Directory to save extracted embeddings")
    parser.add_argument("--model_name", default="google/gemma-3-1b-it",
                       help="HuggingFace model name")
    
    args = parser.parse_args()
    
    # Check if vocabulary file exists
    if not os.path.exists(args.vocab_file):
        print(f"❌ Vocabulary file not found: {args.vocab_file}")
        print("Please ensure gemma3_vocabulary.json exists in the token-gen directory")
        return
    
    # Create extractor
    extractor = FullVocabularyEmbeddingExtractor(args.model_name)
    
    # Run extraction
    try:
        file_info = extractor.run_extraction(args.vocab_file, args.output_dir)
        print(f"\n🎉 Full vocabulary extraction completed!")
        print(f"📁 Check the '{args.output_dir}' folder for results")
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
