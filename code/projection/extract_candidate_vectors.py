#!/usr/bin/env python3
"""
Gemma 3 1B Candidate Vector Extraction

This script extracts candidate vectors from all MLP layers of Gemma 3 1B.
For Gemma 3 1B with 26 layers, we extract layers 0–25 (all layers).

Each MLP layer has:
- Input dimension: 1152 (hidden_size) 
- Intermediate dimension: 6912 (intermediate_size)
- Total candidate vectors per layer: 6912
- Total candidate vectors (all layers): 26 * 6912 = 179,712

The MLP structure in Gemma:
- gate_proj: Linear(1152 -> 6912)  
- up_proj: Linear(1152 -> 6912)
- down_proj: Linear(6912 -> 1152)
- Activation: SiLU (Swish)

We extract the down_proj weight matrix columns as candidate concept vectors.
Each column has 1152 dimensions matching token embedding dimensions.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
import gc
from tqdm import tqdm

# Add your Hugging Face token here
HF_TOKEN = "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"  # Replace with your actual token

# Set environment variables for personal Hugging Face cache
os.environ['HF_HOME'] = '/media/hdd/usr/martinelli/.cache/huggingface'

class GemmaCandidateVectorExtractor:
    """Extract candidate vectors from Gemma 3 1B MLP layers"""
    
    def __init__(self, model_name: str = "google/gemma-3-1b-it", device: str = "cuda:1"):
        """
        Initialize the extractor
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # Gemma 3 1B architecture specifics
        self.num_layers = 26
        self.hidden_size = 1152
        self.intermediate_size = 6912
        
        # Extract from all layers (layer indices 0-25)
        self.target_layers = list(range(self.num_layers))  # all layers
        self.total_candidates = len(self.target_layers) * self.intermediate_size  # columns from down_proj
        
        print(f"🎯 Target layers: {self.target_layers}")
        print(f"📊 Total candidate vectors: {self.total_candidates:,}")
    
    def load_model(self):
        """Load Gemma 3 1B model and tokenizer"""
        print(f"🚀 Loading {self.model_name}...")
        print(f"🎯 Using device: {self.device}")
        
        # Set CUDA device
        if "cuda" in self.device:
            torch.cuda.set_device(int(self.device.split(':')[1]))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN)
        
        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map={"": int(self.device.split(':')[1])} if "cuda" in self.device else self.device,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        
        self.config = self.model.config
        
        # Verify architecture
        assert self.config.num_hidden_layers == self.num_layers, f"Expected {self.num_layers} layers, got {self.config.num_hidden_layers}"
        assert self.config.hidden_size == self.hidden_size, f"Expected hidden size {self.hidden_size}, got {self.config.hidden_size}"
        assert self.config.intermediate_size == self.intermediate_size, f"Expected intermediate size {self.intermediate_size}, got {self.config.intermediate_size}"
        
        print(f"✅ Model loaded successfully!")
        print(f"📋 Verified architecture: {self.num_layers} layers, {self.hidden_size}d hidden, {self.intermediate_size}d intermediate")
    
    def extract_mlp_weights(self) -> Dict[int, torch.Tensor]:
        """
        Extract MLP down_proj weight matrices from target layers
        
        Returns:
            Dict mapping layer_idx -> down_proj weights (1152 x 6912)
            We extract columns (candidate vectors) from these matrices
        """
        print(f"\n🔍 Extracting MLP down_proj weights from layers {self.target_layers}...")
        
        mlp_weights = {}
        
        for layer_idx in tqdm(self.target_layers, desc="Extracting weights"):
            # Get the transformer layer
            layer = self.model.layers[layer_idx]
            
            # Extract MLP down_proj weights
            # down_proj shape: (hidden_size=1152, intermediate_size=6912)
            down_proj_weight = layer.mlp.down_proj.weight.data.clone()
            
            # Store in CPU to save GPU memory
            mlp_weights[layer_idx] = down_proj_weight.cpu()
            
            print(f"  Layer {layer_idx:2d}: down_proj shape {down_proj_weight.shape}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ Extracted weights from {len(mlp_weights)} layers")
        return mlp_weights
    
    def create_candidate_vector_database(self, mlp_weights: Dict[int, torch.Tensor]) -> Dict:
        """
        Create a structured database of all candidate vectors
        
        Args:
            mlp_weights: Dict of layer_idx -> weight tensors
            
        Returns:
            Database with candidate vectors and metadata
        """
        print(f"\n📚 Creating candidate vector database...")
        
        candidate_db = {
            "metadata": {
                "model_name": self.model_name,
                "extraction_date": "2025-08-06",
                "total_layers": self.num_layers,
                "target_layers": self.target_layers,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "total_candidates": self.total_candidates,
                "vector_dimension": self.hidden_size,  # Each column has hidden_size dimensions
                "extraction_method": "mlp_down_proj_weights"
            },
            "vectors": {},
            "layer_info": {}
        }
        
        vector_id = 0
        
        for layer_idx in tqdm(self.target_layers, desc="Processing layers"):
            weight_matrix = mlp_weights[layer_idx]  # Shape: (1152, 6912)
            
            layer_vectors = []
            
            # Extract each COLUMN as a candidate vector from down_proj matrix
            # Each column is a candidate vector vℓi
            for col_idx in range(self.intermediate_size):  # 6912 columns
                vector = weight_matrix[:, col_idx].numpy().astype(np.float32)  # Extract column
                
                # Create unique identifier
                vector_key = f"L{layer_idx:02d}_C{col_idx:04d}"  # C for column
                
                candidate_db["vectors"][vector_key] = {
                    "id": vector_id,
                    "layer": layer_idx,
                    "column": col_idx,  # column index in down_proj matrix
                    "vector": vector.tolist(),  # Convert to list for JSON serialization
                    "norm": float(np.linalg.norm(vector))
                }
                
                layer_vectors.append(vector_key)
                vector_id += 1
            
            # Store layer information
            candidate_db["layer_info"][f"layer_{layer_idx}"] = {
                "layer_index": layer_idx,
                "num_vectors": len(layer_vectors),
                "vector_keys": layer_vectors[:10] + ["..."] if len(layer_vectors) > 10 else layer_vectors,  # Sample for brevity
                "weight_matrix_shape": list(weight_matrix.shape),
                "mean_vector_norm": float(np.mean([candidate_db["vectors"][key]["norm"] for key in layer_vectors]))
            }
        
        print(f"✅ Created database with {vector_id:,} candidate vectors")
        return candidate_db
    
    def save_candidate_vectors(self, candidate_db: Dict, output_dir: str = "."):
        """
        Save candidate vectors in multiple formats
        
        Args:
            candidate_db: Database of candidate vectors
            output_dir: Directory to save files
        """
        print(f"\n💾 Saving candidate vectors to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save full database as JSON (warning: large file!)
        json_path = os.path.join(output_dir, "candidate_vectors_full.json")
        print(f"  Saving full database to {json_path}")
        with open(json_path, 'w') as f:
            json.dump(candidate_db, f, indent=2)
        
        # 2. Save metadata only
        metadata_path = os.path.join(output_dir, "candidate_vectors_metadata.json")
        metadata = {
            "metadata": candidate_db["metadata"],
            "layer_info": candidate_db["layer_info"]
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 3. Save vectors as numpy arrays for efficient loading
        vectors_array = np.array([
            candidate_db["vectors"][key]["vector"] 
            for key in sorted(candidate_db["vectors"].keys())
        ], dtype=np.float32)
        
        numpy_path = os.path.join(output_dir, "candidate_vectors.npy")
        np.save(numpy_path, vectors_array)
        
        # 4. Save vector index mapping
        index_mapping = {
            i: key for i, key in enumerate(sorted(candidate_db["vectors"].keys()))
        }
        mapping_path = os.path.join(output_dir, "vector_index_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(index_mapping, f, indent=2)
        
        print(f"✅ Saved candidate vectors in multiple formats:")
        print(f"    📄 Full database: {json_path}")
        print(f"    📋 Metadata only: {metadata_path}")
        print(f"    🔢 NumPy array: {numpy_path} (shape: {vectors_array.shape})")
        print(f"    🗂️ Index mapping: {mapping_path}")
        
        # Memory usage info
        file_size_mb = os.path.getsize(numpy_path) / (1024 * 1024)
        print(f"    💾 Vector array size: {file_size_mb:.1f} MB")
        
        return {
            "json_path": json_path,
            "metadata_path": metadata_path,
            "numpy_path": numpy_path,
            "mapping_path": mapping_path,
            "vectors_shape": vectors_array.shape
        }
    
    def extract_and_save(self, output_dir: str = ".") -> Dict:
        """
        Complete pipeline: load model, extract vectors, save results
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths and metadata
        """
        print("🎯 Starting Gemma 3 1B Candidate Vector Extraction Pipeline")
        print("=" * 60)
        
        # Step 1: Load model
        self.load_model()
        
        # Step 2: Extract MLP weights
        mlp_weights = self.extract_mlp_weights()
        
        # Step 3: Create candidate database
        candidate_db = self.create_candidate_vector_database(mlp_weights)
        
        # Step 4: Save results
        file_info = self.save_candidate_vectors(candidate_db, output_dir)
        
        # Clean up memory
        del mlp_weights
        del candidate_db
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 60)
        print("✅ EXTRACTION COMPLETE!")
        print("=" * 60)
        print(f"📊 Extracted {self.total_candidates:,} candidate vectors")
        print(f"📁 Saved to: {output_dir}")
        
        return file_info

def main():
    """Main extraction function"""
    # Configuration
    output_dir = "extracted_vectors"
    
    # Create extractor
    extractor = GemmaCandidateVectorExtractor()
    
    # Run extraction pipeline
    file_info = extractor.extract_and_save(output_dir)
    
    print(f"\n🎉 Candidate vector extraction completed!")
    print(f"📁 Check the '{output_dir}' folder for results")

if __name__ == "__main__":
    main()
