#!/usr/bin/env python3
"""
Test script to verify the validation scripts work correctly
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    
    # Install NLTK and ROUGE
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "nltk", "rouge-score"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Download NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
    
    return True

def test_qa_data_loading():
    """Test loading QA data"""
    print("\n📋 Testing QA data loading...")
    
    import json
    base_path = Path("/media/hdd/usr/martinelli/concept-vectors-gemma3")
    qa_file = base_path / "code/concept-val/qa.json"
    
    try:
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        
        print(f"✅ Loaded {len(qa_data)} concepts from qa.json")
        
        # Test concept extraction
        concept_qa_map = {}
        for item in qa_data:
            concept_name = item.get('concept')
            if concept_name:
                qa_pairs = item.get('qa', [])
                questions = [pair.get('q', '') for pair in qa_pairs if pair.get('q')]
                concept_qa_map[concept_name] = questions
                print(f"  📝 {concept_name}: {len(questions)} questions")
        
        print(f"✅ Successfully processed {len(concept_qa_map)} concepts")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load QA data: {e}")
        return False

def test_vector_loading():
    """Test loading concept vectors from test results"""
    print("\n🎯 Testing concept vector loading...")
    
    import json
    base_path = Path("/media/hdd/usr/martinelli/concept-vectors-gemma3")
    results_file = base_path / "code/projection/test_results/selective_test_results.json"
    
    try:
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        concept_vectors = {}
        
        if 'concept_analyses' in results_data:
            for concept_name, analysis in results_data['concept_analyses'].items():
                if 'top_candidates' in analysis and analysis['top_candidates']:
                    top_vector = analysis['top_candidates'][0]
                    vector_key = top_vector.get('vector_key')
                    
                    if vector_key:
                        try:
                            parts = vector_key.split('_')
                            layer = int(parts[0][1:])
                            dimension = int(parts[1][1:])
                            concept_vectors[concept_name] = (layer, dimension)
                            print(f"  🎯 {concept_name}: Layer {layer}, Dimension {dimension}")
                        except (ValueError, IndexError):
                            print(f"  ⚠️ Could not parse vector key {vector_key} for {concept_name}")
        
        print(f"✅ Successfully loaded {len(concept_vectors)} concept vectors")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load concept vectors: {e}")
        return False

def test_validation_import():
    """Test importing validation modules"""
    print("\n🔧 Testing validation module imports...")
    
    try:
        # Add the current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Test simple validation import
        from simple_concept_validation import SimpleConceptValidator, load_concept_vectors_from_results
        print("✅ Simple validation module imported successfully")
        
        # Test advanced validation import  
        from advanced_concept_validation import ConceptVectorValidator, load_concept_vectors_from_test_results
        print("✅ Advanced validation module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to import validation modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Concept Vector Validation Setup")
    print("="*50)
    
    all_passed = True
    
    # Test 1: Install dependencies
    if not install_dependencies():
        all_passed = False
    
    # Test 2: Test QA data loading
    if not test_qa_data_loading():
        all_passed = False
    
    # Test 3: Test vector loading
    if not test_vector_loading():
        all_passed = False
    
    # Test 4: Test validation imports
    if not test_validation_import():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! Validation scripts are ready to use.")
        print("\nTo run validation:")
        print("  Simple:   python simple_concept_validation.py")
        print("  Advanced: python advanced_concept_validation.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()
