#!/usr/bin/env python3
"""
Test Script for Keyword Generation and Validation

This script runs the complete pipeline:
1. generate_keywords.py - Generate keywords for test concepts
2. validate_keywords.py - Validate and map keywords to token IDs
"""

import os
import sys

# Set environment variables if not already set
if not os.getenv('HF_TOKEN'):
    print("❌ HF_TOKEN not set. Please set it first:")
    print("export HF_TOKEN=your_token_here")
    sys.exit(1)

print("🚀 Starting complete keyword generation and validation pipeline...")
print("=" * 80)

# Step 1: Generate keywords
try:
    print("STEP 1: KEYWORD GENERATION")
    print("-" * 40)
    from generate_keywords import main as generate_main
    print("� Starting keyword generation...")
    results = generate_main()
    print("✅ Keyword generation completed successfully!")
    
    # Print a summary
    if results:
        print(f"\n📊 Generation Summary:")
        total_keywords = 0
        for concept, data in results.items():
            desc = data.get('description', '')[:50] + '...' if len(data.get('description', '')) > 50 else data.get('description', '')
            keywords = data.get('keywords', [])
            total_keywords += len(keywords)
            print(f"  {concept}: {len(keywords)} keywords")
            print(f"    Description: {desc}")
            print(f"    Keywords: {keywords[:5]}{'...' if len(keywords) > 5 else ''}")
        print(f"\n📈 Total: {len(results)} concepts, {total_keywords} keywords generated")
    
except Exception as e:
    print(f"❌ Error in keyword generation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)

# Step 2: Validate keywords
try:
    print("STEP 2: KEYWORD VALIDATION")
    print("-" * 40)
    from validate_keywords import main as validate_main
    print("🔄 Starting keyword validation...")
    validate_main()
    print("✅ Keyword validation completed successfully!")
    
except Exception as e:
    print(f"❌ Error in keyword validation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("📁 All output files created in token-results/:")
print("  Generation outputs:")
print("    - generated_keywords.json")
print("    - generated_keywords_with_descriptions.json")
print("    - tokenized_keywords.json")
print("    - generation_summary.json")
print("  Validation outputs:")
print("    - concept_keyword_ids.json")
print("    - concept_keyword_ids_summary.txt")
print("    - validation_report.json")
print("\n🚀 Ready for next pipeline stage!")
