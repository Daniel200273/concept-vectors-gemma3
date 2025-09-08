#!/usr/bin/env python3
"""
Keyword-to-Token-ID Mapping Script (All Vocabulary Token Variants)

This script processes generated keywords and creates comprehensive (vocabulary_token, token_id) 
tuples for each concept by validating keywords against the Gemma 3 vocabulary.

NEW FEATURE: Instead of saving original keywords, this script now saves ALL actual
vocabulary token variations found for each keyword (upper/lower case, with/without prefixes, etc.)

For each concept:
  1. Check if each keyword exists in vocabulary (with fuzzy matching)
  2. Find ALL matching vocabulary token variants for each keyword  
  3. Create (vocabulary_token, token_id) tuples for every variant found
  4. Save comprehensive mappings with actual vocabulary tokens (~200+ tuples per concept)

Usage:
    python validate_keywords.py

Output:
    - concept_keyword_ids.json: ALL (vocabulary_token, token_id) tuples for each concept
    - concept_keyword_ids_summary.txt: Human-readable summary with variant counts
    - validation_report.json: Detailed validation statistics
"""

import json
import os
from collections import defaultdict

def load_vocabulary():
    """Load the Gemma 3 vocabulary from gemma3_vocabulary.json."""
    print("Loading Gemma 3 vocabulary...")
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

def get_stopwords():
    """Return a comprehensive list of stopwords to filter out from descriptions."""
    stopwords = {
        # Articles
        'a', 'an', 'the',
        # Conjunctions
        'and', 'or', 'but', 'so', 'yet', 'for', 'nor',
        # Prepositions
        'in', 'on', 'at', 'by', 'to', 'of', 'with', 'from', 'into', 'onto', 'upon',
        'through', 'over', 'under', 'above', 'below', 'between', 'among', 'within',
        'without', 'during', 'before', 'after', 'since', 'until', 'while', 'across',
        'around', 'behind', 'beside', 'beyond', 'inside', 'outside', 'toward', 'towards',
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what',
        # Auxiliary verbs
        'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'can', 
        'could', 'must', 'ought',
        # Common verbs
        'get', 'got', 'getting', 'gets', 'make', 'made', 'making', 'makes', 'take', 'took',
        'taking', 'takes', 'come', 'came', 'coming', 'comes', 'go', 'went', 'going', 'goes',
        # Adverbs
        'very', 'quite', 'rather', 'too', 'so', 'more', 'most', 'much', 'many', 'little',
        'few', 'less', 'least', 'all', 'some', 'any', 'no', 'not', 'only', 'just', 'even',
        'also', 'still', 'already', 'yet', 'again', 'once', 'twice', 'here', 'there',
        'where', 'when', 'how', 'why', 'now', 'then', 'today', 'tomorrow', 'yesterday',
        # Other common words
        'as', 'if', 'than', 'like', 'such', 'way', 'ways', 'time', 'times', 'place', 'places',
        'thing', 'things', 'people', 'person', 'world', 'life', 'work', 'works', 'working',
        'day', 'days', 'year', 'years', 'part', 'parts', 'use', 'used', 'using', 'uses',
        'see', 'seen', 'saw', 'look', 'looks', 'looking', 'looked', 'find', 'found', 'finding',
        'know', 'knew', 'known', 'knowing', 'think', 'thought', 'thinking', 'thinks',
        'say', 'said', 'saying', 'says', 'tell', 'told', 'telling', 'tells', 'give', 'gave',
        'given', 'giving', 'gives', 'show', 'showed', 'shown', 'showing', 'shows',
        # Numbers (as words)
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'first', 'second', 'third', 'last', 'next', 'new', 'old', 'good', 'bad', 'great',
        'small', 'large', 'big', 'little', 'long', 'short', 'high', 'low', 'right', 'left',
        'same', 'different', 'other', 'another', 'each', 'every', 'both', 'either', 'neither',
        # Common transitions and connectives
        'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'nonetheless',
        'consequently', 'accordingly', 'thus', 'hence', 'indeed', 'certainly', 'particularly',
        'especially', 'specifically', 'generally', 'typically', 'usually', 'often', 'sometimes',
        'always', 'never', 'perhaps', 'maybe', 'probably', 'possibly', 'definitely',
        # Academic/formal terms that are too generic
        'refers', 'including', 'encompasses', 'involves', 'characterized', 'known', 'considered',
        'described', 'defined', 'called', 'named', 'termed', 'various', 'different', 'multiple',
        'several', 'numerous', 'important', 'significant', 'major', 'primary', 'main', 'key',
        'basic', 'fundamental', 'essential', 'necessary', 'related', 'associated', 'connected'
    }
    return stopwords

def load_generated_keywords():
    """Load the generated keywords from token-results/generated_keywords.json."""
    print("Loading generated keywords...")
    
    # Check in token-results folder first, then fallback to current directory
    keywords_file = 'token-results/generated_keywords.json'
    if not os.path.exists(keywords_file):
        keywords_file = 'generated_keywords.json'
        if not os.path.exists(keywords_file):
            print("ERROR: generated_keywords.json not found!")
            print("Please run generate_keywords.py first to create the keywords file.")
            return None
    
    print(f"Reading keywords from: {keywords_file}")
    
    with open(keywords_file, 'r') as f:
        keywords_data = json.load(f)
    
    print(f"Loaded keywords for {len(keywords_data)} concepts")
    
    # Count total keywords
    total_keywords = sum(len(keywords) for keywords in keywords_data.values())
    print(f"Total keywords across all concepts: {total_keywords}")
    
    return keywords_data

def find_token_variants(keyword, vocab_dict):
    """
    Find ALL actual vocabulary token variants for a keyword that exist in the vocabulary.
    Returns list of (variant, token_id) tuples that match (deduplicated).
    Uses efficient lookups instead of scanning the entire vocabulary.
    """
    matches = []
    seen_tokens = set()  # Track tokens we've already found
    keyword_lower = keyword.lower()
    
    # List of potential variants to check (efficient direct lookups)
    potential_variants = [
        keyword,                    # exact match
        keyword_lower,              # lowercase
        keyword.title(),            # Title case
        keyword.upper(),            # UPPERCASE
        f"_{keyword}",              # underscore prefix
        f"▁{keyword}",              # SentencePiece underscore
        f"Ġ{keyword}",              # GPT-style space prefix
        f"_{keyword_lower}",        # underscore + lowercase
        f"▁{keyword_lower}",        # SentencePiece + lowercase
        f"Ġ{keyword_lower}",        # GPT + lowercase
        f"_{keyword.title()}",      # underscore + title
        f"▁{keyword.title()}",      # SentencePiece + title
        f"Ġ{keyword.title()}",      # GPT + title
        f"_{keyword.upper()}",      # underscore + upper
        f"▁{keyword.upper()}",      # SentencePiece + upper
        f"Ġ{keyword.upper()}",      # GPT + upper
    ]
    
    # Check each potential variant with direct lookup (O(1) instead of O(n))
    for variant in potential_variants:
        if variant in vocab_dict and variant not in seen_tokens:
            matches.append((variant, vocab_dict[variant]))
            seen_tokens.add(variant)
    
    # Only do expensive scan if no matches found (fallback for subword matching)
    if not matches:
        for token in vocab_dict:
            if token not in seen_tokens:
                clean_token = token.lstrip('_▁Ġ').lower()
                if (clean_token.startswith(keyword_lower) and 
                    len(clean_token) <= len(keyword_lower) + 3):
                    matches.append((token, vocab_dict[token]))
                    seen_tokens.add(token)
                    if len(matches) >= 3:  # Limit subword matches
                        break
    
    return matches

def validate_keywords(keywords_data, vocabulary):
    """Validate keywords and create (keyword, token_id) tuples for each concept with fuzzy matching."""
    print("\nValidating keywords and extracting token IDs (with fuzzy matching)...")
    
    # Convert vocabulary to a dict for faster lookup: {token: id}
    vocab_dict = vocabulary  # vocabulary is already {token: id}
    
    validation_results = {}
    total_valid = 0
    total_invalid = 0
    invalid_keywords_by_concept = defaultdict(list)
    all_invalid_keywords = set()
    fuzzy_matches = defaultdict(list)  # Track fuzzy matches for analysis
    
    total_concepts = len(keywords_data)
    
    for concept_num, (concept, keywords) in enumerate(keywords_data.items(), 1):
        valid_tuples = []  # List of (keyword, [token_id1, token_id2, ...]) tuples
        invalid_keywords = []
        concept_fuzzy_matches = []
        
        # Load description and split into words to append to keywords
        combined_keywords = list(keywords)  # Start with original keywords
        
        # Try to get description from generated_keywords_with_descriptions.json
        desc_file = 'token-results/generated_keywords_with_descriptions.json'
        if not os.path.exists(desc_file):
            desc_file = 'generated_keywords_with_descriptions.json'
        
        if os.path.exists(desc_file):
            try:
                with open(desc_file, 'r') as f:
                    desc_data = json.load(f)
                
                if concept in desc_data and 'description' in desc_data[concept]:
                    description = desc_data[concept]['description']
                    
                    # Get stopwords for filtering
                    stopwords = get_stopwords()
                    
                    # Split description into words and clean them
                    # Remove punctuation and split
                    clean_desc = description.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
                    clean_desc = clean_desc.replace(';', ' ').replace(':', ' ').replace('(', ' ').replace(')', ' ')
                    clean_desc = clean_desc.replace('[', ' ').replace(']', ' ').replace('"', ' ').replace("'", ' ')
                    
                    desc_words = []
                    for word in clean_desc.split():
                        word = word.strip().lower()
                        # Filter conditions:
                        # 1. Word must be longer than 2 characters
                        # 2. Word must not be in stopwords
                        # 3. Word must be alphabetic (no numbers or special chars)
                        # 4. Word must not be too short or too long
                        if (len(word) > 2 and 
                            len(word) < 20 and
                            word not in stopwords and 
                            word.isalpha()):
                            desc_words.append(word)
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_desc_words = []
                    for word in desc_words:
                        if word not in seen:
                            seen.add(word)
                            unique_desc_words.append(word)
                    
                    # Append filtered description words to keywords
                    combined_keywords.extend(unique_desc_words)
                    print(f"    📝 Added {len(unique_desc_words)} filtered description words for '{concept}' (filtered out {len(desc_words) - len(unique_desc_words)} stopwords/duplicates)")
            except Exception as e:
                print(f"    ⚠️ Could not load description for '{concept}': {e}")
        
        for keyword in combined_keywords:
            # Find possible token variants
            token_variants = find_token_variants(keyword, vocab_dict)
            
            if token_variants:
                # Store both the original keyword and the actual vocabulary tokens found
                # Remove duplicate token variants while preserving order
                seen_variants = set()
                unique_variants = []
                for vocab_token, token_id in token_variants:
                    variant_key = (vocab_token, token_id)
                    if variant_key not in seen_variants:
                        unique_variants.append((vocab_token, token_id))
                        seen_variants.add(variant_key)
                
                valid_tuples.append((keyword, unique_variants))
                total_valid += 1
                
                # Track if this included fuzzy matches (not exact)
                exact_match = any(token == keyword for token, _ in token_variants)
                if not exact_match:
                    fuzzy_match_info = {
                        'original_keyword': keyword,
                        'matched_tokens': [token for token, _ in unique_variants],
                        'token_ids': [token_id for _, token_id in unique_variants],
                        'all_variants': unique_variants
                    }
                    concept_fuzzy_matches.append(fuzzy_match_info)
                    fuzzy_matches[concept].append(fuzzy_match_info)
            else:
                invalid_keywords.append(keyword)
                invalid_keywords_by_concept[concept].append(keyword)
                all_invalid_keywords.add(keyword)
                total_invalid += 1
        
        validation_results[concept] = {
            'total_keywords': len(combined_keywords),
            'original_keywords_count': len(keywords),
            'description_words_count': len(combined_keywords) - len(keywords),
            'valid_tuples': valid_tuples,
            'invalid_keywords': invalid_keywords,
            'fuzzy_matches': concept_fuzzy_matches,
            'valid_count': len(valid_tuples),
            'invalid_count': len(invalid_keywords),
            'fuzzy_count': len(concept_fuzzy_matches),
            'validity_percentage': (len(valid_tuples) / len(combined_keywords) * 100) if combined_keywords else 0
        }
        
        # Show a small sample of valid tuples with token counts
        sample_size = min(3, len(valid_tuples))
        sample_display = []
        for keyword, variants in valid_tuples[:sample_size]:
            if len(variants) == 1:
                vocab_token, token_id = variants[0]
                sample_display.append(f"'{keyword}' -> '{vocab_token}': {token_id}")
            else:
                tokens_str = ', '.join([f"'{token}'" for token, _ in variants])
                sample_display.append(f"'{keyword}' -> [{tokens_str}] ({len(variants)} variants)")
        
        total_token_mappings = sum(len(variants) for _, variants in valid_tuples)
        fuzzy_info = f" ({len(concept_fuzzy_matches)} fuzzy)" if concept_fuzzy_matches else ""
        desc_info = f" (+{len(combined_keywords) - len(keywords)} desc words)" if len(combined_keywords) > len(keywords) else ""
        print(f"\n[{concept_num}/{total_concepts}] {concept}: {len(valid_tuples)}/{len(combined_keywords)} valid{fuzzy_info}{desc_info} ({validation_results[concept]['validity_percentage']:.1f}%) - {total_token_mappings} total mappings")
        print(f"  Sample: {', '.join(sample_display)}")
    
    return validation_results, total_valid, total_invalid, invalid_keywords_by_concept, all_invalid_keywords, fuzzy_matches

def print_validation_summary(validation_results, total_valid, total_invalid, all_invalid_keywords, fuzzy_matches):
    """Print a comprehensive validation summary."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total_keywords = total_valid + total_invalid
    overall_percentage = (total_valid / total_keywords * 100) if total_keywords > 0 else 0
    
    # Count total fuzzy matches
    total_fuzzy = sum(len(matches) for matches in fuzzy_matches.values())
    exact_matches = total_valid - total_fuzzy
    
    print(f"Total keywords processed: {total_keywords}")
    print(f"Valid keywords: {total_valid} ({overall_percentage:.1f}%)")
    print(f"  - Exact matches: {exact_matches}")
    print(f"  - Fuzzy matches: {total_fuzzy}")
    print(f"Invalid keywords: {total_invalid} ({100-overall_percentage:.1f}%)")
    print(f"Unique invalid tokens: {len(all_invalid_keywords)}")
    
    # Concept-level statistics
    print(f"\nConcepts processed: {len(validation_results)}")
    
    # Find best and worst performing concepts
    concept_scores = [(concept, results['validity_percentage']) 
                     for concept, results in validation_results.items()]
    concept_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nBest performing concepts:")
    for concept, score in concept_scores[:5]:
        fuzzy_count = validation_results[concept].get('fuzzy_count', 0)
        fuzzy_info = f" ({fuzzy_count} fuzzy)" if fuzzy_count > 0 else ""
        print(f"  {concept}: {score:.1f}% valid{fuzzy_info}")
    
    print(f"\nWorst performing concepts:")
    for concept, score in concept_scores[-5:]:
        fuzzy_count = validation_results[concept].get('fuzzy_count', 0)
        fuzzy_info = f" ({fuzzy_count} fuzzy)" if fuzzy_count > 0 else ""
        print(f"  {concept}: {score:.1f}% valid{fuzzy_info}")
    
    # Fuzzy match analysis
    if total_fuzzy > 0:
        print(f"\nFUZZY MATCH EXAMPLES:")
        fuzzy_examples = []
        for concept, matches in fuzzy_matches.items():
            for match in matches[:2]:  # First 2 per concept
                tokens_str = ', '.join([f"'{token}'" for token in match['matched_tokens']])
                ids_str = ', '.join([str(id) for id in match['token_ids']])
                fuzzy_examples.append(f"  '{match['original_keyword']}' -> {tokens_str} (IDs: {ids_str})")
        
        for example in fuzzy_examples[:10]:  # Show first 10 examples
            print(example)

def print_invalid_keywords_analysis(all_invalid_keywords, invalid_keywords_by_concept):
    """Analyze and print information about invalid keywords."""
    print("\n" + "="*80)
    print("INVALID KEYWORDS ANALYSIS")
    print("="*80)
    
    print(f"Total unique invalid keywords: {len(all_invalid_keywords)}")
    
    if len(all_invalid_keywords) > 0:
        print(f"\nMost common invalid keywords:")
        
        # Count frequency of invalid keywords across concepts
        invalid_frequency = defaultdict(int)
        for concept, invalid_list in invalid_keywords_by_concept.items():
            for keyword in invalid_list:
                invalid_frequency[keyword] += 1
        
        # Sort by frequency
        frequent_invalid = sorted(invalid_frequency.items(), key=lambda x: x[1], reverse=True)
        
        for keyword, frequency in frequent_invalid[:20]:  # Top 20
            concepts_with_keyword = [concept for concept, invalid_list in invalid_keywords_by_concept.items() 
                                   if keyword in invalid_list]
            print(f"  '{keyword}': appears in {frequency} concepts")
        
        # Analyze patterns in invalid keywords
        print(f"\nInvalid keyword patterns:")
        
        # Check for common issues
        starts_with_capital = sum(1 for kw in all_invalid_keywords if kw and kw[0].isupper())
        has_spaces = sum(1 for kw in all_invalid_keywords if ' ' in kw)
        has_punctuation = sum(1 for kw in all_invalid_keywords if any(c in kw for c in '.,!?;:'))
        very_long = sum(1 for kw in all_invalid_keywords if len(kw) > 20)
        
        print(f"  Starting with capital letter: {starts_with_capital}")
        print(f"  Containing spaces: {has_spaces}")
        print(f"  Containing punctuation: {has_punctuation}")
        print(f"  Very long (>20 chars): {very_long}")

def save_cleaned_keywords(validation_results, output_filename='concept_keyword_ids.json'):
    """Save actual vocabulary token-ID tuples for each concept with all variants found."""
    # Create output directory and use it for the filename
    output_dir = "token-results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\nSaving concept token-ID tuples to {output_path}...")
    
    concept_keyword_ids = {}
    total_original_keywords = 0
    total_token_mappings = 0
    
    for concept, results in validation_results.items():
        # Extract actual vocabulary tokens found (now includes description words)
        vocab_token_tuples = []
        for keyword, variants in results['valid_tuples']:
            # variants is a list of (vocab_token, token_id) tuples
            for vocab_token, token_id in variants:
                vocab_token_tuples.append((vocab_token, token_id))
                total_token_mappings += 1
            total_original_keywords += 1
        
        concept_keyword_ids[concept] = vocab_token_tuples
    
    with open(output_path, 'w') as f:
        json.dump(concept_keyword_ids, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {total_original_keywords} words (keywords + description) -> {total_token_mappings} vocabulary tokens across {len(concept_keyword_ids)} concepts")
    print(f"Average {total_token_mappings/total_original_keywords:.2f} vocabulary token variants per word")
    
    # Also save a human-readable summary
    summary_filename = output_filename.replace('.json', '_summary.txt')
    summary_path = os.path.join(output_dir, summary_filename)
    with open(summary_path, 'w') as f:
        f.write("CONCEPT VOCABULARY TOKEN-ID MAPPING SUMMARY (KEYWORDS + DESCRIPTIONS)\n")
        f.write("=" * 70 + "\n\n")
        
        for concept, tuples in concept_keyword_ids.items():
            # Group by vocabulary token (since we now store vocab tokens, not original keywords)
            token_groups = defaultdict(list)
            for vocab_token, token_id in tuples:
                token_groups[vocab_token].append(token_id)
            
            results = validation_results[concept]
            
            f.write(f"Concept: {concept}\n")
            f.write(f"Total vocabulary tokens found: {len(token_groups)}\n")
            f.write(f"Total token mappings: {len(tuples)}\n")
            f.write(f"Original keywords: {results.get('original_keywords_count', 0)}\n")
            f.write(f"Description words: {results.get('description_words_count', 0)}\n")
            
            # Show original keywords that led to fuzzy matches for this concept
            fuzzy_matches = results.get('fuzzy_matches', [])
            if fuzzy_matches:
                f.write(f"Fuzzy matches: {len(fuzzy_matches)}\n")
                f.write("Fuzzy match examples (original -> vocabulary tokens):\n")
                for match in fuzzy_matches[:5]:  # Show first 5 fuzzy matches
                    tokens_str = ', '.join([f"'{token}'" for token in match['matched_tokens']])
                    ids_str = ', '.join([str(id) for id in match['token_ids']])
                    f.write(f"  '{match['original_keyword']}' -> {tokens_str} (IDs: {ids_str})\n")
            
            f.write("Sample vocabulary token mappings:\n")
            for vocab_token, token_ids in list(token_groups.items())[:10]:  # Show first 10
                if len(token_ids) == 1:
                    f.write(f"  '{vocab_token}' -> ID: {token_ids[0]}\n")
                else:
                    f.write(f"  '{vocab_token}' -> IDs: {token_ids} (appears {len(token_ids)} times)\n")
            f.write("\n" + "-" * 50 + "\n\n")
    
    print(f"Also saved human-readable summary to {summary_path}")
    return concept_keyword_ids

def save_validation_report(validation_results, invalid_keywords_by_concept, 
                          total_valid, total_invalid, output_filename='validation_report.json'):
    """Save detailed validation report as JSON."""
    output_dir = "token-results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Saving detailed validation report to {output_path}...")
    
    report = {
        'summary': {
            'total_concepts': len(validation_results),
            'total_keywords': total_valid + total_invalid,
            'total_valid': total_valid,
            'total_invalid': total_invalid,
            'overall_validity_percentage': (total_valid / (total_valid + total_invalid) * 100) if (total_valid + total_invalid) > 0 else 0
        },
        'concept_details': validation_results,
        'invalid_keywords_by_concept': dict(invalid_keywords_by_concept)
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

def main():
    """Main validation function."""
    print("Keyword Vocabulary Validation")
    print("="*50)
    
    # Load required data
    vocabulary = load_vocabulary()
    if vocabulary is None:
        return
    
    keywords_data = load_generated_keywords()
    if keywords_data is None:
        return
    
    # Perform validation
    validation_results, total_valid, total_invalid, invalid_keywords_by_concept, all_invalid_keywords, fuzzy_matches = validate_keywords(
        keywords_data, vocabulary
    )
    
    # Print comprehensive results
    print_validation_summary(validation_results, total_valid, total_invalid, all_invalid_keywords, fuzzy_matches)
    print_invalid_keywords_analysis(all_invalid_keywords, invalid_keywords_by_concept)
    
    # Save results
    save_cleaned_keywords(validation_results)
    save_validation_report(validation_results, invalid_keywords_by_concept, total_valid, total_invalid)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("Files created in token-results/:")
    print("  - concept_keyword_ids.json: (vocabulary_token, token_id) tuples for each concept")
    print("  - concept_keyword_ids_summary.txt: Human-readable summary")
    print("  - validation_report.json: Detailed validation analysis")
    
    if total_invalid > 0:
        print(f"\nRecommendation: {total_invalid} invalid keywords found.")
        print("Consider reviewing the prompt or model output to improve vocabulary compliance.")
    else:
        print("\n✓ All keywords are valid vocabulary tokens!")

if __name__ == "__main__":
    main()
