# Keyword Generation Prompt for Concept Vector Research

## Master Prompt Template

```
You are an expert in computational linguistics and concept identification. I need you to generate exactly 200 tokens/keywords that are closely related to the concept "{CONCEPT_NAME}" for use in language model concept vector identification.

**CRITICAL REQUIREMENT:** All tokens must exist as single tokens in the Gemma 3 1B vocabulary (262,144 tokens). Avoid multi-word phrases that would be split during tokenization.

The keywords should help a neural network identify when this concept is being discussed or referenced in text. Include diverse types of tokens:

**Requirements:**
1. Generate exactly 200 unique tokens
2. Each token must be a single vocabulary item (not split by tokenizer)
3. Include various token types as specified below
4. Prioritize tokens that would appear in the same context as the concept
5. Consider how the concept might be referenced indirectly
6. Include both common and specific terms
7. Format as a simple comma-separated list

**Token Selection Guidelines:**
- Use single words, not phrases (e.g., "Harry" not "Harry Potter")
- Include common abbreviations and acronyms
- Consider subword tokens (e.g., prefixes, suffixes)
- Avoid spaces, hyphens in multi-word expressions
- Check that tokens would likely exist in a 262K vocabulary

**Token Categories to Include:**

**Primary Identifiers (15-20 tokens):**
- Main names, titles, or direct references
- Official names and common abbreviations
- Key proper nouns

**Core Related Terms (30-40 tokens):**
- Central concepts directly associated
- Technical terminology specific to this domain
- Essential vocabulary

**Contextual Words (40-50 tokens):**
- Words that commonly appear in the same sentences/paragraphs
- Action words, descriptive terms, and modifiers
- Situational context indicators

**Domain-Specific Vocabulary (30-40 tokens):**
- Specialized terms from the relevant field
- Professional jargon and technical language
- Industry-specific terminology

**Associated Entities (20-30 tokens):**
- Related people, places, organizations
- Connected concepts and subcategories
- Comparative or contrasting terms

**Common Phrases & Compounds (15-25 tokens):**
- Single-word compounds that exist as vocabulary tokens
- Common abbreviations and acronyms
- Subword tokens and prefixes/suffixes
- Avoid hyphenated or space-separated phrases

**Emotional/Descriptive Language (10-15 tokens):**
- Adjectives commonly used to describe this concept
- Emotional associations and connotations
- Qualitative descriptors

**Format your response as:**
"{CONCEPT_NAME}": [
"token1", "token2", "token3", ... (exactly 200 tokens)
]

**Concept to process: "{CONCEPT_NAME}"**

Remember: These tokens will be used to identify concept vectors in neural networks, so think about what words would co-occur with this concept in natural language text.
```

## Category-Specific Variations

### For Technology Concepts:

```
Additional focus areas for technology concepts:
- Programming languages, frameworks, tools
- Technical specifications and standards
- Company names and product names
- Version numbers and acronyms
- Implementation details and use cases
- Related protocols and methodologies
```

### For Entertainment/Media Concepts:

```
Additional focus areas for entertainment concepts:
- Character names and relationships
- Plot elements and locations
- Creator/author/director names
- Related franchises and spin-offs
- Fan terminology and slang
- Production companies and platforms
```

### For Geographic Concepts:

```
Additional focus areas for geographic concepts:
- Cities, regions, and landmarks
- Cultural and linguistic terms
- Historical references
- Political and administrative terms
- Economic and demographic indicators
- Natural features and climate terms
```

### For Scientific Concepts:

```
Additional focus areas for scientific concepts:
- Technical processes and methods
- Related theories and principles
- Measurement units and scales
- Research terminology
- Applications and use cases
- Historical discoveries and researchers
```

## Batch Processing Prompt

```
I have a list of {N} concepts for which I need keyword generation. Please process them one by one, generating exactly 200 tokens for each concept following the guidelines above.

Concepts to process:
1. {CONCEPT_1}
2. {CONCEPT_2}
3. {CONCEPT_3}
... (continue for all concepts)

For each concept, provide the output in this JSON format:
"{CONCEPT_NAME}": [
"token1", "token2", "token3", ... (exactly 200 tokens)
]

Process all concepts maintaining consistent quality and coverage across all token categories.
```

## Quality Check Prompt

```
Review the following keyword list for the concept "{CONCEPT_NAME}" and ensure:

1. Exactly 200 unique tokens
2. No duplicates or near-duplicates
3. All tokens are relevant to concept identification
4. Good coverage across different token types
5. Mix of common and specific terms
6. Proper formatting

Keywords: [paste generated keywords here]

Suggest improvements or corrections if needed.
```

## Usage Instructions

1. **Single Concept**: Use the Master Prompt Template, replacing `{CONCEPT_NAME}` with your specific concept
2. **Multiple Concepts**: Use the Batch Processing Prompt for efficiency
3. **Quality Assurance**: Use the Quality Check Prompt to verify results
4. **Category-Specific**: Add the relevant category variation to the master prompt

## Example Usage

For "Harry Potter":

```
You are an expert in computational linguistics and concept identification. I need you to generate exactly 200 tokens/keywords that are closely related to the concept "Harry Potter" for use in language model concept vector identification.

[... rest of master prompt ...]

**Concept to process: "Harry Potter"**
```

This prompt structure ensures comprehensive, high-quality keyword generation suitable for your concept vector research with Gemma 3 1B.
