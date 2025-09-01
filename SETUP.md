# Setup Instructions

## Environment Variables

This project requires a HuggingFace token to access the Gemma models. Follow these steps to set it up:

1. **Get your HuggingFace token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token or copy an existing one

2. **Set the environment variable:**
   
   **Option A: Export in your shell (temporary)**
   ```bash
   export HF_TOKEN=your_actual_token_here
   ```
   
   **Option B: Add to your shell profile (permanent)**
   ```bash
   echo 'export HF_TOKEN=your_actual_token_here' >> ~/.bashrc
   source ~/.bashrc
   ```
   
   **Option C: Use a .env file**
   ```bash
   cp .env.example .env
   # Edit .env and set your actual token
   # Then load it with: source .env
   ```

3. **Verify the setup:**
   ```bash
   echo $HF_TOKEN
   ```

## Security Note

- Never commit actual tokens to version control
- Use environment variables to keep tokens secure
- The `.env.example` file shows the required format without exposing actual tokens

## Running the Scripts

Once the environment variable is set, you can run any of the scripts that require HuggingFace model access:

```bash
python code/concept_val_test/advanced_concept_validation.py
python code/projection/run_pipeline.py
# etc.
```
