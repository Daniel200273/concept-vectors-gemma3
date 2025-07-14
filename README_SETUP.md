# Gemma Concept Vector Finding - Environment Setup Guide

## Quick Setup with Conda (Recommended)

### Option 1: Automated Setup

Run the setup script:

```bash
cd /Users/daniel/Desktop/thesis_material
chmod +x setup_environment.sh
./setup_environment.sh
```

### Option 2: Manual Setup

1. **Create conda environment from file:**

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**

   ```bash
   conda activate gemma_concept_env
   ```

3. **Install Jupyter kernel:**

   ```bash
   python -m ipykernel install --user --name=gemma_concept_env --display-name="Gemma Concept Vectors"
   ```

4. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

## GPU vs CPU Setup

### For GPU Support (NVIDIA CUDA)

- Keep the `pytorch-cuda=11.8` line in `environment.yml`
- Requires NVIDIA GPU with CUDA 11.8+ support

### For CPU-Only Usage

- Remove or comment out the `pytorch-cuda=11.8` line in `environment.yml`
- The setup will automatically use CPU-only PyTorch

## Environment Management

### Daily Usage

```bash
# Activate environment
conda activate gemma_concept_env

# Start Jupyter
jupyter notebook

# Deactivate when done
conda deactivate
```

### Updating Dependencies

```bash
conda activate gemma_concept_env
conda env update -f environment.yml
```

### Removing Environment

```bash
conda env remove --name gemma_concept_env
```

## Troubleshooting

### If conda is not installed:

1. Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Install following the instructions for macOS
3. Restart terminal and try again

### If you get memory errors:

- The Gemma model requires ~3-4GB GPU memory or ~6GB RAM
- Consider using a smaller model variant or CPU-only mode

### If packages are not found:

```bash
conda update conda
conda env update -f environment.yml --prune
```

## File Structure

```
thesis_material/
├── code/
│   └── concept-vectors-gemma3.ipynb
├── environment.yml          # Conda environment specification
├── requirements.txt          # Pip requirements (backup)
├── setup_environment.sh      # Automated setup script
└── README_SETUP.md          # This file
```
