# Cluster Setup Guide for Concept Vectors Gemma3

## Quick Setup on Computing Cluster

### 1. Clone the Repository

```bash
git clone https://github.com/Daniel200273/concept-vectors-gemma3.git
cd concept-vectors-gemma3
```

### 2. Create Conda Environment

```bash
# Create environment from yml file (recommended)
conda env create -f environment.yml

# Activate the environment
conda activate gemma_concept_env
```

### 3. Alternative: Manual Environment Creation

If the yml file doesn't work on your cluster:

```bash
# Create basic environment
conda create -n gemma_concept_env python=3.10 -y
conda activate gemma_concept_env

# Install PyTorch (adjust for your cluster's CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

### 4. GPU Setup (For CUDA Clusters)

If your cluster has GPUs and CUDA support:

```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import AutoTokenizer; print('Transformers OK')"
```

### 6. Run Jupyter (if needed)

```bash
# Start jupyter lab/notebook
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Or use the cluster's preferred method for jupyter
```

## Common Cluster-Specific Notes

- **SLURM clusters**: You may need to request GPU nodes with `salloc` or `srun`
- **Module systems**: Load required modules first (e.g., `module load cuda/12.1`)
- **Storage**: Ensure you have enough space (~10GB for models + data)
- **Memory**: Gemma 3 1B requires ~4-8GB GPU memory or 8-12GB system RAM

## Troubleshooting

1. **Import errors**: Check that all packages are installed in the correct environment
2. **CUDA issues**: Verify PyTorch CUDA version matches your cluster's CUDA
3. **Memory errors**: Use smaller batch sizes or enable CPU offloading
4. **Network issues**: Some clusters require proxy settings for downloading models

## Quick Test

```bash
cd code
python gemma3-playground.py
```
