#!/bin/bash
# Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

export PATH="$HOME/miniconda3/bin:$PATH"

# Create and activate conda env
if ! conda env list | grep -q olmoe; then
    conda create -y -n olmoe python=3.10
fi
source ~/miniconda3/bin/activate
conda activate olmoe

# Install dependencies
pip install --upgrade pip
pip install wandb torchmetrics datasets scikit-learn safetensors msgspec ninja

# Install CUDA-specific packages
export TORCH_CUDA_ARCH_LIST="8.6"

# Set WANDB API key if available
export WANDB_MODE=online
export WANDB_API_KEY=$(grep WANDB_API_KEY .env 2>/dev/null | cut -d '=' -f 2)

pip install -U "huggingface_hub[cli]"
# Download tokenized data directly
mkdir -p data
hf download --repo-type dataset iliasslasri/tokenized-OLMoE-mix --include "*.npy" --local-dir data/
git submodule update --init --recursive
cd OLMo
pip install -e .
cd ..
pip install git+https://github.com/Muennighoff/megablocks.git@olmoe

pip cache purge
conda clean --all -y

# Set PyTorch distributed environment variables (example for single node, 1 GPU)
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Launch your training job (edit as needed)
python OLMo/scripts/train.py configs/olmoe-dense.yml