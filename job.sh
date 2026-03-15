#!/bin/bash
#SBATCH --job-name=64exp8
#SBATCH --output=64exp8.out
#SBATCH --error=64exp8.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate
conda activate olmoe

export PYTHONNOUSERSITE=1
PIP="$HOME/miniconda3/envs/olmoe/bin/pip"

# $PIP install wandb torchmetrics datasets scikit-learn safetensors msgspec

export CUDA_HOME=/usr/local/cuda-12.5
export TORCH_CUDA_ARCH_LIST="8.6"
# $PIP install ninja
# $PIP install stanford-stk
# $PIP install git+https://github.com/Muennighoff/megablocks.git@olmoe --no-build-isolation --no-deps

export WANDB_MODE=online
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env 2>/dev/null | cut -d '=' -f 2)

bash scripts/olmoe-gantry.sh