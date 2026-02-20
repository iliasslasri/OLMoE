#!/bin/bash
#SBATCH --job-name=olmoe-multigpu
#SBATCH --output=olmoe-multigpu.out
#SBATCH --error=olmoe-multigpu.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate
conda activate olmoe
export WANDB_MODE=online
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)

bash scripts/olmoe-gantry.sh