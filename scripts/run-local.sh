#!/usr/bin/env bash
set -ex

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load W&B API key
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)
export WANDB_MODE=online

export OLMO_TASK=model
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/OLMo:${PYTHONPATH}"
export OMP_NUM_THREADS=4

CONDA_PYTHON=/home/tristan/miniconda3/envs/olmoe_env/bin/python

$CONDA_PYTHON -m torch.distributed.run \
  --nproc-per-node=1 --nnodes=1 --node_rank=0 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
  OLMo/scripts/train.py configs/debug/olmoe-local.yml
