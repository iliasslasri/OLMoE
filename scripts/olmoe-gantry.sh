#!/usr/bin/env bash
set -ex

# Use the small debug config
CONFIG_PATH=configs/olmoe-small.yml
ARGS='--save-overwrite --fsdp.sharding_strategy=FULL_SHARD'

# Set environment variables for local execution
# export CUDA_VISIBLE_DEVICES=0 # Let SLURM or the user determine this
export OMP_NUM_THREADS=8
export OLMO_TASK=model
export PYTHONNOUSERSITE=1
export PYTHONPATH=$(pwd)/OLMo
export WANDB_MODE=${WANDB_MODE:-offline}

# Run with torchrun on a single node using the olmoe environment
/home/infres/lasri-22/miniconda3/envs/olmoe/bin/python -m torch.distributed.run --nnodes 1 --node_rank 0 --nproc-per-node 2 \
  --rdzv_id=12347 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
  OLMo/scripts/train.py ${CONFIG_PATH} ${ARGS}