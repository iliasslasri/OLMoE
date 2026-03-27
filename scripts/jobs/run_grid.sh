#!/usr/bin/env bash
# run_grid.sh — Sequential grid runner for Phase 1 MoE scaling law experiment.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_grid.sh configs/grid/d064_*.yml &
#   CUDA_VISIBLE_DEVICES=1 bash run_grid.sh configs/grid/d128_*.yml configs/grid/d192_*.yml &
#   CUDA_VISIBLE_DEVICES=2 bash run_grid.sh configs/grid/d256_*.yml &

set -euo pipefail
mkdir -p logs/grid

for cfg in "$@"; do
    run_name=$(basename "$cfg" .yml)
    log="logs/grid/${run_name}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START  ${run_name}"
    python -m torch.distributed.run --nproc-per-node=1 OLMo/scripts/train.py "$cfg" \
        > "$log" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISH ${run_name}"
done
