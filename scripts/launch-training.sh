#!/usr/bin/env bash
# =============================================================================
# launch-training.sh  —  runs ON the Vast.ai instance
#
# Launches torchrun in a detached tmux session so it survives SSH disconnect.
# Called via deploy-vast.sh after remote-setup.sh completes.
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMoE}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/config.yml}"
SESSION="olmoe_train"

cd "$REPO_DIR"

# Load W&B key
WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)

# Count GPUs
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "[launch] Detected $N_GPU GPU(s)"

# Check PCIe P2P availability (determines NCCL transport efficiency)
echo "[launch] GPU topology:"
nvidia-smi topo -m 2>/dev/null || true
echo "[launch] PCIe P2P access:"
python3 -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    for j in range(n):
        if i != j:
            ok = torch.cuda.can_device_access_peer(i, j)
            print(f'  GPU{i}->GPU{j}: P2P={ok}')
" 2>/dev/null || true

# Build the training command
TRAIN_CMD="
export PYTHONPATH=${REPO_DIR}/OLMo:\$PYTHONPATH
export OLMO_TASK=model
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=${WANDB_API_KEY}
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=0

cd ${REPO_DIR}

python3 -m torch.distributed.run \\
  --nproc-per-node=${N_GPU} \\
  --nnodes=1 --node_rank=0 \\
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \\
  OLMo/scripts/train.py ${TRAIN_CONFIG} \\
  --save_overwrite \\
  2>&1 | tee runs/training_\$(date +%Y%m%d_%H%M%S).log
"

# Kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Launch in new detached tmux session
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "$TRAIN_CMD" Enter

echo "[launch] Training running in tmux session '$SESSION'"
echo "[launch] Attach with: tmux attach -t $SESSION"
echo "[launch] Logs: ${REPO_DIR}/runs/training_*.log"

# Brief wait to confirm it started
sleep 5
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] tmux session is alive — training started successfully."
else
  echo "WARNING: tmux session died immediately. Check logs." >&2
fi
