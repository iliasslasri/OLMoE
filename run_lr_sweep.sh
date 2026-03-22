#!/bin/bash
# run_lr_sweep.sh — SLURM wrapper for one LR sweep scale
#
# Accepts one argument: the d_model to sweep (64 | 128 | 192 | 256)
# Runs all 5 learning rates for that scale sequentially.
#
# Usage (interactive):
#   CUDA_VISIBLE_DEVICES=0 bash run_lr_sweep.sh 64
#   CUDA_VISIBLE_DEVICES=0 bash run_lr_sweep.sh 128
#   CUDA_VISIBLE_DEVICES=1 bash run_lr_sweep.sh 192
#   CUDA_VISIBLE_DEVICES=2 bash run_lr_sweep.sh 256
#
# As SLURM jobs (one job per scale, each on its own GPU):
#   sbatch --gres=gpu:1 run_lr_sweep.sh 64
#   sbatch --gres=gpu:1 run_lr_sweep.sh 128
#   sbatch --gres=gpu:1 run_lr_sweep.sh 192
#   sbatch --gres=gpu:1 run_lr_sweep.sh 256
#
#SBATCH --job-name=lr-sweep
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

DIM="${1:?Usage: $0 <d_model>  (64 | 128 | 192 | 256)}"

# ── Activate environment (same pattern as job.sh) ─────────────────────────────
source ~/miniconda3/bin/activate 2>/dev/null || true

for _mod_init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
                 /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash \
                 /usr/local/Modules/init/bash /opt/modules/init/bash; do
    [ -f "$_mod_init" ] && source "$_mod_init" && break
done
module load cuda/12.9 2>/dev/null || true

# ── W&B credentials ───────────────────────────────────────────────────────────
export WANDB_MODE=online
if [ -f ~/.env ]; then
    export WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' ~/.env | cut -d= -f2- | tr -d '[:space:]')
fi

# ── Run the sweep ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

python "$SCRIPT_DIR/train_lr_sweep.py" --dim "$DIM"
