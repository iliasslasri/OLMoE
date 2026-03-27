#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00

# Usage: sbatch job_grid.sh <dim> [tok_glob]
#   dim:      model width (e.g. 64 | 128 | 192 | 256)
#   tok_glob: token label pattern, default "*" (all tokens)
#             use quoted space-separated labels for multiple: "T8M T16M"
#
# Examples:
#   sbatch job_grid.sh 64              # all token budgets for d=64
#   sbatch job_grid.sh 128 "T8M T16M" # phase-1 subset for d=128
#   sbatch job_grid.sh 256 T16M        # single token budget

DIM="${1:?Usage: sbatch job_grid.sh <dim> [tok_glob]}"
TOKS="${2:-*}"
OLMOE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${OLMOE_DIR}/logs/grid"

echo "=== Grid | d_model=${DIM} tokens=${TOKS} | job ${SLURM_JOB_ID} | node ${SLURM_NODELIST} ==="
echo "Started: $(date)"

set --
source ~/miniconda3/bin/activate

for _mod in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
            /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash \
            /usr/local/Modules/init/bash /opt/modules/init/bash; do
    [ -f "$_mod" ] && source "$_mod" && break
done
module load cuda/12.9 2>/dev/null || true

export WANDB_MODE=online
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)

VENV_DIR="/tmp/olmoe_venv_grid_d${DIM}_${TOKS// /_}"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Python: $(python --version)"

CONFIGS=""
for TOK in ${TOKS}; do
    CONFIGS="${CONFIGS} ${OLMOE_DIR}/configs/grid/d$(printf '%03d' ${DIM})_*_${TOK}.yml"
done

cd "${OLMOE_DIR}"
python "${OLMOE_DIR}/scripts/train_server_grid.py" --configs ${CONFIGS}

echo "=== Job finished: $(date) ==="
