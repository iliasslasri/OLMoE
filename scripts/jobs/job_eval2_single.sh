#!/bin/bash
#SBATCH --job-name=eval2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=35:30:00

# Usage: sbatch job_eval2_single.sh <path/to/config.yml>
CONFIG="${1:?Usage: sbatch job_eval2_single.sh <config.yml>}"
OLMOE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${OLMOE_DIR}/logs/eval2"

echo "=== Phase-2 Eval | $(basename ${CONFIG}) | job ${SLURM_JOB_ID} | node ${SLURM_NODELIST} ==="
echo "Started: $(date)"

# ── Environment setup (mirrors job_grid.sh) ────────────────────────────────────
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

# Per-job venv in /tmp — isolated, no NFS contention
VENV_DIR="/tmp/olmoe_venv_eval2_$(basename ${CONFIG%.yml})"
rm -rf "${VENV_DIR}"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Python: $(python --version)"

cd "${OLMOE_DIR}"
python "${OLMOE_DIR}/train_server_grid.py" --configs "${CONFIG}"

echo "=== Job finished: $(date) ==="
