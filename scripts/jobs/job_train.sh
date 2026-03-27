#!/bin/bash
#SBATCH --job-name=olmoe-train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Usage: sbatch job_train.sh <mode>
#   mode: random | lb_0001 | lb_001 | loss_free
#
# Examples:
#   sbatch job_train.sh random
#   sbatch job_train.sh lb_0001
#   sbatch job_train.sh lb_001
#   sbatch job_train.sh loss_free

MODE="${1:?Usage: sbatch job_train.sh <mode>  (random|lb_0001|lb_001|loss_free)}"
OLMOE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

case "${MODE}" in
    random)    SCRIPT="train_server.py" ;;
    lb_0001)   SCRIPT="train_server_lb_0001.py" ;;
    lb_001)    SCRIPT="train_server_lb_001.py" ;;
    loss_free) SCRIPT="train_server_loss_free.py" ;;
    *) echo "ERROR: unknown mode '${MODE}'. Choose: random | lb_0001 | lb_001 | loss_free"; exit 1 ;;
esac

echo "=== OLMoE train | mode=${MODE} | job ${SLURM_JOB_ID} | node ${SLURM_NODELIST} ==="
echo "Started: $(date)"

source ~/miniconda3/bin/activate

for _mod in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
            /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash \
            /usr/local/Modules/init/bash /opt/modules/init/bash; do
    [ -f "$_mod" ] && source "$_mod" && break
done
module load cuda/12.9 2>/dev/null || true

export WANDB_MODE=online
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)

VENV_DIR="/tmp/olmoe_venv_${MODE}"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Python: $(python --version)"

python "${OLMOE_DIR}/scripts/${SCRIPT}"

echo "=== Job finished: $(date) ==="
