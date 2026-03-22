#!/bin/bash
#SBATCH --job-name=olmoe-loss-free
#SBATCH --output=olmoe-loss-free.out
#SBATCH --error=olmoe-loss-free.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate

# Initialize the module system (not sourced by default in SLURM batch jobs)
for _mod_init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
                 /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash \
                 /usr/local/Modules/init/bash /opt/modules/init/bash; do
    [ -f "$_mod_init" ] && source "$_mod_init" && break
done
module load cuda/12.9 2>/dev/null || true

export WANDB_MODE=online
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)

# Per-job venv in node-local /tmp — isolated from shared conda, no NFS contention
VENV_DIR="/tmp/olmoe_venv_loss_free"
python -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python /home/infres/zakil-22/OLMoE/train_server_loss_free.py
