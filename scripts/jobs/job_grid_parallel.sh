#!/bin/bash
#SBATCH --job-name=grid-parallel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ── Usage ──────────────────────────────────────────────────────────────────────
# sbatch job_grid_parallel.sh <config_glob> [n_parallel]
#
# Examples:
#   sbatch job_grid_parallel.sh "configs/grid/d008_*_T300M.yml" 20
#   sbatch job_grid_parallel.sh "configs/grid/d016_*_T150M.yml" 25
#   sbatch job_grid_parallel.sh "configs/grid/d032_*_T75M.yml"  10
#
# n_parallel defaults to the total number of matching configs (all at once).
# If n_parallel < total, configs are processed in sequential batches of n_parallel.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG_GLOB="${1:?Usage: sbatch job_grid_parallel.sh <config_glob> [n_parallel]}"
N_PARALLEL="${2:-0}"   # 0 = all at once

OLMOE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STAGGER_S=15           # seconds between consecutive launches (avoids pip race)

mkdir -p "${OLMOE_DIR}/logs/grid"
echo "=== Parallel grid | job ${SLURM_JOB_ID} | node ${SLURM_NODELIST} ==="
echo "Config glob : ${CONFIG_GLOB}"
echo "N parallel  : ${N_PARALLEL} (0=all)"
echo "Started     : $(date)"

# ── Env setup ─────────────────────────────────────────────────────────────────
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

VENV_DIR="/tmp/olmoe_venv_parallel"
rm -rf "${VENV_DIR}"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Python: $(python --version)"

# ── Collect configs ───────────────────────────────────────────────────────────
cd "${OLMOE_DIR}"
mapfile -t ALL_CONFIGS < <(ls ${CONFIG_GLOB} 2>/dev/null | sort)
TOTAL=${#ALL_CONFIGS[@]}

if [[ ${TOTAL} -eq 0 ]]; then
    echo "ERROR: no configs found for glob '${CONFIG_GLOB}'"
    echo "       Run: python generate_grid_configs.py"
    exit 1
fi

[[ ${N_PARALLEL} -eq 0 ]] && N_PARALLEL=${TOTAL}
echo "Found ${TOTAL} configs — running ${N_PARALLEL} in parallel per batch"

# ── Parallel launcher ─────────────────────────────────────────────────────────
batch_start=0
while [[ ${batch_start} -lt ${TOTAL} ]]; do
    batch_end=$(( batch_start + N_PARALLEL ))
    [[ ${batch_end} -gt ${TOTAL} ]] && batch_end=${TOTAL}
    batch=( "${ALL_CONFIGS[@]:${batch_start}:$(( batch_end - batch_start ))}" )

    echo ""
    echo "── Batch [$(( batch_start+1 ))–${batch_end}] / ${TOTAL} ──"
    pids=()
    for idx in "${!batch[@]}"; do
        cfg="${batch[$idx]}"
        name=$(basename "${cfg}" .yml)

        # Skip already-completed runs (log file non-empty)
        log="${OLMOE_DIR}/logs/grid/${name}.log"
        if [[ -s "${log}" ]]; then
            echo "  [skip] ${name} — already done"
            continue
        fi

        # Stagger to avoid concurrent pip/shadow-cuda setup races
        [[ ${idx} -gt 0 ]] && sleep ${STAGGER_S}

        echo "  [launch] ${name}"
        python "${OLMOE_DIR}/train_server_grid.py" \
            --configs "${cfg}" \
            > "${log}" 2>&1 &
        pids+=($!)
    done

    echo "  Waiting for ${#pids[@]} processes (pids: ${pids[*]})…"
    for pid in "${pids[@]}"; do
        wait "${pid}"
        status=$?
        [[ ${status} -ne 0 ]] && echo "  [warn] pid ${pid} exited with code ${status}"
    done
    echo "  Batch done."

    batch_start=${batch_end}
done

echo ""
echo "=== All batches finished: $(date) ==="
