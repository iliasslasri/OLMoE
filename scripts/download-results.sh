#!/usr/bin/env bash
# =============================================================================
# download-results.sh  —  runs LOCALLY, sleeps 24h then downloads results
#
# Called by deploy-vast.sh in background:
#   nohup bash scripts/download-results.sh SSH_HOST SSH_PORT INSTANCE_ID \
#         REMOTE_DIR LOCAL_DIR > /tmp/vast_download.log 2>&1 &
# =============================================================================
set -euo pipefail

SSH_HOST="${1}"
SSH_PORT="${2}"
INSTANCE_ID="${3}"
REMOTE_DIR="${4:-/workspace/OLMoE}"
LOCAL_DIR="${5:-./vast_results}"

REMOTE_USER="root"
SSH_OPTS="-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p $SSH_PORT"
RSYNC_SSH="ssh -o StrictHostKeyChecking=no -p $SSH_PORT"

echo "[download] Started at $(date)"
echo "[download] Will download from ${REMOTE_USER}@${SSH_HOST}:${SSH_PORT}"
echo "[download] Remote dir: $REMOTE_DIR"
echo "[download] Local dir:  $LOCAL_DIR"
echo "[download] Sleeping 24 hours (86400 seconds)..."

sleep 86400

echo "[download] Waking up at $(date) — starting download..."
mkdir -p "${LOCAL_DIR}/runs" "${LOCAL_DIR}/wandb"

# Download checkpoints and training logs
echo "[download] Syncing runs/ (checkpoints + logs)..."
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  "${REMOTE_USER}@${SSH_HOST}:${REMOTE_DIR}/runs/" \
  "${LOCAL_DIR}/runs/"

# Download W&B offline logs (from the wandb/ dir inside the repo)
echo "[download] Syncing wandb/ logs..."
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  "${REMOTE_USER}@${SSH_HOST}:${REMOTE_DIR}/wandb/" \
  "${LOCAL_DIR}/wandb/" 2>/dev/null || echo "[download] No wandb/ dir found (W&B may be online-only)."

# Also grab ~/.local/share/wandb if it exists
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  "${REMOTE_USER}@${SSH_HOST}:~/.local/share/wandb/" \
  "${LOCAL_DIR}/wandb_local/" 2>/dev/null || true

echo "[download] Download complete at $(date)!"
echo "[download] Results saved to: ${LOCAL_DIR}/"
echo ""
echo "[download] ===== SUMMARY ====="
du -sh "${LOCAL_DIR}/"* 2>/dev/null || true
echo "[download] =================="
echo ""
echo "[download] Instance $INSTANCE_ID is still running."
echo "[download] To terminate it:  vastai destroy instance $INSTANCE_ID"
