#!/usr/bin/env bash
# =============================================================================
# deploy-vast.sh  —  Full deployment pipeline for Vast.ai
#
# Usage (from repo root):
#   bash scripts/deploy-vast.sh
#
# Requires:
#   - vastai CLI configured (vastai show instances should work)
#   - WANDB_API_KEY in ~/.env  (format: WANDB_API_KEY=...)
#   - Local tokenized data in data/tokenized/
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REMOTE_USER="root"
REMOTE_DIR="/workspace/OLMoE"
DOCKER_IMAGE="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel"
DISK_GB=100
TRAIN_CONFIG="configs/config.yml"
LOCAL_RESULTS_DIR="./vast_results"

# ---------------------------------------------------------------------------
# Load credentials
# ---------------------------------------------------------------------------
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)
fi
if [[ -z "$WANDB_API_KEY" ]]; then
  echo "ERROR: WANDB_API_KEY not found in environment or ~/.env" >&2; exit 1
fi
echo "[deploy] W&B key loaded."

# ---------------------------------------------------------------------------
# STEP 1 — Find and rent cheapest 2x RTX 3090
# ---------------------------------------------------------------------------
echo "[1/6] Finding cheapest 2x A100 SXM4 80GB offer..."

OFFER_JSON=$(vastai search offers \
  'gpu_name=A100_SXM4 num_gpus=2 disk_space>=100 reliability>=0.95 inet_up>=100 inet_down>=100' \
  --order "dph_total asc" --limit 1 --raw)

OFFER_ID=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'])")
OFFER_PRICE=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['dph_total'])")
OFFER_COUNTRY=$(echo "$OFFER_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0].get('geolocation','unknown'))")

echo "[1/6] Best offer: ID=$OFFER_ID  \$$OFFER_PRICE/hr  ($OFFER_COUNTRY)"
echo "[1/6] 24h cost estimate: \$$(python3 -c "print(round($OFFER_PRICE*24,2))")"

# ---------------------------------------------------------------------------
# STEP 2 — Create the instance
# ---------------------------------------------------------------------------
echo "[2/6] Creating instance..."

CREATE_OUTPUT=$(vastai create instance "$OFFER_ID" \
  --image "$DOCKER_IMAGE" \
  --disk "$DISK_GB" \
  --ssh --direct --raw)

INSTANCE_ID=$(echo "$CREATE_OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_contract',d.get('id','')))")
echo "[2/6] Instance created: ID=$INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/vast_instance_id.txt

# ---------------------------------------------------------------------------
# STEP 3 — Wait for SSH to become available
# ---------------------------------------------------------------------------
echo "[3/6] Waiting for instance to be running and SSH-ready..."

SSH_HOST=""
SSH_PORT=""
for i in $(seq 1 60); do
  sleep 10
  INST_JSON=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || echo "null")
  STATUS=$(echo "$INST_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('actual_status','unknown'))" 2>/dev/null || echo "unknown")
  echo "  [wait] attempt $i/60 — status: $STATUS"

  if [[ "$STATUS" == "running" ]]; then
    SSH_HOST=$(echo "$INST_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_host',''))")
    SSH_PORT=$(echo "$INST_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_port',''))")
    if [[ -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then
      # Try actual SSH
      if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
             -p "$SSH_PORT" "${REMOTE_USER}@${SSH_HOST}" "echo OK" 2>/dev/null; then
        echo "[3/6] SSH is up: ${REMOTE_USER}@${SSH_HOST}:${SSH_PORT}"
        break
      fi
    fi
  fi
done

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
  echo "ERROR: Instance never became ready. Check: vastai show instance $INSTANCE_ID" >&2
  exit 1
fi

# Save connection info for other scripts
cat > /tmp/vast_conn.env <<EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
REMOTE_USER=$REMOTE_USER
REMOTE_DIR=$REMOTE_DIR
EOF
echo "[3/6] Connection info saved to /tmp/vast_conn.env"

SSH_CMD="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST}"

# ---------------------------------------------------------------------------
# STEP 4 — Clone repo + rsync tokenized data
# ---------------------------------------------------------------------------
echo "[4/6] Cloning OLMoE repo on remote..."
$SSH_CMD bash <<'REMOTE_CLONE'
set -euo pipefail
mkdir -p /workspace
cd /workspace
if [ ! -d "OLMoE" ]; then
  git clone --recurse-submodules https://github.com/iliasslasri/OLMoE.git
fi
cd OLMoE/OLMo
git fetch origin routing/moe-strategies
git checkout routing/moe-strategies
REMOTE_CLONE

echo "[4/6] Rsyncing tokenized data (this may take a few minutes)..."
$SSH_CMD "mkdir -p ${REMOTE_DIR}/data/tokenized"
rsync -avz --progress \
  -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
  "${REPO_ROOT}/data/tokenized/" \
  "${REMOTE_USER}@${SSH_HOST}:${REMOTE_DIR}/data/tokenized/"

# ---------------------------------------------------------------------------
# STEP 5 — Remote setup (no conda — system Python from Docker image)
# ---------------------------------------------------------------------------
echo "[5/6] Running remote setup (pip install, wandb login)..."
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
  scripts/remote-setup.sh \
  "${REMOTE_USER}@${SSH_HOST}:${REMOTE_DIR}/scripts/remote-setup.sh"

$SSH_CMD "WANDB_API_KEY=$WANDB_API_KEY REPO_DIR=$REMOTE_DIR bash ${REMOTE_DIR}/scripts/remote-setup.sh"

# ---------------------------------------------------------------------------
# STEP 6 — Launch training in tmux + start local 24h download timer
# ---------------------------------------------------------------------------
echo "[6/6] Launching training in tmux session..."
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
  scripts/launch-training.sh \
  "${REMOTE_USER}@${SSH_HOST}:${REMOTE_DIR}/scripts/launch-training.sh"

$SSH_CMD "REPO_DIR=$REMOTE_DIR TRAIN_CONFIG=$TRAIN_CONFIG bash ${REMOTE_DIR}/scripts/launch-training.sh"

echo ""
echo "================================================================"
echo " Training launched!"
echo " Monitor: ssh -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST}"
echo "          tmux attach -t olmoe_train"
echo " Instance ID: $INSTANCE_ID"
echo "================================================================"
echo ""

# Start local 24h download timer in background
echo "[deploy] Starting 24h background download timer..."
WANDB_API_KEY="$WANDB_API_KEY" \
nohup bash scripts/download-results.sh \
  "$SSH_HOST" "$SSH_PORT" "$INSTANCE_ID" \
  "$REMOTE_DIR" "$LOCAL_RESULTS_DIR" \
  > /tmp/vast_download.log 2>&1 &

echo "[deploy] Download timer PID: $!"
echo "[deploy] Log: /tmp/vast_download.log"
echo "[deploy] Results will land in: ${LOCAL_RESULTS_DIR}/"
