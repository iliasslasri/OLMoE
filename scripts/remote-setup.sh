#!/usr/bin/env bash
# =============================================================================
# remote-setup.sh  —  runs ON the Vast.ai instance (system Python, no conda)
#
# Called via deploy-vast.sh as:
#   ssh ... "WANDB_API_KEY=... REPO_DIR=... bash remote-setup.sh"
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMoE}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY not set" >&2; exit 1
fi

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[setup] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git rsync tmux wget ninja-build

# ---------------------------------------------------------------------------
# 2. Fix libcuda.so symlink (required by Triton on Docker containers)
#    The driver ships libcuda.so.1 but Triton looks for libcuda.so without suffix.
# ---------------------------------------------------------------------------
echo "[setup] Fixing libcuda.so symlink for Triton..."
LIBCUDA=$(find /lib /usr/lib -name 'libcuda.so.1' 2>/dev/null | head -1)
if [[ -n "$LIBCUDA" ]]; then
  LIBDIR=$(dirname "$LIBCUDA")
  ln -sf "$LIBCUDA" "${LIBDIR}/libcuda.so"
  echo "[setup] Symlinked ${LIBDIR}/libcuda.so -> $LIBCUDA"
else
  echo "[setup] WARNING: libcuda.so.1 not found — Triton may fail at runtime"
fi

# ---------------------------------------------------------------------------
# 3. Install Python dependencies into system Python
#    (Docker image is pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel)
# ---------------------------------------------------------------------------
echo "[setup] Installing OLMo and train dependencies..."
pip install --quiet -e "${REPO_DIR}/OLMo[train]"

echo "[setup] Installing megablocks (sparse MoE kernels)..."
pip install --quiet \
  git+https://github.com/Tristan22400/megablocks.git@routing/auxiliary-loss-free

# ---------------------------------------------------------------------------
# 4. Install flash-attention via prebuilt wheel (avoids ~20-min source build).
#    Wheel is matched to torch 2.3.x + CUDA 12.x + Python 3.10 + Linux x86_64.
#    Falls back to source compilation if the prebuilt wheel is unavailable.
# ---------------------------------------------------------------------------
echo "[setup] Installing flash-attention (prebuilt wheel)..."
TORCH_VER=$(python3 -c "import torch; v=torch.__version__; print(v[:3])")  # e.g. "2.3"
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda[:2])")        # e.g. "12"
PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")  # e.g. "cp310"

FA_VERSION="2.7.3"
WHEEL="flash_attn-${FA_VERSION}+cu${CUDA_VER}torch${TORCH_VER}cxx11abiFALSE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VERSION}/${WHEEL}"

echo "[setup] Trying prebuilt wheel: $WHEEL"
if pip install --quiet "$WHEEL_URL" 2>/dev/null; then
  echo "[setup] flash-attn installed from prebuilt wheel."
else
  echo "[setup] Prebuilt wheel not found — falling back to source build (takes ~20 min)..."
  pip install --quiet ninja packaging
  MAX_JOBS=4 pip install flash-attn --no-build-isolation
fi

# ---------------------------------------------------------------------------
# 5. W&B login
# ---------------------------------------------------------------------------
echo "[setup] Logging into Weights & Biases..."
wandb login --relogin "$WANDB_API_KEY"

# Persist key for the training launch script
echo "WANDB_API_KEY=${WANDB_API_KEY}" > ~/.env

echo "[setup] Done. All dependencies installed."
