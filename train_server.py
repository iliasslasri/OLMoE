#!/usr/bin/env python3
"""OLMoE Training on a Remote Server

Fine-tune / pre-train an OLMoE (Mixture-of-Experts) model using pre-tokenized `.npy` shards.

Author: Tristan Martin & Iliass Lasri | Date: 2025

Before running — checklist:
- Set WANDB_API_KEY environment variable (or set WANDB_RUN_MODE="offline" below)
- Point DATA_GLOB_PATTERN to your .npy shard files
- Set WORKING_DIR to a directory with sufficient disk space
- Ensure CUDA is available and nvidia-smi is on PATH
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# All user-facing variables live here. Edit as needed.

# ── Execution mode ─────────────────────────────────────────────────────────────
# Set True to run on a single GPU (FSDP NO_SHARD — no sharding, checkpoint-compatible).
SINGLE_GPU = True

# ── Run identity ───────────────────────────────────────────────────────────────
# Change this to name your run. Used for save folder, W&B run name, and config.
RUN_NAME             = "random_balancing_maxvio"

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKING_DIR          = "/home/infres/zakil-22/olmoe_runs"          # base working directory (for saves, pip overrides)
DATA_GLOB_PATTERN    = "/home/infres/zakil-22/OLMoE/data/**/*.npy" # path to .npy shards
REPO_URL             = "https://github.com/iliasslasri/OLMoE.git"
OLMOE_BRANCH         = "config/fsdp-shard-grad-op"                # branch to clone for OLMoE repo
OLMOE_DIR            = "/home/infres/zakil-22/OLMoE"              # existing repo on server (skip clone if present)
PIP_OVERRIDE_DIR     = f"{WORKING_DIR}/_pip_overrides"            # isolated pip installs
TOKENIZER_URL        = "https://huggingface.co/allenai/gpt-neox-olmo-dolma-v1_5/resolve/main/tokenizer.json"
TOKENIZER_LOCAL_PATH = "tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
SAVE_FOLDER          = f"{WORKING_DIR}/runs/{RUN_NAME}"           # checkpoint output

# ── OLMo submodule (points to Tristan22400/OLMo fork) ─────────────────────────
OLMO_REPO_URL        = "https://github.com/Tristan22400/OLMo.git"
OLMO_BRANCH          = "routing/moe-strategies"

# ── Megablocks fork (routing strategies) ───────────────────────────────────────
MEGABLOCKS_PIP_URL   = "git+https://github.com/Tristan22400/megablocks.git@routing/auxiliary-loss-free"

# ── Training hyperparameters ───────────────────────────────────────────────────
GLOBAL_TRAIN_BATCH_SIZE      = 16           # total tokens per optimizer step across all GPUs
DEVICE_TRAIN_MICROBATCH_SIZE = 4            # per-GPU per-step (controls peak VRAM)
MAX_SEQUENCE_LENGTH          = 1024         # context window
MAX_TRAINING_STEPS           = 50000        # stop after this many optimizer steps (int or "2ep" for epochs)
PRECISION                    = "fp32"       # fp32 | amp_bf16 | amp_fp16
ACTIVATION_CHECKPOINTING     = "fine_grained"  # fine_grained | null (see Section 3 comments)
MOE_DROPLESS                 = True         # dropless MoE requires megablocks sparse kernels
MOE_MLP_IMPL                 = "sparse"    # sparse | dense

# ── Single-GPU overrides ─────────────────────────────────────────────────────
if SINGLE_GPU:
    PRECISION = "fp32"

# ── MoE Architecture ──────────────────────────────────────────────────────────
MOE_NUM_EXPERTS              = 8            # total number of experts per MoE layer
MOE_TOP_K                    = 2            # experts selected per token (active experts)

# ── MoE Routing Strategy ──────────────────────────────────────────────────────
# "learned"    → standard auxiliary load balancing loss (baseline)
# "loss_free"  → auxiliary-loss-free additive bias (Wang et al., 2024)
# "random"     → uniform random assignment (ablation baseline)
MOE_ROUTING_TYPE             = "random"

# ── MoE Gate Function ─────────────────────────────────────────────────────────
# "softmax"  → experts compete, scores sum to 1 (standard)
# "sigmoid"  → independent gates, each score in [0,1]
# Only used when MOE_ROUTING_TYPE == "loss_free".
MOE_GATE_TYPE                = "softmax"

# ── MoE Loss ──────────────────────────────────────────────────────────────────
# Alpha (weight) for the auxiliary load-balancing loss.
# For "learned" routing this directly scales the LB loss term.
# For "loss_free" it is used for stats logging only (no gradient contribution).
MOE_LOSS_WEIGHT              = 0.001        # α in L_total = L_CE + α * L_LB

# ── MoE Z-Loss ───────────────────────────────────────────────────────────────
# Weight for the router z-loss (ST-MoE, Zoph et al. 2022).
# Regularizes router logit magnitudes: L_z = w * mean(log(sum(exp(logits)))^2).
# Set to None to disable. Active for both "learned" and "loss_free" routing.
MOE_ZLOSS_WEIGHT             = 0.001       # None to disable, 0.001 is default

# ── Data loading ──────────────────────────────────────────────────────────────
# num_workers=0 means synchronous loading on the main process (GPU stalls).
# Increase to overlap data loading with GPU compute.
DATA_NUM_WORKERS             = 2            # dataloader worker processes
DATA_PREFETCH_FACTOR         = 2            # batches prefetched per worker
DATA_PERSISTENT_WORKERS      = True         # keep workers alive between epochs

# ── Checkpoint save ───────────────────────────────────────────────────────────
# How often (in steps) to save a checkpoint. Only the last one is kept.
# The old checkpoint is deleted only after the new one is fully written.
SAVE_INTERVAL_STEPS          = 50000

# OLMo will gracefully stop and save a checkpoint when time_limit is reached.
# Set to None to disable, or to a value slightly below your job's wall time.
TIME_LIMIT_SECONDS           = 86400       # e.g. ~7h; set None to disable

# ── Checkpoint resume ─────────────────────────────────────────────────────────
# Set to True to resume from a checkpoint.
# CHECKPOINT_PATH: path to a step directory containing model/ and optim/ subdirs.
#   - e.g. "/home/.../runs/loss_free_balancing/step2350"
#   - If None and RESUME_TRAINING is True, OLMo looks for the latest checkpoint in SAVE_FOLDER.
RESUME_TRAINING              = False
CHECKPOINT_PATH              = None  # e.g. "/home/.../step2350"

# ── W&B ────────────────────────────────────────────────────────────────────────
WANDB_ENTITY_NAME  = "iliass-lasri-team"
WANDB_PROJECT_NAME = "olmoe-1"
WANDB_RUN_MODE     = "online"            # online | offline | disabled

# ── System ─────────────────────────────────────────────────────────────────────
OMP_NUM_THREADS_TRAIN = "4"              # OpenMP threads during training
MIN_DISK_FREE_GB      = 8                # warn if less than this available

# ── Imports ────────────────────────────────────────────────────────────────────
import glob
import importlib
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys

# torch and yaml are imported after pip installs (Section 2) to allow running
# from a base environment that may not have them pre-installed.

# =============================================================================
# ## 1. System Check
# =============================================================================

if not shutil.which("nvidia-smi"):
    raise EnvironmentError(
        "\n'nvidia-smi' not found — no GPU attached or not on PATH.\n"
        "Fix: ensure CUDA drivers are installed and nvidia-smi is accessible."
    )
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)

print(f"Python  : {sys.version}")
_smi_out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
N_GPU = len(_smi_out.splitlines()) if _smi_out else 0
print(f"GPUs    : {N_GPU}")
assert N_GPU > 0, "No GPU detected. Ensure CUDA is available and drivers are installed."

disk_usage_output = subprocess.run(
    ['df', '-h', WORKING_DIR], capture_output=True, text=True
).stdout
print(disk_usage_output)

free_gb = float(subprocess.run(
    ['df', '--output=avail', '-BG', WORKING_DIR],
    capture_output=True, text=True
).stdout.strip().split()[-1].replace('G', ''))

if free_gb < MIN_DISK_FREE_GB:
    print(f"WARNING: only {free_gb:.1f} GB free — may be tight.")
else:
    print(f"Disk OK: {free_gb:.1f} GB free.")

# =============================================================================
# ## 2. Environment Setup
# =============================================================================

# ── Resolve tokenized data paths ──────────────────────────────────────────────
TOKENIZED_DATA_PATHS = sorted(
    p for p in glob.glob(DATA_GLOB_PATTERN, recursive=True)
    if os.path.isfile(p) and os.path.getsize(p) > 0
)

assert len(TOKENIZED_DATA_PATHS) > 0, (
    f"\nNo .npy files found at {DATA_GLOB_PATTERN}.\n"
    "Possible fixes:\n"
    "  1. Update DATA_GLOB_PATTERN to point to your .npy files\n"
    f"  2. Run: find {os.path.dirname(DATA_GLOB_PATTERN.split('*')[0])} -name '*.npy'"
)

print(f"Found {len(TOKENIZED_DATA_PATHS)} tokenized shard(s):")
for p in TOKENIZED_DATA_PATHS:
    size_mb = os.path.getsize(p) / 1e6
    print(f"  {p}  ({size_mb:.0f} MB)")

# ── W&B credentials ───────────────────────────────────────────────────────────
wandb_api_key = os.environ.get("WANDB_API_KEY")
if wandb_api_key:
    os.environ["WANDB_API_KEY"] = wandb_api_key
elif WANDB_RUN_MODE == "online":
    print("WARNING: WANDB_API_KEY not set — switching W&B to offline mode.")
    WANDB_RUN_MODE = "offline"
os.environ["WANDB_ENTITY"]  = WANDB_ENTITY_NAME
os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
os.environ["WANDB_MODE"]    = WANDB_RUN_MODE
print("W&B configured.")

# ── Clone repo + submodule ────────────────────────────────────────────────────
os.makedirs(WORKING_DIR, exist_ok=True)

if not os.path.exists(OLMOE_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", OLMOE_BRANCH,
                    REPO_URL, OLMOE_DIR], check=True)
    # Override submodule URL to use our OLMo fork
    subprocess.run(
        ["git", "config", "submodule.OLMo.url", OLMO_REPO_URL],
        cwd=OLMOE_DIR, check=True,
    )
    # Full clone (not --depth 1) so the branch is reachable
    subprocess.run(["git", "submodule", "update", "--init"], cwd=OLMOE_DIR, check=True)
    # Fetch and checkout the routing/moe-strategies branch with our config+train changes
    subprocess.run(["git", "fetch", "origin", OLMO_BRANCH],
                   cwd=f"{OLMOE_DIR}/OLMo", check=True)
    subprocess.run(["git", "checkout", OLMO_BRANCH],
                   cwd=f"{OLMOE_DIR}/OLMo", check=True)
else:
    print("Repo already present, skipping clone.")

os.chdir(OLMOE_DIR)
print("cwd:", os.getcwd())
print("main:", subprocess.run(["git", "log", "--oneline", "-1"],
                               capture_output=True, text=True).stdout.strip())
print("OLMo:", subprocess.run(["git", "log", "--oneline", "-1"],
                               capture_output=True, text=True, cwd="OLMo").stdout.strip())
print("OLMo branch:", subprocess.run(["git", "branch", "--show-current"],
                               capture_output=True, text=True, cwd="OLMo").stdout.strip())

# ── Dependency installer helpers ──────────────────────────────────────────────
#
# Why this complexity? See root causes:
#   - ai2-olmo-core==0.1.0 pins huggingface_hub ~0.36.x; transformers needs >=1.3.0
#   - OLMo[train] may downgrade torch (conflicts with system torch)
#   - megablocks must compile against the post-downgrade torch ABI
#   - System pyarrow/pandas break if numpy downgrades in-kernel

OLMO_SRC = f"{OLMOE_DIR}/OLMo"
os.makedirs(PIP_OVERRIDE_DIR, exist_ok=True)

# Redirect pip/setuptools temp files to local node disk to avoid NFS quota issues.
os.environ["TMPDIR"] = "/tmp"
os.environ["PIP_CACHE_DIR"] = "/tmp/pip_cache"


def pip_install(*args):
    """Install packages quietly via pip, raising on failure."""
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"pip failed: {' '.join(args)}")


def pip_install_isolated(*args):
    """Install packages (--no-deps) into PIP_OVERRIDE_DIR to shadow system copies."""
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "--target", PIP_OVERRIDE_DIR, "--no-deps"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"pip_install_isolated failed: {' '.join(args)}")


def purge_cached_modules(*prefixes):
    """Remove cached modules from sys.modules to force reimport."""
    stale = [k for k in sys.modules
             if any(k == p or k.startswith(p + ".") for p in prefixes)]
    for k in stale:
        del sys.modules[k]
    if stale:
        print(f"  Purged {len(stale)} cached module(s): {prefixes}")


# ── Step 1: OLMo with training extras ─────────────────────────────────────────
print("[1/5] Installing OLMo[train]...")
pip_install("-e", "OLMo[train]")

if OLMO_SRC not in sys.path:
    sys.path.insert(1, OLMO_SRC)
    print(f"  Added {OLMO_SRC} to sys.path")

# ── Step 2: megablocks (our fork with routing strategies) ──────────────────────
# megablocks compiles CUDA extensions — ensure CUDA_HOME is set before building.
if "CUDA_HOME" not in os.environ or not os.path.exists(os.environ["CUDA_HOME"]):
    for _cuda_candidate in [
        "/projects/share/apps/cuda/cuda-12.9",   # ENST cluster default (D) — GCC 14 compatible
        "/projects/share/apps/cuda/cuda-12.5",   # ENST cluster — GCC 13 compatible
        "/projects/share/apps/cuda/cuda-12.4.1", # ENST cluster — GCC 13 compatible
        "/projects/share/apps/cuda/cuda-12.3",   # ENST cluster
        "/projects/share/apps/cuda/cuda-12.1",   # ENST cluster (GCC <= 12 only)
        "/usr/local/cuda",
        "/usr/local/cuda-12.1",
        "/usr/cuda",
    ]:
        if os.path.exists(_cuda_candidate):
            os.environ["CUDA_HOME"] = _cuda_candidate
            print(f"  Set CUDA_HOME={_cuda_candidate}")
            break
    else:
        _nvcc = shutil.which("nvcc")
        if _nvcc:
            _cuda_home = str(pathlib.Path(_nvcc).parent.parent)
            os.environ["CUDA_HOME"] = _cuda_home
            print(f"  Set CUDA_HOME={_cuda_home} (from nvcc)")
        else:
            raise EnvironmentError(
                "CUDA_HOME is not set and nvcc was not found. "
                "Cannot compile megablocks CUDA extensions. "
                "Fix: module load cuda / export CUDA_HOME=/usr/local/cuda-XX.X"
            )

print("[2/5] Installing megablocks (routing strategies fork, force-recompile)...")
print(f"  Source: {MEGABLOCKS_PIP_URL}")
# P100 = sm_60; explicitly set arch list so nvcc doesn't try to compile for
# unsupported architectures. Also ensure CUDA_HOME/bin is on PATH for nvcc.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")  # P100

# Create an nvcc wrapper that injects -allow-unsupported-compiler.
# This is necessary because the system GCC may be newer than the CUDA toolkit
# version supports (e.g. GCC 14 vs CUDA 12.5's limit of GCC 13). The flag is
# safe for P100/sm_60: the restriction is overly conservative, not a hard ABI
# incompatibility for this architecture and C++17 code.
# PyTorch's cpp_extension hard-codes nvcc as CUDA_HOME/bin/nvcc (ignores PATH
# and CUDACXX in most versions). Fix: create a shadow CUDA_HOME in /tmp with
# a wrapper nvcc that injects -allow-unsupported-compiler, symlink everything
# else from the real CUDA_HOME, then point CUDA_HOME at the shadow.
_real_cuda_home = os.environ["CUDA_HOME"]
_real_nvcc = os.path.join(_real_cuda_home, "bin", "nvcc")
_shadow_cuda = "/tmp/shadow_cuda"
shutil.rmtree(_shadow_cuda, ignore_errors=True)
os.makedirs(_shadow_cuda)

# Symlink all top-level entries from the real CUDA_HOME, except /bin
for _entry in os.listdir(_real_cuda_home):
    _src = os.path.join(_real_cuda_home, _entry)
    _dst = os.path.join(_shadow_cuda, _entry)
    if _entry == "bin":
        # Recreate bin dir with symlinks for every file except nvcc
        os.makedirs(_dst)
        for _b in os.listdir(_src):
            if _b != "nvcc":
                os.symlink(os.path.join(_src, _b), os.path.join(_dst, _b))
    else:
        os.symlink(_src, _dst)

# Write our wrapper nvcc that injects -allow-unsupported-compiler
_wrapper_nvcc = os.path.join(_shadow_cuda, "bin", "nvcc")
with open(_wrapper_nvcc, "w") as _f:
    _f.write(f'#!/bin/bash\nexec "{_real_nvcc}" -allow-unsupported-compiler "$@"\n')
os.chmod(_wrapper_nvcc, 0o755)

os.environ["CUDA_HOME"] = _shadow_cuda
os.environ["PATH"] = os.path.join(_shadow_cuda, "bin") + ":" + os.environ.get("PATH", "")

print(f"  CUDA_HOME (shadow) = {_shadow_cuda}")
print(f"  real nvcc          = {_real_nvcc}")
print(f"  wrapper nvcc       = {_wrapper_nvcc}")
print(f"  TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
# Use --no-build-isolation so megablocks compiles against our installed torch
# (avoids pip creating a separate build env with its own torch download).
# Pre-install ninja for faster/more verbose compilation output.
pip_install("ninja")
cmd = [
    sys.executable, "-m", "pip", "install",
    "--force-reinstall", "--no-build-isolation",
    MEGABLOCKS_PIP_URL,
]
# Stream output live so ninja errors are visible immediately
print("  Running:", " ".join(cmd))
result = subprocess.run(cmd, text=True)
if result.returncode != 0:
    raise RuntimeError("megablocks install failed — see output above.")
print("  megablocks installed OK")

# ── Step 3: Reinstall torchvision matching the (possibly downgraded) torch ────
print("[3/5] Detecting torch version after OLMo install...")
detect = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True, check=True,
)
installed_torch_version = detect.stdout.strip()
cuda_tag = installed_torch_version.split("+")[1] if "+" in installed_torch_version else "cpu"
print(f"  Detected torch=={installed_torch_version}, CUDA tag: {cuda_tag}")

print("[4/5] Reinstalling torchvision for detected torch...")
pip_install("--force-reinstall", "torchvision",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}")
purge_cached_modules("torchvision")

# ── Step 4: Force-upgrade huggingface_hub ─────────────────────────────────────
print("[5/5] Fixing huggingface_hub / tokenizers versions...")
pip_install_isolated("huggingface_hub>=1.3.0,<2.0")
pip_install_isolated("tokenizers>=0.22.0,<=0.23.0")

if PIP_OVERRIDE_DIR not in sys.path:
    sys.path.insert(0, PIP_OVERRIDE_DIR)
    print(f"  Inserted {PIP_OVERRIDE_DIR} at sys.path[0]")

purge_cached_modules("huggingface_hub", "tokenizers")

# ── Verify dependencies ───────────────────────────────────────────────────────
# NOTE: megablocks is NOT verified here. Its .so was compiled against the
# downgraded torch, but the kernel still has the old torch in memory.
# The training subprocess loads it correctly from a fresh interpreter.
print("Verifying imports...")
try:
    from huggingface_hub import is_offline_mode  # noqa: F401
    print("  ✓ huggingface_hub version OK")
except ImportError as e:
    raise RuntimeError(f"huggingface_hub fix failed: {e}") from e

for mod in ["omegaconf", "wandb"]:
    importlib.import_module(mod)
    print(f"  ✓ {mod}")

import torch
import yaml
import huggingface_hub
import tokenizers as hf_tok
print(f"  huggingface_hub=={huggingface_hub.__version__}")
print(f"  tokenizers=={hf_tok.__version__}")
print(f"  torch (subprocess)=={installed_torch_version}")
print("All dependencies OK. megablocks verified by dry-run subprocess.")

# ── Download tokenizer ────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(TOKENIZER_LOCAL_PATH), exist_ok=True)

if not os.path.exists(TOKENIZER_LOCAL_PATH):
    print("Downloading tokenizer...")
    result = subprocess.run(
        ["wget", "-q", TOKENIZER_URL, "-O", TOKENIZER_LOCAL_PATH],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(TOKENIZER_LOCAL_PATH):
        raise RuntimeError(
            "Tokenizer download failed. Check internet connectivity.\n" + result.stderr
        )
    size_kb = os.path.getsize(TOKENIZER_LOCAL_PATH) / 1024
    print(f"  Saved to {TOKENIZER_LOCAL_PATH} ({size_kb:.0f} KB)")
    assert size_kb > 100, f"File too small ({size_kb:.0f} KB) — download may have failed."
else:
    print(f"Tokenizer already present at {TOKENIZER_LOCAL_PATH}")

# =============================================================================
# ## 3. Training Configuration
# =============================================================================

# ── Pre-flight batch-size checks ──────────────────────────────────────────────
_effective_gpus = 1 if SINGLE_GPU else N_GPU

if GLOBAL_TRAIN_BATCH_SIZE % _effective_gpus != 0:
    raise ValueError(
        f"GLOBAL_TRAIN_BATCH_SIZE ({GLOBAL_TRAIN_BATCH_SIZE}) "
        f"must be divisible by number of GPUs ({_effective_gpus})"
    )
device_batch = GLOBAL_TRAIN_BATCH_SIZE // _effective_gpus
if device_batch % DEVICE_TRAIN_MICROBATCH_SIZE != 0:
    raise ValueError(
        f"device_batch ({device_batch}) must be divisible by "
        f"DEVICE_TRAIN_MICROBATCH_SIZE ({DEVICE_TRAIN_MICROBATCH_SIZE})"
    )
grad_accum_steps = device_batch // DEVICE_TRAIN_MICROBATCH_SIZE
print(f"Batch: global={GLOBAL_TRAIN_BATCH_SIZE}, GPUs={_effective_gpus}, per-GPU={device_batch}, "
      f"microbatch={DEVICE_TRAIN_MICROBATCH_SIZE}, grad_accum={grad_accum_steps}")

# ── Patch YAML config ─────────────────────────────────────────────────────────
# Activation checkpointing notes for MoE (block_type: moe):
#   fine_grained → checkpoints attn+norms per block, skips MoE FFN (~20% overhead)
#   null         → no checkpointing (highest memory usage)
#   one_in_two / one_in_four / whole_layer → DENSE-only, raises error with MoE

CONFIG_PATH = pathlib.Path(OLMOE_DIR) / "configs/olmoe-small.yml"
assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
# Per-run config written to /tmp to avoid concurrent jobs overwriting the shared template
RUN_CONFIG_PATH = pathlib.Path("/tmp") / f"olmoe_config_{RUN_NAME}.yml"

cfg = yaml.safe_load(CONFIG_PATH.read_text())

# ── Identity ──────────────────────────────────────────────────────────────────
cfg["run_name"]                      = RUN_NAME
cfg["wandb"]["name"]                 = RUN_NAME
cfg["save_folder"]                   = SAVE_FOLDER

# ── Data / model ──────────────────────────────────────────────────────────────
cfg["data"]["paths"]                 = list(TOKENIZED_DATA_PATHS)
cfg["model"]["moe_num_experts"]      = MOE_NUM_EXPERTS
cfg["model"]["moe_top_k"]           = MOE_TOP_K
cfg["model"]["moe_dropless"]         = MOE_DROPLESS
cfg["model"]["moe_mlp_impl"]         = MOE_MLP_IMPL
cfg["model"]["max_sequence_length"]  = MAX_SEQUENCE_LENGTH
cfg["model"]["moe_routing_type"]     = MOE_ROUTING_TYPE
cfg["model"]["moe_gate_type"]        = MOE_GATE_TYPE
cfg["model"]["moe_loss_weight"]      = MOE_LOSS_WEIGHT
cfg["model"]["moe_zloss_weight"]     = MOE_ZLOSS_WEIGHT
cfg["model"]["moe_log_expert_assignment"] = True  # required for MaxVio_batch logging

# ── Data loading ──────────────────────────────────────────────────────────────
cfg["data"]["num_workers"]           = DATA_NUM_WORKERS
cfg["data"]["prefetch_factor"]       = DATA_PREFETCH_FACTOR if DATA_NUM_WORKERS > 0 else None
cfg["data"]["persistent_workers"]    = DATA_PERSISTENT_WORKERS if DATA_NUM_WORKERS > 0 else False

# ── Training ──────────────────────────────────────────────────────────────────
cfg["max_duration"]                  = MAX_TRAINING_STEPS
cfg["precision"]                     = PRECISION
cfg["activation_checkpointing"]      = ACTIVATION_CHECKPOINTING
cfg["global_train_batch_size"]       = GLOBAL_TRAIN_BATCH_SIZE
cfg["device_train_microbatch_size"]  = DEVICE_TRAIN_MICROBATCH_SIZE
cfg["compile"]                       = None

# ── Distributed strategy ──────────────────────────────────────────────────────
if SINGLE_GPU:
    # Single-GPU: keep FSDP FULL_SHARD — with world_size=1, nothing actually
    # gets sharded, but this ensures the checkpoint loader takes the correct
    # code path (use_orig_params=True with sharded handles) so keys match.
    cfg["fsdp"]["sharding_strategy"] = "FULL_SHARD"
    cfg["fsdp"]["precision"]         = "mixed"
else:
    # Multi-GPU: FSDP with sharding
    cfg["fsdp"]["precision"]         = "pure" if PRECISION == "fp32" else "mixed"

# ── Checkpointing: one rolling checkpoint, delete old after new is committed ──
cfg["save_overwrite"]                 = True   # allow re-runs with the same RUN_NAME
cfg["save_interval"]                  = SAVE_INTERVAL_STEPS
cfg["save_interval_ephemeral"]        = None
cfg["save_num_checkpoints_to_keep"]   = 1
cfg["time_limit"]                     = TIME_LIMIT_SECONDS
cfg["extra_steps_after_cancel"]       = 0

# ── Resume ────────────────────────────────────────────────────────────────────
if RESUME_TRAINING:
    if CHECKPOINT_PATH is not None:
        # Load from a specific step directory
        cfg["load_path"]                = CHECKPOINT_PATH
        cfg["try_load_latest_save"]     = False
    else:
        # Auto-detect latest checkpoint in SAVE_FOLDER
        cfg["load_path"]                = None
        cfg["try_load_latest_save"]     = True
    cfg["no_pre_train_checkpoint"]      = False
else:
    cfg["load_path"]                    = None
    cfg["try_load_latest_save"]         = False
    cfg["no_pre_train_checkpoint"]      = True

# For loss_free routing, ensure z-loss is active (needed for stats even if
# the LB loss is not added to the training objective).
if MOE_ROUTING_TYPE == "loss_free":
    cfg["model"].setdefault("moe_zloss_weight", 0.001)

config_yaml_text = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
RUN_CONFIG_PATH.write_text(config_yaml_text)
print(f"Config written to {RUN_CONFIG_PATH}")
print(f"  run_name        : {RUN_NAME}")
print(f"  save_folder     : {SAVE_FOLDER}")
print(f"  moe_num_experts : {MOE_NUM_EXPERTS}  (top_k={MOE_TOP_K})")
print(f"  moe_routing_type: {MOE_ROUTING_TYPE}")
print(f"  moe_gate_type   : {MOE_GATE_TYPE}")
print(f"  moe_loss_weight : {MOE_LOSS_WEIGHT}")
print(f"  moe_zloss_weight: {MOE_ZLOSS_WEIGHT}")
print(f"  moe_log_expert_assignment: True")
print(f"  precision       : {PRECISION}  (fsdp.precision={cfg['fsdp']['precision']})")
print(f"  fsdp.sharding   : {cfg['fsdp']['sharding_strategy']}")
print(f"  data.num_workers: {DATA_NUM_WORKERS}  (prefetch={DATA_PREFETCH_FACTOR}, persistent={DATA_PERSISTENT_WORKERS})")
print(f"  save_interval   : every {SAVE_INTERVAL_STEPS} steps  (keep last 1)")
print(f"  time_limit      : {TIME_LIMIT_SECONDS}s = {TIME_LIMIT_SECONDS/3600:.1f}h")
print(f"  resume_training : {RESUME_TRAINING}")
if RESUME_TRAINING:
    print(f"  load_path       : {cfg.get('load_path', '(auto from SAVE_FOLDER)')}")

# ── Verify patched config ─────────────────────────────────────────────────────
verified_config = yaml.safe_load(RUN_CONFIG_PATH.read_text())

config_checks = [
    (verified_config["run_name"] == RUN_NAME, "run_name"),
    (verified_config["save_folder"] == SAVE_FOLDER, "save_folder"),
    (verified_config["data"]["paths"] == list(TOKENIZED_DATA_PATHS), "data.paths mismatch"),
    (verified_config["model"]["moe_num_experts"] == MOE_NUM_EXPERTS, "moe_num_experts"),
    (verified_config["model"]["moe_top_k"] == MOE_TOP_K, "moe_top_k"),
    (verified_config["model"]["max_sequence_length"] == MAX_SEQUENCE_LENGTH, "max_sequence_length"),
    (verified_config["model"]["moe_dropless"] == MOE_DROPLESS, "moe_dropless"),
    (verified_config["model"]["moe_routing_type"] == MOE_ROUTING_TYPE, "moe_routing_type"),
    (verified_config["model"]["moe_gate_type"] == MOE_GATE_TYPE, "moe_gate_type"),
    (verified_config["model"]["moe_loss_weight"] == MOE_LOSS_WEIGHT, "moe_loss_weight"),
    (verified_config["model"]["moe_zloss_weight"] == MOE_ZLOSS_WEIGHT, "moe_zloss_weight"),
    (verified_config["model"]["moe_log_expert_assignment"] == True, "moe_log_expert_assignment"),
    (verified_config["activation_checkpointing"] == ACTIVATION_CHECKPOINTING, "activation_checkpointing"),
    (verified_config["global_train_batch_size"] == GLOBAL_TRAIN_BATCH_SIZE, "global_train_batch_size"),
    (verified_config["precision"] == PRECISION, "precision"),
    (verified_config["save_overwrite"] == True, "save_overwrite"),
    (verified_config["save_interval"] == SAVE_INTERVAL_STEPS, "save_interval"),
    (verified_config["save_num_checkpoints_to_keep"] == 1, "save_num_checkpoints_to_keep"),
    (verified_config["time_limit"] == TIME_LIMIT_SECONDS, "time_limit"),
]

# Resume checks
if RESUME_TRAINING:
    if CHECKPOINT_PATH is not None:
        config_checks.append((verified_config.get("load_path") == CHECKPOINT_PATH, "load_path mismatch"))
    else:
        config_checks.append((verified_config.get("try_load_latest_save") == True, "try_load_latest_save"))
else:
    config_checks.append((verified_config.get("no_pre_train_checkpoint") == True, "no_pre_train_checkpoint"))

errors = [msg for ok, msg in config_checks if not ok]
if errors:
    raise RuntimeError("Config verification FAILED:\n  " + "\n  ".join(errors))

CHECKPOINTING_DESCRIPTIONS = {
    "fine_grained": "attn+norms only, skips MoE FFN (~20% compute overhead)",
    "null": "none (highest memory usage)",
}
ROUTING_DESCRIPTIONS = {
    "learned": "standard auxiliary load balancing loss",
    "loss_free": "auxiliary-loss-free additive bias (Wang et al., 2024)",
    "random": "uniform random assignment (ablation baseline)",
}
GATE_DESCRIPTIONS = {
    "softmax": "experts compete, scores sum to 1",
    "sigmoid": "independent gates, each score in [0,1]",
}
ckpt_str = verified_config["activation_checkpointing"]
routing_str = verified_config["model"]["moe_routing_type"]
gate_str = verified_config["model"]["moe_gate_type"]
zloss_w = verified_config["model"].get("moe_zloss_weight")
fsdp_prec = verified_config["fsdp"]["precision"]
fsdp_shard = verified_config["fsdp"]["sharding_strategy"]
print("=== Config OK ===")
print(f"  run_name             : {verified_config['run_name']}")
print(f"  save_folder          : {verified_config['save_folder']}")
print(f"  max_duration         : {verified_config['max_duration']}")
print(f"  precision            : {verified_config['precision']}  (fsdp={fsdp_prec})")
print(f"  fsdp.sharding        : {fsdp_shard}")
print(f"  max_sequence_length  : {verified_config['model']['max_sequence_length']}")
print(f"  moe_num_experts      : {verified_config['model']['moe_num_experts']}  (top_k={verified_config['model']['moe_top_k']})")
print(f"  moe_dropless         : {verified_config['model']['moe_dropless']}")
print(f"  moe_mlp_impl         : {verified_config['model']['moe_mlp_impl']}")
print(f"  moe_routing_type     : {routing_str}  ({ROUTING_DESCRIPTIONS.get(routing_str, '?')})")
print(f"  moe_gate_type        : {gate_str}  ({GATE_DESCRIPTIONS.get(gate_str, '?')})")
print(f"  moe_loss_weight      : {verified_config['model']['moe_loss_weight']}  (α for LB loss)")
print(f"  moe_zloss_weight     : {zloss_w}  ({'disabled' if zloss_w is None else 'active'})")
print(f"  activation_ckpt      : {ckpt_str}  ({CHECKPOINTING_DESCRIPTIONS.get(str(ckpt_str), '?')})")
print(f"  compile              : {verified_config.get('compile')}")
print(f"  global_batch_size    : {verified_config['global_train_batch_size']}")
print(f"  microbatch_size      : {verified_config['device_train_microbatch_size']}")
print(f"  save_interval        : every {verified_config['save_interval']} steps  (keep last 1, overwrite=True)")
print(f"  time_limit           : {verified_config['time_limit']}s = {verified_config['time_limit']/3600:.1f}h")
print(f"  resume_training      : {RESUME_TRAINING}")
if RESUME_TRAINING:
    print(f"  load_path            : {verified_config.get('load_path', '(auto from save_folder)')}")
print(f"  data.paths ({len(verified_config['data']['paths'])} file(s)):")
for p in verified_config["data"]["paths"]:
    print(f"    {p}")

# ── Dry-run: import + config validation (no GPU needed) ───────────────────────
# Runs train.py without torchrun. Expected exit: ValueError at
# dist.init_process_group() — means all imports + config checks passed.

dry_run_env = os.environ.copy()
dry_run_env["OLMO_TASK"]       = "model"
dry_run_env["PYTHONPATH"]      = f"{PIP_OVERRIDE_DIR}:{OLMO_SRC}:" + dry_run_env.get("PYTHONPATH", "")
dry_run_env["OMP_NUM_THREADS"] = "1"

result = subprocess.run(
    [sys.executable, f"{OLMOE_DIR}/OLMo/scripts/train.py", str(RUN_CONFIG_PATH)],
    capture_output=True, text=True, env=dry_run_env, cwd=OLMOE_DIR,
)

print(f"Dry run exit code: {result.returncode}")
print("--- STDOUT ---")
print(result.stdout[:2000] or "(no stdout)")
print("--- STDERR (last 2000 chars) ---")
print(result.stderr[-2000:] or "(no stderr)")

# ── Classify dry-run exit ─────────────────────────────────────────────────────
IMPORT_ERROR_MARKERS = ["ModuleNotFoundError", "ImportError", "cannot import name"]
CONFIG_ERROR_MARKERS = [
    "OmegaConf", "yaml", "KeyError", "missing mandatory value",
    "ConfigAttributeError", "MissingMandatoryValue",
]

has_import_error = any(m in result.stderr for m in IMPORT_ERROR_MARKERS)
has_config_error = any(m in result.stderr for m in CONFIG_ERROR_MARKERS)
reached_distributed = "RANK expected, but not set" in result.stderr

if has_import_error:
    raise RuntimeError("Import error in dry run — fix dependencies before training.")
if has_config_error:
    raise RuntimeError("Config error in dry run — check config patches.")
if reached_distributed:
    print("\n✓ Dry run passed: all imports and config loaded OK.")
    print("  (Script stopped at dist.init_process_group — expected without torchrun.)")
elif result.returncode == 0:
    print("\n✓ Dry run passed cleanly.")
else:
    raise RuntimeError(f"Unexpected failure (exit {result.returncode}). See STDERR above.")

# =============================================================================
# ## 4. Launch Training
# =============================================================================

import threading

def find_free_port():
    """Find and return a free TCP port on localhost."""
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# ── Checkpoint watchdog ───────────────────────────────────────────────────────
_watchdog_stop = threading.Event()

def _resolve_latest_checkpoint(save_path: pathlib.Path) -> str:
    """Resolve the latest checkpoint name from the save directory.
    Handles both cases:
      - `latest` is a text file containing the step dir name (older OLMo)
      - `latest` is a directory/symlink to the step dir (olmo_core checkpointer)
    """
    latest = save_path / "latest"
    if not latest.exists():
        return None
    if latest.is_file() and not latest.is_symlink():
        return latest.read_text().strip()
    # latest is a directory or symlink — resolve to the target name
    resolved = latest.resolve()
    return resolved.name

def _checkpoint_watchdog(save_dir: str, poll_interval: int = 30):
    save_path = pathlib.Path(save_dir)
    last_latest = None
    while not _watchdog_stop.is_set():
        try:
            current_latest = _resolve_latest_checkpoint(save_path)
            if current_latest is not None and current_latest != last_latest and last_latest is not None:
                # New checkpoint fully committed — purge all older step dirs
                step_dirs = [
                    d for d in save_path.iterdir()
                    if d.is_dir() and re.match(r"step\d+", d.name)
                    and d.name != current_latest
                ]
                for d in step_dirs:
                    print(f"[watchdog] Removing old checkpoint: {d.name}", flush=True)
                    shutil.rmtree(d, ignore_errors=True)
                if step_dirs:
                    free_gb = shutil.disk_usage(WORKING_DIR).free / 1e9
                    print(f"[watchdog] Kept: {current_latest}  |  disk free: {free_gb:.1f} GB",
                          flush=True)
            if current_latest is not None:
                last_latest = current_latest
        except Exception as exc:
            print(f"[watchdog] Warning: {exc}", flush=True)
        _watchdog_stop.wait(timeout=poll_interval)


_n_procs = 1 if SINGLE_GPU else N_GPU

# ── GPU topology + P2P check (multi-GPU only) ────────────────────────────────
if not SINGLE_GPU:
    print("=== GPU Topology ===")
    topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True)
    print(topo.stdout or topo.stderr)

    print("=== PCIe P2P access ===")
    for i in range(N_GPU):
        for j in range(N_GPU):
            if i != j:
                ok = torch.cuda.can_device_access_peer(i, j)
                print(f"  GPU{i}->GPU{j}: P2P={'ENABLED' if ok else 'BLOCKED (will use SHM fallback)'}")

rdzv_port = find_free_port()
print(f"\nUsing rdzv port: {rdzv_port}")
print(f"Launching on {_n_procs} GPU(s)  (FSDP sharding: {cfg['fsdp']['sharding_strategy']})")
print(f"Checkpoints → {SAVE_FOLDER}  (watchdog will keep only the latest)")

train_env = os.environ.copy()
train_env["OLMO_TASK"]               = "model"
train_env["PYTHONPATH"]              = f"{PIP_OVERRIDE_DIR}:{OLMOE_DIR}/OLMo:" + train_env.get("PYTHONPATH", "")
train_env["OMP_NUM_THREADS"]         = OMP_NUM_THREADS_TRAIN
train_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# NCCL transport tuning
train_env["NCCL_P2P_DISABLE"]        = "0"
train_env["NCCL_P2P_LEVEL"]          = "SYS"
train_env["NCCL_SHM_DISABLE"]        = "0"

train_cmd = [
    sys.executable, "-m", "torch.distributed.run",
    f"--nproc-per-node={_n_procs}",
    "--nnodes=1", "--node_rank=0",
    "--rdzv_backend=c10d",
    f"--rdzv_endpoint=localhost:{rdzv_port}",
    "OLMo/scripts/train.py",
    str(RUN_CONFIG_PATH),
]
print("Command:", " ".join(train_cmd))
print("-" * 60)

# ── Run training (streams output live) ────────────────────────────────────────
_watchdog_stop.clear()
watchdog_thread = threading.Thread(
    target=_checkpoint_watchdog, args=(SAVE_FOLDER,), daemon=True
)
watchdog_thread.start()

process = subprocess.Popen(
    train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, env=train_env,
)
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
finally:
    _watchdog_stop.set()
    watchdog_thread.join(timeout=10)

if process.returncode != 0:
    raise RuntimeError(f"Training failed (exit code {process.returncode})")
print("\nTraining completed successfully.")
