#!/usr/bin/env python3
"""train_lr_sweep.py — LR Sweep Launcher (one scale, five learning rates)

Runs five sequential training jobs for a single model scale, sweeping over
the learning rates needed for the Chinchilla-style optimal-LR analysis.

Usage
-----
    python train_lr_sweep.py --dim 64    # d_model=64,  500 K tokens
    python train_lr_sweep.py --dim 128   # d_model=128, 1 M tokens
    python train_lr_sweep.py --dim 192   # d_model=192, 2 M tokens
    python train_lr_sweep.py --dim 256   # d_model=256, 5 M tokens

Environment
-----------
Assumes the OLMo submodule is already installed (run train_server.py once to
bootstrap the server).  This script only sets PYTHONPATH and env vars;
it does NOT reinstall packages.

Results
-------
After all runs: python plot_lr_sweep.py  (fetches final loss from W&B)
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Edit these if your server layout differs from the default.

import argparse
import os
import pathlib
import socket
import subprocess
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
OLMOE_DIR  = pathlib.Path(__file__).parent.resolve()
OLMO_SRC   = OLMOE_DIR / "OLMo"
CONFIG_DIR = OLMOE_DIR / "configs" / "lr_sweep"
LOG_DIR    = OLMOE_DIR / "logs"
TOKENIZER_PATH = "tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"

# ── W&B ────────────────────────────────────────────────────────────────────────
WANDB_ENTITY       = "iliass-lasri-team"
WANDB_PROJECT      = "olmoe-1"
WANDB_GROUP        = "lr_sweep"
WANDB_RUN_MODE     = "online"   # set to "offline" if no internet on compute node

# ── Scale registry ─────────────────────────────────────────────────────────────
# Maps d_model → (n_steps, train_tokens, n_heads)
SCALES = {
    64:  dict(steps=122,  tokens=500_000,   n_heads=1),
    128: dict(steps=244,  tokens=1_000_000, n_heads=2),
    192: dict(steps=488,  tokens=2_000_000, n_heads=3),
    256: dict(steps=1220, tokens=5_000_000, n_heads=4),
}

# ── Learning rates (must match configs/lr_sweep/ filenames) ───────────────────
LR_LIST = [
    (5e-4,   "5e-4"),
    (1.5e-3, "1.5e-3"),
    (5e-3,   "5e-3"),
    (1.5e-2, "1.5e-2"),
    (5e-2,   "5e-2"),
]

# =============================================================================
# ## Parse arguments
# =============================================================================

parser = argparse.ArgumentParser(description="LR sweep for one model scale")
parser.add_argument(
    "--dim", type=int, required=True, choices=sorted(SCALES),
    help="d_model to sweep (64 | 128 | 192 | 256)",
)
parser.add_argument(
    "--gpu", type=int, default=None,
    help="GPU index to use (sets CUDA_VISIBLE_DEVICES). "
         "Ignored if CUDA_VISIBLE_DEVICES is already set.",
)
args = parser.parse_args()

DIM   = args.dim
SCALE = SCALES[DIM]

# =============================================================================
# ## Environment setup
# =============================================================================

os.chdir(OLMOE_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)
(OLMOE_DIR / "results").mkdir(parents=True, exist_ok=True)

# ── GPU visibility ─────────────────────────────────────────────────────────────
if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"CUDA_VISIBLE_DEVICES={args.gpu}")
else:
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set — all GPUs visible)')}")

# ── Python path (OLMo submodule) ───────────────────────────────────────────────
if not OLMO_SRC.exists():
    raise FileNotFoundError(
        f"OLMo submodule not found at {OLMO_SRC}.\n"
        "Run train_server.py once to bootstrap the environment."
    )
if str(OLMO_SRC) not in sys.path:
    sys.path.insert(0, str(OLMO_SRC))

# ── W&B credentials ───────────────────────────────────────────────────────────
wandb_key = os.environ.get("WANDB_API_KEY", "")
if not wandb_key:
    env_file = pathlib.Path.home() / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("WANDB_API_KEY="):
                wandb_key = line.split("=", 1)[1].strip()
                break
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key
elif WANDB_RUN_MODE == "online":
    print("WARNING: WANDB_API_KEY not found — switching W&B to offline mode.")
    os.environ["WANDB_MODE"] = "offline"

os.environ.setdefault("WANDB_ENTITY",  WANDB_ENTITY)
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
os.environ.setdefault("WANDB_MODE",    WANDB_RUN_MODE)

# ── Training subprocess environment ───────────────────────────────────────────
TRAIN_ENV = os.environ.copy()
TRAIN_ENV["OLMO_TASK"]               = "model"
TRAIN_ENV["OMP_NUM_THREADS"]         = "4"
TRAIN_ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
TRAIN_ENV["PYTHONPATH"]              = f"{OLMO_SRC}:{TRAIN_ENV.get('PYTHONPATH', '')}"

# =============================================================================
# ## FLOP reporting
# =============================================================================

def compute_c_train(d_model: int, train_tokens: int) -> float:
    """Total training FLOPs for one scale (C = 3 * M_fwd * D)."""
    n_layers, seq_len, top_k, n_experts = 10, 2048, 2, 8
    mlp_ratio, vocab_size = 4, 50280
    d_expert = mlp_ratio * d_model
    c_attn   = 8 * d_model**2 + 4 * seq_len * d_model
    c_ffn    = 6 * top_k * d_model * d_expert
    c_router = 2 * d_model * n_experts
    c_logit  = 2 * d_model * vocab_size
    m_fwd    = n_layers * (c_attn + c_ffn + c_router) + c_logit
    return 3 * m_fwd * train_tokens


C_TRAIN = compute_c_train(DIM, SCALE["tokens"])

# =============================================================================
# ## Summary
# =============================================================================

print("=" * 60)
print(f"  LR Sweep — d_model={DIM}")
print(f"  n_layers=10  n_heads={SCALE['n_heads']}  mlp_ratio=4")
print(f"  seq_len=2048  top_k=2  n_experts=8")
print(f"  train_tokens={SCALE['tokens']:,}  steps={SCALE['steps']}")
print(f"  C_train = {C_TRAIN:.4e} FLOPs")
print(f"  W&B: {WANDB_ENTITY}/{WANDB_PROJECT}  group={WANDB_GROUP}")
print(f"  Learning rates: {[lr_str for _, lr_str in LR_LIST]}")
print("=" * 60)


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# =============================================================================
# ## Run experiments
# =============================================================================

failed = []

for lr_val, lr_str in LR_LIST:
    run_name = f"d{DIM:03d}_lr{lr_str}"
    config_path = CONFIG_DIR / f"{run_name}.yml"
    log_path    = LOG_DIR / f"{run_name}.log"

    if not config_path.exists():
        print(f"\n[SKIP] Config not found: {config_path}")
        print("       Run: python generate_lr_sweep_configs.py")
        failed.append(run_name)
        continue

    rdzv_port = find_free_port()

    print(f"\n{'─'*60}")
    print(f"  [{LR_LIST.index((lr_val, lr_str))+1}/{len(LR_LIST)}]  {run_name}")
    print(f"  lr={lr_str}   log → logs/{run_name}.log")
    print(f"{'─'*60}")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc-per-node=1",
        "--nnodes=1", "--node_rank=0",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=localhost:{rdzv_port}",
        "OLMo/scripts/train.py",
        str(config_path),
    ]
    print("  cmd:", " ".join(cmd))

    with open(log_path, "w") as log_fh:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=TRAIN_ENV,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_fh.write(line)
        process.wait()

    if process.returncode != 0:
        msg = f"[FAILED] {run_name} exited with code {process.returncode}"
        print(msg)
        with open(log_path, "a") as log_fh:
            log_fh.write(msg + "\n")
        failed.append(run_name)
        # Continue to next LR rather than aborting the sweep
    else:
        print(f"[DONE]   {run_name}")

# =============================================================================
# ## Final summary
# =============================================================================

print("\n" + "=" * 60)
print(f"  Sweep complete for d_model={DIM}")
print(f"  C_train = {C_TRAIN:.4e} FLOPs")
n_ok = len(LR_LIST) - len(failed)
print(f"  Successful: {n_ok}/{len(LR_LIST)}")
if failed:
    print(f"  Failed runs: {failed}")
print()
print("  Fetch results and plot:")
print("    python plot_lr_sweep.py")
print("=" * 60)
