#!/usr/bin/env python3
"""train_lr_sweep.py — LR Sweep Launcher (one scale, five learning rates)

Runs five sequential training jobs for a single model scale, sweeping over
the learning rates needed for the Chinchilla-style optimal-LR analysis.

Environment setup mirrors train_server.py exactly:
  installs OLMo[train], compiles megablocks against the node's CUDA,
  and fixes huggingface_hub before launching any training subprocess.

Usage
-----
    python train_lr_sweep.py --dim 64    # d_model=64,  500 K tokens
    python train_lr_sweep.py --dim 128   # d_model=128, 1 M tokens
    python train_lr_sweep.py --dim 192   # d_model=192, 2 M tokens
    python train_lr_sweep.py --dim 256   # d_model=256, 5 M tokens

Results
-------
After all runs: python plot_lr_sweep.py  (fetches final loss from W&B)
"""

import argparse
import os
import pathlib
import shutil
import socket
import subprocess
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
OLMOE_DIR      = pathlib.Path(__file__).parent.resolve()
OLMO_SRC       = OLMOE_DIR / "OLMo"
CONFIG_DIR     = OLMOE_DIR / "configs" / "lr_sweep"
LOG_DIR        = OLMOE_DIR / "logs"
PIP_OVERRIDE_DIR = str(OLMOE_DIR / "_pip_overrides")   # isolated package overrides

# ── Megablocks fork (same URL as train_server.py) ──────────────────────────────
MEGABLOCKS_PIP_URL = "git+https://github.com/Tristan22400/megablocks.git@routing/auxiliary-loss-free"

# ── W&B ────────────────────────────────────────────────────────────────────────
WANDB_ENTITY   = "iliass-lasri-team"
WANDB_PROJECT  = "olmoe-1"
WANDB_GROUP    = "lr_sweep_v3"
WANDB_RUN_MODE = "online"

# ── Scale registry ─────────────────────────────────────────────────────────────
SCALES = {
    4:   dict(steps=9,    tokens=294_912,     n_heads=1),
    8:   dict(steps=23,   tokens=753_664,     n_heads=1),
    16:  dict(steps=45,   tokens=1_474_560,   n_heads=1),
    24:  dict(steps=97,   tokens=3_178_496,   n_heads=1),
    32:  dict(steps=213,  tokens=6_979_584,   n_heads=1),
    48:  dict(steps=339,  tokens=11_108_352,  n_heads=1),
    64:  dict(steps=488,  tokens=15_990_784,  n_heads=1),
    128: dict(steps=1220, tokens=39_976_960,  n_heads=2),
    192: dict(steps=1952, tokens=63_963_136,  n_heads=3),
    256: dict(steps=3660, tokens=119_930_880, n_heads=4),
}

# ── Learning rates (must match configs/lr_sweep/ filenames) ───────────────────
LR_LIST = [
    (3e-4, "3e-4"),
    (1e-3, "1e-3"),
    (3e-3, "3e-3"),
    (8e-3, "8e-3"),
    (2e-2, "2e-2"),
    (6e-2, "6e-2"),
    (2e-1, "2e-1"),
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
    help="GPU index (sets CUDA_VISIBLE_DEVICES). Ignored if already set.",
)
args = parser.parse_args()

DIM   = args.dim
SCALE = SCALES[DIM]

# =============================================================================
# ## 1. Basic setup
# =============================================================================

os.chdir(OLMOE_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)
(OLMOE_DIR / "results").mkdir(parents=True, exist_ok=True)
os.makedirs(PIP_OVERRIDE_DIR, exist_ok=True)

if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

# Redirect pip/setuptools temp files to local disk (avoids NFS quota issues)
os.environ["TMPDIR"]        = "/tmp"
os.environ["PIP_CACHE_DIR"] = "/tmp/pip_cache"

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

# =============================================================================
# ## 2. Install dependencies  (mirrors train_server.py sections 1–5)
# =============================================================================

def pip_install(*args):
    """Install packages quietly, raising on failure."""
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"pip failed: {' '.join(args)}")


def pip_install_isolated(*args):
    """Install into PIP_OVERRIDE_DIR (--no-deps) to shadow system copies."""
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "--target", PIP_OVERRIDE_DIR, "--no-deps"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"pip_install_isolated failed: {' '.join(args)}")


def purge_cached_modules(*prefixes):
    stale = [k for k in sys.modules
             if any(k == p or k.startswith(p + ".") for p in prefixes)]
    for k in stale:
        del sys.modules[k]
    if stale:
        print(f"  Purged {len(stale)} cached module(s): {prefixes}")


# ── Step 1: OLMo with training extras ─────────────────────────────────────────
print("[1/4] Installing OLMo[train]...")
pip_install("-e", "OLMo[train]")
if str(OLMO_SRC) not in sys.path:
    sys.path.insert(1, str(OLMO_SRC))

# ── Step 2: megablocks (CUDA compilation — P100 / sm_60) ──────────────────────
print("[2/4] Setting up CUDA and installing megablocks...")

# Detect CUDA_HOME
if "CUDA_HOME" not in os.environ or not os.path.exists(os.environ.get("CUDA_HOME", "")):
    for _cand in [
        "/projects/share/apps/cuda/cuda-12.9",
        "/projects/share/apps/cuda/cuda-12.5",
        "/projects/share/apps/cuda/cuda-12.4.1",
        "/projects/share/apps/cuda/cuda-12.3",
        "/projects/share/apps/cuda/cuda-12.1",
        "/usr/local/cuda",
        "/usr/local/cuda-12.1",
        "/usr/cuda",
    ]:
        if os.path.exists(_cand):
            os.environ["CUDA_HOME"] = _cand
            print(f"  CUDA_HOME = {_cand}")
            break
    else:
        _nvcc = shutil.which("nvcc")
        if _nvcc:
            os.environ["CUDA_HOME"] = str(pathlib.Path(_nvcc).parent.parent)
        else:
            raise EnvironmentError(
                "CUDA_HOME not found. Run: module load cuda / export CUDA_HOME=..."
            )

# Shadow nvcc: injects -allow-unsupported-compiler (GCC > CUDA limit on ENST)
_real_cuda   = os.environ["CUDA_HOME"]
_real_nvcc   = os.path.join(_real_cuda, "bin", "nvcc")
_shadow_cuda = f"/tmp/shadow_cuda_lrsweep_d{DIM}"
shutil.rmtree(_shadow_cuda, ignore_errors=True)
os.makedirs(_shadow_cuda)

for _entry in os.listdir(_real_cuda):
    _src = os.path.join(_real_cuda, _entry)
    _dst = os.path.join(_shadow_cuda, _entry)
    if _entry == "bin":
        os.makedirs(_dst)
        for _b in os.listdir(_src):
            if _b != "nvcc":
                os.symlink(os.path.join(_src, _b), os.path.join(_dst, _b))
    else:
        os.symlink(_src, _dst)

_wrapper = os.path.join(_shadow_cuda, "bin", "nvcc")
with open(_wrapper, "w") as _f:
    _f.write(f'#!/bin/bash\nexec "{_real_nvcc}" -allow-unsupported-compiler "$@"\n')
os.chmod(_wrapper, 0o755)

os.environ["CUDA_HOME"] = _shadow_cuda
os.environ["PATH"]      = os.path.join(_shadow_cuda, "bin") + ":" + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")   # P100

print(f"  shadow CUDA_HOME = {_shadow_cuda}")
print(f"  TORCH_CUDA_ARCH_LIST = {os.environ['TORCH_CUDA_ARCH_LIST']}")
print(f"  megablocks URL: {MEGABLOCKS_PIP_URL}")

pip_install("ninja", "wheel", "setuptools")
cmd = [sys.executable, "-m", "pip", "install",
       "--force-reinstall", "--no-build-isolation", MEGABLOCKS_PIP_URL]
print("  Running:", " ".join(cmd))
result = subprocess.run(cmd, text=True)
if result.returncode != 0:
    raise RuntimeError("megablocks install failed — see output above.")
print("  megablocks installed OK")

# Megablocks pulls numpy 2.x; pin numpy<2 into PIP_OVERRIDE_DIR so it sits
# at the front of PYTHONPATH in training subprocesses, shadowing numpy 2.x.
pip_install_isolated("numpy<2")

# ── Step 3: fix torchvision to match installed torch ──────────────────────────
print("[3/4] Fixing torchvision...")
detect = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True, check=True,
)
torch_ver = detect.stdout.strip()
cuda_tag  = torch_ver.split("+")[1] if "+" in torch_ver else "cpu"
print(f"  torch={torch_ver}  cuda_tag={cuda_tag}")
pip_install("--force-reinstall", "torchvision",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}")
purge_cached_modules("torchvision")

# ── Step 4: fix huggingface_hub / tokenizers ──────────────────────────────────
print("[4/4] Fixing huggingface_hub / tokenizers...")
pip_install_isolated("huggingface_hub>=1.3.0,<2.0")
pip_install_isolated("tokenizers>=0.22.0,<=0.23.0")

if PIP_OVERRIDE_DIR not in sys.path:
    sys.path.insert(0, PIP_OVERRIDE_DIR)
purge_cached_modules("huggingface_hub", "tokenizers")

print("All dependencies ready.\n")

# =============================================================================
# ## 3. FLOP reporting
# =============================================================================

def compute_c_train(d_model: int, train_tokens: int) -> float:
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
# ## 4. Summary
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

# ── Training subprocess environment ───────────────────────────────────────────
TRAIN_ENV = os.environ.copy()
TRAIN_ENV["OLMO_TASK"]               = "model"
TRAIN_ENV["OMP_NUM_THREADS"]         = "4"
TRAIN_ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
TRAIN_ENV["NCCL_P2P_DISABLE"]        = "0"
TRAIN_ENV["NCCL_P2P_LEVEL"]          = "SYS"
TRAIN_ENV["PYTHONPATH"]              = (
    f"{PIP_OVERRIDE_DIR}:{OLMO_SRC}:" + TRAIN_ENV.get("PYTHONPATH", "")
)


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# =============================================================================
# ## 5. Run experiments
# =============================================================================

failed = []

for lr_val, lr_str in LR_LIST:
    run_name    = f"d{DIM:03d}_lr{lr_str}"
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
    else:
        print(f"[DONE]   {run_name}")

# =============================================================================
# ## 6. Final summary
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
