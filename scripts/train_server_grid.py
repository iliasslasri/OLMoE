#!/usr/bin/env python3
"""train_server_grid.py — Phase 1 Grid Runner.

Installs all dependencies (mirrors train_lr_sweep.py exactly), then runs a list
of YAML configs sequentially on the current GPU.

Usage:
    python train_server_grid.py --configs configs/grid/d064_*.yml
    python train_server_grid.py --configs configs/grid/d128_*.yml configs/grid/d192_*.yml
    python train_server_grid.py --configs configs/grid/d256_*.yml
"""

import argparse
import glob
import os
import pathlib
import shutil
import socket
import subprocess
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
OLMOE_DIR        = pathlib.Path(__file__).parent.resolve()
OLMO_SRC         = OLMOE_DIR / "OLMo"
LOG_DIR          = OLMOE_DIR / "logs" / "grid"
PIP_OVERRIDE_DIR = str(OLMOE_DIR / "_pip_overrides")

MEGABLOCKS_PIP_URL = "git+https://github.com/Tristan22400/megablocks.git@routing/auxiliary-loss-free"

WANDB_ENTITY  = "iliass-lasri-team"
WANDB_PROJECT = "olmoe-1"
WANDB_MODE    = "online"

# =============================================================================
# ## Args
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--configs", nargs="+", required=True,
    help="YAML config files to run (glob patterns expanded by shell or Python)",
)
args = parser.parse_args()

# Expand any remaining glob patterns (in case shell didn't)
config_paths = []
for pat in args.configs:
    expanded = sorted(glob.glob(pat))
    config_paths.extend(expanded if expanded else [pat])

if not config_paths:
    sys.exit("No config files found.")

print(f"Grid runner: {len(config_paths)} configs queued")
for p in config_paths[:5]:
    print(f"  {p}")
if len(config_paths) > 5:
    print(f"  ... and {len(config_paths)-5} more")

# =============================================================================
# ## 1. Basic setup  (identical to train_lr_sweep.py)
# =============================================================================

os.chdir(OLMOE_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(PIP_OVERRIDE_DIR, exist_ok=True)

os.environ["TMPDIR"]        = "/tmp"
os.environ["PIP_CACHE_DIR"] = "/tmp/pip_cache"

# W&B credentials
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
elif WANDB_MODE == "online":
    print("WARNING: WANDB_API_KEY not found — switching to offline mode.")
    os.environ["WANDB_MODE"] = "offline"

os.environ.setdefault("WANDB_ENTITY",  WANDB_ENTITY)
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
os.environ.setdefault("WANDB_MODE",    WANDB_MODE)

# =============================================================================
# ## 2. Install dependencies  (identical to train_lr_sweep.py)
# =============================================================================

def pip_install(*a):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(a)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
        raise RuntimeError(f"pip failed: {' '.join(a)}")


def pip_install_isolated(*a):
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "--target", PIP_OVERRIDE_DIR, "--no-deps"] + list(a)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
        raise RuntimeError(f"pip_install_isolated failed: {' '.join(a)}")


def purge_cached_modules(*prefixes):
    stale = [k for k in sys.modules
             if any(k == p or k.startswith(p + ".") for p in prefixes)]
    for k in stale:
        del sys.modules[k]
    if stale:
        print(f"  Purged {len(stale)} cached module(s): {prefixes}")


print("[1/4] Installing OLMo[train]...")
pip_install("--force-reinstall", "--no-deps", "-e", "OLMo[train]")
if str(OLMO_SRC) not in sys.path:
    sys.path.insert(1, str(OLMO_SRC))

print("[2/4] Setting up CUDA and installing megablocks...")
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
            raise EnvironmentError("CUDA_HOME not found.")

_real_cuda   = os.environ["CUDA_HOME"]
_real_nvcc   = os.path.join(_real_cuda, "bin", "nvcc")
_shadow_cuda = f"/tmp/shadow_cuda_grid_{os.getpid()}"
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
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")

pip_install("ninja", "wheel", "setuptools")
# Serialize megablocks build across parallel processes — only one compiles at a time.
# Others wait, then find it already installed and skip the heavy build.
import fcntl as _fcntl
_mb_lock_path = "/tmp/megablocks_build.lock"
with open(_mb_lock_path, "w") as _mb_lf:
    _fcntl.flock(_mb_lf, _fcntl.LOCK_EX)
    _already = subprocess.run(
        [sys.executable, "-c", "import megablocks"],
        capture_output=True,
    ).returncode == 0
    if not _already:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--force-reinstall", "--no-build-isolation", MEGABLOCKS_PIP_URL],
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError("megablocks install failed.")
    _fcntl.flock(_mb_lf, _fcntl.LOCK_UN)
pip_install("numpy<2")

print("[3/4] Fixing torchvision...")
detect = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True, check=True,
)
torch_ver = detect.stdout.strip()
cuda_tag  = torch_ver.split("+")[1] if "+" in torch_ver else "cpu"
pip_install("--force-reinstall", "torchvision",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}")
purge_cached_modules("torchvision")

print("[4/4] Fixing huggingface_hub / tokenizers...")
pip_install_isolated("huggingface_hub>=1.3.0,<2.0")
pip_install_isolated("tokenizers>=0.22.0,<=0.23.0")
if PIP_OVERRIDE_DIR not in sys.path:
    sys.path.insert(0, PIP_OVERRIDE_DIR)
purge_cached_modules("huggingface_hub", "tokenizers")

print("All dependencies ready.\n")

# =============================================================================
# ## 3. Train loop
# =============================================================================

TRAIN_ENV = os.environ.copy()
TRAIN_ENV["OLMO_TASK"]               = "model"
TRAIN_ENV["OMP_NUM_THREADS"]         = "4"
TRAIN_ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
TRAIN_ENV["NCCL_P2P_DISABLE"]        = "0"
TRAIN_ENV["NCCL_P2P_LEVEL"]          = "SYS"
TRAIN_ENV["PYTHONPATH"]              = (
    f"{PIP_OVERRIDE_DIR}:{OLMO_SRC}:" + TRAIN_ENV.get("PYTHONPATH", "")
)


def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


failed = []
total  = len(config_paths)

for i, cfg in enumerate(config_paths):
    cfg_path = pathlib.Path(cfg)
    if not cfg_path.exists():
        print(f"\n[SKIP {i+1}/{total}] Config not found: {cfg}")
        failed.append(cfg)
        continue

    run_name = cfg_path.stem
    log_path = LOG_DIR / f"{run_name}.log"

    if log_path.exists() and log_path.stat().st_size > 0:
        last_line = log_path.read_text().strip().splitlines()[-1]
        if last_line.startswith("[DONE]"):
            print(f"\n[SKIP {i+1}/{total}] Already completed: {run_name}")
            continue
        print(f"\n[RETRY {i+1}/{total}] Previous run failed, retrying: {run_name}")

    # Use a user-owned checkpoint dir to avoid /tmp permission issues from other jobs
    import getpass as _getpass
    _user = _getpass.getuser()
    _save_folder = str(OLMOE_DIR / "tmp" / "olmoe_ckpt" / run_name)
    shutil.rmtree(_save_folder, ignore_errors=True)
    pathlib.Path(_save_folder).mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  [{i+1}/{total}]  {run_name}")
    print(f"  log → logs/grid/{run_name}.log")
    print(f"{'─'*60}")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc-per-node=1",
        "--nnodes=1", "--node_rank=0",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=localhost:{find_free_port()}",
        "OLMo/scripts/train.py",
        str(cfg_path),
        f"--save_folder={_save_folder}",
        "--save_num_checkpoints_to_keep=0",
        "--save_num_unsharded_checkpoints_to_keep=0",
        "--save_interval_unsharded=null",
    ]

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
        msg = f"[FAILED] {run_name} — exit code {process.returncode}"
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        failed.append(run_name)
    else:
        print(f"[DONE]   {run_name}")

# =============================================================================
# ## 4. Summary
# =============================================================================

print("\n" + "=" * 60)
n_ok = total - len(failed)
print(f"  Grid runner finished:  {n_ok}/{total} succeeded")
if failed:
    print(f"  Failed ({len(failed)}):")
    for f in failed:
        print(f"    {f}")
print("=" * 60)
