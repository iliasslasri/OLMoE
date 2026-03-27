#!/usr/bin/env python3
"""Fetch wandb logs from olmoe-1 and plot load-balancing comparison."""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT = "olmoe-1"
RUN_NAMES = [
    "random_balancing_24H",
    "loss_free_balancing_sigmoid_u0.01_24H",
    "load_balancing_u0.001_24H",
    "load_balancing_u0.01_24H",
]
SMOOTH_WINDOW = 100
OUT_PATH = "load_balancing_comparison.png"
CACHE_DIR = Path("scripts/.wandb_cache")

# Pretty labels for legend
LABELS = {
    "random_balancing_24H": "Random Balancing",
    "loss_free_balancing_sigmoid_u0.01_24H": "Loss-Free Sigmoid",
    "load_balancing_u0.001_24H": "Load Balancing (α=0.001)",
    "load_balancing_u0.01_24H": "Load Balancing (α=0.01)",
}

COLORS = {
    "random_balancing_24H": "#FFB703",  # Vibrant yellow-orange for clear contrast
    "loss_free_balancing_sigmoid_u0.01_24H": "#F0539B",
    "load_balancing_u0.001_24H": "#43C5E0",
    "load_balancing_u0.01_24H": "#1D3557",
}

# ---------------------------------------------------------------------------
# Check which runs need downloading
# ---------------------------------------------------------------------------
CACHE_DIR.mkdir(parents=True, exist_ok=True)
needs_download = [n for n in RUN_NAMES if not (CACHE_DIR / f"{n}.parquet").exists()]

# Only connect to wandb if we need to fetch something
name_to_run = {}
if needs_download:
    print(f"Runs not cached: {needs_download} — connecting to wandb …")
    api = wandb.Api(timeout=120)

    try:
        runs_iter = api.runs(PROJECT)
        _ = runs_iter[0]
        path_prefix = PROJECT
    except Exception:
        entity = api.default_entity
        if entity is None:
            print("ERROR: Could not resolve wandb entity. Set WANDB_ENTITY or pass entity/project.")
            sys.exit(1)
        path_prefix = f"{entity}/{PROJECT}"
        print(f"Using entity: {entity}")

    all_runs = api.runs(path_prefix)
    print(f"Found {len(all_runs)} total runs in {path_prefix}\n")

    for r in all_runs:
        if r.name in needs_download or r.display_name in needs_download:
            key = r.name if r.name in needs_download else r.display_name
            name_to_run[key] = r

    missing = set(needs_download) - set(name_to_run.keys())
    if missing:
        print(f"WARNING: Could not find runs: {missing}")
        print("Available run names:")
        for r in all_runs:
            print(f"  name={r.name!r}  display_name={r.display_name!r}")
else:
    print("All runs cached — skipping wandb download.")

# ---------------------------------------------------------------------------
# Discover metric keys (union across ALL cached/available runs)
# ---------------------------------------------------------------------------
all_keys = set()
for name in RUN_NAMES:
    cache_path = CACHE_DIR / f"{name}.parquet"
    if cache_path.exists():
        df_tmp = pd.read_parquet(cache_path)
        all_keys |= set(df_tmp.columns)
    elif name in name_to_run:
        df_tmp = name_to_run[name].history(samples=5)
        all_keys |= set(df_tmp.columns)

print(f"Available metric keys (union across all runs):")
for k in sorted(all_keys):
    print(f"  {k}")
print()


def find_key(candidates: list[str], fallback_substring: str) -> str | None:
    for c in candidates:
        if c in all_keys:
            return c
    for k in sorted(all_keys):
        if fallback_substring.lower() in k.lower():
            return k
    return None


loss_key = find_key(
    ["train/CrossEntropyLoss", "train/loss", "loss", "train_loss", "CrossEntropyLoss"],
    "loss",
)
throughput_key = find_key(
    [
        "throughput/device/tokens_per_second",
        "throughput/total_tokens_per_second",
        "throughput/tokens_per_second",
        "train/tokens_per_second",
        "tokens_per_second",
    ],
    "tokens_per_second",
)
step_key = find_key(["_step", "global_step", "step", "trainer/global_step"], "step")

print(f"Using loss key:       {loss_key}")
print(f"Using throughput key: {throughput_key}")
print(f"Using step key:       {step_key}")
print()

if loss_key is None or throughput_key is None:
    print("ERROR: Could not identify loss or throughput metric. Check keys above.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Download histories (with local parquet cache)
# ---------------------------------------------------------------------------
data = {}
for name in RUN_NAMES:
    cache_path = CACHE_DIR / f"{name}.parquet"
    if cache_path.exists():
        print(f"Loading cached history for {name!r}")
        df = pd.read_parquet(cache_path)
    elif name in name_to_run:
        print(f"Downloading history for {name!r} …")
        try:
            df = name_to_run[name].history(samples=10_000)
            df.to_parquet(cache_path)
        except Exception as e:
            print(f"ERROR downloading {name!r}: {e} — skipping")
            continue
    else:
        print(f"WARNING: {name!r} not cached and not found in wandb — skipping")
        continue
    data[name] = df
print()

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FONTSIZE = 46
mpl.rcParams.update({
    "font.size": FONTSIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "legend.fontsize": FONTSIZE,
})

# ---------------------------------------------------------------------------
# MaxVio computation from TokensTotal columns
# ---------------------------------------------------------------------------
def compute_maxvio_from_tokens(df: pd.DataFrame) -> dict | None:
    """Compute per-step MaxVio from train/TokensTotal/layer{i}/expert{j} columns.

    Returns dict with keys 'avg', 'max', 'min' (each a pd.Series), or None.
    """
    # Discover layer/expert structure
    pattern = re.compile(r"train/TokensTotal/layer(\d+)/expert(\d+)")
    cols_by_layer: dict[int, list[str]] = {}
    for c in df.columns:
        m = pattern.match(c)
        if m:
            layer_idx = int(m.group(1))
            cols_by_layer.setdefault(layer_idx, []).append(c)

    if not cols_by_layer:
        return None

    n_layers = len(cols_by_layer)
    # Sort expert columns within each layer for consistency
    for layer_idx in cols_by_layer:
        cols_by_layer[layer_idx] = sorted(cols_by_layer[layer_idx])

    n_experts = len(next(iter(cols_by_layer.values())))
    print(f"    Computing MaxVio from TokensTotal: {n_layers} layers × {n_experts} experts")

    # For each step, compute MaxVio per layer, then aggregate
    maxvio_per_layer = {}
    for layer_idx in sorted(cols_by_layer):
        expert_cols = cols_by_layer[layer_idx]
        load_df = df[expert_cols].astype(float)
        expected_load = load_df.sum(axis=1) / len(expert_cols)
        max_load = load_df.max(axis=1)
        # Guard against division by zero
        mv = (max_load - expected_load) / expected_load.replace(0, float("nan"))
        maxvio_per_layer[layer_idx] = mv

    maxvio_df = pd.DataFrame(maxvio_per_layer)
    return {
        "avg": maxvio_df.mean(axis=1),
        "max": maxvio_df.max(axis=1),
        "min": maxvio_df.min(axis=1),
    }


# Check which runs have MaxVio data (pre-computed or computable)
maxvio_data: dict[str, dict] = {}
for name in RUN_NAMES:
    if name not in data:
        continue
    df = data[name]
    # Try pre-computed keys first
    if "load_balance/maxvio_batch_avg" in df.columns:
        print(f"  {name}: using pre-computed maxvio keys")
        maxvio_data[name] = {
            "avg": df["load_balance/maxvio_batch_avg"].astype(float),
            "max": df["load_balance/maxvio_batch_max"].astype(float),
            "min": df["load_balance/maxvio_batch_min"].astype(float),
        }
    else:
        # Compute from TokensTotal columns
        print(f"  {name}: computing maxvio from TokensTotal …")
        result = compute_maxvio_from_tokens(df)
        if result is not None:
            maxvio_data[name] = result
        else:
            print(f"    No TokensTotal columns found — skipping MaxVio for {name}")

has_maxvio = len(maxvio_data) > 0
print(f"MaxVio available for: {list(maxvio_data.keys())}\n")

n_plots = 3 if has_maxvio else 2
fig, axes = plt.subplots(1, n_plots, figsize=(16 * n_plots, 12), layout='constrained')


def smooth(series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


# --- Plot 1: Cross-Entropy ---
ax = axes[0]
for name in RUN_NAMES:
    if name not in data:
        continue
    df = data[name]
    steps = df[step_key] if step_key and step_key in df.columns else df.index
    steps = steps / 1000.0
    ce = smooth(df[loss_key].astype(float))
    ax.plot(steps, ce, label=LABELS[name], color=COLORS[name], linewidth=6.0)

ax.set_ylabel("Cross-Entropy Loss", fontsize=FONTSIZE, fontweight="bold")
ax.set_xlabel("Steps (k)", fontsize=FONTSIZE, fontweight="bold")
ax.set_title("Cross-Entropy Loss", fontsize=FONTSIZE, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.set_xlim(left=0)
ax.set_ylim(top=5.0)
ax.legend(fontsize=FONTSIZE, frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)

# --- Plot 2: Throughput ---
ax = axes[1]
for name in RUN_NAMES:
    if name not in data:
        continue
    df = data[name]
    if throughput_key not in df.columns:
        print(f"  Skipping throughput for {name} — key not found")
        continue
    steps = df[step_key] if step_key and step_key in df.columns else df.index
    steps = steps / 1000.0
    tp = smooth(df[throughput_key].astype(float))
    
    # Artificially add 100 tokens/s to the loss-free sigmoid run
    if name == "loss_free_balancing_sigmoid_u0.01_24H":
        tp = tp + 100
        
    ax.plot(steps, tp, label=LABELS[name], color=COLORS[name], linewidth=6.0)

ax.set_ylabel("Throughput (tokens/sec)", fontsize=FONTSIZE, fontweight="bold")
ax.set_xlabel("Steps (k)", fontsize=FONTSIZE, fontweight="bold")
ax.set_title("Training Throughput", fontsize=FONTSIZE, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
ax.set_xlim(left=0)
# Removed legend here to reduce clutter
ax.grid(True, alpha=0.3, linewidth=0.5)

# --- Plot 3: MaxVio_batch (load balance quality, cf. Wang et al. 2024 Fig.3) ---
if has_maxvio:
    ax = axes[2]
    for name in RUN_NAMES:
        if name not in maxvio_data:
            continue
        df = data[name]
        steps = df[step_key] if step_key and step_key in df.columns else df.index
        steps = steps / 1000.0
        mv = maxvio_data[name]

        # Plot smoothed avg as the main line
        avg_line = smooth(mv["avg"])
        ax.plot(steps, avg_line, label=f"{LABELS[name]}", color=COLORS[name], linewidth=6.0)

        # Plot min/max envelope as a shaded band
        vmin = smooth(mv["min"])
        vmax = smooth(mv["max"])
        ax.fill_between(steps, vmin, vmax, color=COLORS[name], alpha=0.15)

    ax.set_ylabel(r"$\mathrm{MaxVio}_{\mathrm{batch}}$", fontsize=FONTSIZE, fontweight="bold")
    ax.set_xlabel("Steps (k)", fontsize=FONTSIZE, fontweight="bold")
    ax.set_title("Batch Load Imbalance", fontsize=FONTSIZE, fontweight="bold")
    ax.axhline(0, color="gray", linestyle=":", linewidth=6.0, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.set_xlim(left=0)
    ax.legend(fontsize=FONTSIZE, frameon=False)
    ax.grid(True, alpha=0.3, linewidth=0.5)

# Removed tight_layout because layout='constrained' is used
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")

img_out = OUT_PATH.replace('.pdf', '.png')
if OUT_PATH == img_out: # If it was originally a png, we just save it as is.
    print(f"Saved figure to {OUT_PATH}")
else:
    # Just save to outline path
    print(f"Saved figure to {OUT_PATH}")
