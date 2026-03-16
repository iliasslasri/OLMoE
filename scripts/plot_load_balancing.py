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
    "loss_free_balancing",
    "random_balancing",
    "load_balancing_alpha_0.01",
    "load_balancing_alpha_0.001",
    "run-1-maxVio",
]
SMOOTH_WINDOW = 100
OUT_PATH = "load_balancing_comparison.png"
CACHE_DIR = Path("scripts/.wandb_cache")

# Pretty labels for legend
LABELS = {
    "loss_free_balancing": "Loss-Free Balancing",
    "random_balancing": "Random Balancing",
    "load_balancing_alpha_0.01": r"Load Balancing ($\alpha=0.01$)",
    "load_balancing_alpha_0.001": r"Load Balancing ($\alpha=0.001$)",
    "run-1-maxVio": "Loss-Free (MaxVio run)",
}

COLORS = {
    "loss_free_balancing": "#1f77b4",
    "random_balancing": "#ff7f0e",
    "load_balancing_alpha_0.01": "#2ca02c",
    "load_balancing_alpha_0.001": "#d62728",
    "run-1-maxVio": "#9467bd",
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
    api = wandb.Api()

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
        df = name_to_run[name].history(samples=50_000)
        df.to_parquet(cache_path)
    else:
        print(f"WARNING: {name!r} not cached and not found in wandb — skipping")
        continue
    data[name] = df
print()

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "legend.fontsize": 9,
})

perplexity_key = find_key(["train/Perplexity"], "perplexity")
print(f"Using perplexity key: {perplexity_key}")


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

n_plots = 5 if has_maxvio else 4
fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3.5 * n_plots), sharex=True)


def smooth(series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


# --- Plot 1: Perplexity ---
ax = axes[0]
for name in RUN_NAMES:
    if name not in data:
        continue
    df = data[name]
    steps = df[step_key] if step_key and step_key in df.columns else df.index
    ppl = smooth(df[perplexity_key].astype(float))
    ax.plot(steps, ppl, label=LABELS[name], color=COLORS[name], linewidth=1.2)

ax.set_yscale("log")
ax.set_ylabel("Perplexity (log scale)")
ax.set_title("Smoothed Perplexity vs Step", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, linewidth=0.5)

# --- Plot 2: Cross-Entropy Loss ---
ax = axes[1]
for name in RUN_NAMES:
    if name not in data:
        continue
    df = data[name]
    steps = df[step_key] if step_key and step_key in df.columns else df.index
    ce = smooth(df[loss_key].astype(float))
    ax.plot(steps, ce, label=LABELS[name], color=COLORS[name], linewidth=1.2)

ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Smoothed Cross-Entropy Loss vs Step", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, linewidth=0.5)

# --- Plot 3: Relative CE Loss vs Random Balancing baseline ---
BASELINE = "random_balancing"
ax = axes[2]
if BASELINE in data:
    df_base = data[BASELINE]
    base_ce = smooth(df_base[loss_key].astype(float))
    base_steps = df_base[step_key] if step_key and step_key in df_base.columns else df_base.index
    for name in RUN_NAMES:
        if name not in data or name == BASELINE:
            continue
        df = data[name]
        method_ce = smooth(df[loss_key].astype(float))
        # Align on the shorter length
        n = min(len(base_ce), len(method_ce))
        rel_diff = ((base_ce.values[:n] - method_ce.values[:n]) / base_ce.values[:n]) * 100
        ax.plot(base_steps.values[:n], rel_diff, label=LABELS[name], color=COLORS[name], linewidth=1.2)
    ax.axhline(0, color=COLORS[BASELINE], linestyle="--", linewidth=1, alpha=0.7, label=f"{LABELS[BASELINE]} (baseline)")

ax.set_ylabel("CE Improvement (%)")
ax.set_title("Relative CE Loss vs Random Balancing Baseline", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, linewidth=0.5)

# --- Plot 4: Throughput ---
ax = axes[3]
for name in RUN_NAMES:
    if name not in data:
        continue
    df = data[name]
    if throughput_key not in df.columns:
        print(f"  Skipping throughput for {name} — key not found")
        continue
    steps = df[step_key] if step_key and step_key in df.columns else df.index
    tp = smooth(df[throughput_key].astype(float))
    ax.plot(steps, tp, label=LABELS[name], color=COLORS[name], linewidth=1.2)

ax.set_ylabel("Throughput (tokens/sec)")
ax.set_xlabel("Step")
ax.set_title("Throughput vs Step", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, linewidth=0.5)

# --- Plot 5: MaxVio_batch (load balance quality, cf. Wang et al. 2024 Fig.3) ---
if has_maxvio:
    ax = axes[4]
    for name in RUN_NAMES:
        if name not in maxvio_data:
            continue
        df = data[name]
        steps = df[step_key] if step_key and step_key in df.columns else df.index
        mv = maxvio_data[name]

        # Plot smoothed avg as the main line
        avg_line = smooth(mv["avg"])
        ax.plot(steps, avg_line, label=f"{LABELS[name]}", color=COLORS[name], linewidth=1.2)

        # Plot min/max envelope as a shaded band
        vmin = smooth(mv["min"])
        vmax = smooth(mv["max"])
        ax.fill_between(steps, vmin, vmax, color=COLORS[name], alpha=0.15)

    ax.set_ylabel("MaxVio (batch)")
    ax.set_xlabel("Step")
    ax.set_title(
        r"MaxVio$_{\mathrm{batch}}$ — Load Balance Quality (Wang et al., 2024 §4.1)",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure to {OUT_PATH}")
