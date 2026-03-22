#!/usr/bin/env python3
"""plot_lr_sweep.py — Fetch LR sweep results from W&B and plot optimal-LR scaling.

Pulls all finished runs from the 'lr_sweep' group in the olmoe-1 W&B project,
extracts the final training loss, fits a power-law lr*(C) ~ C^β, and writes:

  results/lr_sweep.csv   — tabular results (one row per run)
  results/lr_sweep.png   — two-panel figure

Usage:
    python plot_lr_sweep.py
    python plot_lr_sweep.py --entity my-team --project my-project --group lr_sweep
    python plot_lr_sweep.py --offline results/lr_sweep.csv  # re-plot from local CSV
"""

import argparse
import os
import sys
import numpy as np

# ── Architecture constants (must match generate_lr_sweep_configs.py) ──────────
N_LAYERS   = 10
SEQ_LEN    = 2048
TOP_K      = 2
N_EXPERTS  = 8
MLP_RATIO  = 4
VOCAB_SIZE = 50280

TOKENS_BY_DIM = {
    64:   500_000,
    128:  1_000_000,
    192:  2_000_000,
    256:  5_000_000,
}

# ── Learning rate order for axis sorting ──────────────────────────────────────
LR_DISPLAY_ORDER = [5e-4, 1.5e-3, 5e-3, 1.5e-2, 5e-2]


# =============================================================================
# ## Helpers
# =============================================================================

def compute_c_train(d_model: int) -> float:
    """Total training FLOPs for a given d_model."""
    D        = TOKENS_BY_DIM[d_model]
    d_expert = MLP_RATIO * d_model
    c_attn   = 8 * d_model**2 + 4 * SEQ_LEN * d_model
    c_ffn    = 6 * TOP_K * d_model * d_expert
    c_router = 2 * d_model * N_EXPERTS
    c_logit  = 2 * d_model * VOCAB_SIZE
    m_fwd    = N_LAYERS * (c_attn + c_ffn + c_router) + c_logit
    return 3 * m_fwd * D


# =============================================================================
# ## W&B fetch
# =============================================================================

def fetch_from_wandb(entity: str, project: str, group: str):
    """Fetch completed runs and return a list of dicts."""
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed: pip install wandb")

    api   = wandb.Api()
    path  = f"{entity}/{project}"
    print(f"Querying W&B: {path}  (group={group}) …")

    runs = api.runs(path, filters={"group": group})
    records = []

    for run in runs:
        if run.state not in ("finished", "crashed"):
            print(f"  [skip] {run.name}  state={run.state}")
            continue

        cfg     = run.config
        d_model = cfg.get("model", {}).get("d_model")
        lr      = cfg.get("optimizer", {}).get("learning_rate")

        if d_model is None or lr is None:
            print(f"  [skip] {run.name}: missing d_model or lr in config")
            continue

        d_model = int(d_model)
        lr      = float(lr)

        # Final training loss — use run.summary (last logged value, fast)
        summary     = run.summary._json_dict
        final_loss  = (
            summary.get("train/loss")
            or summary.get("train_loss")
            or summary.get("loss")
        )
        if final_loss is None:
            # Fallback: scan history for the last non-NaN train/loss
            try:
                hist = run.history(keys=["train/loss"], pandas=True, samples=2000)
                col  = next((c for c in hist.columns if "loss" in c.lower()), None)
                if col:
                    vals       = hist[col].dropna()
                    final_loss = float(vals.iloc[-1]) if len(vals) > 0 else float("nan")
                else:
                    final_loss = float("nan")
            except Exception as e:
                print(f"  [warn] {run.name}: could not fetch history ({e})")
                final_loss = float("nan")

        c_train = compute_c_train(d_model) if d_model in TOKENS_BY_DIM else float("nan")
        n_steps = run.summary.get("_step", None)

        records.append(dict(
            run_name   = run.name,
            d_model    = d_model,
            lr         = lr,
            C_train    = c_train,
            final_loss = float(final_loss) if final_loss is not None else float("nan"),
            n_steps    = n_steps,
            wandb_id   = run.id,
        ))
        print(f"  {run.name:<22}  d={d_model}  lr={lr:.2e}  "
              f"loss={final_loss:.4f}  steps={n_steps}")

    return records


# =============================================================================
# ## Plot
# =============================================================================

def plot(df_records: list, output_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed: pip install matplotlib")

    import collections

    # Group by d_model
    by_dim = collections.defaultdict(list)
    for r in df_records:
        by_dim[r["d_model"]].append(r)

    dims   = sorted(by_dim.keys())
    cmap   = plt.cm.viridis(np.linspace(0.1, 0.9, len(dims)))
    colors = dict(zip(dims, cmap))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: Loss vs LR per scale ─────────────────────────────────────────
    ax  = axes[0]
    opt = {}   # dim → optimal lr

    for d in dims:
        rows   = sorted(by_dim[d], key=lambda r: r["lr"])
        lrs    = [r["lr"] for r in rows]
        losses = [r["final_loss"] for r in rows]
        c      = rows[0]["C_train"]
        color  = colors[d]

        ax.semilogx(lrs, losses, "o-", color=color, lw=1.5,
                    label=f"d={d}  (C={c:.1e})")

        # Mark optimal (min loss)
        best_i = int(np.nanargmin(losses))
        ax.plot(lrs[best_i], losses[best_i], "*", color=color, ms=14, zorder=5)
        opt[d] = lrs[best_i]

    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Final Train Loss", fontsize=12)
    ax.set_title("LR Sweep — Loss vs Learning Rate", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Optimal LR vs C_train (log-log, power-law fit) ───────────────
    ax2    = axes[1]
    c_vals = [by_dim[d][0]["C_train"] for d in dims if d in opt]
    lr_opt = [opt[d] for d in dims if d in opt]

    ax2.loglog(c_vals, lr_opt, "o", color="steelblue", ms=9, zorder=3)
    for d, c, lr in zip(dims, c_vals, lr_opt):
        ax2.annotate(f"d={d}", (c, lr),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Power-law fit: log(lr*) = β log(C) + const
    if len(c_vals) >= 2:
        log_c  = np.log(np.array(c_vals, dtype=float))
        log_lr = np.log(np.array(lr_opt, dtype=float))
        beta, log_a = np.polyfit(log_c, log_lr, 1)
        c_fit  = np.linspace(log_c.min(), log_c.max(), 200)
        ax2.loglog(np.exp(c_fit), np.exp(log_a + beta * c_fit),
                   "--", color="gray", alpha=0.8,
                   label=rf"lr* $\propto$ C$^{{{beta:.2f}}}$")
        ax2.legend(fontsize=11)
        print(f"\nPower-law fit:  lr*(C) ∝ C^{beta:.3f}  "
              f"(prefactor={np.exp(log_a):.3e})")

    ax2.set_xlabel("C_train  (total FLOPs)", fontsize=12)
    ax2.set_ylabel("Optimal Learning Rate", fontsize=12)
    ax2.set_title("Optimal LR Scaling", fontsize=13)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {output_path}")
    plt.show()


# =============================================================================
# ## CSV I/O
# =============================================================================

def records_to_csv(records: list, path: str):
    import csv
    fields = ["run_name", "d_model", "lr", "C_train", "final_loss", "n_steps", "wandb_id"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(records, key=lambda r: (r["d_model"], r["lr"])))
    print(f"Saved CSV  → {path}")


def records_from_csv(path: str) -> list:
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["d_model"]    = int(r["d_model"])
        r["lr"]         = float(r["lr"])
        r["C_train"]    = float(r["C_train"])
        r["final_loss"] = float(r["final_loss"])
        r["n_steps"]    = int(r["n_steps"]) if r.get("n_steps") else None
    return rows


# =============================================================================
# ## Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity",  default="iliass-lasri-team")
    parser.add_argument("--project", default="olmoe-1")
    parser.add_argument("--group",   default="lr_sweep")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--offline", metavar="CSV",
        help="Skip W&B; re-plot from an existing CSV file.",
    )
    args = parser.parse_args()

    csv_path  = os.path.join(args.output_dir, "lr_sweep.csv")
    plot_path = os.path.join(args.output_dir, "lr_sweep.png")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.offline:
        print(f"Offline mode — loading {args.offline}")
        records = records_from_csv(args.offline)
    else:
        records = fetch_from_wandb(args.entity, args.project, args.group)

    if not records:
        print("\nNo records found.  Check that:")
        print("  1. Runs are finished (state=finished)")
        print(f"  2. W&B group is set to '{args.group}' in the YAML configs")
        print("  3. --entity and --project match your W&B workspace")
        sys.exit(1)

    print(f"\nFetched {len(records)} run(s).")
    records_to_csv(records, csv_path)

    # Print table
    header = f"{'run_name':<22}  {'d':>4}  {'lr':>8}  {'C_train':>12}  {'final_loss':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(records, key=lambda x: (x["d_model"], x["lr"])):
        print(f"{r['run_name']:<22}  {r['d_model']:>4}  {r['lr']:>8.2e}  "
              f"{r['C_train']:>12.3e}  {r['final_loss']:>10.4f}")

    plot(records, plot_path)


if __name__ == "__main__":
    main()
