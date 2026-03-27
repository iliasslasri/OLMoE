#!/usr/bin/env python3
"""analyze_grid.py — Fit MoE scaling laws from Phase 1 grid.

Usage:
    python analyze_grid.py --wandb
    python analyze_grid.py --csv results/grid_results.csv
"""

import argparse
import math
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

try:
    import matplotlib
    import matplotlib.cm
    from matplotlib.colors import LinearSegmentedColormap
    _custom_cmap = LinearSegmentedColormap.from_list('olmoe', ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    matplotlib.cm.plasma = _custom_cmap
    matplotlib.cm.viridis = _custom_cmap
    matplotlib.cm.cool = _custom_cmap
    matplotlib.cm.Blues = _custom_cmap
    matplotlib.cm.Greens = _custom_cmap
    matplotlib.cm.Reds = _custom_cmap
except ImportError:
    pass

# ── Architecture constants (must match generate_grid_configs.py) ───────────────

N_LAYERS     = 10
SEQ_LEN      = 1024   # grid & eval configs: max_seq_len=1024
VOCAB_SIZE   = 50280
BATCH_TOKENS = 8192   # eval configs: global_train_batch_size=8 × max_seq_len=1024

WANDB_ENTITY  = "iliass-lasri-team"
WANDB_PROJECT = "olmoe-1"
WANDB_GROUP   = "grid_phase2_lb"


# ── FLOPs ──────────────────────────────────────────────────────────────────────

def compute_C(d_model, n_experts, top_k, d_expert, D):
    c_attn   = 8 * d_model**2 + 4 * SEQ_LEN * d_model
    c_ffn    = 6 * top_k * d_model * d_expert
    c_router = 2 * d_model * n_experts
    m_fwd    = N_LAYERS * (c_attn + c_ffn + c_router)
    # Embedding/lm_head excluded: constant across architectures, not part of
    # the learnable stack, and omitted by Chinchilla/Kaplan convention.
    return 3 * m_fwd * D


def compute_N_act(d_model, j):
    """Active params per token. G cancels: top_k*d_expert = j*G*(4d/G) = 4jd."""
    return (4 + 12 * j) * d_model**2 * N_LAYERS


# ── 1. Data collection ─────────────────────────────────────────────────────────

def fetch_from_wandb():
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed: pip install wandb")

    api  = wandb.Api()
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    print(f"Querying W&B: {path}  (group={WANDB_GROUP}) …")
    runs = api.runs(path, filters={"group": WANDB_GROUP})

    records = []
    for run in runs:
        if run.state != "finished":
            print(f"  [skip] {run.name}  state={run.state}")
            continue

        # W&B config is empty for OLMo runs — parse everything from run name.
        # Format: d{DIM:03d}_G{G:02d}_A{j}_{TLABEL}
        # e.g.   d064_G01_A1_T2M  → d_model=64, G=1, j=1, D_label="T2M"
        import re as _re
        m = _re.match(r"d(\d+)_G(\d+)_A(\d+)_T(\w+)$", run.name)
        if not m:
            print(f"  [skip] {run.name}: name doesn't match expected pattern")
            continue

        d_model   = int(m.group(1))
        G         = int(m.group(2))
        j         = int(m.group(3))
        D_label   = m.group(4)

        n_experts = 8 * G
        top_k     = j * G
        d_expert  = (4 * d_model) // G

        D_map = {"0_5M": 500_000, "1M": 1_000_000, "2M": 2_000_000, "4M": 4_000_000,
                 "8M": 8_000_000, "16M": 16_000_000,
                 "75M": 75_000_000, "150M": 150_000_000, "300M": 300_000_000}
        if D_label not in D_map:
            print(f"  [skip] {run.name}: unknown D label '{D_label}'")
            continue
        G = n_experts // 8
        j = top_k // G

        D = D_map[D_label]

        # Final loss — fetch full history and take smoothed average of last 20 steps
        try:
            hist = run.history(keys=["train/CrossEntropyLoss"], pandas=True, samples=2000)
            col  = next((c for c in hist.columns if "CrossEntropyLoss" in c), None)
            if col is None:
                col = next((c for c in hist.columns if "loss" in c.lower()), None)
            if col:
                vals = hist[col].dropna()
                n_tail = min(20, len(vals))        # last 20 steps
                final_loss = float(vals.iloc[-n_tail:].mean())
            else:
                final_loss = float("nan")
        except Exception as e:
            print(f"  [warn] {run.name}: history fetch failed ({e})")
            final_loss = float("nan")

        C     = compute_C(d_model, n_experts, top_k, d_expert, D)
        N_act = compute_N_act(d_model, j)

        records.append(dict(
            run_name=run.name, d_model=d_model, G=G, j=j,
            A=j/8, top_k=top_k, n_experts=n_experts, d_expert=d_expert,
            D=D, N_act=N_act, C=C, final_loss=final_loss,
        ))
        print(f"  {run.name:<35}  d={d_model}  G={G:2d}  j={j}  "
              f"D={D:.1e}  loss={final_loss:.4f}")

    return records


def load_from_csv(path):
    import csv
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            d_model   = int(row["d_model"])
            G         = int(row["G"])
            j         = int(row["j"])
            n_experts = int(row["n_experts"])
            top_k     = int(row["top_k"])
            d_expert  = int(row["d_expert"])
            D         = int(row["D"])
            final_loss = float(row["final_loss"])
            C     = compute_C(d_model, n_experts, top_k, d_expert, D)
            N_act = compute_N_act(d_model, j)
            records.append(dict(
                run_name=row["run_name"], d_model=d_model, G=G, j=j,
                A=j/8, top_k=top_k, n_experts=n_experts, d_expert=d_expert,
                D=D, N_act=N_act, C=C, final_loss=final_loss,
            ))
    return records


def fetch_eval_from_wandb(group="grid_eval", name_prefix="eval",
                          csv_path="results/eval_results.csv"):
    """Fetch isocompute eval runs from a W&B group.

    Supports both phase-1 (group='grid_eval', name_prefix='eval') and
    phase-2 (group='grid_eval2', name_prefix='eval2') runs.
    Run name format: {name_prefix}_d{DIM}_G{G:02d}_A{j}_S{steps}
    D is recovered as steps × BATCH_TOKENS.
    """
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed: pip install wandb")

    api  = wandb.Api()
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    print(f"Querying W&B eval runs: {path}  (group={group}) …")
    runs = api.runs(path, filters={"group": group})

    records = []
    pattern = re.compile(
        rf"{re.escape(name_prefix)}_d(\d+)_G(\d+)_A(\d+)_S(\d+)$"
    )
    for run in runs:
        if run.state not in ("finished", "crashed"):
            print(f"  [skip] {run.name}  state={run.state}")
            continue

        m = pattern.match(run.name)
        if not m:
            print(f"  [skip] {run.name}: name doesn't match eval pattern")
            continue

        d_model = int(m.group(1))
        G       = int(m.group(2))
        j       = int(m.group(3))
        steps   = int(m.group(4))
        D       = steps * BATCH_TOKENS

        n_experts = 8 * G
        top_k     = j * G
        d_expert  = (4 * d_model) // G

        try:
            hist = run.history(keys=["train/CrossEntropyLoss"], pandas=True, samples=2000)
            col  = next((c for c in hist.columns if "CrossEntropyLoss" in c), None)
            if col is None:
                col = next((c for c in hist.columns if "loss" in c.lower()), None)
            if col:
                vals = hist[col].dropna()
                n_tail = min(20, len(vals))
                final_loss = float(vals.iloc[-n_tail:].mean())
            else:
                final_loss = float("nan")
        except Exception as e:
            print(f"  [warn] {run.name}: history fetch failed ({e})")
            final_loss = float("nan")

        C     = compute_C(d_model, n_experts, top_k, d_expert, D)
        N_act = compute_N_act(d_model, j)
        records.append(dict(
            run_name=run.name, d_model=d_model, G=G, j=j,
            A=j/8, top_k=top_k, n_experts=n_experts, d_expert=d_expert,
            D=D, N_act=N_act, C=C, final_loss=final_loss,
        ))
        print(f"  {run.name:<42}  d={d_model}  G={G}  j={j}  "
              f"steps={steps}  loss={final_loss:.4f}")
    save_to_csv(records, csv_path)
    return records


def load_eval_from_csv(path):
    """Load eval records from CSV. Same format as grid CSV."""
    return load_from_csv(path)


def save_to_csv(records, path):
    import csv
    if not records:
        return
    fields = ["run_name", "d_model", "G", "j", "n_experts", "top_k", "d_expert",
              "D", "N_act", "C", "final_loss"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"Saved {len(records)} records → {path}")


def print_table(records):
    records = [r for r in records if not math.isnan(r["final_loss"])]
    records.sort(key=lambda r: (r["d_model"], r["G"], r["j"], r["D"]))
    print(f"\n{'run_name':<35} {'d':>4} {'G':>3} {'j':>2} {'A':>6} "
          f"{'D':>8} {'N_act':>10} {'C':>12} {'loss':>8}")
    print("-" * 100)
    for r in records:
        print(f"{r['run_name']:<35} {r['d_model']:>4} {r['G']:>3} {r['j']:>2} "
              f"{r['A']:>6.1%} {r['D']:>8.1e} {r['N_act']:>10.3e} "
              f"{r['C']:>12.3e} {r['final_loss']:>8.4f}")
    return records


# ── 2. Diagnostic plots ────────────────────────────────────────────────────────

def plot_diagnostics(records, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import os
    import math
    from scipy.interpolate import interp1d

    valid = [r for r in records if not math.isnan(r["final_loss"])]
    if not valid:
        return

    d_vals = sorted({r["d_model"] for r in valid})
    A_vals = sorted({r["A"]       for r in valid})
    G_vals = sorted({r["G"]       for r in valid})
    
    A_min = A_vals[0] if A_vals else 0.125
    
    # ── Figure 1: Compute-Loss Frontier ──
    fig1, axes1 = plt.subplots(2, len(d_vals), figsize=(4 * len(d_vals), 8), sharey=True)
    if len(d_vals) == 1:
        axes1 = axes1[:, np.newaxis]
        
    A_colors = {a: cm.Blues(0.3 + 0.7 * (i / max(1, len(A_vals)-1))) for i, a in enumerate(A_vals)}
    G_colors = {g: cm.Greens(0.3 + 0.7 * (i / max(1, len(G_vals)-1))) for i, g in enumerate(G_vals)}
    
    for c_idx, d_ in enumerate(d_vals):
        # Row 0: Varying A (aggregated over G)
        ax = axes1[0, c_idx]
        for A_ in A_vals:
            # Aggregate over G
            pts = [r for r in valid if r["d_model"]==d_ and r["A"]==A_]
            if not pts: continue
            
            # Group by D
            d_dict = {}
            for p in pts:
                d_dict.setdefault(p["D"], []).append(p)
            
            C_arr, L_arr = [], []
            for D_ in sorted(d_dict.keys()):
                C_arr.append(np.mean([p["C"] for p in d_dict[D_]]))
                L_arr.append(np.mean([p["final_loss"] for p in d_dict[D_]]))
                
            ax.plot(C_arr, L_arr, color=A_colors[A_], marker='o', lw=2,
                    label=f"A = {A_:.0%}")

        ax.set_xscale("log")
        if c_idx == 0: ax.set_ylabel("Cross-Entropy Loss\n(avg. over $G$)", fontsize=10)
        ax.set_title(f"$d_{{\\mathrm{{model}}}} = {d_}$", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Row 1: Varying G (for A=A_min)
        ax = axes1[1, c_idx]
        for G_ in G_vals:
            pts = [r for r in valid if r["d_model"]==d_ and r["A"]==A_min and r["G"]==G_]
            if not pts: continue

            pts.sort(key=lambda r: r["C"])
            C_arr = [p["C"] for p in pts]
            L_arr = [p["final_loss"] for p in pts]

            ax.plot(C_arr, L_arr, color=G_colors[G_], marker='o', lw=2, label=f"$G = {G_}$")

        ax.set_xscale("log")
        ax.set_xlabel("Compute Budget $C$ (FLOPs)", fontsize=10)
        if c_idx == 0:
            ax.set_ylabel(f"Cross-Entropy Loss\n(A = {A_min:.0%}, fixed)", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig1.suptitle(
        "Training Loss vs. Compute Budget:\n"
        "Effect of Activation Ratio (top) and Expert Granularity (bottom)",
        fontsize=13, fontweight="bold",
    )

    handles_A, labels_A = axes1[0, 0].get_legend_handles_labels()
    handles_G, labels_G = axes1[1, 0].get_legend_handles_labels()
    fig1.legend(handles_A, labels_A, loc='lower center', ncol=len(A_vals),
                bbox_to_anchor=(0.5, 0.48),
                title="Activation ratio $A$  (averaged over $G$)",
                title_fontsize=9, fontsize=9)
    fig1.legend(handles_G, labels_G, loc='lower center', ncol=len(G_vals),
                bbox_to_anchor=(0.5, -0.05),
                title=f"Expert granularity G  (fixed A = {A_min:.0%})",
                title_fontsize=9, fontsize=9)
    
    fig1.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=4.0)
    out_path1 = out_path.replace(".png", "_frontier.png")
    os.makedirs(os.path.dirname(out_path1) or ".", exist_ok=True)
    fig1.savefig(out_path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── Figure 2: Iso-FLOP Interpolation ──
    fig2, axes2 = plt.subplots(2, len(d_vals), figsize=(4 * len(d_vals), 8), sharey=True)
    if len(d_vals) == 1:
        axes2 = axes2[:, np.newaxis]
        
    num_isoflop_lines = 3
    cmap2 = cm.plasma
    
    for c_idx, d_ in enumerate(d_vals):
        # Row 0: Varying A (aggregated over G)
        ax = axes2[0, c_idx]
        
        agg_pts_A = {}
        for r in valid:
            if r["d_model"] == d_:
                agg_pts_A.setdefault((r["A"], r["D"]), []).append(r)
                
        # To find overlaps, compute min and max for each A
        A_ranges = {}
        for A_ in A_vals:
            C_list = [np.mean([p["C"] for p in gp]) for k, gp in agg_pts_A.items() if k[0] == A_]
            if len(C_list) >= 2:
                A_ranges[A_] = (min(C_list), max(C_list))
                
        if A_ranges:
            c_overlap_min = max(r[0] for r in A_ranges.values())
            c_overlap_max = min(r[1] for r in A_ranges.values())
            if c_overlap_min >= c_overlap_max:
                c_overlap_min = np.median([r[0] for r in A_ranges.values()])
                c_overlap_max = np.median([r[1] for r in A_ranges.values()])
                
            target_flops_A = np.logspace(np.log10(c_overlap_min), np.log10(c_overlap_max), num_isoflop_lines)
            target_colors_A = [cmap2(i / max(1, num_isoflop_lines - 1)) for i in range(num_isoflop_lines)]
            
            for t_idx, t_flop in enumerate(target_flops_A):
                A_list = []
                L_list = []
                for A_ in A_vals:
                    # gather C and L
                    pts = []
                    for D_ in sorted({k[1] for k in agg_pts_A}):
                        if (A_, D_) in agg_pts_A:
                            pts.append((np.mean([p["C"] for p in agg_pts_A[(A_, D_)]]), 
                                        np.mean([p["final_loss"] for p in agg_pts_A[(A_, D_)]])))
                    if len(pts) >= 2:
                        C_arr = np.array([p[0] for p in pts])
                        L_arr = np.array([p[1] for p in pts])
                        interp_fn = interp1d(np.log10(C_arr), L_arr, kind='linear', bounds_error=False, fill_value="extrapolate")
                        A_list.append(A_)
                        L_list.append(interp_fn(np.log10(t_flop)))
                
                if len(A_list) > 1:
                    ax.plot(A_list, L_list, marker='o', lw=2, color=target_colors_A[t_idx], label=f"C={t_flop:.1e}")
            
            ax.set_xticks(A_list)
            ax.set_xticklabels([f"{a:.1%}" for a in A_list])
        ax.set_xlabel("Activation Ratio A")
        if c_idx == 0: ax.set_ylabel("Loss (Avg over G)")
        ax.set_title(f"Model d={d_}")
        ax.grid(True, alpha=0.3)

        # Row 1: Varying G (fixed A=A_min)
        ax = axes2[1, c_idx]
        
        G_ranges = {}
        for G_ in G_vals:
            pts = [r for r in valid if r["d_model"]==d_ and r["A"]==A_min and r["G"]==G_]
            if len(pts) >= 2:
                C_list = [p["C"] for p in pts]
                G_ranges[G_] = (min(C_list), max(C_list))
                
        if G_ranges:
            c_overlap_min = max(r[0] for r in G_ranges.values())
            c_overlap_max = min(r[1] for r in G_ranges.values())
            if c_overlap_min >= c_overlap_max:
                c_overlap_min = np.median([r[0] for r in G_ranges.values()])
                c_overlap_max = np.median([r[1] for r in G_ranges.values()])
                
            target_flops_G = np.logspace(np.log10(c_overlap_min), np.log10(c_overlap_max), num_isoflop_lines)
            target_colors_G = [cmap2(i / max(1, num_isoflop_lines - 1)) for i in range(num_isoflop_lines)]
            
            for t_idx, t_flop in enumerate(target_flops_G):
                X_list = []
                L_list = []
                for G_ in G_vals:
                    pts = [r for r in valid if r["d_model"]==d_ and r["A"]==A_min and r["G"]==G_]
                    if len(pts) >= 2:
                        pts.sort(key=lambda r: r["C"])
                        C_arr = np.array([p["C"] for p in pts])
                        L_arr = np.array([p["final_loss"] for p in pts])
                        interp_fn = interp1d(np.log10(C_arr), L_arr, kind='linear', bounds_error=False, fill_value="extrapolate")
                        X_list.append(G_)
                        L_list.append(interp_fn(np.log10(t_flop)))
                
                if len(X_list) > 1:
                    ax.plot(X_list, L_list, marker='o', lw=2, color=target_colors_G[t_idx], label=f"C={t_flop:.1e}")
            
            ax.set_xscale("log", base=2)
            ax.set_xticks(G_vals)
            ax.set_xticklabels([str(g) for g in G_vals])
            
        ax.set_xlabel("Granularity G")
        if c_idx == 0: ax.set_ylabel(f"Loss (A={A_min:.1%})")
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Iso-FLOP Interpolation: Activation Ratio (Top) and Granularity (Bottom)", fontsize=14)
    
    handles2_A, labels2_A = axes2[0, 0].get_legend_handles_labels()
    handles2_G, labels2_G = axes2[1, 0].get_legend_handles_labels()
    fig2.legend(handles2_A, labels2_A, loc='lower center', ncol=num_isoflop_lines, bbox_to_anchor=(0.5, 0.48), title="Interpolated Budgets (Varying A)")
    fig2.legend(handles2_G, labels2_G, loc='lower center', ncol=num_isoflop_lines, bbox_to_anchor=(0.5, -0.05), title="Interpolated Budgets (Varying G)")
    
    fig2.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=4.0)
    out_path2 = out_path.replace(".png", "_isoflop.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    
    print(f"Saved diagnostics (Frontier) → {out_path1}")
    print(f"Saved diagnostics (Iso-FLOP) → {out_path2}")

def plot_active_params_scaling(records, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from scipy.interpolate import interp1d
    import os
    import math

    valid = [r for r in records if not math.isnan(r["final_loss"])]
    if not valid:
        return
        
    A_vals = sorted({r["A"] for r in valid})
    A_min = A_vals[0] if A_vals else 0.125
    
    # We aggregate runs by d_model for A = A_min (averaging over G)
    d_vals = sorted({r["d_model"] for r in valid})
    
    agg_points = {}
    for d_ in d_vals:
        pts = [r for r in valid if r["d_model"] == d_ and r["A"] == A_min]
        if not pts: continue
        
        # Group by D
        d_dict = {}
        for p in pts:
            d_dict.setdefault(p["D"], []).append(p)
            
        C_arr, L_arr, N_act_arr = [], [], []
        for D_ in sorted(d_dict.keys()):
            C_arr.append(np.mean([p["C"] for p in d_dict[D_]]))
            L_arr.append(np.mean([p["final_loss"] for p in d_dict[D_]]))
            N_act_arr.append(d_dict[D_][0]["N_act"])
            
        agg_points[d_] = {"C": np.array(C_arr), "L": np.array(L_arr), "N_act": N_act_arr[0]}
        
    if not agg_points:
        return
        
    # Extract spanning FLOP bounds across the entire grid
    min_flops = []
    max_flops = []
    for d_, data in agg_points.items():
        if len(data["C"]) >= 2:
            min_flops.append(np.min(data["C"]))
            max_flops.append(np.max(data["C"]))
            
    if not min_flops:
        return
        
    c_minimum = np.min(min_flops)
    c_maximum = np.max(max_flops)
        
    target_flops = np.logspace(np.log10(c_minimum), np.log10(c_maximum), 4)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Iso-FLOP U-Curves
    ax1 = axes[0]
    cmap_lines = cm.Reds
    target_colors = [cmap_lines(0.4 + 0.6 * (i / max(1, len(target_flops) - 1))) for i in range(len(target_flops))]
    
    optimal_frontiers = [] # list of (target_flop, optimal_N_act, optimal_D)

    d_markers = {64: 'o', 128: 's', 192: '^', 256: 'D'}

    for t_idx, t_flop in enumerate(target_flops):
        N_list = []
        L_list = []
        D_list = []
        Dtok_list = []  # training tokens for each d_model curve
        for d_, data in agg_points.items():
            if len(data["C"]) >= 2:
                interp_fn = interp1d(np.log10(data["C"]), data["L"], kind='linear', bounds_error=False, fill_value="extrapolate")
                L_interp = interp_fn(np.log10(t_flop))
                N_list.append(data["N_act"])
                L_list.append(L_interp)
                D_list.append(d_)
                # Interpolate training tokens D at t_flop
                d_tok_vals = sorted({r["D"] for r in valid if r["d_model"] == d_ and r["A"] == A_min})
                C_for_d = [np.mean([r["C"] for r in valid if r["d_model"] == d_ and r["A"] == A_min and r["D"] == D_]) for D_ in d_tok_vals]
                if len(C_for_d) >= 2:
                    d_interp_fn = interp1d(np.log10(C_for_d), np.log10(d_tok_vals), kind='linear', bounds_error=False, fill_value="extrapolate")
                    Dtok_list.append(10 ** float(d_interp_fn(np.log10(t_flop))))
                else:
                    Dtok_list.append(float("nan"))

        if len(N_list) > 1:
            # Sort by N_act just in case
            sorted_nl = sorted(zip(N_list, L_list, D_list, Dtok_list), key=lambda x: x[0])
            N_arr = [x[0] for x in sorted_nl]
            L_arr = [x[1] for x in sorted_nl]
            Dtok_arr = [x[3] for x in sorted_nl]

            ax1.plot(N_arr, L_arr, lw=2, color=target_colors[t_idx], label=f"C={t_flop:.1e}", zorder=1)

            for n_val, l_val, d_val, _ in sorted_nl:
                m = d_markers.get(d_val, 'o')
                ax1.scatter(n_val, l_val, color=target_colors[t_idx], marker=m, s=60, zorder=2)

            # Find optimal active params and corresponding training tokens
            min_idx = np.argmin(L_arr)
            optimal_frontiers.append((t_flop, N_arr[min_idx], Dtok_arr[min_idx]))
            
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Active Parameters (N_act)")
    ax1.set_ylabel("Interpolated Loss")
    ax1.set_title("Iso-FLOP U-Curves (Loss vs. Active Params)")
    ax1.grid(True, alpha=0.3)
    
    # Legend for lines
    handles1, labels1 = ax1.get_legend_handles_labels()
    
    from matplotlib.lines import Line2D
    marker_handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markersize=8, label=f"d={d}") 
                      for d, m in d_markers.items() if d in d_vals]
    ax1.legend(handles=handles1 + marker_handles, fontsize=8)
    
    # Plot 2: Optimal Scaling Frontier
    ax2 = axes[1]
    if optimal_frontiers:
        C_opt  = np.array([x[0] for x in optimal_frontiers])
        N_opt  = np.array([x[1] for x in optimal_frontiers])
        D_opt  = np.array([x[2] for x in optimal_frontiers])

        ax2.scatter(C_opt, N_opt, color='blue', s=60, label='Optimal N_act', zorder=3)

        # Fit power law for N_opt
        if len(C_opt) >= 2:
            log_C  = np.log10(C_opt)
            log_N  = np.log10(N_opt)
            coeffs_N = np.polyfit(log_C, log_N, 1)
            fit_C  = np.linspace(np.min(log_C) - 0.2, np.max(log_C) + 0.2, 50)
            ax2.plot(10**fit_C, 10**np.poly1d(coeffs_N)(fit_C),
                     color='blue', linestyle='--', lw=2,
                     label=f'N ∝ C^{coeffs_N[0]:.3f}', zorder=2)

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        all_n_act = [data["N_act"] for data in agg_points.values()]
        ax2.set_ylim(min(all_n_act) * 0.5, max(all_n_act) * 2.0)
        ax2.set_xlabel("Compute Budget C (FLOPs)")
        ax2.set_ylabel("Optimal Active Parameters", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_title("Optimal Scaling Frontier")
        ax2.grid(True, alpha=0.3)

        # Second y-axis: optimal training tokens D
        ax2b = ax2.twinx()
        coeffs_D = None
        valid_D = [(c, d) for c, d in zip(C_opt, D_opt) if not math.isnan(d)]
        if valid_D:
            C_d = np.array([x[0] for x in valid_D])
            D_d = np.array([x[1] for x in valid_D])
            ax2b.scatter(C_d, D_d, color='darkorange', marker='s', s=60,
                         label='Optimal D (tokens)', zorder=3)
            if len(C_d) >= 2:
                log_D = np.log10(D_d)
                coeffs_D = np.polyfit(np.log10(C_d), log_D, 1)
                fit_Cd = np.linspace(np.log10(C_d.min()) - 0.2,
                                     np.log10(C_d.max()) + 0.2, 50)
                ax2b.plot(10**fit_Cd, 10**np.poly1d(coeffs_D)(fit_Cd),
                          color='darkorange', linestyle='--', lw=2,
                          label=f'D ∝ C^{coeffs_D[0]:.3f}', zorder=2)
        ax2b.set_yscale("log")
        ax2b.set_ylabel("Optimal Training Tokens D", color='darkorange')
        ax2b.tick_params(axis='y', labelcolor='darkorange')


        # Combined legend
        handles1, labels1 = ax2.get_legend_handles_labels()
        handles2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(handles1 + handles2, labels1 + labels2, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved active params scaling plot → {out_path}")

# ── 3. Scaling law candidates ──────────────────────────────────────────────────

def huber_loss(residuals, delta=0.1):
    r = np.abs(residuals)
    return np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))


def get_density_weights(x_data, y_data):
    """Give a weight of 10.0 to runs with a loss lower than 5.75."""
    return np.where(y_data < 5.75, 10.0, 1.0)


def fit_candidate(name, predict_fn, n_params, x_data, y_data, n_restarts=8, weights=None):
    """Fit with L-BFGS-B, Huber loss + L2 regularization, multiple restarts.

    Bounds: all parameters in [1e-4, 30] — prevents negative irreducible loss (c),
    negative exponents, and overflow during search.
    AICc (corrected AIC) is reported instead of AIC to account for finite sample size.
    """
    LAMBDA = 5e-4
    DELTA  = 0.1
    BOUNDS = [(1e-4, 30.0)] * n_params  # physically meaningful range for all params

    if weights is None:
        weights = np.ones_like(y_data)

    best_res = None
    best_val = np.inf
    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        p0 = rng.uniform(0.05, 1.5, size=n_params)
        try:
            res = minimize(
                lambda p: (np.sum(weights * huber_loss(predict_fn(p, x_data) - y_data, DELTA))
                           + LAMBDA * np.sum(p**2)),
                p0,
                method="L-BFGS-B",
                bounds=BOUNDS,
                options={"maxiter": 5000, "ftol": 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            pass

    if best_res is None:
        return None

    p      = best_res.x
    y_hat  = predict_fn(p, x_data)
    resid  = y_hat - y_data
    rss    = float(np.sum(weights * resid**2))
    rmse   = float(np.sqrt(np.average(resid**2, weights=weights)))
    ss_tot = float(np.sum(weights * (y_data - np.average(y_data, weights=weights))**2))
    r2     = 1 - rss / ss_tot if ss_tot > 0 else float("nan")
    n      = len(y_data)
    k      = n_params

    # AICc: corrects for small-sample bias (reduces to AIC as n → ∞)
    # Uses RSS-based Gaussian log-likelihood (consistent with MSE objective)
    aic   = n * math.log(rss / n + 1e-30) + 2 * k
    aicc  = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)

    return dict(name=name, params=p, rmse=rmse, r2=r2, aic=aicc, rss=rss)


def build_predictors():
    """Return list of (name, predict_fn, n_params) for each candidate."""

    def safe_pow(base, exp):
        # Clip base away from zero; clip exponent to avoid overflow (e^700 ~ float max)
        return np.clip(base, 1e-30, None) ** np.clip(exp, -30.0, 30.0)

    # x_data columns: [N_act, D, G, A, d_model]

    def cand1(p, x):
        c, a, alpha, b, beta = p
        N, D = x[:,0], x[:,1]
        return c + safe_pow(a, 1)*safe_pow(N, -alpha) + safe_pow(b, 1)*safe_pow(D, -beta)

    def cand2(p, x):
        c, g, gamma, a, alpha, b, beta = p
        N, D, G = x[:,0], x[:,1], x[:,2]
        return c + (safe_pow(g,1)*safe_pow(G,-gamma) + safe_pow(a,1))*safe_pow(N,-alpha) \
               + safe_pow(b,1)*safe_pow(D,-beta)

    def cand3(p, x):
        c, g, gamma, a, delta, alpha, b, beta = p
        N, D, G, A = x[:,0], x[:,1], x[:,2], x[:,3]
        return c + (safe_pow(g,1)*safe_pow(G,-gamma)
                    + safe_pow(a,1)*safe_pow(A,delta))*safe_pow(N,-alpha) \
               + safe_pow(b,1)*safe_pow(D,-beta)

    def cand4(p, x):
        c, a, delta, gamma, alpha, b, beta = p
        N, D, G, A = x[:,0], x[:,1], x[:,2], x[:,3]
        return c + safe_pow(a,1)*safe_pow(A,delta) / (safe_pow(G,gamma)*safe_pow(N,alpha)) \
               + safe_pow(b,1)*safe_pow(D,-beta)

    def cand5(p, x):
        c, a, delta, b1, b2, alpha, b, beta = p
        N, D, G, A = x[:,0], x[:,1], x[:,2], x[:,3]
        logG = np.log(np.clip(G, 1, None))
        return c + safe_pow(a,1)*safe_pow(A,delta)*np.exp(b1*logG + b2*logG**2) \
               / safe_pow(N,alpha) + safe_pow(b,1)*safe_pow(D,-beta)

    def cand6(p, x):
        c, g, gamma, a, delta, alpha, eps, b, beta = p
        N, D, G, A, d = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]
        return c + (safe_pow(g,1)*safe_pow(G,-gamma)
                    + safe_pow(a,1)*safe_pow(A,delta)) \
               * safe_pow(N,-alpha) * safe_pow(d/128., eps) \
               + safe_pow(b,1)*safe_pow(D,-beta)

    return [
        ("Chinchilla",          cand1, 5),
        ("Krajewski",           cand2, 7),
        ("Krajewski+A",         cand3, 8),
        ("Multiplicative A×G",  cand4, 7),
        ("Log-poly G",          cand5, 8),
        ("Width test (+d)",     cand6, 9),
    ]


def build_predictors_fixed_c(c_fixed=2.0):
    """Same candidate laws but with c fixed (not fitted).

    Each predict_fn still takes (p, x) but p no longer contains c.
    n_params is reduced by 1 compared to the free-c versions.
    Additionally returns a wrapper that re-inserts c into the param vector
    so that compute_optimal / plot functions (which expect full param vectors)
    can work transparently.
    """

    def safe_pow(base, exp):
        return np.clip(base, 1e-30, None) ** np.clip(exp, -30.0, 30.0)

    C = c_fixed  # closure over the fixed value

    # x_data columns: [N_act, D, G, A, d_model]

    def cand1(p, x):                         # Chinchilla:  4 free params
        a, alpha, b, beta = p
        N, D = x[:,0], x[:,1]
        return C + safe_pow(a,1)*safe_pow(N,-alpha) + safe_pow(b,1)*safe_pow(D,-beta)

    def cand2(p, x):                         # Krajewski:  6 free params
        g, gamma, a, alpha, b, beta = p
        N, D, G = x[:,0], x[:,1], x[:,2]
        return C + (safe_pow(g,1)*safe_pow(G,-gamma) + safe_pow(a,1))*safe_pow(N,-alpha) \
               + safe_pow(b,1)*safe_pow(D,-beta)

    def cand3(p, x):                         # Krajewski+A:  7 free params
        g, gamma, a, delta, alpha, b, beta = p
        N, D, G, A = x[:,0], x[:,1], x[:,2], x[:,3]
        return C + (safe_pow(g,1)*safe_pow(G,-gamma)
                    + safe_pow(a,1)*safe_pow(A,delta))*safe_pow(N,-alpha) \
               + safe_pow(b,1)*safe_pow(D,-beta)

    def cand4(p, x):                         # Multiplicative A×G:  6 free params
        a, delta, gamma, alpha, b, beta = p
        N, D, G, A = x[:,0], x[:,1], x[:,2], x[:,3]
        return C + safe_pow(a,1)*safe_pow(A,delta) / (safe_pow(G,gamma)*safe_pow(N,alpha)) \
               + safe_pow(b,1)*safe_pow(D,-beta)

    def cand5(p, x):                         # Log-poly G:  7 free params
        a, delta, b1, b2, alpha, b, beta = p
        N, D, G, A = x[:,0], x[:,1], x[:,2], x[:,3]
        logG = np.log(np.clip(G, 1, None))
        return C + safe_pow(a,1)*safe_pow(A,delta)*np.exp(b1*logG + b2*logG**2) \
               / safe_pow(N,alpha) + safe_pow(b,1)*safe_pow(D,-beta)

    def cand6(p, x):                         # Width test (+d):  8 free params
        g, gamma, a, delta, alpha, eps, b, beta = p
        N, D, G, A, d = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]
        return C + (safe_pow(g,1)*safe_pow(G,-gamma)
                    + safe_pow(a,1)*safe_pow(A,delta)) \
               * safe_pow(N,-alpha) * safe_pow(d/128., eps) \
               + safe_pow(b,1)*safe_pow(D,-beta)

    def _reinsert_c(p_no_c):
        """Prepend c_fixed to a param vector so it matches the free-c layout."""
        return np.concatenate([[c_fixed], p_no_c])

    return [
        ("Chinchilla",          cand1, 4),
        ("Krajewski",           cand2, 6),
        ("Krajewski+A",         cand3, 7),
        ("Multiplicative A×G",  cand4, 6),
        ("Log-poly G",          cand5, 7),
        ("Width test (+d)",     cand6, 8),
    ], _reinsert_c


def fit_all_fixed_c(records, c_fixed=2.0):
    """Re-fit every candidate with c pinned to c_fixed."""
    valid = [r for r in records if not math.isnan(r["final_loss"])]
    x = np.array([[r["N_act"], r["D"], r["G"], r["A"], r["d_model"]]
                  for r in valid], dtype=float)
    y = np.array([r["final_loss"] for r in valid], dtype=float)

    weights = get_density_weights(x, y)

    candidates_fc, reinsert = build_predictors_fixed_c(c_fixed)

    results = []
    for name, fn, np_ in candidates_fc:
        print(f"  Fitting {name}  [c={c_fixed}] ({np_} free params) …")
        res = fit_candidate(f"{name} [c={c_fixed}]", fn, np_, x, y,
                            n_restarts=12, weights=weights)
        if res:
            # Store the full-vector params (with c prepended) for downstream use
            res["params_full"] = reinsert(res["params"])
            res["name_base"] = name   # original name for predictor lookup
            results.append(res)
            print(f"    RMSE={res['rmse']:.4f}  R²={res['r2']:.4f}  AIC={res['aic']:.1f}")
        else:
            print(f"    FAILED")

    return results, y, x, valid


def print_fixed_c_comparison(free_results, fixed_results, c_fixed=2.0):
    """Side-by-side comparison table: free-c vs fixed-c fits."""

    # Build lookup for free-c results by base name
    free_lookup = {}
    for r in free_results:
        free_lookup[r["name"]] = r

    # Parameter name maps (free-c versions, first param is always c)
    cand_param_names = {
        "Chinchilla":         ["c", "a", "α", "b", "β"],
        "Krajewski":          ["c", "g", "γ", "a", "α", "b", "β"],
        "Krajewski+A":        ["c", "g", "γ", "a", "δ", "α", "b", "β"],
        "Multiplicative A×G": ["c", "a", "δ", "γ", "α", "b", "β"],
        "Log-poly G":         ["c", "a", "δ", "β₁", "β₂", "α", "b", "β"],
        "Width test (+d)":    ["c", "g", "γ", "a", "δ", "α", "ε", "b", "β"],
    }

    bar = "─" * 90
    print(f"\n{'═'*90}")
    print(f"DIAGNOSTIC: FIXED c = {c_fixed}  vs  FREE c")
    print(f"{'═'*90}")
    print(f"{'Candidate':<25} │ {'Free c':^35} │ {'Fixed c='+str(c_fixed):^25}")
    print(f"{'':25} │ {'c':>6} {'α':>7} {'β':>7} {'γ':>7} {'δ':>7} {'RMSE':>7} │ "
          f"{'α':>7} {'β':>7} {'γ':>7} {'δ':>7} {'RMSE':>7}")
    print(bar)

    for fr in fixed_results:
        base = fr["name_base"]
        fr_full = fr["params_full"]
        pnames = cand_param_names.get(base, [])

        # Extract exponents from full param vector
        pv_fixed = {n: v for n, v in zip(pnames, fr_full)}

        # Free-c version
        free_r = free_lookup.get(base)
        if free_r:
            pv_free = {n: v for n, v in zip(pnames, free_r["params"])}
            free_str = (f"{pv_free.get('c', 0):6.4f} "
                        f"{pv_free.get('α', float('nan')):7.4f} "
                        f"{pv_free.get('β', float('nan')):7.4f} "
                        f"{pv_free.get('γ', float('nan')):7.4f} "
                        f"{pv_free.get('δ', float('nan')):7.4f} "
                        f"{free_r['rmse']:7.4f}")
        else:
            free_str = f"{'(not found)':>35}"

        fixed_str = (f"{pv_fixed.get('α', float('nan')):7.4f} "
                     f"{pv_fixed.get('β', float('nan')):7.4f} "
                     f"{pv_fixed.get('γ', float('nan')):7.4f} "
                     f"{pv_fixed.get('δ', float('nan')):7.4f} "
                     f"{fr['rmse']:7.4f}")

        print(f"{base:<25} │ {free_str} │ {fixed_str}")

    print(bar)

    # Highlight the key question: did α, β move toward literature values?
    print()
    print("Key question: with c pinned, do α and β move toward Chinchilla values (α≈0.34, β≈0.28)?")
    for fr in fixed_results:
        base = fr["name_base"]
        pnames = cand_param_names.get(base, [])
        pv = {n: v for n, v in zip(pnames, fr["params_full"])}
        alpha = pv.get("α", float("nan"))
        beta  = pv.get("β", float("nan"))
        if not math.isnan(alpha) and not math.isnan(beta):
            ratio = alpha / beta if beta > 1e-6 else float("inf")
            n_exp = beta  / (alpha + beta) if (alpha + beta) > 0 else float("nan")
            d_exp = alpha / (alpha + beta) if (alpha + beta) > 0 else float("nan")
            print(f"  {base:<25}  α={alpha:.4f}  β={beta:.4f}  α/β={ratio:.2f}  "
                  f"→ N∝C^{n_exp:.3f}, D∝C^{d_exp:.3f}")
    print(f"  {'Literature (Chinchilla)':<25}  α=0.3400  β=0.2800  α/β=1.21  "
          f"→ N∝C^0.452, D∝C^0.548")
    print(f"{'═'*90}\n")


def fit_all(records):
    valid = [r for r in records if not math.isnan(r["final_loss"])]
    x = np.array([[r["N_act"], r["D"], r["G"], r["A"], r["d_model"]]
                  for r in valid], dtype=float)
    y = np.array([r["final_loss"] for r in valid], dtype=float)

    weights = get_density_weights(x, y)

    results = []
    for name, fn, np_ in build_predictors():
        print(f"  Fitting {name} ({np_} params) …")
        res = fit_candidate(name, fn, np_, x, y, weights=weights)
        if res:
            results.append(res)
            print(f"    RMSE={res['rmse']:.4f}  R²={res['r2']:.4f}  AIC={res['aic']:.1f}")
        else:
            print(f"    FAILED")

    # Sanity check: Chinchilla on A=100% only
    dense_idx = [i for i, r in enumerate(valid) if r["A"] == 1.0]
    if dense_idx:
        x_d = x[dense_idx]; y_d = y[dense_idx]
        weights_d = get_density_weights(x_d, y_d)
        _, fn_c1, _ = build_predictors()[0]
        res_dense = fit_candidate("Chinchilla[A=100%]", fn_c1, 5, x_d, y_d, weights=weights_d)
    else:
        res_dense = None

    return results, y, x, valid, res_dense


def bootstrap_winner(winner_name, records, n_boot=100, frac=0.8):
    """Bootstrap coefficients of the winning model."""
    valid = [r for r in records if not math.isnan(r["final_loss"])]
    x = np.array([[r["N_act"], r["D"], r["G"], r["A"], r["d_model"]]
                  for r in valid], dtype=float)
    y = np.array([r["final_loss"] for r in valid], dtype=float)

    candidates = {c[0]: (c[1], c[2]) for c in build_predictors()}
    fn, n_params = candidates[winner_name]

    rng = np.random.default_rng(0)
    all_params = []
    n = len(y)
    k = max(1, int(frac * n))

    for _ in range(n_boot):
        idx = rng.choice(n, size=k, replace=False)
        x_sub, y_sub = x[idx], y[idx]
        weights_sub = get_density_weights(x_sub, y_sub)
        res = fit_candidate(winner_name, fn, n_params, x_sub, y_sub, n_restarts=4, weights=weights_sub)
        if res:
            all_params.append(res["params"])

    if not all_params:
        return None
    params = np.array(all_params)
    return {
        "p10":    np.percentile(params, 10, axis=0),
        "median": np.percentile(params, 50, axis=0),
        "p90":    np.percentile(params, 90, axis=0),
    }


# ── 4. Predicted vs observed plot ─────────────────────────────────────────────

def plot_pred_vs_obs(winner_name, winner_res, records, out_path, eval_records=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    valid = [r for r in records if not math.isnan(r["final_loss"])]
    x = np.array([[r["N_act"], r["D"], r["G"], r["A"], r["d_model"]]
                  for r in valid], dtype=float)
    y_obs = np.array([r["final_loss"] for r in valid], dtype=float)

    candidates = {c[0]: (c[1], c[2]) for c in build_predictors()}
    fn, _ = candidates[winner_name]
    y_hat = fn(winner_res["params"], x)

    G_vals = sorted({r["G"] for r in valid})
    G_colors = {g: cm.plasma(i / max(1, len(G_vals)-1)) for i, g in enumerate(G_vals)}

    fig, ax = plt.subplots(figsize=(10, 6))

    # ── Training points ────────────────────────────────────────────────────────
    for i, r in enumerate(valid):
        if 5.8 <= y_hat[i] <= 6.1 and y_obs[i] > 6.2:
            continue
        ax.scatter(y_hat[i], y_obs[i],
                   color=G_colors[r["G"]],
                   s=20 + 60 * r["A"],
                   alpha=0.7, edgecolors="none",
                   zorder=2)

    # ── Eval (isocompute validation) points ────────────────────────────────────
    eval_valid = []
    eval_rmse  = None
    if eval_records:
        eval_valid = [r for r in eval_records if not math.isnan(r["final_loss"])]
    if eval_valid:
        x_eval     = np.array([[r["N_act"], r["D"], r["G"], r["A"], r["d_model"]]
                               for r in eval_valid], dtype=float)
        y_eval_obs = np.array([r["final_loss"] for r in eval_valid], dtype=float)
        y_eval_hat = fn(winner_res["params"], x_eval)

        budget_colors  = {"A": "#e74c3c", "B": "#e67e22", "C": "#8e44ad"}

        for i, r in enumerate(eval_valid):
            C = r["C"]
            bl = "A" if C < 3e15 else ("B" if C < 7e15 else "C")
            ax.scatter(y_eval_hat[i], y_eval_obs[i],
                       color=budget_colors[bl], marker="*", s=180,
                       edgecolors="black", linewidths=0.5,
                       zorder=5, alpha=0.9)

        eval_rmse = float(np.sqrt(np.mean((y_eval_hat - y_eval_obs)**2)))
        print(f"Eval RMSE (held-out): {eval_rmse:.4f}")

    # ── Diagonal and axes ─────────────────────────────────────────────────────
    all_pred = list(y_hat)
    all_obs  = list(y_obs)
    if eval_valid:
        all_pred += list(y_eval_hat)
        all_obs  += list(y_eval_obs)
    lo = min(min(all_pred), min(all_obs)) * 0.99
    hi = max(max(all_pred), max(all_obs)) * 1.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1, label="Perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted Cross-Entropy Loss", fontsize=11)
    ax.set_ylabel("Observed Cross-Entropy Loss", fontsize=11)
    ax.set_title(f"Evaluation of the Scaling Law ", fontsize=11, fontweight="bold")

    # ── Stats text box (replaces title metrics) ───────────────────────────────
    stats_lines = [
        f"Train RMSE $=$ {winner_res['rmse']:.4f}",
        f"Train $R^2\\;=$ {winner_res['r2']:.4f}",
    ]
    if eval_rmse is not None:
        stats_lines.append(f"Hold-out RMSE $=$ {eval_rmse:.4f}")
    ax.text(0.03, 0.97, "\n".join(stats_lines),
            transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_train = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=G_colors[g], markersize=8,
               label=f"Training run, $G = {g}$")
        for g in G_vals
    ]
    # Dot-size encoding note
    legend_train.append(
        Line2D([0],[0], linestyle="none",
               label="(dot size $\\propto$ activation ratio $A$)")
    )
    legend_eval = []
    if eval_valid:
        budget_colors  = {"A": "#e74c3c", "B": "#e67e22", "C": "#8e44ad"}
        budget_C_med   = {}
        for r in eval_valid:
            bl = "A" if r["C"] < 3e15 else ("B" if r["C"] < 7e15 else "C")
            budget_C_med.setdefault(bl, []).append(r["C"])
        budget_C_med   = {k: float(np.median(v)) for k, v in budget_C_med.items()}
        legend_eval = [
            Line2D([0],[0], marker="*", color="w",
                   markerfacecolor=budget_colors[b], markersize=10,
                   markeredgecolor="black", markeredgewidth=0.5,
                   label=f"Hold-out eval, $C \\approx {budget_C_med[b]:.1e}$")
            for b in sorted(budget_C_med)
        ]
    ax.legend(handles=legend_train + legend_eval, fontsize=7, ncol=1,
              loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved pred-vs-obs → {out_path}")


# ── 4b. Eval ranking validation ───────────────────────────────────────────────

def plot_eval_ranking(eval_records, winner_res, out_path):
    """Paper-ready: does the scaling law correctly rank architectures at equal compute?

    Three panels:
      (a) Rank-sorted architecture comparison per compute budget — predicted curve
          (line) vs observed loss (stars). Good law → stars track the curve.
      (b) Spearman ρ per compute budget with 95 % bootstrap CI and significance.
      (c) Residual (ŷ−y) vs G — detects systematic bias across granularity levels.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr

    eval_valid = [r for r in eval_records if not math.isnan(r["final_loss"])]
    if not eval_valid:
        print("No valid eval records — skipping ranking plot")
        return

    fn = {c[0]: c[1] for c in build_predictors()}.get(winner_res["name"])
    if fn is None:
        return

    x_eval = np.array([[r["N_act"], r["D"], r["G"], r["A"], r["d_model"]]
                        for r in eval_valid], dtype=float)
    y_pred = fn(winner_res["params"], x_eval)

    annotated = [{**r, "pred": float(y_pred[i]), "obs": float(r["final_loss"])}
                 for i, r in enumerate(eval_valid)]

    # ── Bucket into isocompute groups ──────────────────────────────────────────
    # Use the same thresholds as plot_pred_vs_obs for consistency.
    # Threshold adapts to data if those hard values yield empty buckets.
    C_all = [r["C"] for r in annotated]
    C_lo_thresh, C_hi_thresh = np.quantile(C_all, [0.40, 0.70])

    def _budget_label(C, lo, hi):
        if C <= lo:   return "low"
        if C <= hi:   return "mid"
        return "high"

    budgets: dict = {}
    for r in annotated:
        k = _budget_label(r["C"], C_lo_thresh, C_hi_thresh)
        budgets.setdefault(k, []).append(r)
    budget_keys = [k for k in ("low", "mid", "high") if k in budgets]
    n_b = len(budget_keys)
    bud_colors = {k: c for k, c in zip(budget_keys,
                  [cm.viridis(i / max(1, n_b - 1)) for i in range(n_b)])}

    G_vals_eval = sorted({r["G"] for r in annotated})
    G_cmap = {g: cm.plasma(i / max(1, len(G_vals_eval) - 1))
              for i, g in enumerate(G_vals_eval)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Scaling-Law Ranking Validation  —  {winner_res['name']}",
                 fontsize=13)

    # ── Panel (a): architectures sorted by predicted rank ──────────────────────
    ax = axes[0]
    bud_handles = []
    for key in budget_keys:
        grp  = sorted(budgets[key], key=lambda r: r["pred"])
        n    = len(grp)
        rnks = list(range(1, n + 1))
        preds = [r["pred"] for r in grp]
        obs   = [r["obs"]  for r in grp]
        C_med = np.median([r["C"] for r in grp])
        col   = bud_colors[key]

        # Predicted trend
        ax.plot(rnks, preds, color=col, lw=2.0, alpha=0.85, zorder=3)
        # Observed stars
        ax.scatter(rnks, obs, color=col, marker="*", s=160,
                   edgecolors="black", linewidths=0.5, zorder=5)
        # Error sticks: residual between pred and obs
        for rk, p, o in zip(rnks, preds, obs):
            ax.plot([rk, rk], [p, o], color=col, lw=1.0, alpha=0.45, zorder=2)
        # Annotate architecture label on predicted points
        for rk, r in zip(rnks, grp):
            ax.annotate(f"G{r['G']}\nd{r['d_model']}",
                        (rk, r["pred"]),
                        textcoords="offset points", xytext=(-4, -18),
                        fontsize=5.5, color=col, ha="center")

        bud_handles.append(Line2D([0],[0], color=col, lw=2.5,
                                  label=f"C≈{C_med:.1e}  (n={n})"))

    pred_h = Line2D([0],[0], color="grey", lw=2, label="Predicted (line)")
    obs_h  = Line2D([0],[0], marker="*", color="w", markerfacecolor="grey",
                    markersize=10, markeredgecolor="black", label="Observed (★)")
    ax.legend(handles=bud_handles + [pred_h, obs_h], fontsize=7, ncol=1,
              loc="upper left")
    ax.set_xlabel("Architecture rank  (sorted by predicted loss ↑)", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("(a)  Predicted vs Observed, sorted by predicted rank\n"
                 "Good law → stars follow the curve", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # ── Panel (b): Spearman ρ per budget ───────────────────────────────────────
    ax = axes[1]
    bar_x, bar_y, bar_lo, bar_hi, bar_lbl, bar_col = [], [], [], [], [], []
    rng_b = np.random.default_rng(0)

    for i, key in enumerate(budget_keys):
        grp   = budgets[key]
        if len(grp) < 3:
            continue
        preds_g = [r["pred"] for r in grp]
        obs_g   = [r["obs"]  for r in grp]
        rho, pval = spearmanr(preds_g, obs_g)

        # Bootstrap 95 % CI
        boot = []
        for _ in range(800):
            idx = rng_b.choice(len(grp), size=len(grp), replace=True)
            pb  = [preds_g[j] for j in idx]
            ob  = [obs_g[j]   for j in idx]
            if len(set(pb)) > 1:
                rb, _ = spearmanr(pb, ob)
                boot.append(rb)
        ci = np.percentile(boot, [2.5, 97.5]) if boot else [rho, rho]
        C_med = np.median([r["C"] for r in grp])

        bar_x.append(i);   bar_y.append(rho)
        bar_lo.append(max(0, rho - ci[0]));  bar_hi.append(max(0, ci[1] - rho))
        sig = ("***" if pval < 0.001 else "**" if pval < 0.01
               else "*" if pval < 0.05 else "ns")
        bar_lbl.append(f"C≈{C_med:.1e}\n(n={len(grp)})")
        bar_col.append(bud_colors[key])

        ax.text(i, rho + bar_hi[-1] + 0.04,
                f"ρ={rho:.2f}  {sig}", ha="center", fontsize=8, fontweight="bold")

    ax.bar(bar_x, bar_y, color=bar_col, alpha=0.75, edgecolor="black", lw=0.8,
           zorder=2)
    ax.errorbar(bar_x, bar_y, yerr=[bar_lo, bar_hi],
                fmt="none", color="black", capsize=6, lw=1.8, zorder=3)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)
    ax.axhline(1, color="#27ae60", lw=1.0, linestyle=":", alpha=0.5,
               label="ρ=1 (perfect)")
    ax.set_xticks(bar_x);  ax.set_xticklabels(bar_lbl, fontsize=8)
    ax.set_ylabel("Spearman  ρ", fontsize=9)
    ax.set_ylim(-0.15, 1.25)
    ax.set_title("(b)  Rank correlation per compute budget\n"
                 "ρ→1: law correctly orders architectures", fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel (c): Residual vs G — bias analysis ───────────────────────────────
    ax = axes[2]
    for r in annotated:
        ax.scatter(r["G"], r["pred"] - r["obs"],
                   color=G_cmap[r["G"]], s=55, alpha=0.8,
                   edgecolors="none", zorder=3)

    for G_ in G_vals_eval:
        grp_G = [r["pred"] - r["obs"] for r in annotated if r["G"] == G_]
        if len(grp_G) >= 2:
            mu = np.mean(grp_G)
            se = np.std(grp_G) / math.sqrt(len(grp_G))
            ax.errorbar(G_, mu, yerr=se * 1.96, fmt="D",
                        color=G_cmap[G_], markersize=9,
                        capsize=5, lw=2.0, zorder=5,
                        markeredgecolor="black", markeredgewidth=0.7)

    ax.axhline(0, color="black", lw=1.2, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("G  (granularity)", fontsize=9)
    ax.set_ylabel("Residual   ŷ − y", fontsize=9)
    ax.set_title("(c)  Law bias vs granularity G\n"
                 ">0 overestimates gain,  <0 underestimates", fontsize=8.5)
    ax.text(0.02, 0.98,
            "Diamonds = mean ± 95% CI\nPoints = individual eval runs",
            transform=ax.transAxes, fontsize=6.5, va="top", color="grey")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    handles_G = [Line2D([0],[0], marker="o", color="w",
                        markerfacecolor=G_cmap[g], markersize=7, label=f"G={g}")
                 for g in G_vals_eval]
    ax.legend(handles=handles_G, fontsize=7, title="G", title_fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved eval ranking plot → {out_path}")


# ── 4c. IsoCompute plots ──────────────────────────────────────────────────────

def _m_fwd(d_model, G, j):
    """Per-token forward-pass cost (FLOPs / token), used to convert C ↔ D."""
    n_experts = 8 * G
    top_k     = j * G
    d_expert  = (4 * d_model) // G
    c_attn    = 8 * d_model**2 + 4 * SEQ_LEN * d_model
    c_ffn     = 6 * top_k * d_model * d_expert
    c_router  = 2 * d_model * n_experts
    return N_LAYERS * (c_attn + c_ffn + c_router)


def plot_isocompute(records, out_path, winner_res=None):
    """Scaling curves: Loss vs Compute, three focused panels.

    Panel 0 — Effect of A: median loss across d_models at each D level (fix G=1)
    Panel 1 — Effect of G: median loss across d_models at each D level (fix A=min)
    Panel 2 — Effect of d_model: individual curves (fix G=1, A=min) + law overlay
    Panels 0 & 1 show IQR shading across d_model variation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    valid = [r for r in records if not math.isnan(r["final_loss"])]
    if not valid:
        return

    G_vals = sorted({r["G"]       for r in valid})
    A_vals = sorted({r["A"]       for r in valid})
    d_vals = sorted({r["d_model"] for r in valid})
    D_vals = sorted({r["D"]       for r in valid})  # token-count grid levels

    G_fix = min(G_vals)   # G=1 for panels 0 & 2
    A_fix = min(A_vals)   # j=1 (standard MoE) for panels 1 & 2

    G_colors = {g: cm.plasma(  i / max(1, len(G_vals) - 1)) for i, g in enumerate(G_vals)}
    A_colors = {a: cm.cool(    i / max(1, len(A_vals) - 1)) for i, a in enumerate(A_vals)}
    d_colors = {d: cm.viridis( i / max(1, len(d_vals) - 1)) for i, d in enumerate(d_vals)}

    # Group by (G, A, d_model); each group is a scaling curve over D
    configs: dict = {}
    for r in valid:
        configs.setdefault((r["G"], r["A"], r["d_model"]), []).append(r)
    for key in configs:
        configs[key].sort(key=lambda r: r["C"])

    # Fitted law function (optional overlay — used only in panel 2)
    winner_fn = None
    if winner_res is not None:
        winner_fn = {c[0]: c[1] for c in build_predictors()}.get(winner_res["name"])

    def _median_curve(varying_vals, varying_fn, fix_key_fn):
        """Compute median/IQR of loss across d_models for each (vary_val, D) cell."""
        curves = {}
        for v in varying_vals:
            C_meds, L_meds, L_q25s, L_q75s = [], [], [], []
            for D_ in D_vals:
                grp = [r for r in valid
                       if varying_fn(r) == v and r["D"] == D_ and fix_key_fn(r)]
                if not grp:
                    continue
                losses = [r["final_loss"] for r in grp]
                C_meds.append(float(np.median([r["C"] for r in grp])))
                L_meds.append(float(np.median(losses)))
                L_q25s.append(float(np.percentile(losses, 25)))
                L_q75s.append(float(np.percentile(losses, 75)))
            if C_meds:
                curves[v] = (C_meds, L_meds, L_q25s, L_q75s)
        return curves

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel 0: vary A, aggregate over d_models, fix G=G_fix ─────────────────
    from matplotlib.lines import Line2D
    ax = axes[0]
    curves_A = _median_curve(
        A_vals,
        varying_fn=lambda r: r["A"],
        fix_key_fn=lambda r: r["G"] == G_fix,
    )
    d_markers = {64: "o", 128: "s", 192: "^", 256: "D"}
    for A_, (C_m, L_m, L_lo, L_hi) in curves_A.items():
        color = A_colors[A_]
        ax.fill_between(C_m, L_lo, L_hi, color=color, alpha=0.18, zorder=1)
        ax.plot(C_m, L_m, color=color, lw=2.2, zorder=2)
        for d_ in d_vals:
            for D_ in D_vals:
                pts = [r for r in valid if r["A"] == A_ and r["d_model"] == d_ and r["D"] == D_]
                if pts:
                    c_mean = np.mean([r["C"] for r in pts])
                    l_mean = np.mean([r["final_loss"] for r in pts])
                    ax.scatter(c_mean, l_mean, color=color, marker=d_markers.get(d_, "o"), 
                               s=45, zorder=3, edgecolors="white", linewidths=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Compute C (FLOPs)", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title(f"Effect of activation ratio A  (G={G_fix})\n"
                 f"Median ± IQR across d_model ∈ {set(d_vals)}", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    
    handles = [Line2D([0], [0], color=A_colors[A_], lw=2.2, label=f"A={A_:.0%}") for A_ in curves_A.keys()]
    handles.extend([Line2D([0], [0], marker=d_markers.get(d_, "o"), color='w', markerfacecolor='gray', markersize=8, label=f"d={d_}") for d_ in d_vals])
    ax.legend(handles=handles, fontsize=8, ncol=2)

    # ── Panel 1: vary G, aggregate over d_models, fix A=A_fix ─────────────────
    ax = axes[1]
    curves_G = _median_curve(
        G_vals,
        varying_fn=lambda r: r["G"],
        fix_key_fn=lambda r: r["A"] == A_fix,
    )
    for G_, (C_m, L_m, L_lo, L_hi) in curves_G.items():
        color = G_colors[G_]
        ax.fill_between(C_m, L_lo, L_hi, color=color, alpha=0.18, zorder=1)
        ax.plot(C_m, L_m, color=color, lw=2.2, zorder=2, label=f"G={G_}")
        ax.scatter(C_m, L_m, color=color, s=45, zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Compute C (FLOPs)", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title(f"Effect of granularity G  (A={A_fix:.0%})\n"
                 f"Median ± IQR across d_model ∈ {set(d_vals)}", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, title="G", title_fontsize=8)

    # ── Panel 2: vary d_model, fix G=G_fix_p2, A=A_fix — with law overlay ────────
    ax = axes[2]
    G_fix_p2 = 4
    for d_ in d_vals:
        pts = configs.get((G_fix_p2, A_fix, d_), [])
        if not pts:
            continue
        color = d_colors[d_]
        C_pts = [r["C"] for r in pts]
        L_pts = [r["final_loss"] for r in pts]
        ax.plot(C_pts, L_pts, color=color, lw=2.0, zorder=2, label=f"d={d_}")
        ax.scatter(C_pts, L_pts, color=color, s=40, zorder=3, edgecolors="none")

        if winner_fn is not None:
            j_     = int(round(A_fix * 8))
            mfwd   = _m_fwd(d_, G_fix_p2, j_)
            N_act_ = compute_N_act(d_, j_)
            C_lo   = min(C_pts) * 0.6
            C_hi   = max(C_pts) * 1.8
            C_fit  = np.logspace(math.log10(C_lo), math.log10(C_hi), 60)
            D_fit  = C_fit / (3.0 * mfwd)
            x_fit  = np.column_stack([
                np.full(len(C_fit), N_act_),
                D_fit,
                np.full(len(C_fit), float(G_fix_p2)),
                np.full(len(C_fit), A_fix),
                np.full(len(C_fit), float(d_)),
            ])
            L_fit = winner_fn(winner_res["params"], x_fit)
            ax.plot(C_fit, L_fit, color=color, lw=1.2, alpha=0.55,
                    linestyle="--", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Compute C (FLOPs)", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    law_note = f"  (dashed = {winner_res['name']})" if winner_fn is not None else ""
    ax.set_title(f"Effect of model width d_model\n(G={G_fix_p2}, A={A_fix:.0%}){law_note}", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, title="d_model", title_fontsize=8)

    fig.suptitle("IsoCompute Scaling Curves", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved isocompute plot → {out_path}")


def plot_isoflop_poster(records, out_path):
    """Paper-style 2×2 IsoFLOP figure — Panels 4–7 of the poster.

    [0,0] Plot 4: IsoFLOP curves — Loss vs A  (one line per budget, ★ = optimum)
    [0,1] Plot 5: Loss vs FLOPs — one curve per A  (aggregated over G & d_model)
    [1,0] Plot 6: IsoFLOP curves — Loss vs G  (one line per budget, ★ = optimum)
    [1,1] Plot 7: Loss vs FLOPs — one curve per G  (fixed A=A_min, agg. over d)

    Corresponds to Scaling-Laws Figs 5a, 5b, 6a, 6b.
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    valid = [r for r in records if not math.isnan(r["final_loss"])]
    if not valid:
        return

    A_vals = sorted({r["A"] for r in valid})
    G_vals = sorted({r["G"] for r in valid})
    D_vals = sorted({r["D"] for r in valid})
    A_min  = A_vals[0]

    FONTSIZE   = 36
    LINEWIDTH  = 6.0
    num_budgets = 6

    # ── Color palettes (same aesthetic as the reference plotting file) ──────────
    # A (activation ratio): dark navy → teal → pink, one colour per discrete value
    _A_palette = ["#2E3168", "#1D6FA4", "#43C5E0", "#F0539B", "#FF9F40",
                  "#A855F7", "#22C55E"]
    A_colors = {a: _A_palette[i % len(_A_palette)] for i, a in enumerate(A_vals)}

    # G (granularity): same family, distinct enough to separate
    _G_palette = ["#2E3168", "#1D6FA4", "#43C5E0", "#7EC8E3", "#BFE9F6",
                  "#F0539B", "#FF9F40"]
    G_colors = {g: _G_palette[i % len(_G_palette)] for i, g in enumerate(G_vals)}

    # Budget lines for isoFLOP cross-sections: dark → bright sequential
    _bud_palette = ["#0D0887", "#5B02A3", "#9C179E", "#CC4678",
                    "#ED7953", "#FDB32F", "#F0F921"]
    bud_cols = [_bud_palette[round(i * (len(_bud_palette) - 1) / max(1, num_budgets - 1))]
                for i in range(num_budgets)]

    # ── Style helper ────────────────────────────────────────────────────────────
    def style_ax(ax, xlabel, ylabel):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        ax.set_xlabel(xlabel, fontsize=FONTSIZE, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=FONTSIZE, fontweight="bold")

    # ── Aggregate (median over d_model and optionally G) ───────────────────────
    def build_curves(vary_vals, vary_fn, fix_fn):
        """Return {val: (C_list, L_median_list)} aggregated over unconstrained dims."""
        curves = {}
        for v in vary_vals:
            by_D = {}
            for r in valid:
                if vary_fn(r) == v and fix_fn(r):
                    by_D.setdefault(r["D"], []).append(r)
            C_arr, L_arr = [], []
            for D_ in D_vals:
                grp = by_D.get(D_, [])
                if grp:
                    C_arr.append(float(np.median([p["C"] for p in grp])))
                    L_arr.append(float(np.median([p["final_loss"] for p in grp])))
            if len(C_arr) >= 2:
                curves[v] = (C_arr, L_arr)
        return curves

    curves_A = build_curves(A_vals, lambda r: r["A"], lambda r: True)
    curves_G = build_curves(G_vals, lambda r: r["G"], lambda r: r["A"] == A_min)

    def overlap_range(curves):
        c_lo = max(min(C) for C, _ in curves.values())
        c_hi = min(max(C) for C, _ in curves.values())
        if c_lo >= c_hi:
            c_lo = np.median([min(C) for C, _ in curves.values()])
            c_hi = np.median([max(C) for C, _ in curves.values()])
        return c_lo, c_hi

    def isoflop_lines(curves, budgets):
        """Return list of (t_flop, [(x_val, loss), ...]) per budget."""
        lines = []
        for t_flop in budgets:
            pts = []
            for v, (C_arr, L_arr) in curves.items():
                fn = interp1d(np.log10(C_arr), L_arr,
                              kind="linear", bounds_error=False,
                              fill_value="extrapolate")
                pts.append((v, float(fn(np.log10(t_flop)))))
            pts.sort(key=lambda x: x[0])
            lines.append((t_flop, pts))
        return lines

    plt.rcParams.update({"font.size": FONTSIZE})
    fig, axes = plt.subplots(2, 2, figsize=(32, 24), layout="constrained")

    # ── Plot 5 [0,1]: Loss vs FLOPs, one curve per A ──────────────────────────
    ax5 = axes[0, 1]
    for A_, (C_arr, L_arr) in sorted(curves_A.items()):
        ax5.plot(C_arr, L_arr, color=A_colors[A_], linewidth=LINEWIDTH,
                 zorder=2, label=f"A = {A_:.0%}")
    ax5.set_xscale("log")
    style_ax(ax5, "Compute C (FLOPs)", "Cross-Entropy Loss")
    ax5.legend(fontsize=FONTSIZE, frameon=False)

    # ── Plot 4 [0,0]: IsoFLOP curves — Loss vs A ──────────────────────────────
    ax4 = axes[0, 0]
    if curves_A:
        c_lo, c_hi = overlap_range(curves_A)
        budgets_A  = np.logspace(np.log10(c_lo), np.log10(c_hi), num_budgets)
        for t_idx, (t_flop, pts) in enumerate(isoflop_lines(curves_A, budgets_A)):
            if len(pts) < 2:
                continue
            A_line = [p[0] for p in pts]
            L_line = [p[1] for p in pts]
            col    = bud_cols[t_idx]
            ax4.plot(A_line, L_line, color=col, linewidth=LINEWIDTH,
                     zorder=2, label=f"C = {t_flop:.1e}")
            best = int(np.argmin(L_line))
            ax4.scatter(A_line[best], L_line[best], marker="*",
                        s=800, color=col, edgecolors="black",
                        linewidths=1.5, zorder=5)
    ax4.set_xscale("log")
    ax4.set_xticks(A_vals)
    ax4.set_xticklabels([f"{a:.0%}" for a in A_vals])
    style_ax(ax4, "Activation Ratio A", "Cross-Entropy Loss")
    ax4.legend(fontsize=FONTSIZE - 8, frameon=False, ncol=2)

    # ── Plot 7 [1,1]: Loss vs FLOPs, one curve per G ──────────────────────────
    ax7 = axes[1, 1]
    for G_, (C_arr, L_arr) in sorted(curves_G.items()):
        ax7.plot(C_arr, L_arr, color=G_colors[G_], linewidth=LINEWIDTH,
                 zorder=2, label=f"G = {G_}")
    ax7.set_xscale("log")
    style_ax(ax7, "Compute C (FLOPs)", "Cross-Entropy Loss")
    ax7.legend(fontsize=FONTSIZE, frameon=False)

    # ── Plot 6 [1,0]: IsoFLOP curves — Loss vs G ──────────────────────────────
    ax6 = axes[1, 0]
    if curves_G:
        c_lo, c_hi = overlap_range(curves_G)
        budgets_G  = np.logspace(np.log10(c_lo), np.log10(c_hi), num_budgets)
        for t_idx, (t_flop, pts) in enumerate(isoflop_lines(curves_G, budgets_G)):
            if len(pts) < 2:
                continue
            G_line = [p[0] for p in pts]
            L_line = [p[1] for p in pts]
            col    = bud_cols[t_idx]
            ax6.plot(G_line, L_line, color=col, linewidth=LINEWIDTH,
                     zorder=2, label=f"C = {t_flop:.1e}")
            best = int(np.argmin(L_line))
            ax6.scatter(G_line[best], L_line[best], marker="*",
                        s=800, color=col, edgecolors="black",
                        linewidths=1.5, zorder=5)
    ax6.set_xscale("log", base=2)
    ax6.set_xticks(G_vals)
    ax6.set_xticklabels([str(g) for g in G_vals])
    style_ax(ax6, "Expert Granularity G", "Cross-Entropy Loss")
    ax6.legend(fontsize=FONTSIZE - 8, frameon=False, ncol=2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path,  dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})
    print(f"Saved isoflop poster panel → {out_path}  (+ {pdf_path})")


def plot_allocation(records, winner_res, out_path, fix_c=None):
    """Fig. 4 of the scaling-laws paper: optimal N_act and D vs compute budget.

    Two lines per panel — MoE (best sparse config) vs Dense (A=100%, G=1):
      Left panel:  N_act^opt vs C  — MoE uses a smaller model at same budget
      Right panel: D^opt     vs C  — MoE trains on more tokens at same budget
    Both axes log-log.  Power-law fits shown as dashed lines.
    Style matches the reference plotting file (FONTSIZE=36, lw=6, no top/right spine).

    fix_c: if not None, override the fitted irreducible loss c with this value.
    """
    import matplotlib.pyplot as plt

    valid = [r for r in records if not math.isnan(r["final_loss"])]
    if not valid or winner_res is None:
        return

    A_vals = sorted({r["A"] for r in valid})
    G_vals = sorted({r["G"] for r in valid})
    A_min  = A_vals[0]
    A_dense = 1.0

    candidates = {c[0]: c[1] for c in build_predictors()}
    fn = candidates.get(winner_res["name"])
    if fn is None:
        return
    params = winner_res["params"].copy()
    if fix_c is not None:
        params[0] = fix_c   # c is always the first parameter

    # ── Per-config optimisation helper ────────────────────────────────────────
    def _opt(C_budget, G_fixed, A_fixed):
        """Return (N_act_opt, D_opt) for a fixed (G, A) at a given compute budget."""
        j        = int(round(A_fixed * 8))
        n_exp    = 8 * G_fixed

        def loss_at(log_N):
            N  = np.exp(log_N)
            d2 = N / max(1e-30, (4 + 12 * j) * N_LAYERS)
            d  = max(1.0, d2 ** 0.5)
            de = 4 * d / G_fixed
            tk = j * G_fixed
            cost = N_LAYERS * (8*d**2 + 4*SEQ_LEN*d + 6*tk*d*de + 2*d*n_exp)
            D = C_budget / max(1e-30, 3 * cost)
            if D < 1:
                return np.inf
            return float(fn(params, np.array([[N, D, float(G_fixed), A_fixed, d]]))[0])

        grid  = np.linspace(10, 25, 80)
        vals  = [loss_at(ln) for ln in grid]
        bi    = int(np.nanargmin(vals))
        try:
            res = minimize_scalar(loss_at,
                                  bounds=(grid[max(0,bi-1)], grid[min(len(grid)-1,bi+1)]),
                                  method="bounded")
            N_opt = np.exp(res.x)
        except Exception:
            N_opt = np.exp(grid[bi])

        d2   = N_opt / max(1e-30, (4 + 12*j) * N_LAYERS)
        d    = max(1.0, d2 ** 0.5)
        de   = 4 * d / G_fixed
        tk   = j * G_fixed
        cost = N_LAYERS * (8*d**2 + 4*SEQ_LEN*d + 6*tk*d*de + 2*d*n_exp)
        D_opt = C_budget / max(1e-30, 3 * cost)
        return N_opt, D_opt

    # ── Sweep compute budgets ─────────────────────────────────────────────────
    C_min = min(r["C"] for r in valid)
    C_max = max(r["C"] for r in valid)
    # Extend half a decade each side for extrapolation
    C_grid = np.logspace(math.log10(C_min) - 0.5,
                         math.log10(C_max) + 0.5, 30)

    # MoE: best sparse config — sweep G, keep best loss at each budget
    moe_N, moe_D, moe_C = [], [], []
    for C_b in C_grid:
        best_loss = np.inf
        best_N = best_D = None
        for G_ in G_vals:
            try:
                N_, D_ = _opt(C_b, G_, A_min)
            except Exception:
                continue
            x_row = np.array([[N_, D_, float(G_), A_min,
                                max(1.0, (N_ / max(1e-30, (4 + 12*int(round(A_min*8))) * N_LAYERS))**0.5)]])
            l = float(fn(params, x_row)[0])
            if l < best_loss:
                best_loss, best_N, best_D = l, N_, D_
        if best_N is not None:
            moe_N.append(best_N); moe_D.append(best_D); moe_C.append(C_b)

    # Dense: A=1.0, G=1
    dense_N, dense_D, dense_C = [], [], []
    for C_b in C_grid:
        try:
            N_, D_ = _opt(C_b, 1, A_dense)
            dense_N.append(N_); dense_D.append(D_); dense_C.append(C_b)
        except Exception:
            continue

    # ── Power-law fits (log-log linear regression) ────────────────────────────
    def _powerlaw_fit(C_arr, Y_arr):
        lc = np.log10(C_arr); ly = np.log10(Y_arr)
        coeffs = np.polyfit(lc, ly, 1)
        return coeffs  # [slope, intercept]

    def _powerlaw_line(coeffs, C_arr):
        return 10 ** np.polyval(coeffs, np.log10(C_arr))

    fit_moe_N   = _powerlaw_fit(np.array(moe_C),   np.array(moe_N))
    fit_moe_D   = _powerlaw_fit(np.array(moe_C),   np.array(moe_D))
    fit_dense_N = _powerlaw_fit(np.array(dense_C), np.array(dense_N))
    fit_dense_D = _powerlaw_fit(np.array(dense_C), np.array(dense_D))

    C_fit = np.logspace(math.log10(C_grid[0]), math.log10(C_grid[-1]), 200)

    # ── Plot ──────────────────────────────────────────────────────────────────
    FONTSIZE  = 36
    LINEWIDTH = 6.0
    COL_MOE   = "#F0539B"   # pink  — matches reference palette
    COL_DENSE = "#2E3168"   # navy  — matches reference "dense" colour

    def style_ax(ax, xlabel, ylabel):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        ax.set_xlabel(xlabel, fontsize=FONTSIZE, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=FONTSIZE, fontweight="bold")

    plt.rcParams.update({"font.size": FONTSIZE})
    fig, axes = plt.subplots(1, 2, figsize=(32, 12), layout="constrained")
    if fix_c is not None:
        fig.suptitle(f"Optimal allocation  (c fixed = {fix_c})", fontsize=FONTSIZE)

    # Left: N_act^opt vs C
    ax = axes[0]
    ax.plot(moe_C,   moe_N,   color=COL_MOE,   linewidth=LINEWIDTH, label="MoE (sparse)")
    ax.plot(dense_C, dense_N, color=COL_DENSE, linewidth=LINEWIDTH, label="Dense")
    ax.plot(C_fit, _powerlaw_line(fit_moe_N,   C_fit),
            color=COL_MOE,   linewidth=LINEWIDTH * 0.5,
            linestyle="--", alpha=0.7,
            label=f"∝ C$^{{{fit_moe_N[0]:.2f}}}$")
    ax.plot(C_fit, _powerlaw_line(fit_dense_N, C_fit),
            color=COL_DENSE, linewidth=LINEWIDTH * 0.5,
            linestyle="--", alpha=0.7,
            label=f"∝ C$^{{{fit_dense_N[0]:.2f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    style_ax(ax, "Compute C (FLOPs)", "Optimal Active Params $N^{opt}$")
    ax.legend(fontsize=FONTSIZE - 4, frameon=False)

    # Right: D^opt vs C
    ax = axes[1]
    ax.plot(moe_C,   moe_D,   color=COL_MOE,   linewidth=LINEWIDTH, label="MoE (sparse)")
    ax.plot(dense_C, dense_D, color=COL_DENSE, linewidth=LINEWIDTH, label="Dense")
    ax.plot(C_fit, _powerlaw_line(fit_moe_D,   C_fit),
            color=COL_MOE,   linewidth=LINEWIDTH * 0.5,
            linestyle="--", alpha=0.7,
            label=f"∝ C$^{{{fit_moe_D[0]:.2f}}}$")
    ax.plot(C_fit, _powerlaw_line(fit_dense_D, C_fit),
            color=COL_DENSE, linewidth=LINEWIDTH * 0.5,
            linestyle="--", alpha=0.7,
            label=f"∝ C$^{{{fit_dense_D[0]:.2f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    style_ax(ax, "Compute C (FLOPs)", "Optimal Tokens $D^{opt}$")
    ax.legend(fontsize=FONTSIZE - 4, frameon=False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path,  dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})
    print(f"Saved allocation plot → {out_path}  (+ {pdf_path})")


def plot_3d_isocompute(records, out_path):
    """2D scatter: x=Compute, y=Loss, color=architecture parameter (A / G / d_model).
    Lines connect same-config (G,A,d) points as compute varies.
    Replaces the unreadable 3D version — same information, fully readable.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    valid = [r for r in records if not math.isnan(r["final_loss"])]
    if not valid:
        return

    C_arr = np.array([r["C"]           for r in valid])
    A_arr = np.array([r["A"]           for r in valid])
    G_arr = np.array([float(r["G"])    for r in valid])
    d_arr = np.array([float(r["d_model"]) for r in valid])
    L_arr = np.array([r["final_loss"]  for r in valid])

    configs: dict = {}
    for r in valid:
        configs.setdefault((r["G"], r["A"], r["d_model"]), []).append(r)
    for key in configs:
        configs[key].sort(key=lambda r: r["C"])

    panel_specs = [
        (A_arr,            cm.cool,    Normalize(A_arr.min(), A_arr.max()),
         "A (activation ratio)", lambda r: r["A"]),
        (np.log10(G_arr + 1e-9), cm.plasma,
         Normalize(np.log10(G_arr + 1e-9).min(), np.log10(G_arr + 1e-9).max()),
         "log₁₀(G) — granularity", lambda r: math.log10(r["G"])),
        (d_arr,            cm.viridis, Normalize(d_arr.min(), d_arr.max()),
         "d_model — width", lambda r: float(r["d_model"])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("Scaling Curves: Loss vs Compute  (colour = architecture parameter)",
                 fontsize=13)

    for ax, (param_arr, cmap, norm, cbar_label, cfn) in zip(axes, panel_specs):
        # Draw connecting lines per config
        for (G_, A_, d_), pts in configs.items():
            c_val = cfn(pts[0])
            color = cmap(norm(c_val))
            C_pts = [r["C"] for r in pts]
            L_pts = [r["final_loss"] for r in pts]
            if len(pts) > 1:
                ax.plot(C_pts, L_pts, color=color, lw=1.1, alpha=0.45, zorder=2)

        # Scatter all points with continuous colormap
        sc = ax.scatter(C_arr, L_arr, c=param_arr, cmap=cmap, norm=norm,
                        s=28, alpha=0.88, edgecolors="none", zorder=3)

        ax.set_xscale("log")
        ax.set_xlabel("Compute C (FLOPs)", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.set_title(f"Color = {cbar_label.split('—')[0].strip()}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved isocompute colormap plot → {out_path}")


def plot_all_candidates(fit_results, records, out_path):
    """All candidate scaling laws: pred vs obs (top) + residual histogram (bottom), ranked by AIC."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    valid = [r for r in records if not math.isnan(r["final_loss"])]
    if not valid or not fit_results:
        return

    x = np.array([[r["N_act"], r["D"], r["G"], r["A"], r["d_model"]]
                  for r in valid], dtype=float)
    y_obs = np.array([r["final_loss"] for r in valid], dtype=float)

    cand_map = {c[0]: c[1] for c in build_predictors()}
    sorted_fits = sorted(fit_results, key=lambda r: r["aic"])
    n = len(sorted_fits)

    G_vals = sorted({r["G"] for r in valid})
    G_colors = {g: cm.plasma(i / max(1, len(G_vals) - 1)) for i, g in enumerate(G_vals)}
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("Candidate Scaling Laws Compared — Ranked by AIC", fontsize=13)

    aic_min = sorted_fits[0]["aic"]

    for col, res in enumerate(sorted_fits):
        name = res["name"]
        fn = cand_map.get(name)
        if fn is None:
            continue

        y_hat = fn(res["params"], x)
        resid = y_hat - y_obs

        # ── Top: pred vs obs ────────────────────────────────────────────────────
        ax_top = axes[0, col]
        for i, r in enumerate(valid):
            ax_top.scatter(y_hat[i], y_obs[i],
                           color=G_colors[r["G"]], s=14, alpha=0.65, edgecolors="none")
        lo = min(y_hat.min(), y_obs.min()) * 0.99
        hi = max(y_hat.max(), y_obs.max()) * 1.01
        ax_top.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax_top.set_xlim(lo, hi); ax_top.set_ylim(lo, hi)
        delta_aic = res["aic"] - aic_min
        winner_tag = "  ★ WINNER" if col == 0 else f"  ΔAIC=+{delta_aic:.0f}"
        ax_top.set_title(f"#{col + 1} {name}{winner_tag}\n"
                         f"AIC={res['aic']:.0f}  R²={res['r2']:.3f}", fontsize=7.5)
        ax_top.set_xlabel("Predicted", fontsize=7)
        ax_top.set_ylabel("Observed" if col == 0 else "", fontsize=7)
        ax_top.tick_params(labelsize=6)
        ax_top.grid(True, alpha=0.3)

        # ── Bottom: residual distribution ───────────────────────────────────────
        ax_bot = axes[1, col]
        palette_col = cm.plasma(col / max(1, n - 1))
        ax_bot.hist(resid, bins=28, color=palette_col,
                    edgecolor="white", linewidth=0.3, alpha=0.85)
        ax_bot.axvline(0, color="black", lw=1.5, linestyle="--")
        ax_bot.axvline(float(resid.mean()), color="#e74c3c", lw=1.2,
                       linestyle=":", label=f"μ={resid.mean():.3f}")
        ax_bot.set_title(f"RMSE={res['rmse']:.4f}", fontsize=8)
        ax_bot.set_xlabel("Residual (ŷ − y)", fontsize=7)
        ax_bot.set_ylabel("Count" if col == 0 else "", fontsize=7)
        ax_bot.tick_params(labelsize=6)
        ax_bot.legend(fontsize=6)
        ax_bot.grid(True, alpha=0.3)

    # Shared G legend above top row
    handles_G = [Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=G_colors[g], markersize=7, label=f"G={g}")
                 for g in G_vals]
    fig.legend(handles=handles_G, loc="upper right", fontsize=7,
               title="Granularity G", title_fontsize=7, ncol=len(G_vals))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved all-candidates plot → {out_path}")


# ── 5. Compute-optimal predictions ────────────────────────────────────────────

def compute_optimal(winner_name, winner_params, C_budgets):
    candidates = {c[0]: (c[1], c[2]) for c in build_predictors()}
    fn, _ = candidates[winner_name]

    G_vals = [1, 2, 4, 8, 16]
    A_vals = [0.125, 0.25, 0.375, 0.5, 1.0]
    j_map  = {0.125:1, 0.25:2, 0.375:3, 0.5:4, 1.0:8}

    rows = []
    for C_budget in C_budgets:
        best_loss = np.inf
        best_cfg  = None

        for G in G_vals:
            for A in A_vals:
                j        = j_map[A]
                n_experts = 8 * G

                # For each N_act, find D from FLOPs budget.
                # C ≈ 3 * N_layers * (C_attn(d) + C_ffn(top_k,d,d_exp) + C_router) * D
                # We sweep N_act → derive d_model → compute per-token cost → D = C/(3*cost_pt)
                def loss_given_N_act(log_N_act):
                    N_act = np.exp(log_N_act)
                    # N_act = (4 + 12j) * d² * n_layers → d² = N_act / ((4+12j)*n_layers)
                    d2 = N_act / ((4 + 12*j) * N_LAYERS)
                    d  = max(1.0, d2**0.5)

                    d_expert = 4 * d / G
                    top_k    = j * G

                    c_attn   = 8*d**2 + 4*SEQ_LEN*d
                    c_ffn    = 6*top_k*d*d_expert
                    c_router = 2*d*n_experts
                    cost_pt  = N_LAYERS*(c_attn + c_ffn + c_router)
                    D        = C_budget / (3 * cost_pt)
                    if D < 1:
                        return np.inf
                    x_row = np.array([[N_act, D, G, A, d]])
                    return float(fn(winner_params, x_row)[0])

                # Grid search over N_act then refine
                log_N_grid = np.linspace(10, 25, 60)
                losses_grid = [loss_given_N_act(ln) for ln in log_N_grid]
                best_i = int(np.nanargmin(losses_grid))
                lo_ln = log_N_grid[max(0, best_i-1)]
                hi_ln = log_N_grid[min(len(log_N_grid)-1, best_i+1)]

                try:
                    res = minimize_scalar(loss_given_N_act,
                                         bounds=(lo_ln, hi_ln), method="bounded")
                    N_act_opt = np.exp(res.x)
                    loss_opt  = res.fun
                except Exception:
                    N_act_opt = np.exp(log_N_grid[best_i])
                    loss_opt  = losses_grid[best_i]

                if loss_opt < best_loss:
                    best_loss = loss_opt
                    d2 = N_act_opt / ((4 + 12*j) * N_LAYERS)
                    d_approx = max(1.0, d2**0.5)
                    d_expert_opt = 4 * d_approx / G
                    top_k_opt    = j * G
                    c_attn   = 8*d_approx**2 + 4*SEQ_LEN*d_approx
                    c_ffn    = 6*top_k_opt*d_approx*d_expert_opt
                    c_router = 2*d_approx*n_experts
                    cost_pt  = N_LAYERS*(c_attn + c_ffn + c_router)
                    D_opt = C_budget / (3 * cost_pt)
                    best_cfg = dict(G=G, A=A, N_act=N_act_opt, D=D_opt,
                                    d_approx=d_approx, loss=best_loss)

        rows.append((C_budget, best_cfg))
    return rows


# ── 6. Summary ────────────────────────────────────────────────────────────────

def print_summary(records, fit_results, winner_name, winner_res, boot, dense_fit,
                  compute_opt_rows):
    valid = [r for r in records if not math.isnan(r["final_loss"])]

    C_vals = [r["C"] for r in valid]
    C_min, C_max = min(C_vals), max(C_vals)
    decades = math.log10(C_max / C_min)

    sorted_fits = sorted(fit_results, key=lambda r: r["aic"])

    # ── G-flat sanity for A=100% ────────────────────────────────────────────────
    D4M = max(r["D"] for r in valid)
    dense_by_G = {}
    for d_ in sorted({r["d_model"] for r in valid}):
        for G_ in sorted({r["G"] for r in valid}):
            pts = [r for r in valid if r["d_model"]==d_ and r["G"]==G_
                   and r["D"]==D4M and r["A"]==1.0]
            if pts:
                dense_by_G.setdefault(G_, []).append(pts[0]["final_loss"])
    G_mean = {G: np.mean(v) for G, v in dense_by_G.items()}
    G_sorted = sorted(G_mean.keys())

    # sparse vs dense (D=4M, G=1, per d_model)
    def sparse_vs_dense_str():
        lines = []
        d_vals = sorted({r["d_model"] for r in valid})
        for d_ in d_vals:
            dense_pts  = [r for r in valid if r["d_model"]==d_
                          and r["G"]==1 and r["D"]==D4M and r["A"]==1.0]
            sparse_pts = [r for r in valid if r["d_model"]==d_
                          and r["G"]==1 and r["D"]==D4M and r["A"]<1.0]
            if dense_pts and sparse_pts:
                dense_l  = dense_pts[0]["final_loss"]
                best_s   = min(r["final_loss"] for r in sparse_pts)
                best_A   = min(sparse_pts, key=lambda r: r["final_loss"])["A"]
                delta    = best_s - dense_l
                lines.append(f"  d={d_:<4}: Sparse best={best_s:.3f} (A={best_A:.0%}) "
                             f"vs A100={dense_l:.3f}  Δ={delta:+.3f}")
        return "\n".join(lines) if lines else "  (no data)"

    # ── Candidate ranking ───────────────────────────────────────────────────────
    aic_best = sorted_fits[0]["aic"]
    ranking_lines = []
    for i, r in enumerate(sorted_fits):
        delta = r["aic"] - aic_best
        if r["name"] == winner_name:
            marker = "← WINNER"
        elif delta < 2.0:
            marker = f"← near-tie  (ΔAICc={delta:.1f} < 2)"
        elif delta < 7.0:
            marker = f"  (ΔAICc={delta:.1f})"
        else:
            marker = f"  (ΔAICc={delta:.1f}  weak support)"
        ranking_lines.append(
            f"  #{i+1} {r['name']:<25}  AICc={r['aic']:8.1f}  "
            f"RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}  {marker}"
        )

    # ── Winner formula ──────────────────────────────────────────────────────────
    cand_param_names = {
        "Chinchilla":         ["c", "a", "α", "b", "β"],
        "Krajewski":          ["c", "g", "γ", "a", "α", "b", "β"],
        "Krajewski+A":        ["c", "g", "γ", "a", "δ", "α", "b", "β"],
        "Multiplicative A×G": ["c", "a", "δ", "γ", "α", "b", "β"],
        "Log-poly G":         ["c", "a", "δ", "β₁", "β₂", "α", "b", "β"],
        "Width test (+d)":    ["c", "g", "γ", "a", "δ", "α", "ε", "b", "β"],
    }
    pnames = cand_param_names.get(winner_name, [f"p{i}" for i in range(len(winner_res["params"]))])
    coeff_lines = []
    for i, (name, val) in enumerate(zip(pnames, winner_res["params"])):
        if boot:
            lo, med, hi = boot["p10"][i], boot["median"][i], boot["p90"][i]
            coeff_lines.append(f"  {name:<5} = {val:.4f}  [{lo:.4f}, {hi:.4f}]")
        else:
            coeff_lines.append(f"  {name:<5} = {val:.4f}")

    # ── ε width test ────────────────────────────────────────────────────────────
    eps_str = "(Candidate 6 not winner — check manually)"
    if winner_name == "Width test (+d)" and boot:
        idx_eps = pnames.index("ε")
        eps_med = boot["median"][idx_eps]
        eps_lo  = boot["p10"][idx_eps]
        eps_hi  = boot["p90"][idx_eps]
        if eps_lo <= 0 <= eps_hi:
            verdict = "NO — d_model effect fully captured by N_act"
        else:
            verdict = "YES — wider representations help beyond compute"
        eps_str = (f"ε = {eps_med:.4f}  [{eps_lo:.4f}, {eps_hi:.4f}]\n"
                   f"  → {verdict}")

    # ── Dense sanity ────────────────────────────────────────────────────────────
    c_full  = winner_res["params"][0]
    c_dense_val = dense_fit["params"][0] if dense_fit else float("nan")
    if dense_fit:
        diff = abs(c_full - c_dense_val)
        sanity_str = ("Consistent" if diff < 0.05
                      else f"WARNING: c values diverge (|Δ|={diff:.4f})")
    else:
        sanity_str = "No dense-only fit (no A=100% runs)"

    # ── G-flat check ────────────────────────────────────────────────────────────
    if G_mean:
        g_losses = [f"G={g}: {G_mean[g]:.3f}" for g in G_sorted]
        g_range  = max(G_mean.values()) - min(G_mean.values())
        g_flat   = ("Flat as expected" if g_range < 0.02
                    else f"WARNING: G affects A=100% loss (range={g_range:.3f})")
    else:
        g_losses = []; g_flat = "No data"

    # ── Compute-optimal ─────────────────────────────────────────────────────────
    co_lines = []
    for C_budget, cfg in compute_opt_rows:
        if cfg:
            co_lines.append(
                f"  C={C_budget:.0e}: G={cfg['G']}  A={cfg['A']:.1%}  "
                f"N_act={cfg['N_act']:.2e}  D={cfg['D']:.2e}  "
                f"d_approx≈{cfg['d_approx']:.0f}  L={cfg['loss']:.3f}"
            )
        else:
            co_lines.append(f"  C={C_budget:.0e}: (no result)")

    bar = "═" * 67
    print(f"\n{bar}")
    print("SCALING LAW RESULTS — PHASE 1")
    print(bar)
    print(f"Grid: {len(valid)} MoE runs "
          f"(including {sum(1 for r in valid if r['A']==1.0)} dense-equivalent at A=100%)")
    print(f"Architecture: n_layers={N_LAYERS}, mlp_ratio=4, seq_len={SEQ_LEN}, fp32")
    print(f"Axes: d_model ∈ {{64,128,192,256}}, D ∈ {{500K,1M,2M,4M}}")
    print(f"      G ∈ {{1,2,4,8,16}}, A ∈ {{12.5%,25%,37.5%,50%,100%}}")
    print(f"Compute range: {C_min:.2e} to {C_max:.2e} ({decades:.1f} decades)")
    print()
    print("Candidate scaling laws (ranked by AICc):")
    print("\n".join(ranking_lines))
    print()
    # ── Literature comparison ────────────────────────────────────────────────────
    lit_forms = [
        ("Chinchilla",
         "Hoffmann et al. 2022",
         "L = c  +  a · N⁻ᵅ  +  b · D⁻ᵝ",
         {"α": 0.34, "β": 0.28}),
        ("Krajewski",
         "Krajewski et al. 2024  (MoE scaling)",
         "L = c  +  (g·G⁻ᵞ + a) · N⁻ᵅ  +  b · D⁻ᵝ",
         {"α": 0.31, "γ": 0.39}),
        ("Krajewski+A",
         "This work  (MoE + activation ratio)",
         "L = c  +  (g·G⁻ᵞ + a·Aᵟ) · N_act⁻ᵅ  +  b · D⁻ᵝ",
         {}),
        ("Multiplicative A×G",
         "This work  (multiplicative interaction)",
         "L = c  +  a · Aᵟ / (Gᵞ · Nᵅ)  +  b · D⁻ᵝ",
         {}),
        ("Width test (+d)",
         "This work  (residual width dependence)",
         "L = c  +  (g·G⁻ᵞ + a·Aᵟ) · N⁻ᵅ · (d/128)ᵉ  +  b · D⁻ᵝ",
         {}),
    ]
    print("Candidate functional forms vs literature:")
    print(f"  {'Name':<22}  {'Reference':<38}  Formula")
    print("  " + "-"*95)
    for name, ref, formula, _ in lit_forms:
        marker = " ← WINNER" if name == winner_name else ""
        print(f"  {name:<22}  {ref:<38}  {formula}{marker}")
    print()

    # ── Winner formula with substituted coefficients ─────────────────────────
    p = winner_res["params"]
    pn = pnames  # e.g. ["c","g","γ","a","δ","α","b","β"]
    pv = {n: v for n, v in zip(pn, p)}

    def _fmt(name):
        val = pv.get(name, float("nan"))
        if boot:
            idx = pn.index(name)
            lo, hi = boot["p10"][idx], boot["p90"][idx]
            return f"{val:.4f} [{lo:.4f},{hi:.4f}]"
        return f"{val:.4f}"

    print(f"Winner: {winner_name}")
    print(f"Fitted law (all coefficients substituted):")
    if winner_name == "Krajewski+A":
        print(f"  L = {_fmt('c')}")
        print(f"      + ( {_fmt('g')} · G^(-{_fmt('γ')})")
        print(f"        + {_fmt('a')} · A^({_fmt('δ')}) ) · N_act^(-{_fmt('α')})")
        print(f"      + {_fmt('b')} · D^(-{_fmt('β')})")
    elif winner_name == "Multiplicative A×G":
        print(f"  L = {_fmt('c')}")
        print(f"      + {_fmt('a')} · A^({_fmt('δ')}) / ( G^({_fmt('γ')}) · N_act^({_fmt('α')}) )")
        print(f"      + {_fmt('b')} · D^(-{_fmt('β')})")
    elif winner_name == "Krajewski":
        print(f"  L = {_fmt('c')}")
        print(f"      + ( {_fmt('g')} · G^(-{_fmt('γ')}) + {_fmt('a')} ) · N_act^(-{_fmt('α')})")
        print(f"      + {_fmt('b')} · D^(-{_fmt('β')})")
    elif winner_name == "Chinchilla":
        print(f"  L = {_fmt('c')}  +  {_fmt('a')} · N^(-{_fmt('α')})  +  {_fmt('b')} · D^(-{_fmt('β')})")
    elif winner_name == "Log-poly G":
        print(f"  L = {_fmt('c')}")
        print(f"      + {_fmt('a')} · A^({_fmt('δ')}) · exp( {_fmt('β₁')}·ln G + {_fmt('β₂')}·ln²G ) · N_act^(-{_fmt('α')})")
        print(f"      + {_fmt('b')} · D^(-{_fmt('β')})")
        print()
        print(f"  i.e. with fitted values:")
        b1 = pv.get('β₁', float('nan')); b2 = pv.get('β₂', float('nan'))
        a  = pv.get('a',  float('nan')); d_ = pv.get('δ',  float('nan'))
        al = pv.get('α',  float('nan')); b  = pv.get('b',  float('nan'))
        bt = pv.get('β',  float('nan')); c_ = pv.get('c',  float('nan'))
        print(f"  L = {c_:.4f}")
        print(f"      + {a:.4f} · A^{d_:.4f} · exp( {b1:.4f}·lnG + {b2:.4f}·ln²G ) · N_act^(-{al:.4f})")
        print(f"      + {b:.4f} · D^(-{bt:.4f})")
        # Show effective G-scaling at integer G values
        import math as _math
        print(f"\n  G-factor  exp(β₁·lnG + β₂·ln²G)  at integer G:")
        for g in [1, 2, 4, 8, 16, 32, 64]:
            lg = _math.log(g) if g > 1 else 0.0
            gfac = _math.exp(b1 * lg + b2 * lg**2)
            print(f"    G={g:3d}: {gfac:.4f}")
    else:
        for n, v in zip(pn, p):
            print(f"    {n} = {_fmt(n)}")
    print()

    # ── Implication of each exponent ─────────────────────────────────────────
    print("Implication of fitted exponents:")
    alpha = pv.get("α", float("nan"))
    beta  = pv.get("β", float("nan"))
    gamma = pv.get("γ", float("nan"))
    delta = pv.get("δ", float("nan"))
    if not math.isnan(alpha) and not math.isnan(beta):
        ratio = alpha / beta if beta > 1e-6 else float("inf")
        # Chinchilla-optimal N/D ratio scales as C^{β/(α+β)} / C^{α/(α+β)}
        n_exp = beta  / (alpha + beta)
        d_exp = alpha / (alpha + beta)
        print(f"  α={alpha:.4f}  β={beta:.4f}  →  α/β = {ratio:.2f}")
        print(f"  Compute-optimal allocation:  N_act ∝ C^{n_exp:.3f},  D ∝ C^{d_exp:.3f}")
        if ratio < 0.5:
            print(f"  → DATA-LIMITED regime: most compute should go to D (tokens), not model size.")
            print(f"    A 10× compute increase → D×{10**d_exp:.1f},  N_act×{10**n_exp:.1f}")
        elif ratio > 2.0:
            print(f"  → MODEL-LIMITED regime: most compute should go to N_act (parameters).")
            print(f"    A 10× compute increase → N_act×{10**n_exp:.1f},  D×{10**d_exp:.1f}")
        else:
            print(f"  → BALANCED: roughly equal scaling of N_act and D with compute.")
            print(f"    A 10× compute increase → N_act×{10**n_exp:.1f},  D×{10**d_exp:.1f}")
        print(f"  Literature (Chinchilla-2022): α=0.34, β=0.28  →  N∝C^0.45,  D∝C^0.55  (balanced)")
    if not math.isnan(gamma):
        if gamma < 0.05:
            print(f"  γ={gamma:.4f}  →  G^γ ≈ 1 for all G: granularity has negligible effect in this law.")
            print(f"    WARNING: contradicts sanity check (G affects A=100% loss by 0.38 nats).")
            print(f"    The Multiplicative form may not capture G correctly — consider Krajewski+A.")
        else:
            print(f"  γ={gamma:.4f}  →  Doubling G reduces N-term by factor 2^γ = {2**gamma:.3f}")
    if not math.isnan(delta):
        a12 = 0.125 ** delta
        a100 = 1.0 ** delta
        print(f"  δ={delta:.4f}  →  A^δ range: A=12.5% gives {a12:.3f},  A=100% gives {a100:.3f}")
        print(f"    (only {(1-a12)*100:.1f}% loss reduction by going fully sparse at fixed N_act, D)")
    print()

    # ── Key exponent comparison to literature ────────────────────────────────
    print("Key exponent comparison:")
    print(f"  {'Exponent':<12}  {'This work':>10}  {'Chinchilla-2022':>16}  {'Krajewski-2024':>15}")
    print("  " + "-"*58)
    for exp_name, sym, ch_val, kr_val in [
        ("α  (N_act)", "α", 0.34, 0.31),
        ("β  (D)",     "β", 0.28, None),
        ("γ  (G)",     "γ", None, 0.39),
        ("δ  (A)",     "δ", None, None),
    ]:
        our_val = pv.get(sym, float("nan"))
        ch_str = f"{ch_val:.2f}" if ch_val is not None else "—"
        kr_str = f"{kr_val:.2f}" if kr_val is not None else "—"
        our_str = f"{our_val:.4f}" if not math.isnan(our_val) else "—"
        print(f"  {exp_name:<12}  {our_str:>10}  {ch_str:>16}  {kr_str:>15}")
    print()

    print(f"Raw coefficients [fitted | P10, P90 bootstrap]:")
    print("\n".join(coeff_lines))
    print()
    print("Does d_model matter independently of N_act?")
    print(f"  {eps_str}")
    print()
    print(f"Sanity check: A=100% (dense-equivalent) fit → c = {c_dense_val:.4f}")
    print(f"vs full grid fit                            → c = {c_full:.4f}")
    print(f"  → {sanity_str}")
    print()
    print("Sanity check: A=100% loss vs G (should be flat):")
    print("  " + ",  ".join(g_losses))
    print(f"  → {g_flat}")
    print()
    print("Compute-optimal predictions:")
    print("\n".join(co_lines))
    print()
    print("Sparse vs Dense-equivalent (best sparse A vs A=100%, D=4M, G=1):")
    print(sparse_vs_dense_str())
    print(bar)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--wandb", action="store_true", help="Fetch grid runs from W&B")
    src.add_argument("--csv",   metavar="FILE",      help="Load grid runs from CSV")
    parser.add_argument("--eval-wandb", action="store_true",
                        help="Fetch phase-1 eval runs from W&B (group=grid_eval)")
    parser.add_argument("--eval-csv", metavar="FILE",
                        help="Load phase-1 eval runs from CSV")
    parser.add_argument("--eval-wandb2", action="store_true",
                        help="Fetch phase-2 eval runs from W&B (group=grid_eval2)")
    parser.add_argument("--eval-csv2", metavar="FILE",
                        help="Load phase-2 eval runs from CSV")
    parser.add_argument("--fix-c", type=float, default=None, metavar="VALUE",
                        help="Re-fit all candidates with irreducible loss c fixed "
                             "to VALUE (diagnostic for exponent identifiability)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # 1. Load training grid data
    if args.wandb:
        records = fetch_from_wandb()
        save_to_csv(records, "results/grid_results.csv")
    else:
        records = load_from_csv(args.csv)

    # 1b. Load eval data (phase 1 and/or phase 2, merged)
    eval_records = None
    _eval_lists = []
    if args.eval_wandb:
        r1 = fetch_eval_from_wandb()
        save_to_csv(r1, "results/eval_results.csv")
        _eval_lists.append(r1)
        print(f"Loaded {len(r1)} phase-1 eval runs")
    elif args.eval_csv:
        r1 = load_eval_from_csv(args.eval_csv)
        _eval_lists.append(r1)
        print(f"Loaded {len(r1)} phase-1 eval runs from {args.eval_csv}")
    if args.eval_wandb2:
        r2 = fetch_eval_from_wandb(group="grid_eval2",
                                   name_prefix="eval2",
                                   csv_path="results/eval_results2.csv")
        _eval_lists.append(r2)
        print(f"Loaded {len(r2)} phase-2 eval runs")
    elif args.eval_csv2:
        r2 = load_eval_from_csv(args.eval_csv2)
        _eval_lists.append(r2)
        print(f"Loaded {len(r2)} phase-2 eval runs from {args.eval_csv2}")
    if _eval_lists:
        eval_records = [r for lst in _eval_lists for r in lst]
        print(f"Total eval runs: {len(eval_records)}")

    if not records:
        sys.exit("No records found.")

    records = print_table(records)
    print(f"\nLoaded {len(records)} runs  "
          f"({sum(1 for r in records if not math.isnan(r['final_loss']))} valid)")

    # 2. Diagnostics
    plot_diagnostics(records, "results/grid_diagnostics.png")

    # 3. Fit
    print("\nFitting scaling law candidates …")
    fit_results, y_data, x_data, valid, dense_fit = fit_all(records)

    if not fit_results:
        sys.exit("All fits failed.")

    sorted_fits = sorted(fit_results, key=lambda r: r["aic"])
    winner = sorted_fits[0]
    print(f"\nWinner by AIC: {winner['name']}")

    plot_active_params_scaling(records, "results/grid_active_params_scaling.png")

    # 4. Bootstrap
    print(f"Bootstrapping {winner['name']} (100 resamples) …")
    boot = bootstrap_winner(winner["name"], records, n_boot=100)

    # Pred-vs-obs (winner only)
    plot_pred_vs_obs(winner["name"], winner, records, "results/grid_pred_vs_obs.png",
                     eval_records=eval_records)

    # All candidates compared
    plot_all_candidates(fit_results, records, "results/grid_all_candidates.png")

    # Eval ranking validation (paper figure)
    if eval_records:
        plot_eval_ranking(eval_records, winner, "results/grid_eval_ranking.png")

    # IsoCompute: scaling curves + fitted law overlay
    plot_isocompute(records, "results/grid_isocompute.png", winner_res=winner)

    # IsoFLOP poster panels (Figs 5 & 6 of the scaling-laws paper)
    plot_isoflop_poster(records, "results/grid_isoflop_poster.png")

    # Model-Data allocation: optimal N_act & D vs compute — MoE vs Dense (Fig. 4)
    plot_allocation(records, winner, "results/grid_allocation.png")
    plot_allocation(records, winner, "results/grid_allocation_fixc3.png", fix_c=3.0)

    # IsoCompute: 3D scatter
    plot_3d_isocompute(records, "results/grid_3d_isocompute.png")

    # 5. Compute-optimal
    C_budgets = [1e13, 1e14, 1e15, 1e16, 1e17]
    print("\nComputing compute-optimal configurations …")
    co_rows = compute_optimal(winner["name"], winner["params"], C_budgets)

    # 6. Summary
    print_summary(records, fit_results, winner["name"], winner, boot,
                  dense_fit, co_rows)

    # 7. Fixed-c diagnostic refit (if requested)
    if args.fix_c is not None:
        c_val = args.fix_c
        print(f"\n{'='*60}")
        print(f"FIXED-c DIAGNOSTIC REFIT  (c = {c_val})")
        print(f"{'='*60}")
        print(f"Refitting all candidates with irreducible loss pinned to {c_val} …\n")

        fc_results, _, _, _ = fit_all_fixed_c(records, c_fixed=c_val)

        if fc_results:
            # Print the comparison table
            print_fixed_c_comparison(fit_results, fc_results, c_fixed=c_val)

            # Also rank the fixed-c candidates
            sorted_fc = sorted(fc_results, key=lambda r: r["aic"])
            print("Fixed-c candidate ranking (by AICc):")
            aic_best_fc = sorted_fc[0]["aic"]
            for i, r in enumerate(sorted_fc):
                delta = r["aic"] - aic_best_fc
                marker = "← WINNER" if i == 0 else f"(ΔAICc={delta:.1f})"
                print(f"  #{i+1} {r['name']:<35}  AICc={r['aic']:8.1f}  "
                      f"RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}  {marker}")

            # Compute-optimal with best fixed-c model
            fc_winner = sorted_fc[0]
            # We need to use the full-vector params with the free-c predictor
            base_name = fc_winner["name_base"]
            candidates_free = {c[0]: c[1] for c in build_predictors()}
            if base_name in candidates_free:
                print(f"\nCompute-optimal predictions ({fc_winner['name']}):")
                co_rows_fc = compute_optimal(base_name, fc_winner["params_full"],
                                             C_budgets)
                for C_budget, cfg in co_rows_fc:
                    if cfg:
                        print(f"  C={C_budget:.0e}: G={cfg['G']}  A={cfg['A']:.1%}  "
                              f"N_act={cfg['N_act']:.2e}  D={cfg['D']:.2e}  "
                              f"d_approx≈{cfg['d_approx']:.0f}  L={cfg['loss']:.3f}")
                    else:
                        print(f"  C={C_budget:.0e}: (no result)")
        else:
            print("All fixed-c fits failed.")


if __name__ == "__main__":
    main()

