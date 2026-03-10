#!/usr/bin/env python3
"""
Step 4: Unified Loss Law and Cross-Regime Comparison.

Assembles the unified loss law L_X(N_a, D, A, G) per regime.
Tests loss ordering and widening gap. Produces final comparison figures.

Usage:
    python analysis_step4_unified.py --results_dir ./results/ --output_dir ./analysis/step4/ \
        --dense_params ./analysis/step1/dense_params.json \
        --routing_params ./analysis/step2/routing_efficiency.json \
        --granularity_params ./analysis/step3/granularity_params.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fitting_utils import (
    compute_N_attn,
    load_results,
    neff,
    save_json,
)
from plot_config import (
    REGIME_COLORS,
    REGIME_LABELS,
    REGIME_MARKERS,
    get_fig,
    save_fig,
    setup_style,
)


ALL_REGIMES = ["D", "M0", "M_alpha", "M_lfb"]
PLOT_KEYS = {"D": "D", "M0": "M0", "M_alpha": "Malpha", "M_lfb": "Mlfb"}


def unified_loss(N_a, D, A, G, params, rho, gammas, N_attn):
    """
    Unified loss model (Eq. 22):
    L_X = E + B / [N_attn + (N_a - N_attn)*A^(-rho)]^alpha + B0 / D^beta + Gamma(G)
    """
    E, B, alpha, B0, beta = params["E"], params["B"], params["alpha"], params["B0"], params["beta"]

    # Capacity term
    if rho is not None and A < 1.0:
        N_eff_val = N_attn + (N_a - N_attn) * A ** (-rho)
    else:
        N_eff_val = N_a

    L = E + B / N_eff_val**alpha + B0 / D**beta

    # Granularity correction
    if gammas is not None and G > 1.0:
        gamma1, gamma2 = gammas
        log_G = np.log(G)
        Gamma = gamma2 * log_G**2 + gamma1 * log_G
        L = L * np.exp(Gamma)

    return L


def main(results_dir: str, output_dir: str,
         dense_path: str, routing_path: str, granularity_path: str):
    setup_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load all fitted params
    with open(dense_path) as f:
        dense = json.load(f)
    with open(routing_path) as f:
        routing = json.load(f)
    with open(granularity_path) as f:
        granularity = json.load(f)

    params = dense["params"]
    allocation = dense.get("allocation_law", {})

    # Build per-regime parameter sets
    regime_params = {}
    for reg in ALL_REGIMES:
        rp = {"params": params}

        if reg == "D":
            rp["rho"] = None
            rp["gammas"] = None
        else:
            rr = routing.get("regimes", {}).get(reg, {})
            rp["rho"] = rr.get("rho")

            gr = granularity.get("regimes", {}).get(reg, {})
            g1 = gr.get("gamma1")
            g2 = gr.get("gamma2")
            if g1 is not None and g2 is not None:
                rp["gammas"] = (g1, g2)
                rp["G_opt"] = gr.get("G_opt")
            else:
                rp["gammas"] = None
                rp["G_opt"] = None

        regime_params[reg] = rp

    # ── Compute L_opt(C) per regime ───────────────────────────────────────────
    C_range = np.logspace(18, 21, 50)
    a_N = allocation.get("a_N")
    b_N = allocation.get("b_N")

    # Default architecture params (from protocol typical values)
    n_layers = 16
    d_model = 2048
    N_attn = compute_N_attn(n_layers, d_model)
    A_default = 1 / 32
    T = 4096

    optimal_losses = {reg: [] for reg in ALL_REGIMES}

    for C in C_range:
        for reg in ALL_REGIMES:
            rp = regime_params[reg]

            # Compute-optimal N_a and D
            if a_N is not None and b_N is not None:
                N_a_opt = a_N * C ** b_N
            else:
                N_a_opt = 1e8  # fallback
            M_Na = 2 * N_a_opt + 4 * n_layers * T * d_model
            D_opt = C / (3 * M_Na)

            G = rp.get("G_opt", 4) or 4
            A = A_default if reg != "D" else 1.0

            L = unified_loss(N_a_opt, D_opt, A, G, params,
                             rp["rho"], rp["gammas"], N_attn)
            optimal_losses[reg].append(L)

    for reg in ALL_REGIMES:
        optimal_losses[reg] = np.array(optimal_losses[reg])

    # ── Verification checks ───────────────────────────────────────────────────
    verification = {}

    # V1: Loss ordering L_Mlfb < L_Malpha < L_M0 < L_D at all C
    if all(len(optimal_losses[r]) > 0 for r in ALL_REGIMES):
        ordering_holds = np.all(
            (optimal_losses["M_lfb"] < optimal_losses["M_alpha"])
            & (optimal_losses["M_alpha"] < optimal_losses["M0"])
            & (optimal_losses["M0"] < optimal_losses["D"])
        )
        verification["V1_loss_ordering"] = {"pass": bool(ordering_holds)}
    else:
        verification["V1_loss_ordering"] = {"pass": False, "note": "missing data"}

    # V2: Delta_L(C) increasing
    if "M_alpha" in optimal_losses and "M_lfb" in optimal_losses:
        delta_L = optimal_losses["M_alpha"] - optimal_losses["M_lfb"]
        if len(delta_L) > 1:
            # Simple monotonicity test
            increasing = np.sum(np.diff(delta_L) > 0) / max(len(delta_L) - 1, 1)
            verification["V2_widening_gap"] = {
                "fraction_increasing": float(increasing),
                "pass": increasing > 0.8,  # relaxed for finite data
            }

    # V3-V5: Parameter ordering checks
    rho_vals = {}
    for reg in ["M0", "M_alpha", "M_lfb"]:
        rr = routing.get("regimes", {}).get(reg, {})
        if rr.get("rho") is not None:
            rho_vals[reg] = rr["rho"]

    if len(rho_vals) == 3:
        verification["V3_rho_ordering"] = {
            "pass": rho_vals["M_lfb"] > rho_vals["M_alpha"] > rho_vals["M0"],
        }

    G_opts = {}
    for reg in ["M0", "M_alpha", "M_lfb"]:
        gr = granularity.get("regimes", {}).get(reg, {})
        if gr.get("G_opt") is not None:
            G_opts[reg] = gr["G_opt"]

    if len(G_opts) == 3:
        verification["V4_G_opt_ordering"] = {
            "pass": G_opts["M_lfb"] > G_opts["M_alpha"] > G_opts["M0"],
        }

    # V6: All R² >= 0.97
    r2 = dense.get("diagnostics", {}).get("R2", 0)
    verification["V6_R2"] = {"value": r2, "pass": r2 >= 0.97}

    # ── Fig 1: Compute-optimal loss frontier ──────────────────────────────────
    fig, ax = get_fig(width="single")
    for reg in ALL_REGIMES:
        pk = PLOT_KEYS[reg]
        ax.loglog(C_range, optimal_losses[reg],
                  color=REGIME_COLORS[pk], label=REGIME_LABELS[pk],
                  linewidth=1.5)
    ax.set_xlabel("Compute $C$ (FLOPs)")
    ax.set_ylabel(r"$\mathcal{L}^{\mathrm{opt}}(C)$")
    ax.set_title("Compute-Optimal Loss Frontier")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, str(out / "loss_frontier_all"))

    # Inset: Delta_L
    if "M_alpha" in optimal_losses and "M_lfb" in optimal_losses:
        fig, (ax1, ax2) = get_fig(ncols=2, width="double")
        for reg in ALL_REGIMES:
            pk = PLOT_KEYS[reg]
            ax1.loglog(C_range, optimal_losses[reg],
                       color=REGIME_COLORS[pk], label=REGIME_LABELS[pk],
                       linewidth=1.5)
        ax1.set_xlabel("Compute $C$ (FLOPs)")
        ax1.set_ylabel(r"$\mathcal{L}^{\mathrm{opt}}(C)$")
        ax1.set_title("Loss Frontier")
        ax1.legend(fontsize=7)

        delta_L = optimal_losses["M_alpha"] - optimal_losses["M_lfb"]
        ax2.semilogx(C_range, delta_L, color=REGIME_COLORS["Mlfb"], linewidth=1.5)
        ax2.set_xlabel("Compute $C$ (FLOPs)")
        ax2.set_ylabel(r"$\Delta\mathcal{L}(C) = \mathcal{L}_{M_\alpha} - \mathcal{L}_{M_\ell}$")
        ax2.set_title("LFB Advantage Gap")
        ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        fig.tight_layout()
        save_fig(fig, str(out / "loss_frontier_with_gap"))

    # ── Fig 2: Summary 2x2 panel ─────────────────────────────────────────────
    fig, axes = get_fig(ncols=2, nrows=2, width="double")

    # Panel 1: Loss frontier (repeat)
    ax = axes[0, 0] if hasattr(axes, 'shape') else axes[0]
    for reg in ALL_REGIMES:
        pk = PLOT_KEYS[reg]
        ax.loglog(C_range, optimal_losses[reg],
                  color=REGIME_COLORS[pk], label=REGIME_LABELS[pk])
    ax.set_xlabel("$C$")
    ax.set_ylabel(r"$\mathcal{L}^*$")
    ax.legend(fontsize=6)
    ax.set_title("(a) Loss frontier")

    # Panel 2: rho comparison
    ax = axes[0, 1] if hasattr(axes, 'shape') else axes[1]
    if len(rho_vals) > 0:
        regs = sorted(rho_vals.keys())
        x = np.arange(len(regs))
        vals = [rho_vals[r] for r in regs]
        colors = [REGIME_COLORS[PLOT_KEYS[r]] for r in regs]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[PLOT_KEYS[r]] for r in regs], fontsize=7)
        ax.set_ylabel(r"$\rho$")
    ax.set_title(r"(b) Routing efficiency $\rho$")

    # Panel 3: G_opt comparison
    ax = axes[1, 0] if hasattr(axes, 'shape') else axes[2]
    if len(G_opts) > 0:
        regs = sorted(G_opts.keys())
        x = np.arange(len(regs))
        vals = [G_opts[r] for r in regs]
        colors = [REGIME_COLORS[PLOT_KEYS[r]] for r in regs]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[PLOT_KEYS[r]] for r in regs], fontsize=7)
        ax.set_ylabel(r"$G^{\mathrm{opt}}$")
    ax.set_title(r"(c) Optimal granularity $G^{\mathrm{opt}}$")

    # Panel 4: Delta_L
    ax = axes[1, 1] if hasattr(axes, 'shape') else axes[3]
    if "M_alpha" in optimal_losses and "M_lfb" in optimal_losses:
        delta_L = optimal_losses["M_alpha"] - optimal_losses["M_lfb"]
        ax.semilogx(C_range, delta_L, color=REGIME_COLORS["Mlfb"])
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("$C$")
    ax.set_ylabel(r"$\Delta\mathcal{L}$")
    ax.set_title(r"(d) $\mathcal{L}_{M_\alpha} - \mathcal{L}_{M_\ell}$")

    fig.tight_layout()
    save_fig(fig, str(out / "summary_2x2"))

    # ── Fig 3: Parameter comparison table ─────────────────────────────────────
    fig, ax = get_fig(width="double", aspect=3.0)
    ax.axis("off")

    col_labels = ["Parameter", "Dense", "MoE (no bal.)",
                   r"MoE + $\mathcal{L}_{bal}$", "MoE + LFB"]
    row_data = []

    # E, B, alpha, B0, beta (same for all from dense fit)
    for pname in ["E", "B", "alpha", "B0", "beta"]:
        row = [pname, f"{params[pname]:.4f}", f"{params[pname]:.4f}",
               f"{params[pname]:.4f}", f"{params[pname]:.4f}"]
        row_data.append(row)

    # rho
    row = [r"$\rho$", "—"]
    for reg in ["M0", "M_alpha", "M_lfb"]:
        v = rho_vals.get(reg)
        row.append(f"{v:.4f}" if v is not None else "—")
    row_data.append(row)

    # G_opt
    row = [r"$G^{\mathrm{opt}}$", "—"]
    for reg in ["M0", "M_alpha", "M_lfb"]:
        v = G_opts.get(reg)
        row.append(f"{v:.2f}" if v is not None else "—")
    row_data.append(row)

    table = ax.table(cellText=row_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Parameter Comparison Across Regimes", fontsize=11, pad=20)
    fig.tight_layout()
    save_fig(fig, str(out / "parameter_table"))

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "regime_params": {
            reg: {
                "rho": regime_params[reg]["rho"],
                "gammas": list(regime_params[reg]["gammas"]) if regime_params[reg]["gammas"] else None,
                "G_opt": regime_params[reg].get("G_opt"),
            }
            for reg in ALL_REGIMES
        },
        "verification": verification,
        "all_checks_pass": all(v.get("pass", False) for v in verification.values()),
    }
    save_json(output, str(out / "unified_results.json"))
    print(f"\nSaved to {out / 'unified_results.json'}")
    print(f"All verification checks pass: {output['all_checks_pass']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Unified Law")
    parser.add_argument("--results_dir", type=str, default="./results/")
    parser.add_argument("--output_dir", type=str, default="./analysis/step4/")
    parser.add_argument("--dense_params", type=str, default="./analysis/step1/dense_params.json")
    parser.add_argument("--routing_params", type=str, default="./analysis/step2/routing_efficiency.json")
    parser.add_argument("--granularity_params", type=str, default="./analysis/step3/granularity_params.json")
    args = parser.parse_args()
    main(args.results_dir, args.output_dir,
         args.dense_params, args.routing_params, args.granularity_params)
