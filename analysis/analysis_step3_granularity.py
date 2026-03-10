#!/usr/bin/env python3
"""
Step 3: Expert Granularity — Fit Gamma(G) = gamma2*(log G)^2 + gamma1*log G per regime.

Loads Step 3 runs and dense_params.json.
Fits (gamma1, gamma2) per regime via no-intercept OLS.
Computes G_opt = exp(-gamma1 / 2*gamma2) with bootstrap CIs.

Usage:
    python analysis_step3_granularity.py --results_dir ./results/ --output_dir ./analysis/step3/ \
        --dense_params ./analysis/step1/dense_params.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fitting_utils import (
    bootstrap_ci,
    fit_granularity,
    load_results,
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


REGIMES = ["M0", "M_alpha", "M_lfb"]
PLOT_KEYS = {"M0": "M0", "M_alpha": "Malpha", "M_lfb": "Mlfb"}


def main(results_dir: str, output_dir: str, dense_params_path: str):
    setup_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load dense params
    with open(dense_params_path) as f:
        dense = json.load(f)

    # Load Step 3 runs
    step3_runs = load_results(
        results_dir,
        lambda r: r.get("config", {}).get("regime") in REGIMES
                  and r.get("config", {}).get("step_in_protocol") == 3
    )

    if len(step3_runs) == 0:
        step3_runs = load_results(
            results_dir,
            lambda r: r.get("config", {}).get("regime") in REGIMES
                      and r.get("actual_G") is not None
        )

    if len(step3_runs) == 0:
        print("WARNING: No Step 3 runs found.")
        save_json({"status": "no_data"}, str(out / "granularity_params.json"))
        return

    print(f"Loaded {len(step3_runs)} Step 3 runs")

    # Group by regime
    regime_data = {reg: [] for reg in REGIMES}
    for run in step3_runs:
        reg = run["config"]["regime"]
        regime_data[reg].append(run)

    # ── Fit per regime ────────────────────────────────────────────────────────
    results = {"regimes": {}, "verification": {}}

    fig, ax = get_fig(width="single")

    for reg in REGIMES:
        runs = regime_data[reg]
        plot_key = PLOT_KEYS[reg]

        if len(runs) < 3:
            results["regimes"][reg] = {"gamma1": None, "gamma2": None,
                                        "G_opt": None, "note": "insufficient data"}
            continue

        G_vals = np.array([r["actual_G"] for r in runs])
        L_obs = np.array([r["val_loss_final"] for r in runs])

        # Find G=1 baseline loss
        g1_mask = np.abs(G_vals - 1.0) < 0.1
        if np.any(g1_mask):
            L_baseline = np.mean(L_obs[g1_mask])
        else:
            L_baseline = L_obs[np.argmin(np.abs(G_vals - 1.0))]

        # Filter out G=1 for fitting (Gamma(1)=0 by construction)
        fit_mask = G_vals > 1.1
        if np.sum(fit_mask) < 2:
            results["regimes"][reg] = {"note": "insufficient non-G=1 data"}
            continue

        gammas, G_opt = fit_granularity(G_vals[fit_mask], L_obs[fit_mask], L_baseline)
        gamma1, gamma2 = gammas

        print(f"  {reg}: gamma1={gamma1:.4f}, gamma2={gamma2:.4f}, G_opt={G_opt:.2f}")

        # Bootstrap CI
        data = np.column_stack([G_vals[fit_mask], L_obs[fit_mask]])
        def boot_fn(d):
            g, go = fit_granularity(d[:, 0], d[:, 1], L_baseline)
            return np.array([g[0], g[1], go])

        point, lower, upper = bootstrap_ci(data, boot_fn, n_bootstrap=2000)

        results["regimes"][reg] = {
            "gamma1": float(gamma1),
            "gamma2": float(gamma2),
            "G_opt": float(G_opt),
            "ci": {
                "gamma1": {"lower": float(lower[0]), "upper": float(upper[0])},
                "gamma2": {"lower": float(lower[1]), "upper": float(upper[1])},
                "G_opt": {"lower": float(lower[2]), "upper": float(upper[2])},
            },
        }

        # Plot data and fitted curve
        sort_idx = np.argsort(G_vals)
        log_G = np.log(G_vals[sort_idx])
        Gamma_obs = np.log(L_obs[sort_idx]) - np.log(L_baseline)
        ax.scatter(log_G, Gamma_obs,
                   color=REGIME_COLORS[plot_key],
                   marker=REGIME_MARKERS[plot_key],
                   s=30, label=REGIME_LABELS[plot_key], zorder=3)

        # Fitted parabola
        log_G_smooth = np.linspace(0, np.max(log_G) * 1.1, 100)
        Gamma_fit = gamma2 * log_G_smooth**2 + gamma1 * log_G_smooth
        ax.plot(log_G_smooth, Gamma_fit, color=REGIME_COLORS[plot_key],
                linewidth=1.2, linestyle="--")

        # Mark G_opt
        if not np.isnan(G_opt):
            log_G_opt = np.log(G_opt)
            Gamma_opt = gamma2 * log_G_opt**2 + gamma1 * log_G_opt
            ax.plot(log_G_opt, Gamma_opt, "*", color=REGIME_COLORS[plot_key],
                    markersize=12, zorder=4)

    ax.set_xlabel(r"$\log G$")
    ax.set_ylabel(r"$\Gamma(G) = \log \mathcal{L}(G) - \log \mathcal{L}(1)$")
    ax.set_title("Step 3: Granularity U-curves")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, str(out / "granularity_ucurves"))

    # ── G_opt comparison bar chart ────────────────────────────────────────────
    regs_with_data = [r for r in REGIMES if results["regimes"].get(r, {}).get("G_opt") is not None]
    if len(regs_with_data) > 0:
        fig, ax = get_fig(width="single")
        x = np.arange(len(regs_with_data))
        vals = [results["regimes"][r]["G_opt"] for r in regs_with_data]
        colors = [REGIME_COLORS[PLOT_KEYS[r]] for r in regs_with_data]

        yerr_lo, yerr_hi = [], []
        for r in regs_with_data:
            ci = results["regimes"][r].get("ci", {}).get("G_opt", {})
            if ci.get("lower") is not None:
                yerr_lo.append(results["regimes"][r]["G_opt"] - ci["lower"])
                yerr_hi.append(ci["upper"] - results["regimes"][r]["G_opt"])
            else:
                yerr_lo.append(0)
                yerr_hi.append(0)

        ax.bar(x, vals, color=colors, yerr=[yerr_lo, yerr_hi],
               capsize=4, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[PLOT_KEYS[r]] for r in regs_with_data])
        ax.set_ylabel(r"$G^{\mathrm{opt}}$")
        ax.set_title(r"Optimal Granularity $G^{\mathrm{opt}}$ by Regime")
        fig.tight_layout()
        save_fig(fig, str(out / "G_opt_comparison"))

    # ── Verification V1-V5 ────────────────────────────────────────────────────
    v = results["verification"]

    # V1: gamma2 > 0
    for reg in REGIMES:
        rd = results["regimes"].get(reg, {})
        g2 = rd.get("gamma2")
        if g2 is not None:
            v[f"V1_gamma2_positive_{reg}"] = {"value": g2, "pass": g2 > 0}

    # V2: gamma1 < 0
    for reg in REGIMES:
        rd = results["regimes"].get(reg, {})
        g1 = rd.get("gamma1")
        if g1 is not None:
            v[f"V2_gamma1_negative_{reg}"] = {"value": g1, "pass": g1 < 0}

    # V3: G_opt in [2, 16]
    for reg in REGIMES:
        rd = results["regimes"].get(reg, {})
        go = rd.get("G_opt")
        if go is not None:
            v[f"V3_G_opt_range_{reg}"] = {"value": go, "pass": 2 <= go <= 16}

    # V5: Ordering G_opt_Mlfb > G_opt_Malpha > G_opt_M0
    g_opts = {r: results["regimes"].get(r, {}).get("G_opt") for r in REGIMES}
    if all(g is not None for g in g_opts.values()):
        ordering = g_opts["M_lfb"] > g_opts["M_alpha"] > g_opts["M0"]
        v["V5_ordering"] = {
            "G_opt_M0": g_opts["M0"],
            "G_opt_Malpha": g_opts["M_alpha"],
            "G_opt_Mlfb": g_opts["M_lfb"],
            "pass": ordering,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    save_json(results, str(out / "granularity_params.json"))
    print(f"\nSaved to {out / 'granularity_params.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Granularity")
    parser.add_argument("--results_dir", type=str, default="./results/")
    parser.add_argument("--output_dir", type=str, default="./analysis/step3/")
    parser.add_argument("--dense_params", type=str, default="./analysis/step1/dense_params.json")
    args = parser.parse_args()
    main(args.results_dir, args.output_dir, args.dense_params)
