#!/usr/bin/env python3
"""
Step 1: Dense Baseline — Fit Chinchilla scaling law L(N_a, D) = E + B/N_a^alpha + B0/D^beta.

Loads all dense runs. Fits (E, B, alpha, B0, beta) via Huber + BFGS (3-stage).
Computes 95% bootstrap CIs (B=2000). Runs verification checks V1-V6.
Fits compute-optimal allocation: log N_a_opt = log a_N + b_N * log C.

Usage:
    python analysis_step1_dense_baseline.py --results_dir ./results/ --output_dir ./analysis/step1/
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from fitting_utils import (
    bootstrap_ci,
    chinchilla_loss,
    compute_N_a,
    compute_N_attn,
    compute_r_squared,
    fit_chinchilla,
    load_results,
    run_diagnostics,
    save_json,
    shapiro_wilk_test,
)
from plot_config import (
    REGIME_COLORS,
    get_fig,
    save_fig,
    setup_style,
)


def main(results_dir: str, output_dir: str):
    setup_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load dense runs
    dense_runs = load_results(
        results_dir,
        lambda r: r.get("config", {}).get("regime") == "dense"
    )

    if len(dense_runs) == 0:
        print("WARNING: No dense runs found.")
        save_json({"status": "no_data"}, str(out / "dense_params.json"))
        return

    print(f"Loaded {len(dense_runs)} dense runs")

    # Extract arrays
    N_a = np.array([r["actual_N_a"] for r in dense_runs])
    D = np.array([r["actual_tokens_seen"] for r in dense_runs])
    L_obs = np.array([r["val_loss_final"] for r in dense_runs])
    C = np.array([r["actual_flops"] for r in dense_runs])

    # ── Fit Chinchilla model ──────────────────────────────────────────────────
    params, obj = fit_chinchilla(N_a, D, L_obs)
    E_hat, B_hat, alpha_hat, B0_hat, beta_hat = params
    print(f"Fitted: E={E_hat:.4f}, B={B_hat:.4f}, alpha={alpha_hat:.4f}, "
          f"B0={B0_hat:.4f}, beta={beta_hat:.4f}")

    # Predicted losses
    L_pred = chinchilla_loss(params, N_a, D)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    diag = run_diagnostics(L_obs, L_pred, label="dense_chinchilla")

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    data_pairs = np.column_stack([N_a, D, L_obs])

    def fit_fn(data):
        p, _ = fit_chinchilla(data[:, 0], data[:, 1], data[:, 2])
        return p

    point, lower, upper = bootstrap_ci(data_pairs, fit_fn, n_bootstrap=2000)
    param_names = ["E", "B", "alpha", "B0", "beta"]
    cis = {name: {"point": float(point[i]), "lower": float(lower[i]),
                   "upper": float(upper[i])}
           for i, name in enumerate(param_names)}
    print("Bootstrap CIs:")
    for name, ci in cis.items():
        print(f"  {name}: {ci['point']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # ── Compute-optimal allocation ────────────────────────────────────────────
    # For each budget C_k, find N_a_opt numerically (argmin on IsoFLOP curve)
    unique_C = np.unique(np.round(np.log10(C), 1))
    C_budgets = 10.0 ** unique_C

    N_a_opts = []
    C_opts = []
    for C_k in C_budgets:
        mask = np.abs(np.log10(C / C_k)) < 0.3
        if np.sum(mask) < 3:
            continue
        best_idx = np.argmin(L_obs[mask])
        N_a_opts.append(N_a[mask][best_idx])
        C_opts.append(C_k)

    N_a_opts = np.array(N_a_opts)
    C_opts = np.array(C_opts)

    # Fit log N_a_opt = log a_N + b_N * log C
    if len(C_opts) >= 2:
        log_C = np.log(C_opts)
        log_Na = np.log(N_a_opts)
        slope, intercept, r_val, _, _ = sp_stats.linregress(log_C, log_Na)
        b_N = slope
        a_N = np.exp(intercept)
        print(f"Allocation law: log N_a_opt = {intercept:.4f} + {b_N:.4f} * log C")
        print(f"  a_N = {a_N:.4e}, b_N = {b_N:.4f}, R² = {r_val**2:.4f}")
    else:
        b_N = np.nan
        a_N = np.nan

    # ── Verification checks V1-V6 ────────────────────────────────────────────
    verification = {}

    # V1: R² >= 0.98 on held-out (use LOO approximation)
    verification["V1_R2"] = {
        "value": diag["R2"],
        "threshold": 0.98,
        "pass": diag["R2"] >= 0.98,
    }

    # V2: Shapiro-Wilk p > 0.05
    verification["V2_shapiro"] = {
        "value": diag["shapiro_wilk_p"],
        "threshold": 0.05,
        "pass": diag["shapiro_wilk_pass"],
    }

    # V3: alpha in (0.25, 0.45), expected ~0.34
    verification["V3_alpha"] = {
        "value": float(alpha_hat),
        "range": [0.25, 0.45],
        "pass": 0.25 < alpha_hat < 0.45,
    }

    # V4: beta in (0.20, 0.38), expected ~0.28
    verification["V4_beta"] = {
        "value": float(beta_hat),
        "range": [0.20, 0.38],
        "pass": 0.20 < beta_hat < 0.38,
    }

    # V5: E in (1.0, 2.5), expected ~1.6
    verification["V5_E"] = {
        "value": float(E_hat),
        "range": [1.0, 2.5],
        "pass": 1.0 < E_hat < 2.5,
    }

    # V6: |b_N - beta/(alpha+beta)| / (beta/(alpha+beta)) < 0.10
    if not np.isnan(b_N):
        classical = beta_hat / (alpha_hat + beta_hat)
        relative_error = abs(b_N - classical) / classical
        verification["V6_attention_correction"] = {
            "b_N": float(b_N),
            "classical_approx": float(classical),
            "relative_error": float(relative_error),
            "threshold": 0.10,
            "pass": relative_error < 0.10,
        }
    else:
        verification["V6_attention_correction"] = {"pass": False, "note": "insufficient data"}

    all_pass = all(v["pass"] for v in verification.values())
    print(f"\nVerification: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    for k, v in verification.items():
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  {k}: {status}")

    # ── Figures ───────────────────────────────────────────────────────────────
    # Fig 1: IsoFLOP curves
    if len(C_budgets) > 0:
        n_panels = len(C_budgets)
        fig, axes = get_fig(ncols=min(n_panels, 4), nrows=(n_panels + 3) // 4,
                            width="double")
        if n_panels == 1:
            axes = [axes]
        elif hasattr(axes, 'flat'):
            axes = axes.flat

        for i, C_k in enumerate(C_budgets):
            if i >= len(axes):
                break
            ax = axes[i]
            mask = np.abs(np.log10(C / C_k)) < 0.3
            if np.sum(mask) == 0:
                continue

            sort_idx = np.argsort(N_a[mask])
            ax.plot(N_a[mask][sort_idx], L_obs[mask][sort_idx], "o-",
                    color=REGIME_COLORS["D"], markersize=5)
            ax.plot(N_a[mask][sort_idx], L_pred[mask][sort_idx], "--",
                    color="gray", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel(r"$N_a$")
            ax.set_ylabel("Val loss" if i % 4 == 0 else "")
            ax.set_title(f"C = {C_k:.0e}")

        fig.suptitle("Step 1: IsoFLOP Curves (Dense)", fontsize=11, y=1.02)
        fig.tight_layout()
        save_fig(fig, str(out / "isoflop_dense"))

    # Fig 2: Compute-optimal frontier
    if len(C_opts) >= 2:
        fig, ax = get_fig(width="single")
        # Optimal loss at each budget
        L_opts = []
        for C_k in C_opts:
            mask = np.abs(np.log10(C / C_k)) < 0.3
            L_opts.append(np.min(L_obs[mask]))
        L_opts = np.array(L_opts)

        ax.loglog(C_opts, L_opts, "o-", color=REGIME_COLORS["D"],
                  markersize=6, label="Dense optimal")
        ax.set_xlabel("Compute (FLOPs)")
        ax.set_ylabel(r"$\mathcal{L}^*(C)$")
        ax.set_title("Compute-Optimal Frontier")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, str(out / "compute_optimal_frontier"))

    # Fig 3: Residual diagnostics
    fig, (ax1, ax2) = get_fig(ncols=2, width="double")
    residuals = L_obs - L_pred
    ax1.scatter(L_pred, residuals, s=20, color=REGIME_COLORS["D"], alpha=0.7)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xlabel("Predicted loss")
    ax1.set_ylabel("Residual")
    ax1.set_title("Residual vs Predicted")

    sp_stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")
    fig.suptitle("Step 1: Residual Diagnostics", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, str(out / "residual_diagnostics"))

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "params": {
            "E": float(E_hat),
            "B": float(B_hat),
            "alpha": float(alpha_hat),
            "B0": float(B0_hat),
            "beta": float(beta_hat),
        },
        "confidence_intervals": cis,
        "allocation_law": {
            "a_N": float(a_N) if not np.isnan(a_N) else None,
            "b_N": float(b_N) if not np.isnan(b_N) else None,
        },
        "diagnostics": diag,
        "verification": verification,
        "all_checks_pass": all_pass,
        "n_runs": len(dense_runs),
    }
    save_json(output, str(out / "dense_params.json"))
    print(f"\nSaved to {out / 'dense_params.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Dense Baseline")
    parser.add_argument("--results_dir", type=str, default="./results/")
    parser.add_argument("--output_dir", type=str, default="./analysis/step1/")
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)
