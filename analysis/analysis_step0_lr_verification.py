#!/usr/bin/env python3
"""
Step 0: Learning Rate Verification.

Loads the 9 LR verification runs (3 budgets x 3 learning rates).
Checks that the predicted eta_opt(C) = 1.1576 * C^(-0.1529) falls
within the near-optimal region (<0.25% excess loss) at each budget.

Usage:
    python analysis_step0_lr_verification.py --results_dir ./results/ --output_dir ./analysis/step0/
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fitting_utils import load_results, save_json
from plot_config import setup_style, get_fig, save_fig, REGIME_COLORS


def eta_opt_predicted(C: float) -> float:
    """Reference LR law: eta_opt(C) = 1.1576 * C^(-0.1529)."""
    return 1.1576 * C ** (-0.1529)


def main(results_dir: str, output_dir: str):
    setup_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load LR verification runs (step_in_protocol == 0 or regime contains "lr_")
    all_runs = load_results(results_dir, lambda r: r.get("config", {}).get("step_in_protocol") == 0)

    if len(all_runs) == 0:
        # Fallback: try to identify by run_name pattern
        all_runs = load_results(results_dir, lambda r: "lr_verify" in r.get("run_name", "").lower())

    if len(all_runs) == 0:
        print("WARNING: No LR verification runs found. Creating empty output.")
        save_json({"status": "no_data", "budgets": []}, str(out / "lr_verification.json"))
        return

    # Group by compute budget
    budgets = {}
    for run in all_runs:
        C = run["config"]["compute_budget_flops"]
        C_key = f"{C:.0e}"
        if C_key not in budgets:
            budgets[C_key] = {"C": C, "runs": []}
        budgets[C_key]["runs"].append(run)

    # Analyse each budget
    results = {"budgets": [], "overall_pass": True}

    fig, axes = get_fig(ncols=len(budgets), width="double")
    if len(budgets) == 1:
        axes = [axes]

    for idx, (C_key, data) in enumerate(sorted(budgets.items())):
        C = data["C"]
        runs = data["runs"]

        # Extract LR and final val loss
        lrs = np.array([r["config"]["learning_rate"] for r in runs])
        losses = np.array([r["val_loss_final"] for r in runs])

        # Best LR and near-optimal band
        best_idx = np.argmin(losses)
        best_lr = lrs[best_idx]
        best_loss = losses[best_idx]
        threshold = best_loss * 1.0025  # 0.25% excess
        near_optimal_mask = losses <= threshold

        # Predicted LR
        eta_pred = eta_opt_predicted(C)

        # Check if prediction falls in near-optimal region
        # Interpolate: is there a near-optimal LR close to eta_pred?
        pred_in_band = np.any(near_optimal_mask & (np.abs(np.log(lrs / eta_pred)) < 0.5))

        budget_result = {
            "C": C,
            "C_key": C_key,
            "best_lr": float(best_lr),
            "best_loss": float(best_loss),
            "eta_predicted": float(eta_pred),
            "near_optimal_threshold": float(threshold),
            "near_optimal_lrs": lrs[near_optimal_mask].tolist(),
            "prediction_in_band": bool(pred_in_band),
            "pass": bool(pred_in_band),
        }
        results["budgets"].append(budget_result)
        if not pred_in_band:
            results["overall_pass"] = False

        # Plot
        ax = axes[idx]
        sort_idx = np.argsort(lrs)
        ax.plot(lrs[sort_idx], losses[sort_idx], "o-", color=REGIME_COLORS["D"],
                markersize=6, linewidth=1.2)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8,
                    label=f"0.25% band")
        ax.axvline(eta_pred, color=REGIME_COLORS["Mlfb"], linestyle=":",
                    linewidth=1.0, label=rf"$\eta^{{\mathrm{{pred}}}}$={eta_pred:.2e}")
        ax.fill_between(lrs[sort_idx],
                         best_loss, threshold,
                         alpha=0.1, color="green")
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Val loss" if idx == 0 else "")
        ax.set_title(f"C = {C_key}")
        ax.legend(fontsize=7)

    fig.suptitle("Step 0: LR Verification", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, str(out / "lr_verification"))

    # Determine accepted law
    if results["overall_pass"]:
        results["accepted_law"] = {"a_eta": 1.1576, "b_eta": -0.1529}
    else:
        results["accepted_law"] = None
        results["note"] = "LR law rejected; refit (a_eta, b_eta) locally before proceeding."

    save_json(results, str(out / "lr_verification.json"))
    print(f"Step 0 complete. Overall pass: {results['overall_pass']}")
    print(f"Results saved to {out / 'lr_verification.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 0: LR Verification")
    parser.add_argument("--results_dir", type=str, default="./results/")
    parser.add_argument("--output_dir", type=str, default="./analysis/step0/")
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)
