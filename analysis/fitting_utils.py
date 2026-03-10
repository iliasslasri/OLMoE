"""
Statistical fitting utilities for MoE scaling law analysis.

Implements:
- Huber loss with adaptive threshold (Eq. 8 from protocol)
- 3-stage optimisation pipeline (grid → L-BFGS-B → select)
- Bootstrap confidence intervals
- Diagnostic checks (R², Shapiro-Wilk, residual plots)
"""

import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, stats


# ── Huber loss ────────────────────────────────────────────────────────────────

def compute_huber_threshold(residuals: np.ndarray) -> float:
    """Adaptive Huber threshold: delta = 1.35 * sigma_MAD."""
    sigma_mad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    return 1.35 * sigma_mad


def huber_loss(residuals: np.ndarray, delta: float) -> float:
    """Huber loss (Eq. 8): sum of element-wise Huber penalties."""
    abs_r = np.abs(residuals)
    quadratic = 0.5 * residuals**2
    linear = delta * (abs_r - 0.5 * delta)
    return np.sum(np.where(abs_r <= delta, quadratic, linear))


# ── Chinchilla loss model L(N_a, D) = E + B/N_a^alpha + B0/D^beta ────────────

def chinchilla_loss(params: np.ndarray, N_a: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Compute predicted loss for the 5-parameter Chinchilla model (Eq. 9)."""
    E, B, alpha, B0, beta = params
    return E + B / np.power(N_a, alpha) + B0 / np.power(D, beta)


def chinchilla_objective(params: np.ndarray, N_a: np.ndarray, D: np.ndarray,
                         L_obs: np.ndarray, delta: float) -> float:
    """Huber objective for Chinchilla fit (Eq. 11)."""
    residuals = L_obs - chinchilla_loss(params, N_a, D)
    return huber_loss(residuals, delta)


# ── 3-stage optimisation pipeline ─────────────────────────────────────────────

def fit_chinchilla(N_a: np.ndarray, D: np.ndarray, L_obs: np.ndarray,
                   n_grid: int = 5) -> Tuple[np.ndarray, float]:
    """
    Fit (E, B, alpha, B0, beta) via 3-stage pipeline:
    1. Coarse grid search over (alpha, beta) with OLS for (B, B0) in log-space.
    2. L-BFGS-B refinement from top-n_grid initialisers.
    3. Select solution with lowest objective (ties broken by ||theta||_2).

    Returns (params, objective_value).
    """
    # Initialise E as min observed loss minus epsilon
    E0 = np.min(L_obs) - 1e-3

    # Preliminary OLS to get residuals for Huber threshold
    # Simple OLS: log(L - E0) ~ log(B) - alpha*log(N_a) (ignoring D term)
    prelim_residuals = L_obs - E0
    delta = compute_huber_threshold(prelim_residuals)
    if delta < 1e-8:
        delta = 0.01  # fallback

    # Stage 1: coarse grid over (alpha, beta)
    alphas = np.linspace(0.1, 1.0, 20)
    betas = np.linspace(0.1, 1.0, 20)
    best_results = []

    for a in alphas:
        for b in betas:
            # For fixed (E0, a, b), solve for (B, B0) via least squares
            # L_obs - E0 ≈ B * N_a^(-a) + B0 * D^(-b)
            y = L_obs - E0
            X = np.column_stack([np.power(N_a, -a), np.power(D, -b)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            B_est, B0_est = coeffs
            if B_est <= 0 or B0_est <= 0:
                continue
            params0 = np.array([E0, B_est, a, B0_est, b])
            obj = chinchilla_objective(params0, N_a, D, L_obs, delta)
            best_results.append((obj, params0))

    best_results.sort(key=lambda x: x[0])
    top_starts = best_results[:n_grid]

    if len(top_starts) == 0:
        raise RuntimeError("Grid search found no valid initialisations")

    # Stage 2: L-BFGS-B refinement
    bounds = [(0.0, None), (0.0, None), (0.01, 2.0), (0.0, None), (0.01, 2.0)]
    refined = []

    for _, p0 in top_starts:
        try:
            result = optimize.minimize(
                chinchilla_objective, p0,
                args=(N_a, D, L_obs, delta),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 5000, "ftol": 1e-12},
            )
            if result.success or result.fun < 1e10:
                refined.append((result.fun, result.x))
        except Exception:
            continue

    if len(refined) == 0:
        raise RuntimeError("L-BFGS-B refinement failed for all initialisations")

    # Stage 3: select best (ties broken by ||theta||_2)
    refined.sort(key=lambda x: (x[0], np.linalg.norm(x[1])))
    best_obj, best_params = refined[0]

    return best_params, best_obj


# ── Routing efficiency fit (Step 2) ──────────────────────────────────────────

def neff(N_a: float, N_attn: float, A: np.ndarray, rho: float) -> np.ndarray:
    """Effective capacity N_eff(A) = N_attn + (N_a - N_attn) * A^(-rho). (Eq. 14)"""
    return N_attn + (N_a - N_attn) * np.power(A, -rho)


def routing_efficiency_loss(rho: float, A_vals: np.ndarray, L_obs: np.ndarray,
                            E_hat: float, B_hat: float, alpha_hat: float,
                            B0_hat: float, beta_hat: float,
                            N_a: float, N_attn: float, D: float,
                            delta: float) -> float:
    """Objective for fitting rho (Eq. 16): Huber loss with frozen Chinchilla params."""
    N_eff_vals = neff(N_a, N_attn, A_vals, rho)
    L_pred = E_hat + B_hat / np.power(N_eff_vals, alpha_hat) + B0_hat / np.power(D, beta_hat)
    residuals = L_obs - L_pred
    return huber_loss(residuals, delta)


def fit_rho(A_vals: np.ndarray, L_obs: np.ndarray,
            chinchilla_params: np.ndarray,
            N_a: float, N_attn: float, D: float) -> Tuple[float, float]:
    """
    Fit rho by coarse grid search + golden-section refinement.
    Returns (rho_hat, objective_value).
    """
    E_hat, B_hat, alpha_hat, B0_hat, beta_hat = chinchilla_params
    delta = compute_huber_threshold(L_obs - np.mean(L_obs))
    if delta < 1e-8:
        delta = 0.01

    # Coarse grid
    rhos = np.linspace(0.0, 1.0, 1001)
    objs = np.array([
        routing_efficiency_loss(r, A_vals, L_obs, E_hat, B_hat, alpha_hat,
                                B0_hat, beta_hat, N_a, N_attn, D, delta)
        for r in rhos
    ])
    best_idx = np.argmin(objs)

    # Golden-section refinement
    lo = max(0.0, rhos[max(0, best_idx - 1)])
    hi = min(1.0, rhos[min(len(rhos) - 1, best_idx + 1)])

    result = optimize.minimize_scalar(
        routing_efficiency_loss, bounds=(lo, hi), method="bounded",
        args=(A_vals, L_obs, E_hat, B_hat, alpha_hat, B0_hat, beta_hat,
              N_a, N_attn, D, delta),
        options={"xatol": 1e-6},
    )

    return result.x, result.fun


# ── Granularity fit (Step 3) ──────────────────────────────────────────────────

def gamma_model(log_G: np.ndarray, gamma1: float, gamma2: float) -> np.ndarray:
    """Gamma(G) = gamma2 * (log G)^2 + gamma1 * log G. (Eq. 19, no intercept)"""
    return gamma2 * log_G**2 + gamma1 * log_G


def fit_granularity(G_vals: np.ndarray, L_obs: np.ndarray, L_baseline: float
                    ) -> Tuple[np.ndarray, float]:
    """
    Fit (gamma1, gamma2) via no-intercept OLS on Gamma(G) = log L(G) - log L(1).

    Returns (np.array([gamma1, gamma2]), G_opt).
    """
    Gamma = np.log(L_obs) - np.log(L_baseline)
    log_G = np.log(G_vals)

    # No-intercept OLS: Gamma = gamma2 * (log G)^2 + gamma1 * log G
    # Design matrix: [log_G, log_G^2]
    X = np.column_stack([log_G, log_G**2])
    coeffs, _, _, _ = np.linalg.lstsq(X, Gamma, rcond=None)
    gamma1, gamma2 = coeffs

    # G_opt = exp(-gamma1 / (2*gamma2))
    if gamma2 > 0 and gamma1 < 0:
        G_opt = np.exp(-gamma1 / (2 * gamma2))
    else:
        G_opt = np.nan
        warnings.warn(f"Invalid granularity fit: gamma1={gamma1:.4f}, gamma2={gamma2:.4f}")

    return np.array([gamma1, gamma2]), G_opt


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(data_pairs: np.ndarray, fit_fn: Callable,
                 n_bootstrap: int = 2000, ci_level: float = 0.95,
                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-parametric bootstrap confidence intervals.

    Args:
        data_pairs: (n, k) array of observations to resample (rows).
        fit_fn: Callable that takes resampled data and returns parameter array.
        n_bootstrap: Number of resamples.
        ci_level: Confidence level.

    Returns:
        (point_estimate, lower_ci, upper_ci) for each parameter.
    """
    rng = np.random.default_rng(seed)
    n = len(data_pairs)
    alpha = (1 - ci_level) / 2

    point = fit_fn(data_pairs)
    n_params = len(point)
    boot_params = np.zeros((n_bootstrap, n_params))

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_params[b] = fit_fn(data_pairs[idx])
        except Exception:
            boot_params[b] = np.nan

    # Remove failed bootstraps
    valid = ~np.any(np.isnan(boot_params), axis=1)
    boot_params = boot_params[valid]

    if len(boot_params) < 100:
        warnings.warn(f"Only {len(boot_params)} valid bootstrap resamples")

    lower = np.percentile(boot_params, 100 * alpha, axis=0)
    upper = np.percentile(boot_params, 100 * (1 - alpha), axis=0)

    return point, lower, upper


# ── Diagnostics ───────────────────────────────────────────────────────────────

def compute_r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - ss_res / ss_tot


def shapiro_wilk_test(residuals: np.ndarray) -> Tuple[float, float]:
    """Shapiro-Wilk test on log-residuals. Returns (statistic, p-value)."""
    log_res = np.log(np.abs(residuals) + 1e-15)
    if len(log_res) < 3:
        return 0.0, 0.0
    stat, p = stats.shapiro(log_res)
    return stat, p


def run_diagnostics(y_obs: np.ndarray, y_pred: np.ndarray,
                    label: str = "") -> Dict:
    """Run standard diagnostic checks. Returns dict of results."""
    residuals = y_obs - y_pred
    r2 = compute_r_squared(y_obs, y_pred)
    sw_stat, sw_p = shapiro_wilk_test(residuals)

    results = {
        "label": label,
        "R2": r2,
        "R2_pass": r2 >= 0.97,
        "shapiro_wilk_stat": sw_stat,
        "shapiro_wilk_p": sw_p,
        "shapiro_wilk_pass": sw_p > 0.05,
        "max_residual": float(np.max(np.abs(residuals))),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
    }
    return results


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_results(results_dir: str, filter_fn: Optional[Callable] = None
                 ) -> List[Dict]:
    """Load all results.json files from results_dir, optionally filtering."""
    results_path = Path(results_dir)
    all_results = []
    for rj in sorted(results_path.glob("*/results.json")):
        with open(rj) as f:
            data = json.load(f)
        if filter_fn is None or filter_fn(data):
            all_results.append(data)
    return all_results


def save_json(data: Dict, path: str):
    """Save dict as JSON with numpy-safe serialisation."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


# ── FLOP computation (Eq. 5) ──────────────────────────────────────────────────

def compute_flops(N_a: float, D: float, T: int, L: int,
                  d_model: int, d_expert: int, K: int) -> float:
    """Exact total training FLOPs (Eq. 5 from protocol)."""
    return 3 * D * L * (8 * d_model**2 + 4 * T * d_model + 4 * K * d_model * d_expert)


def compute_N_a(L: int, d_model: int, d_expert: int, K: int) -> float:
    """Active parameters N_a = L * (4*d_model^2 + 2*K*d_model*d_expert). (Eq. 1)"""
    return L * (4 * d_model**2 + 2 * K * d_model * d_expert)


def compute_N_attn(L: int, d_model: int) -> float:
    """Attention parameters N_attn = 4*L*d_model^2."""
    return 4 * L * d_model**2
