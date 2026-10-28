"""Declared convex objectives; no test data or outcome-dependent search choices."""

import hashlib
import time
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit

ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
SEEDS = [101, 202, 303, 404, 505]
PRIORS = [0.01, 0.05, 0.20, 0.50]
METHODS = [
    "unchanged",
    "prior_odds",
    "intercept",
    "slope_intercept",
    "coefficient_refit",
    "known_prior",
]
OPTIONS = {"ftol": 1e-14, "gtol": 1e-9, "maxiter": 5000, "maxls": 50}


class OptimizationFailure(RuntimeError):
    def __init__(self, evidence):
        super().__init__("An optimization attempt failed its declared gate")
        self.evidence = evidence


def partitions(groups, seed):
    names = []
    for group in groups:
        bucket = (
            int(hashlib.sha256(f"{group}:{seed}".encode()).hexdigest()[:8], 16) % 20
        )
        names.append(
            "fit"
            if bucket < 10
            else (
                "validation"
                if bucket < 13
                else "calibration" if bucket < 16 else "test"
            )
        )
    return np.array(names)


def weights(y, prior=None):
    if prior is None:
        return np.full(len(y), 1 / len(y))
    assert 0 < prior < 1 and set(y) == {0, 1}
    return np.where(y == 1, prior / np.sum(y == 1), (1 - prior) / np.sum(y == 0))


def standardize(x):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    scale[scale == 0] = 1
    return mean, scale


def design(x, mean, scale):
    return np.ascontiguousarray(np.column_stack(((x - mean) / scale, np.ones(len(x)))))


def linear(x, beta):
    return np.einsum("ij,j->i", x, beta, optimize=False)


def objective(beta, x, y, w, alpha):
    eta = linear(x, beta)
    loss = np.logaddexp(0, eta) - y * eta
    value = float(np.sum(w * loss) + alpha * np.sum(beta[:-1] ** 2) / 2)
    grad = np.einsum("ij,i->j", x, w * (expit(eta) - y), optimize=False)
    grad[:-1] += alpha * beta[:-1]
    return value, grad


def projected_residual(beta, gradient, bounds=None):
    residual = gradient.copy()
    if bounds is not None:
        for j, (lower, upper) in enumerate(bounds):
            if beta[j] <= lower + 1e-12 and gradient[j] > 0:
                residual[j] = 0
            if beta[j] >= upper - 1e-12 and gradient[j] < 0:
                residual[j] = 0
    return residual


def newton_polish(beta, x, y, w, alpha, bounds=None):
    beta = beta.copy(); trace = []
    for iteration in range(21):
        value, gradient = objective(beta, x, y, w, alpha)
        residual = projected_residual(beta, gradient, bounds)
        norm = float(np.max(abs(residual)))
        entry = {"iteration": iteration, "coefficients": beta.tolist(), "objective": value, "projected_gradient_inf": norm, "line_trials": []}
        trace.append(entry)
        if norm <= 1e-12:
            return beta, trace
        if iteration == 20:
            break
        p = expit(linear(x, beta))
        hessian = x.T @ (x * (w * p * (1 - p))[:, None])
        hessian[np.arange(len(beta) - 1), np.arange(len(beta) - 1)] += alpha
        free = np.where(residual != 0)[0]
        step = np.zeros(len(beta)); step[free] = np.linalg.solve(hessian[np.ix_(free, free)], gradient[free])
        accepted = False
        for halving in range(20):
            proposal = beta - (0.5 ** halving) * step
            if bounds is not None:
                proposal = np.clip(proposal, [b[0] for b in bounds], [b[1] for b in bounds])
            next_value, next_gradient = objective(proposal, x, y, w, alpha)
            next_norm = float(np.max(abs(projected_residual(proposal, next_gradient, bounds))))
            accepted = next_value <= value + 2e-15 and next_norm < norm
            entry["line_trials"].append({"halving": halving, "coefficients": proposal.tolist(), "objective": next_value, "projected_gradient_inf": next_norm, "accepted": bool(accepted)})
            if accepted:
                beta = proposal; break
        if not accepted:
            break
    raise OptimizationFailure({"method": "newton_polish", "alpha": alpha, "bounds": bounds, "trace": trace})


def solve_ridge(x, y, w, alpha):
    prevalence = float(np.sum(w * y))
    starts = [
        np.zeros(x.shape[1]),
        np.r_[np.full(x.shape[1] - 1, 0.01), np.log(prevalence / (1 - prevalence))],
    ]
    attempts = []
    for index, start in enumerate(starts):
        began = time.perf_counter()
        result = minimize(
            objective,
            start,
            args=(x, y, w, alpha),
            jac=True,
            method="L-BFGS-B",
            options=OPTIONS,
        )
        original_value, original_grad = objective(result.x, x, y, w, alpha)
        beta, trace = newton_polish(result.x, x, y, w, alpha) if result.success else (result.x, [])
        value, grad = objective(beta, x, y, w, alpha)
        attempts.append(
            {
                "start_id": index,
                "start": start.tolist(),
                "coefficients": beta.tolist(),
                "lbfgsb": {"coefficients": result.x.tolist(), "objective": original_value, "gradient_inf": float(np.max(abs(original_grad)))},
                "newton_trace": trace,
                "objective": value,
                "gradient_inf": float(np.max(np.abs(grad))),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "seconds": time.perf_counter() - began,
            }
        )
    if not all(
        a["success"] and a["gradient_inf"] <= 1e-12 for a in attempts
    ) or not np.isclose(
        attempts[0]["objective"], attempts[1]["objective"], atol=1e-8, rtol=1e-8
    ):
        raise OptimizationFailure(
            {"method": "ridge", "alpha": alpha, "attempts": attempts}
        )
    chosen = min(attempts, key=lambda a: (a["objective"], a["start_id"]))
    return {
        "alpha": alpha,
        "attempts": attempts,
        "chosen_start": chosen["start_id"],
        "coefficients": chosen["coefficients"],
    }


def solve_intercept(eta, y, w):
    def derivative(b):
        return float(np.sum(w * (expit(eta + b) - y)))

    lower, upper = derivative(-20), derivative(20)
    b = (
        -20.0
        if lower >= 0
        else 20.0 if upper <= 0 else float(brentq(derivative, -20, 20, xtol=1e-13))
    )
    return {
        "offset": b,
        "objective": float(np.sum(w * (np.logaddexp(0, eta + b) - y * (eta + b)))),
        "derivative": derivative(b),
        "boundary": bool(abs(b) == 20),
    }


def solve_platt(eta, y, w, odds_offset):
    x = np.column_stack((eta, np.ones(len(eta))))
    attempts = []
    for index, start in enumerate(([1, 0], [1, odds_offset], [0.5, 0])):
        began = time.perf_counter()
        result = minimize(
            objective,
            start,
            args=(x, y, w, 0.0),
            jac=True,
            bounds=[(0, 5), (-20, 20)],
            method="L-BFGS-B",
            options=OPTIONS,
        )
        original_value, original_grad = objective(result.x, x, y, w, 0.0)
        beta, trace = newton_polish(result.x, x, y, w, 0.0, [(0, 5), (-20, 20)]) if result.success else (result.x, [])
        value, grad = objective(beta, x, y, w, 0.0)
        attempts.append(
            {
                "start_id": index,
                "start": list(start),
                "coefficients": beta.tolist(),
                "lbfgsb": {"coefficients": result.x.tolist(), "objective": original_value, "gradient_inf": float(np.max(abs(original_grad)))},
                "newton_trace": trace,
                "objective": value,
                "gradient": grad.tolist(),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "seconds": time.perf_counter() - began,
            }
        )
    if not all(a["success"] for a in attempts) or not np.allclose(
        [a["objective"] for a in attempts],
        attempts[0]["objective"],
        atol=1e-8,
        rtol=1e-8,
    ):
        raise OptimizationFailure({"method": "slope_intercept", "attempts": attempts})
    chosen = min(attempts, key=lambda a: (a["objective"], a["start_id"]))
    return {
        "attempts": attempts,
        "chosen_start": chosen["start_id"],
        "slope": chosen["coefficients"][0],
        "offset": chosen["coefficients"][1],
    }


def metrics(y, eta, w, prior):
    p = expit(eta)
    alert = p >= 10 / 11
    tp = float(np.sum(w[(y == 1) & alert]))
    fp = float(np.sum(w[(y == 0) & alert]))
    tn = float(np.sum(w[(y == 0) & ~alert]))
    fn = float(np.sum(w[(y == 1) & ~alert]))
    return {
        "log_loss": float(np.sum(w * (np.logaddexp(0, eta) - y * eta))),
        "brier_mse": float(np.sum(w * (p - y) ** 2)),
        "mean_probability": float(np.sum(w * p)),
        "calibration_offset": float(np.sum(w * p) - prior),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / prior,
        "false_positive_rate": fp / (1 - prior),
        "review_fraction": tp + fp,
        "decision_loss": 10 * fp + fn,
        "n_positive": int(np.sum(y)),
        "n_negative": int(np.sum(y == 0)),
    }
