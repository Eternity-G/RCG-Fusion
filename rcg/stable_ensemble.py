"""Selection-fitted convex aggregation of full and posterior actions."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def mix_actions(actions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=float); weights = np.asarray(weights, dtype=float)
    if actions.ndim != 3 or weights.shape != (actions.shape[0],):
        raise ValueError("expected actions [A,N,K] and one weight per action")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("weights must lie on the probability simplex")
    return np.einsum("a,ank->nk", weights, actions)


def fit_simplex_weights(actions: np.ndarray, labels: np.ndarray, l2: float = 0.) -> np.ndarray:
    """Fit nonnegative weights on the selection split only."""
    actions = np.asarray(actions, dtype=float); labels = np.asarray(labels, dtype=int)
    if actions.ndim != 3 or actions.shape[1] != len(labels):
        raise ValueError("actions and labels are not aligned")

    def softmax(value):
        shifted = value-value.max(); probability = np.exp(shifted)
        return probability/probability.sum()

    def objective(logits):
        weight = softmax(logits); probability = mix_actions(actions, weight)
        nll = -np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        return nll+float(l2)*np.square(weight).sum()

    result = minimize(objective, np.zeros(actions.shape[0]), method="BFGS",
                      options={"maxiter": 2000})
    if not np.isfinite(result.fun):
        raise RuntimeError("simplex optimization failed")
    return softmax(result.x)


def select_safe_shrinkage(full: np.ndarray, candidate: np.ndarray,
                          labels: np.ndarray, grid=None):
    """Minimize selection NLL without reducing selection accuracy."""
    if grid is None:
        grid = np.linspace(0, 1, 11)
    labels = np.asarray(labels, dtype=int); rows = np.arange(len(labels))
    baseline_accuracy = np.mean(full.argmax(1) == labels); records = []
    for rho in grid:
        probability = (1-float(rho))*full+float(rho)*candidate
        records.append({"rho": float(rho),
                        "accuracy": float(np.mean(probability.argmax(1) == labels)),
                        "nll": float(-np.log(np.clip(
                            probability[rows, labels], 1e-12, 1)).mean())})
    feasible = [item for item in records if item["accuracy"] >= baseline_accuracy]
    best = min(feasible, key=lambda item: (item["nll"], item["rho"]))
    return best["rho"], records
