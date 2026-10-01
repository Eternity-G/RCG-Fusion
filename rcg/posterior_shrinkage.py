"""Bounded posterior-residual fusion."""
from __future__ import annotations

import math

import numpy as np
import torch


def posterior_shrinkage(full_probability: torch.Tensor,
                        posterior_probability: torch.Tensor,
                        alpha: float = .25) -> torch.Tensor:
    if not 0 <= alpha < 1:
        raise ValueError("alpha must lie in [0,1)")
    if full_probability.shape != posterior_probability.shape:
        raise ValueError("full and posterior probabilities must have equal shapes")
    return (1-float(alpha))*full_probability+float(alpha)*posterior_probability


def worst_case_loss_increase(alpha: float) -> float:
    """Pointwise CE increase bound relative to the full-fusion prediction."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must lie in [0,1)")
    return -math.log1p(-float(alpha))


def select_alpha_by_nll(full_probability, posterior_probability, labels,
                        grid=(0., .25, .5, .75)) -> tuple[float, dict[float, float]]:
    """Select a global trust-region radius using selection labels only.

    Ties are resolved toward the smaller correction, so the full-fusion path is
    the deterministic fallback when the posterior adds no selection-set value.
    """
    full = np.asarray(full_probability, dtype=np.float64)
    posterior = np.asarray(posterior_probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if full.shape != posterior.shape or full.ndim != 2 or len(labels) != len(full):
        raise ValueError("probabilities and labels have incompatible shapes")
    candidates = sorted(set(float(alpha) for alpha in grid))
    if not candidates or candidates[0] < 0 or candidates[-1] >= 1:
        raise ValueError("alpha grid must be nonempty and contained in [0,1)")
    rows = np.arange(len(labels)); scores = {}
    for alpha in candidates:
        mixture = (1-alpha)*full+alpha*posterior
        scores[alpha] = float(-np.log(np.clip(mixture[rows, labels], 1e-12, 1.)).mean())
    selected = min(candidates, key=lambda alpha: (scores[alpha], alpha))
    return selected, scores


def posterior_risk_certificate(full_probability: torch.Tensor,
                               posterior_probability: torch.Tensor,
                               alpha: float = .25,
                               epsilon: float = 1e-8
                               ) -> dict[str, torch.Tensor]:
    """Return posterior expected gain and its convexity lower bound."""
    full = full_probability.clamp(epsilon, 1.)
    posterior = posterior_probability.clamp(epsilon, 1.)
    mixture = posterior_shrinkage(full, posterior, alpha).clamp(epsilon, 1.)
    expected_gain = (posterior*(mixture.log()-full.log())).sum(-1)
    kl = (posterior*(posterior.log()-full.log())).sum(-1)
    return {"probability": mixture, "posterior_expected_gain": expected_gain,
            "convexity_lower_bound": float(alpha)*kl,
            "worst_case_loss_increase": torch.full_like(
                expected_gain, worst_case_loss_increase(alpha))}


def contribution_adaptive_shrinkage(full_probability: torch.Tensor,
                                    posterior_probability: torch.Tensor,
                                    alpha_max: float = .75,
                                    gamma: float = 2.,
                                    tau: float = .01,
                                    epsilon: float = 1e-8
                                    ) -> dict[str, torch.Tensor]:
    """Shrink toward the posterior according to analytic benefit probability.

    The trust score is computed for the maximum admissible action.  Raising it
    to ``gamma`` implements a smooth risk-averse gate without a discontinuous
    sample selector.  No label is used by this function.
    """
    if gamma < 1:
        raise ValueError("gamma must be at least one")
    full = full_probability.clamp(epsilon, 1.)
    posterior = posterior_probability.clamp(epsilon, 1.)
    maximum = posterior_shrinkage(full, posterior, alpha_max).clamp(epsilon, 1.)
    label_gain = maximum.log()-full.log()
    safe_probability = (posterior*(label_gain > tau).to(posterior.dtype)).sum(-1)
    alpha = float(alpha_max)*safe_probability.pow(float(gamma))
    mixture = ((1-alpha[:, None])*full
               + alpha[:, None]*posterior).clamp(epsilon, 1.)
    expected_gain = (posterior*(mixture.log()-full.log())).sum(-1)
    kl = (posterior*(posterior.log()-full.log())).sum(-1)
    return {"probability": mixture, "alpha": alpha,
            "safe_probability": safe_probability,
            "posterior_expected_gain": expected_gain,
            "convexity_lower_bound": alpha*kl,
            "worst_case_loss_increase": torch.full_like(
                expected_gain, worst_case_loss_increase(alpha_max))}
