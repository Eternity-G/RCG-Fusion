"""Multi-coalition posterior projection and risk-controlled shrinkage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


def project_simplex(value: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of the final dimension onto the probability simplex."""
    if value.ndim < 1:
        raise ValueError("simplex projection requires at least one dimension")
    sorted_value, _ = torch.sort(value, dim=-1, descending=True)
    cumulative = sorted_value.cumsum(-1)-1
    count = torch.arange(1, value.shape[-1]+1, device=value.device,
                         dtype=value.dtype)
    positive = sorted_value-cumulative/count > 0
    rho = positive.sum(-1).clamp_min(1)-1
    theta = cumulative.gather(-1, rho[..., None]).squeeze(-1) / (rho+1).to(value.dtype)
    return (value-theta[..., None]).clamp_min(0)


@dataclass(frozen=True)
class ProjectionResult:
    pool_indices: torch.Tensor
    weights: torch.Tensor
    probability: torch.Tensor
    objective_history: torch.Tensor


def posterior_convex_projection(posterior: torch.Tensor,
                                coalition_probabilities: torch.Tensor,
                                coalition_scores: torch.Tensor, *, top_k: int = 3,
                                full_index: int = -1, iterations: int = 50,
                                learning_rate: float = .1,
                                valid_coalitions: torch.Tensor | None = None,
                                epsilon: float = 1e-6) -> ProjectionResult:
    """Project a posterior onto the convex hull of top-k coalitions plus full.

    Updates that increase the per-sample convex objective are rejected.  This
    retains the requested fixed initial step while making monotonicity explicit.
    """
    if coalition_probabilities.ndim != 3 or posterior.ndim != 2:
        raise ValueError("expected posterior [N,K] and probabilities [N,C,K]")
    n, coalitions, _ = coalition_probabilities.shape
    if full_index < 0:
        full_index = coalitions-1
    if not 0 <= full_index < coalitions:
        raise ValueError("invalid full coalition index")
    if iterations < 1 or learning_rate <= 0:
        raise ValueError("positive iterations and learning rate are required")
    scores = coalition_scores.clone()
    if valid_coalitions is not None:
        valid = valid_coalitions.to(device=scores.device, dtype=torch.bool)
        if valid.ndim == 1:
            valid = valid[None].expand(n, -1)
        scores = scores.masked_fill(~valid, -torch.inf)
    scores[:, full_index] = -torch.inf
    available = torch.isfinite(scores).sum(-1)
    k = min(int(top_k), coalitions-1)
    if (available < k).any():
        raise ValueError("fewer valid non-full coalitions than requested top_k")
    top = scores.topk(k, dim=1).indices
    full = torch.full((n, 1), full_index, dtype=torch.long, device=top.device)
    pool = torch.cat([top, full], 1)
    gather = pool[..., None].expand(-1, -1, coalition_probabilities.shape[-1])
    probability = coalition_probabilities.gather(1, gather).clamp(epsilon, 1.)
    weights = torch.zeros((n, k+1), dtype=probability.dtype, device=probability.device)
    weights[:, -1] = 1.

    def state(current):
        mixture = (current[..., None]*probability).sum(1).clamp(epsilon, 1.)
        objective = -(posterior*mixture.log()).sum(-1)
        return mixture, objective

    mixture, objective = state(weights)
    history = [objective]
    for _ in range(iterations):
        gradient = -(posterior[:, None, :]*probability
                     / mixture[:, None, :]).sum(-1)
        proposed = project_simplex(weights-learning_rate*gradient)
        proposed_mixture, proposed_objective = state(proposed)
        accept = proposed_objective <= objective+1e-10
        weights = torch.where(accept[:, None], proposed, weights)
        mixture = torch.where(accept[:, None], proposed_mixture, mixture)
        objective = torch.where(accept, proposed_objective, objective)
        history.append(objective)
    return ProjectionResult(pool, weights, mixture,
                            torch.stack(history, dim=1))


def shrink_probability(full_probability: torch.Tensor,
                       projected_probability: torch.Tensor,
                       alpha: float) -> torch.Tensor:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    return (1-float(alpha))*full_probability+float(alpha)*projected_probability


def analytic_action(posterior: torch.Tensor, action_probability: torch.Tensor,
                    full_probability: torch.Tensor, *, tau: float = .01,
                    clip: float = 1., epsilon: float = 1e-6
                    ) -> dict[str, torch.Tensor]:
    action = action_probability.clamp(epsilon, 1.)
    full = full_probability.clamp(epsilon, 1.)
    label_benefit = (action.log()-full.log()).clamp(-clip, clip)
    gain = (posterior*label_benefit.clamp_min(0)).sum(-1)
    harm = (posterior*(-label_benefit).clamp_min(0)).sum(-1)
    return {
        "label_benefit": label_benefit,
        "expected_benefit": (posterior*label_benefit).sum(-1),
        "safe_probability": (posterior*(label_benefit > tau).to(posterior.dtype)).sum(-1),
        "gain": gain,
        "harm": harm,
    }


def select_alpha(labels: np.ndarray, full_probability: np.ndarray,
                 projected_probability: np.ndarray,
                 candidates: Sequence[float] = (.25, .5, .75, 1.),
                 tolerance: float = 1e-3) -> tuple[float, list[dict[str, float]]]:
    labels = np.asarray(labels, dtype=int)
    rows = np.arange(len(labels)); results = []
    for alpha in candidates:
        probability = shrink_probability(
            torch.as_tensor(full_probability),
            torch.as_tensor(projected_probability), alpha).numpy()
        nll = float(-np.log(np.clip(probability[rows, labels], 1e-12, 1.)).mean())
        results.append({"alpha": float(alpha), "selection_nll": nll})
    best = min(item["selection_nll"] for item in results)
    chosen = min(item["alpha"] for item in results
                 if item["selection_nll"] <= best+tolerance)
    return float(chosen), results


def build_projection(posterior: np.ndarray, coalition_probabilities: np.ndarray,
                     expected_gain: np.ndarray, expected_harm: np.ndarray, *,
                     beta: float = 2., top_k: int = 3, alpha: float = 1.,
                     iterations: int = 50, learning_rate: float = .1
                     ) -> dict[str, np.ndarray]:
    q = torch.as_tensor(posterior, dtype=torch.float64)
    probability = torch.as_tensor(coalition_probabilities, dtype=torch.float64)
    score = torch.as_tensor(expected_gain-beta*expected_harm, dtype=torch.float64)
    projection = posterior_convex_projection(
        q, probability, score, top_k=top_k, iterations=iterations,
        learning_rate=learning_rate)
    full = probability[:, -1]
    action_probability = shrink_probability(full, projection.probability, alpha)
    action = analytic_action(q, action_probability, full)
    return {
        "pool_indices": projection.pool_indices.numpy(),
        "weights": projection.weights.numpy(),
        "projected_probability": projection.probability.numpy(),
        "action_probability": action_probability.numpy(),
        "projection_objective_history": projection.objective_history.numpy(),
        **{key: value.numpy() for key, value in action.items()},
        "action_value": (action["gain"]-beta*action["harm"]).numpy(),
    }
