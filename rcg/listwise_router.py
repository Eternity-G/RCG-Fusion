"""Cross-fitted soft coalition targets and analytic-residual listwise routing."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np
import torch
from scipy.special import softmax
from torch import nn
from torch.nn import functional as F

from .posterior_analytic import analytic_contribution


def build_soft_coalition_targets(losses_by_teacher: np.ndarray, *,
                                   oracle_temperature: float = .1,
                                   epsilon: float = .01,
                                   stable_fraction: float = .8
                                   ) -> dict[str, np.ndarray]:
    """Create listwise, pairwise and stable-improvement OOF targets.

    ``losses_by_teacher`` must contain only predictions from teachers that did
    not train on the corresponding row.  The function is label-agnostic once
    those OOF losses have been materialized.
    """
    losses = np.asarray(losses_by_teacher, dtype=np.float64)
    if losses.ndim != 3 or losses.shape[0] < 1 or losses.shape[2] < 2:
        raise ValueError("losses must have shape [teacher, sample, coalition]")
    if not np.isfinite(losses).all() or oracle_temperature <= 0:
        raise ValueError("losses must be finite and temperature positive")
    if not 0.5 <= stable_fraction <= 1:
        raise ValueError("stable_fraction must lie in [0.5,1]")
    teacher_distribution = softmax(-losses/oracle_temperature, axis=2)
    soft_oracle = teacher_distribution.mean(0)
    entropy = -(soft_oracle*np.log(np.clip(soft_oracle, 1e-12, 1))).sum(1)
    agreement = np.clip(1-entropy/math.log(losses.shape[2]), 0, 1)
    # P[i,j] is the fraction of teachers for which coalition i is materially
    # better than coalition j.  P[i,j]+P[j,i] can be below one due to ties.
    pairwise = (losses[:, :, :, None]+epsilon < losses[:, :, None, :]).mean(0)
    full = losses.shape[2]-1
    improvement = losses[:, :, full, None]-losses
    stable_probability = (improvement > epsilon).mean(0)
    stable = (stable_probability >= stable_fraction).astype(np.float32)
    mean_loss = losses.mean(0)
    return {
        "soft_oracle": soft_oracle.astype(np.float32),
        "pairwise_preference": pairwise.astype(np.float32),
        "stable_improvement": stable,
        "stable_probability": stable_probability.astype(np.float32),
        "agreement_weight": agreement.astype(np.float32),
        "loss_mean": mean_loss.astype(np.float32),
        "oracle_coalition_index": mean_loss.argmin(1).astype(np.int64),
    }


class AnalyticResidualListwiseRouter(nn.Module):
    """Rank coalitions from an analytic utility prior plus a bounded residual."""

    def __init__(self, dims: Sequence[int], classes: int, n_coalitions: int,
                 hidden: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = .2, residual_scale: float = .5):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.n_modalities = len(dims)
        self.classes = classes
        self.n_coalitions = n_coalitions
        self.residual_scale = float(residual_scale)
        self.feature_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for d in dims
        ])
        self.probability_projector = nn.Sequential(
            nn.Linear(classes+2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mask_projector = nn.Linear(len(dims), hidden)
        self.size_embedding = nn.Embedding(len(dims)+1, hidden)
        self.modality_embedding = nn.Parameter(torch.randn(len(dims), hidden)*.02)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden*2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.posterior_head = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, classes))
        self.residual_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        # Start exactly from the analytic prior.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def entropy(probability: torch.Tensor) -> torch.Tensor:
        p = probability.clamp_min(1e-8)
        return -(p*p.log()).sum(-1)

    def forward(self, xs: Sequence[torch.Tensor], coalition_probabilities: torch.Tensor,
                masks: torch.Tensor,
                posterior_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, coalitions, classes = coalition_probabilities.shape
        if coalitions != self.n_coalitions or classes != self.classes:
            raise ValueError("coalition probabilities do not match router configuration")
        if masks.shape != (coalitions, self.n_modalities):
            raise ValueError("coalition masks have incompatible shape")
        raw = torch.stack([
            projector(x) for projector, x in zip(self.feature_projectors, xs)
        ], 1)
        raw = raw+self.modality_embedding[None].to(raw.dtype)
        masks = masks.to(raw.device, raw.dtype)
        active = masks[None, :, :, None]
        pooled = (raw[:, None]*active).sum(2)/active.sum(2).clamp_min(1)
        cp = coalition_probabilities.clamp(1e-6, 1-1e-6)
        cp_state = torch.cat([
            cp, cp.max(-1).values[..., None], self.entropy(cp)[..., None]
        ], -1)
        query = (pooled+self.probability_projector(cp_state)
                 +self.mask_projector(masks)[None]
                 +self.size_embedding(masks.sum(1).long())[None])
        encoded = self.encoder(query)
        full_candidates = torch.nonzero(masks.sum(1) == self.n_modalities).flatten()
        if len(full_candidates) != 1:
            raise ValueError("exactly one full coalition is required")
        full_index = int(full_candidates.item())
        full = encoded[:, full_index]
        posterior_logits = self.posterior_head(
            torch.cat([full, raw.mean(1)], -1))
        if posterior_override is None:
            posterior = posterior_logits.softmax(-1)
        else:
            if posterior_override.shape != (batch, classes):
                raise ValueError("posterior override has incompatible shape")
            posterior = posterior_override.clamp_min(1e-8)
            posterior = posterior/posterior.sum(-1, keepdim=True)
            posterior_logits = posterior.log()
        analytic = analytic_contribution(
            posterior, cp, full_index=full_index)["expected_benefit"]
        residual = self.residual_scale*torch.tanh(
            self.residual_head(encoded).squeeze(-1))
        # Scores are defined relative to full fusion for identifiability.
        residual = residual-residual[:, full_index, None]
        score = analytic+residual
        score = score-score[:, full_index, None]
        return {"score": score, "analytic_score": analytic,
                "residual": residual, "posterior_logits": posterior_logits,
                "posterior": posterior,
                "full_index": torch.as_tensor(full_index, device=score.device)}


def listwise_router_objective(output: dict[str, torch.Tensor], labels: torch.Tensor,
                              targets: dict[str, torch.Tensor], *,
                              route_temperature: float = .2,
                              pair_weight: float = .5,
                              stable_weight: float = .2,
                              posterior_weight: float = 1.,
                              residual_weight: float = .02
                              ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    score = output["score"]
    target = targets["soft_oracle"]
    weight = targets["agreement_weight"].clamp_min(.05)
    list_per_row = -(target*F.log_softmax(score/route_temperature, -1)).sum(-1)
    listwise = (weight*list_per_row).sum()/weight.sum()

    preference = targets["pairwise_preference"]
    pair_total = preference+preference.transpose(1, 2)
    pair_target = preference/pair_total.clamp_min(1e-8)
    upper = torch.triu(torch.ones_like(pair_total, dtype=torch.bool), diagonal=1)
    useful = upper & (pair_total > 0)
    difference = score[:, :, None]-score[:, None, :]
    if useful.any():
        pair_loss = F.binary_cross_entropy_with_logits(
            difference[useful]/route_temperature, pair_target[useful], reduction="none")
        pairwise = (pair_loss*pair_total[useful]).sum()/pair_total[useful].sum()
    else:
        pairwise = score.sum()*0

    stable_target = targets["stable_improvement"][:, :-1]
    stable_logit = (score[:, :-1]-score[:, -1, None])/route_temperature
    stable = F.binary_cross_entropy_with_logits(stable_logit, stable_target)
    posterior = F.cross_entropy(output["posterior_logits"], labels)
    probability = output["posterior"]
    one_hot = F.one_hot(labels, probability.shape[-1]).to(probability.dtype)
    brier = ((probability-one_hot)**2).sum(-1).mean()
    residual = output["residual"].square().mean()
    total = (listwise+pair_weight*pairwise+stable_weight*stable
             +posterior_weight*(posterior+.1*brier)+residual_weight*residual)
    return total, {"listwise": listwise, "pairwise": pairwise, "stable": stable,
                   "posterior_nll": posterior, "posterior_brier": brier,
                   "residual_l2": residual}


def ranking_metrics(score: np.ndarray, losses: np.ndarray, epsilon: float = .01,
                    relevance_temperature: float = .1
                    ) -> dict[str, float]:
    """Decision-aligned ranking metrics computed without external packages."""
    score = np.asarray(score); losses = np.asarray(losses)
    if score.shape != losses.shape or score.ndim != 2:
        raise ValueError("score and losses must be equal [sample,coalition] arrays")
    order = np.argsort(-score, axis=1)
    oracle = losses.argmin(1)
    top1 = (order[:, 0] == oracle).mean()
    top2 = np.asarray([oracle[i] in order[i, :2] for i in range(len(oracle))]).mean()
    # Exponential relevance produces a bounded NDCG whose ideal ordering is
    # exactly the ascending-loss coalition ordering.
    relevance = np.exp(-(losses-losses.min(1, keepdims=True))/relevance_temperature)
    discounts = 1/np.log2(np.arange(2, losses.shape[1]+2))
    dcg = (np.take_along_axis(relevance, order, axis=1)*discounts).sum(1)
    ideal = (np.sort(relevance, axis=1)[:, ::-1]*discounts).sum(1)
    ndcg = np.mean(dcg/np.maximum(ideal, 1e-12))
    correct, count = 0., 0
    for i, j in combinations(range(losses.shape[1]), 2):
        useful = np.abs(losses[:, i]-losses[:, j]) > epsilon
        if useful.any():
            correct += ((score[useful, i] > score[useful, j])
                        == (losses[useful, i] < losses[useful, j])).sum()
            count += useful.sum()
    return {"top1": float(top1), "top2": float(top2), "ndcg": float(ndcg),
            "pairwise_accuracy": float(correct/max(count, 1))}


def anchor_preserving_candidates(analytic_score: np.ndarray, residual_score: np.ndarray,
                                 top_k: int = 3) -> np.ndarray:
    """Keep the analytic winner first and use the residual rank only as support.

    This construction makes the deployed candidate Top-1 exactly equal to the
    analytic Top-1, so a noisy listwise residual cannot overwrite the anchor.
    """
    analytic_score = np.asarray(analytic_score)
    residual_score = np.asarray(residual_score)
    if analytic_score.shape != residual_score.shape or analytic_score.ndim != 2:
        raise ValueError("analytic and residual scores must be equal 2-D arrays")
    if not 1 <= top_k <= analytic_score.shape[1]:
        raise ValueError("top_k is outside the coalition count")
    anchor = analytic_score.argmax(1)
    residual_order = np.argsort(-residual_score, axis=1)
    result = np.empty((len(anchor), top_k), dtype=np.int64)
    result[:, 0] = anchor
    for row in range(len(anchor)):
        supplement = [index for index in residual_order[row] if index != anchor[row]]
        result[row, 1:] = supplement[:top_k-1]
    return result


def candidate_recall(candidates: np.ndarray, losses: np.ndarray) -> float:
    candidates = np.asarray(candidates); losses = np.asarray(losses)
    oracle = losses.argmin(1)
    return float(np.mean([oracle[row] in candidates[row]
                          for row in range(len(oracle))]))
