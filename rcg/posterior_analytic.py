"""Posterior-analytic conditional contribution estimation.

Only the label posterior is learned.  Conditional coalition benefit, its
positive/negative parts, and the probability of a beneficial switch are
deterministic functionals of that posterior and the coalition predictions.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RelationalPosteriorEstimator(nn.Module):
    """Estimate P(Y|X) from modality features and all coalition predictions."""

    def __init__(self, dims: Sequence[int], classes: int, n_coalitions: int,
                 hidden: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = .2, use_relations: bool = True):
        super().__init__()
        self.n_modalities = len(dims)
        self.classes = classes
        self.n_coalitions = n_coalitions
        self.use_relations = use_relations
        self.feature_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for d in dims
        ])
        self.probability_projector = nn.Sequential(
            nn.Linear(classes, 64), nn.ReLU(), nn.Linear(64, hidden)
        )
        self.scalar_projector = nn.Linear(3, hidden)
        self.modality_embedding = nn.Parameter(torch.randn(len(dims), hidden) * .02)
        self.mask_projector = nn.Linear(len(dims), hidden)
        self.size_embedding = nn.Embedding(len(dims) + 1, hidden)
        self.relation_projector = nn.Sequential(
            nn.Linear(10, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.coalition_probability = nn.Sequential(
            nn.Linear(classes + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.posterior_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, classes)
        )

    @staticmethod
    def entropy(probability: torch.Tensor) -> torch.Tensor:
        p = probability.clamp_min(1e-8)
        return -(p * p.log()).sum(-1)

    def forward(self, xs: Sequence[torch.Tensor], coalition_probabilities: torch.Tensor,
                masks: torch.Tensor, availability: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor]:
        modalities = self.n_modalities
        masks = masks.to(xs[0].device, xs[0].dtype)
        singleton_indices = [
            int(torch.nonzero((masks == F.one_hot(
                torch.tensor(i, device=masks.device), modalities)).all(1),
                as_tuple=False)[0])
            for i in range(modalities)
        ]
        singleton = coalition_probabilities[:, singleton_indices]
        if availability is None:
            availability = torch.ones(
                (len(xs[0]), modalities), device=xs[0].device, dtype=xs[0].dtype)
        else:
            availability = availability.to(xs[0].device, xs[0].dtype)
            if availability.shape != (len(xs[0]), modalities):
                raise ValueError("availability has incompatible shape")
            if (availability.sum(1) == 0).any():
                raise ValueError("at least one modality must be available")
        uniform = torch.full_like(singleton, 1./self.classes)
        singleton = torch.where(availability[:, :, None].bool(), singleton, uniform)
        entropy = self.entropy(singleton)
        confidence = singleton.max(-1).values
        raw = torch.stack([
            projector(x) for projector, x in zip(self.feature_projectors, xs)
        ], 1)
        scalars = torch.stack([entropy, confidence, availability], -1)
        tokens = (raw + self.probability_projector(singleton)
                  + self.scalar_projector(scalars)
                  + self.modality_embedding[None].to(raw.dtype))
        tokens = tokens*availability[:, :, None]

        active = masks[None, :, :, None]*availability[:, None, :, None]
        pooled = (tokens[:, None] * active).sum(2) / active.sum(2).clamp_min(1)
        relation_sum = torch.zeros(
            (len(xs[0]), len(masks), 5), device=raw.device, dtype=raw.dtype)
        relation_max = torch.full_like(relation_sum, -torch.inf)
        pair_count = torch.zeros(
            (len(xs[0]), len(masks), 1), device=raw.device, dtype=raw.dtype)
        for i, j in combinations(range(modalities), 2):
            pi, pj = singleton[:, i].clamp_min(1e-8), singleton[:, j].clamp_min(1e-8)
            middle = .5 * (pi + pj)
            values = torch.stack([
                .5 * ((pi * (pi.log()-middle.log())).sum(1)
                      + (pj * (pj.log()-middle.log())).sum(1)),
                F.cosine_similarity(raw[:, i], raw[:, j], dim=-1),
                pi.argmax(1).eq(pj.argmax(1)).to(raw.dtype),
                (confidence[:, i]-confidence[:, j]).abs(),
                (entropy[:, i]-entropy[:, j]).abs(),
            ], -1)[:, None]
            on = ((masks[:, i] * masks[:, j]).bool()[None, :, None]
                  & (availability[:, i] * availability[:, j]).bool()[:, None, None])
            relation_sum += values * on
            relation_max = torch.where(on, torch.maximum(relation_max, values), relation_max)
            pair_count += on.to(raw.dtype)
        relation_mean = relation_sum / pair_count.clamp_min(1)
        relation_max = torch.where(
            pair_count > 0, relation_max, torch.zeros_like(relation_max))
        relations = self.relation_projector(
            torch.cat([relation_mean, relation_max], -1))
        if not self.use_relations:
            relations = torch.zeros_like(relations)

        cp = coalition_probabilities.clamp(1e-6, 1-1e-6)
        cp_state = torch.cat([
            cp, cp.max(-1).values[..., None], self.entropy(cp)[..., None]
        ], -1)
        query = (pooled + relations + self.mask_projector(masks)[None]
                 + self.size_embedding(masks.sum(1).long())[None]
                 + self.coalition_probability(cp_state))
        valid_query = (masks[None] <= availability[:, None]).all(2)
        encoded = self.encoder(query, src_key_padding_mask=~valid_query)
        reference_match = (masks[None] == availability[:, None]).all(2)
        if not reference_match.any(1).all():
            raise ValueError("availability must correspond to a nonempty coalition mask")
        reference_index = reference_match.float().argmax(1)
        rows = torch.arange(len(encoded), device=encoded.device)
        reference = encoded[rows, reference_index]
        global_token = tokens.sum(1)/availability.sum(1, keepdim=True).clamp_min(1)
        logits = self.posterior_head(
            torch.cat([reference, global_token], -1))
        return {"posterior_logits": logits, "posterior": logits.softmax(-1)}


def posterior_objective(logits: torch.Tensor, labels: torch.Tensor,
                        brier_weight: float = .1
                        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Strictly proper posterior objective used by the v2 estimator."""
    ce = F.cross_entropy(logits, labels)
    probability = logits.softmax(-1)
    target = F.one_hot(labels, logits.shape[-1]).to(probability.dtype)
    brier = ((probability-target)**2).sum(-1).mean()
    return ce+brier_weight*brier, {"nll": ce, "brier": brier}


def analytic_contribution(posterior: torch.Tensor,
                          coalition_probabilities: torch.Tensor,
                          *, full_index: int = -1, tau: float = .01,
                          clip: float = 1., epsilon: float = 1e-6
                          ) -> dict[str, torch.Tensor]:
    """Compute coherent posterior functionals of coalition CE improvement."""
    if full_index < 0:
        full_index = coalition_probabilities.shape[1]-1
    probability = coalition_probabilities.clamp(epsilon, 1.)
    label_benefit = (probability.log()
                     - probability[:, full_index, None, :].log()).clamp(-clip, clip)
    q = posterior[:, None, :]
    gain = (q * label_benefit.clamp_min(0)).sum(-1)
    harm = (q * (-label_benefit).clamp_min(0)).sum(-1)
    expected = (q * label_benefit).sum(-1)
    safe_probability = (q * (label_benefit > tau).to(q.dtype)).sum(-1)
    return {
        "label_benefit": label_benefit,
        "expected_benefit": expected,
        "safe_probability": safe_probability,
        "gain": gain,
        "harm": harm,
    }


def analytic_candidate(posteriors: Sequence[np.ndarray],
                       coalition_probabilities: np.ndarray,
                       masks: Sequence[Sequence[int]], *, beta: float = 2.,
                       tau: float = .01, availability: Sequence[int] | None = None
                       ) -> dict[str, np.ndarray]:
    """Select a fixed candidate using posterior-integrated asymmetric utility."""
    if not posteriors:
        raise ValueError("at least one posterior is required")
    probability = torch.as_tensor(coalition_probabilities, dtype=torch.float64)
    member = []
    for posterior in posteriors:
        member.append(analytic_contribution(
            torch.as_tensor(posterior, dtype=torch.float64), probability, tau=tau))
    posterior = np.mean(posteriors, axis=0)
    aggregate = analytic_contribution(
        torch.as_tensor(posterior, dtype=torch.float64), probability, tau=tau)
    result = {key: value.numpy() for key, value in aggregate.items()}
    member_expected = np.stack([
        value["expected_benefit"].numpy() for value in member
    ])
    result["posterior"] = posterior
    result["expected_std"] = (member_expected.std(0, ddof=1)
                              if len(member_expected) > 1
                              else np.zeros_like(member_expected[0]))
    score = result["gain"]-float(beta)*result["harm"]
    full_index = len(masks)-1
    valid = np.ones(len(masks), dtype=bool)
    if availability is not None:
        valid = np.asarray([
            all(not keep or availability[i] for i, keep in enumerate(mask))
            for mask in masks
        ])
    valid[full_index] = False
    score[:, ~valid] = -np.inf
    candidate = score.argmax(1)
    rows = np.arange(len(candidate))
    result.update({
        "candidate": candidate,
        "candidate_value": score[rows, candidate],
        "candidate_safe_probability": result["safe_probability"][rows, candidate],
        "candidate_expected_benefit": result["expected_benefit"][rows, candidate],
        "candidate_expected_std": result["expected_std"][rows, candidate],
        "score": score,
    })
    return result
