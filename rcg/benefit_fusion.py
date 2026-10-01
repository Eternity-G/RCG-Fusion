"""Posterior-structured conditional benefit estimation and CRC selection.

This module is the revised Innovation 2/3 implementation.  It intentionally
does not replace the legacy realized-loss predictor in :mod:`rcg.rcg_fusion`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def teacher_record_weights(n_teachers: int, n_samples: int) -> np.ndarray:
    """Per-record weights whose sum is one within every original sample."""
    if n_teachers < 1 or n_samples < 1:
        raise ValueError("positive teacher and sample counts are required")
    return np.full((n_teachers, n_samples), 1/n_teachers, dtype=np.float32)


class PosteriorStructuredBenefitEstimator(nn.Module):
    """Estimate expected coalition benefit and its probability of being safe."""

    def __init__(self, dims: Sequence[int], classes: int, n_coalitions: int,
                 hidden: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = .2, use_relations: bool = True,
                 use_structured: bool = True,
                 output_priors: Sequence[float] = (.28, .068, .184)):
        super().__init__()
        self.n_modalities = len(dims)
        self.classes = classes
        self.n_coalitions = n_coalitions
        self.use_relations = use_relations
        self.use_structured = use_structured
        self.feature_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for d in dims
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
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes)
        )
        # safe logit, positive gain, harm, structured residual
        self.benefit_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 4)
        )
        # OOF-derived priors prevent the gain/harm heads from starting at the
        # inappropriate symmetric value 0.5.
        nn.init.zeros_(self.benefit_head[-1].weight)
        safe_prior, gain_prior, harm_prior = [min(max(float(x), 1e-4), 1-1e-4) for x in output_priors]
        logit = lambda p: math.log(p/(1-p))
        self.benefit_head[-1].bias.data.copy_(
            torch.tensor([logit(safe_prior), logit(gain_prior), logit(harm_prior), 0.]))

    @staticmethod
    def entropy(probability: torch.Tensor) -> torch.Tensor:
        p = probability.clamp_min(1e-8)
        return -(p * p.log()).sum(-1)

    def forward(self, xs: Sequence[torch.Tensor], coalition_probabilities: torch.Tensor,
                masks: torch.Tensor, availability: torch.Tensor | None = None,
                posterior_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch = len(xs[0]); modalities = self.n_modalities
        masks = masks.to(xs[0].device, xs[0].dtype)
        singleton_indices = [
            int(torch.nonzero((masks == F.one_hot(torch.tensor(i, device=masks.device), modalities)).all(1),
                              as_tuple=False)[0])
            for i in range(modalities)
        ]
        singleton = coalition_probabilities[:, singleton_indices]
        entropy = self.entropy(singleton)
        confidence = singleton.max(-1).values
        if availability is None:
            availability = torch.ones_like(confidence)
        raw = torch.stack([projector(x) for projector, x in zip(self.feature_projectors, xs)], 1)
        scalars = torch.stack([entropy, confidence, availability], -1)
        tokens = (raw + self.probability_projector(singleton) + self.scalar_projector(scalars)
                  + self.modality_embedding[None].to(raw.dtype))

        active = masks[None, :, :, None]
        pooled = (tokens[:, None] * active).sum(2) / active.sum(2).clamp_min(1)
        relation_sum = torch.zeros((batch, len(masks), 5), device=raw.device, dtype=raw.dtype)
        relation_max = torch.full_like(relation_sum, -torch.inf)
        pair_count = torch.zeros((1, len(masks), 1), device=raw.device, dtype=raw.dtype)
        for i, j in combinations(range(modalities), 2):
            pi, pj = singleton[:, i].clamp_min(1e-8), singleton[:, j].clamp_min(1e-8)
            middle = .5 * (pi + pj)
            values = torch.stack([
                .5 * ((pi * (pi.log()-middle.log())).sum(1) +
                      (pj * (pj.log()-middle.log())).sum(1)),
                F.cosine_similarity(raw[:, i], raw[:, j], dim=-1),
                pi.argmax(1).eq(pj.argmax(1)).to(raw.dtype),
                (confidence[:, i]-confidence[:, j]).abs(),
                (entropy[:, i]-entropy[:, j]).abs(),
            ], -1)[:, None]
            on = (masks[:, i] * masks[:, j]).bool()[None, :, None]
            relation_sum += values * on
            relation_max = torch.where(on, torch.maximum(relation_max, values), relation_max)
            pair_count += on.to(raw.dtype)
        relation_mean = relation_sum / pair_count.clamp_min(1)
        relation_max = torch.where(pair_count > 0, relation_max, torch.zeros_like(relation_max))
        relations = self.relation_projector(torch.cat([relation_mean, relation_max], -1))
        if not self.use_relations:
            relations = torch.zeros_like(relations)

        cp = coalition_probabilities.clamp(1e-6, 1-1e-6)
        cp_state = torch.cat([cp, cp.max(-1).values[..., None], self.entropy(cp)[..., None]], -1)
        query = (pooled + relations + self.mask_projector(masks)[None]
                 + self.size_embedding(masks.sum(1).long())[None]
                 + self.coalition_probability(cp_state))
        encoded = self.encoder(query)
        full_index = len(masks)-1
        global_token = tokens.mean(1)
        posterior_logits = self.posterior_head(torch.cat([encoded[:, full_index], global_token], -1))
        if posterior_override is None:
            posterior = posterior_logits.softmax(-1)
        else:
            posterior = posterior_override.clamp_min(1e-6)
        log_ratio = cp.log()-cp[:, full_index, None, :].log()
        structured = (posterior[:, None, :] * log_ratio).sum(-1).clamp(-1, 1)
        raw_output = self.benefit_head(encoded)
        expected = structured + .5 * torch.tanh(raw_output[..., 3])
        if not self.use_structured:
            expected = .5 * torch.tanh(raw_output[..., 3])
        return {
            "posterior_logits": posterior_logits,
            "posterior": posterior,
            "structured_benefit": structured,
            "safe_logit": raw_output[..., 0],
            "safe_probability": torch.sigmoid(raw_output[..., 0]),
            "gain": torch.sigmoid(raw_output[..., 1]),
            "harm": torch.sigmoid(raw_output[..., 2]),
            "expected_benefit": expected,
        }


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    pt = targets*probability + (1-targets)*(1-probability)
    return ((1-pt).pow(gamma)*ce).mean()


def benefit_objective(output: dict[str, torch.Tensor], labels: torch.Tensor,
                      coalition_losses: torch.Tensor, *, tau: float = .01,
                      full_index: int = -1, posterior_weight: float = 1.
                      ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-task objective for conditional expected benefit."""
    if full_index < 0:
        full_index = coalition_losses.shape[1]-1
    keep = torch.arange(coalition_losses.shape[1], device=coalition_losses.device) != full_index
    benefit = (coalition_losses[:, full_index, None]-coalition_losses)[:, keep].clamp(-1, 1)
    safe_target = (benefit > tau).to(benefit.dtype)
    gain_target, harm_target = benefit.clamp_min(0), (-benefit).clamp_min(0)
    posterior = F.cross_entropy(output["posterior_logits"], labels)
    safe = focal_bce(output["safe_logit"][:, keep], safe_target)
    gain = F.smooth_l1_loss(output["gain"][:, keep], gain_target)
    harm = F.smooth_l1_loss(output["harm"][:, keep], harm_target)
    structured = F.smooth_l1_loss(output["expected_benefit"][:, keep], benefit)
    target_difference = benefit[:, :, None]-benefit[:, None, :]
    prediction = output["expected_benefit"][:, keep]
    prediction_difference = prediction[:, :, None]-prediction[:, None, :]
    useful = target_difference.abs() > tau
    ranking = (F.softplus(-torch.sign(target_difference[useful]) *
                          prediction_difference[useful]).mean()
               if useful.any() else prediction.sum()*0)
    total = posterior_weight*posterior + safe + gain + harm + .5*ranking + .25*structured
    return total, {"posterior": posterior, "safe": safe, "gain": gain, "harm": harm,
                   "ranking": ranking, "structured": structured}


def ensemble_candidate(outputs: Sequence[dict[str, np.ndarray]], masks: Sequence[Sequence[int]],
                       kappa: float = .5, availability: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Freeze one candidate per sample before CRC threshold calibration."""
    gains = np.stack([x["gain"] for x in outputs])
    harms = np.stack([x["harm"] for x in outputs])
    safe = np.stack([x["safe_probability"] for x in outputs])
    net_members = gains-harms
    mean_net, std_net = net_members.mean(0), net_members.std(0, ddof=1) if len(outputs) > 1 else np.zeros_like(net_members[0])
    score = mean_net-kappa*std_net
    full_index = len(masks)-1
    valid = np.ones(len(masks), dtype=bool)
    if availability is not None:
        valid = np.asarray([all(not keep or availability[i] for i, keep in enumerate(mask))
                            for mask in masks])
    valid[full_index] = False
    score[:, ~valid] = -np.inf
    candidate = score.argmax(1)
    rows = np.arange(len(candidate))
    return {"candidate": candidate, "candidate_value": score[rows, candidate],
            "candidate_safe_probability": safe.mean(0)[rows, candidate],
            "mean_net": mean_net, "std_net": std_net,
            "safe_probability": safe.mean(0)}


@dataclass(frozen=True)
class CRCCalibration:
    threshold: float
    event_threshold: float
    harm_threshold: float
    alpha_event: float
    alpha_harm: float
    n_calibration: int
    event_bound: float
    harm_bound: float


def crc_risk_curve(safety_score: np.ndarray, eligible: np.ndarray,
                   observed_benefit: np.ndarray) -> list[dict[str, float]]:
    safety_score = np.asarray(safety_score); eligible = np.asarray(eligible, dtype=bool)
    observed_benefit = np.asarray(observed_benefit)
    thresholds = np.r_[np.unique(safety_score[eligible]) if eligible.any() else [], np.inf]
    n = len(safety_score); rows = []
    for threshold in thresholds:
        switch = eligible & (safety_score >= threshold)
        event = switch & (observed_benefit <= 0)
        harm = switch * np.minimum(np.maximum(-observed_benefit, 0), 1)
        empirical_event, empirical_harm = event.mean(), harm.mean()
        rows.append({
            "threshold": float(threshold), "switch_rate": float(switch.mean()),
            "empirical_event_risk": float(empirical_event),
            "empirical_harm_risk": float(empirical_harm),
            "crc_event_bound": float(n/(n+1)*empirical_event+1/(n+1)),
            "crc_harm_bound": float(n/(n+1)*empirical_harm+1/(n+1)),
        })
    return rows


def fit_crc(safety_score: np.ndarray, eligible: np.ndarray, observed_benefit: np.ndarray,
            alpha_event: float = .05, alpha_harm: float = .02) -> tuple[CRCCalibration, list[dict[str, float]]]:
    curve = crc_risk_curve(safety_score, eligible, observed_benefit)
    event_ok = [row for row in curve if row["crc_event_bound"] <= alpha_event]
    harm_ok = [row for row in curve if row["crc_harm_bound"] <= alpha_harm]
    event_threshold = min((row["threshold"] for row in event_ok), default=math.inf)
    harm_threshold = min((row["threshold"] for row in harm_ok), default=math.inf)
    threshold = max(event_threshold, harm_threshold)
    chosen = next(row for row in curve if row["threshold"] == threshold)
    calibration = CRCCalibration(
        threshold, event_threshold, harm_threshold, alpha_event, alpha_harm,
        len(safety_score), chosen["crc_event_bound"], chosen["crc_harm_bound"]
    )
    return calibration, curve


def apply_crc(candidate: dict[str, np.ndarray], calibration: CRCCalibration,
              full_index: int) -> dict[str, np.ndarray]:
    switch = ((candidate["candidate_value"] > 0) &
              (candidate["candidate_safe_probability"] >= calibration.threshold))
    selected = np.where(switch, candidate["candidate"], full_index)
    return {"selected": selected, "switch": switch}
