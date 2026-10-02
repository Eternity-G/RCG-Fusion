"""Core models and statistics for Reliability--Contribution Gap guided fusion.

The module is deliberately independent of a particular dataset.  It supports
two or three modalities (and, mechanically, any small number for which exact
coalition enumeration is affordable).  Labels are used to construct training
targets and evaluation records, but never by :class:`CoalitionRiskPredictor`
or :class:`ConformalSafeSelector` at test-time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def nonempty_coalitions(n_modalities: int) -> tuple[tuple[int, ...], ...]:
    if n_modalities < 1:
        raise ValueError("n_modalities must be positive")
    return tuple(
        tuple(int(i in members) for i in range(n_modalities))
        for size in range(1, n_modalities + 1)
        for members in combinations(range(n_modalities), size)
    )


def draw_coalition_masks(n: int, n_modalities: int, device: torch.device | str) -> torch.Tensor:
    """Draw 50% full and 50% uniformly distributed incomplete nonempty masks."""
    if n_modalities == 1:
        return torch.ones((n, 1), device=device)
    incomplete = torch.tensor(nonempty_coalitions(n_modalities)[:-1], dtype=torch.float32, device=device)
    sampled = incomplete[torch.randint(len(incomplete), (n,), device=device)]
    full = torch.ones((n, n_modalities), device=device)
    return torch.where(torch.rand((n, 1), device=device) < .5, full, sampled)


def draw_modality_dropout_masks(n: int, n_modalities: int,
                                device: torch.device | str,
                                keep_probability: float = .5) -> torch.Tensor:
    """Draw independent modality-dropout masks and repair empty rows."""
    if n_modalities < 1 or not 0 < keep_probability <= 1:
        raise ValueError("modalities must be positive and keep_probability in (0, 1]")
    mask = (torch.rand((n, n_modalities), device=device) < keep_probability).float()
    empty = mask.sum(1) == 0
    if empty.any():
        rows = torch.nonzero(empty, as_tuple=False).flatten()
        columns = torch.randint(n_modalities, (len(rows),), device=device)
        mask[rows, columns] = 1.
    return mask


class CoalitionAwareBackbone(nn.Module):
    """A shared task model whose explicit mask makes every coalition valid."""

    def __init__(self, dims: Sequence[int], classes: int, hidden: int = 128,
                 heads: int = 4, layers: int = 2, dropout: float = .2):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.n_modalities = len(dims)
        self.classes = classes
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            for d in dims
        ])
        self.modality_embedding = nn.Parameter(torch.randn(len(dims), hidden) * .02)
        self.presence_embedding = nn.Embedding(2, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool_query = nn.Parameter(torch.randn(hidden) * .02)
        self.task_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes)
        )

    def modality_tokens(self, xs: Sequence[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        if len(xs) != self.n_modalities or mask.shape != (len(xs[0]), self.n_modalities):
            raise ValueError("features and coalition mask do not match the configured modalities")
        if (mask.sum(1) == 0).any():
            raise ValueError("the empty coalition is not a valid model input")
        base = torch.stack([projector(x) for projector, x in zip(self.projectors, xs)], 1)
        identity = self.modality_embedding[None].to(base.dtype)
        presence = self.presence_embedding(mask.long())
        return base + identity + presence

    def encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode a set of already identified tokens without positional encoding."""
        encoded = self.encoder(tokens, src_key_padding_mask=~mask.bool())
        scores = torch.einsum("bmh,h->bm", encoded, self.pool_query) / math.sqrt(encoded.shape[-1])
        scores = scores.masked_fill(~mask.bool(), -torch.inf)
        weights = torch.softmax(scores, 1)
        return torch.einsum("bm,bmh->bh", weights, encoded)

    def forward(self, xs: Sequence[torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.modality_tokens(xs, mask)
        pooled = self.encode_tokens(tokens, mask)
        return {"logits": self.task_head(pooled), "pooled": pooled, "tokens": tokens}


@torch.inference_mode()
def evaluate_all_coalitions(model: CoalitionAwareBackbone, xs: Sequence[torch.Tensor],
                            temperatures: Sequence[float] | None = None
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits and probabilities shaped [batch, coalition, class]."""
    masks = nonempty_coalitions(model.n_modalities)
    logits = []
    for mask in masks:
        present = torch.tensor(mask, dtype=xs[0].dtype, device=xs[0].device)[None].expand(len(xs[0]), -1)
        logits.append(model(xs, present)["logits"])
    logits = torch.stack(logits, 1)
    if temperatures is None:
        scaled = logits
    else:
        temperature = torch.as_tensor(temperatures, dtype=logits.dtype, device=logits.device)
        if temperature.numel() != len(masks):
            raise ValueError("one temperature is required per coalition")
        scaled = logits / temperature[None, :, None].clamp_min(1e-6)
    return logits, torch.softmax(scaled, -1)


def coalition_losses(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -probabilities.gather(2, labels[:, None, None].expand(-1, probabilities.shape[1], 1)).squeeze(2).clamp_min(1e-12).log()


def exact_shapley_from_losses(losses: np.ndarray, empty_loss: np.ndarray,
                              masks: Sequence[Sequence[int]]) -> np.ndarray:
    """Exact signed Shapley values for value=-loss."""
    losses = np.asarray(losses)
    lookup = {tuple(mask): losses[:, i] for i, mask in enumerate(masks)}
    n, n_modalities = losses.shape[0], len(masks[0])
    lookup[(0,) * n_modalities] = np.asarray(empty_loss)
    output = np.zeros((n, n_modalities), dtype=np.float64)
    for modality in range(n_modalities):
        others = [i for i in range(n_modalities) if i != modality]
        for size in range(n_modalities):
            coefficient = math.factorial(size) * math.factorial(n_modalities-size-1) / math.factorial(n_modalities)
            for members in combinations(others, size):
                before = tuple(int(i in members) for i in range(n_modalities))
                after = list(before); after[modality] = 1
                output[:, modality] += coefficient * (lookup[before] - lookup[tuple(after)])
    return output


def build_teacher_targets(losses_by_teacher: np.ndarray, probabilities_by_teacher: np.ndarray,
                          labels: np.ndarray, class_prior: np.ndarray,
                          masks: Sequence[Sequence[int]], epsilon: float = .01) -> dict[str, np.ndarray]:
    """Aggregate multi-teacher OOF outputs into stable coalition supervision."""
    losses = np.asarray(losses_by_teacher, dtype=np.float64)
    probabilities = np.asarray(probabilities_by_teacher, dtype=np.float64)
    if losses.ndim != 3 or probabilities.ndim != 4 or losses.shape[:2] != probabilities.shape[:2]:
        raise ValueError("expected [teacher,sample,coalition] losses and [teacher,sample,coalition,class] probabilities")
    mean = losses.mean(0)
    variance = losses.var(0, ddof=1) if losses.shape[0] > 1 else np.zeros_like(mean)
    mean_probability = probabilities.mean(0)
    full_index = len(masks) - 1
    improvement = mean[:, full_index, None] - mean
    oracle = mean.argmin(1)
    regret = mean[:, full_index] - mean.min(1)
    edge_rows, edge_base, edge_added, edge_sign = [], [], [], []
    lookup = {tuple(mask): i for i, mask in enumerate(masks)}
    for base_index, mask in enumerate(masks):
        for modality, keep in enumerate(mask):
            if keep:
                continue
            after = list(mask); after[modality] = 1
            added_index = lookup[tuple(after)]
            utility = mean[:, base_index] - mean[:, added_index]
            edge_rows.append(utility); edge_base.append(base_index); edge_added.append(added_index)
            edge_sign.append(np.where(utility > epsilon, 2, np.where(utility < -epsilon, 0, 1)))
    empty_loss = -np.log(np.maximum(np.asarray(class_prior)[np.asarray(labels)], 1e-12))
    shapley = exact_shapley_from_losses(mean, empty_loss, masks)
    return {
        "loss_mean": mean.astype(np.float32),
        "loss_variance": variance.astype(np.float32),
        "probability_mean": mean_probability.astype(np.float32),
        "improvement_over_full": improvement.astype(np.float32),
        "safe_improvement": (improvement > epsilon).astype(np.float32),
        "edge_contribution": np.stack(edge_rows, 1).astype(np.float32),
        "edge_sign_class": np.stack(edge_sign, 1).astype(np.int64),
        "edge_base_index": np.asarray(edge_base, dtype=np.int64),
        "edge_added_index": np.asarray(edge_added, dtype=np.int64),
        "shapley_contribution": shapley.astype(np.float32),
        "oracle_coalition_index": oracle.astype(np.int64),
        "full_fusion_regret": regret.astype(np.float32),
    }


class CoalitionRiskPredictor(nn.Module):
    """Permutation-equivariant relational predictor over candidate coalitions."""

    def __init__(self, dims: Sequence[int], classes: int, hidden: int = 128,
                 heads: int = 4, layers: int = 2, dropout: float = .2):
        super().__init__()
        self.n_modalities = len(dims)
        self.classes = classes
        self.feature_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for d in dims
        ])
        self.probability_projector = nn.Sequential(nn.Linear(classes, 64), nn.ReLU(), nn.Linear(64, hidden))
        self.scalar_projector = nn.Linear(4, hidden)
        self.modality_embedding = nn.Parameter(torch.randn(len(dims), hidden) * .02)
        self.mask_projector = nn.Linear(len(dims), hidden)
        self.size_embedding = nn.Embedding(len(dims) + 1, hidden)
        self.relation_projector = nn.Sequential(nn.Linear(10, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.coalition_probability = nn.Sequential(nn.Linear(classes + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        # Begin from the calibrated confidence-risk baseline.  Learning then
        # adds a bounded relational correction instead of destroying a useful
        # ordering during the first small-data updates.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.head[-1].bias.data[1] = -1.5  # softplus(-1.5) ~= 0.20

    @staticmethod
    def _entropy(probability: torch.Tensor) -> torch.Tensor:
        p = probability.clamp_min(1e-8)
        return -(p * p.log()).sum(-1)

    def forward(self, xs: Sequence[torch.Tensor], singleton_probabilities: torch.Tensor,
                coalition_probabilities: torch.Tensor, masks: torch.Tensor,
                native_reliability: torch.Tensor | None = None,
                availability: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, modalities, _ = singleton_probabilities.shape
        if modalities != self.n_modalities:
            raise ValueError("singleton probability modality count mismatch")
        entropy = self._entropy(singleton_probabilities)
        confidence = singleton_probabilities.max(-1).values
        if native_reliability is None:
            native_reliability = confidence
        if availability is None:
            availability = torch.ones_like(confidence)
        else:
            availability = availability.to(confidence.device, confidence.dtype)
            if availability.shape != confidence.shape:
                raise ValueError("availability must have shape [batch, modality]")
        scalars = torch.stack([entropy, confidence, native_reliability, availability], -1)
        raw = torch.stack([projector(x) for projector, x in zip(self.feature_projectors, xs)], 1)
        tokens = raw + self.probability_projector(singleton_probabilities) + self.scalar_projector(scalars)
        tokens = tokens + self.modality_embedding[None].to(tokens.dtype)

        masks = masks.to(tokens.device, tokens.dtype)
        active = masks[None, :, :, None]
        pooled = (tokens[:, None] * active).sum(2) / active.sum(2).clamp_min(1)

        relation_sum = torch.zeros((batch, len(masks), 5), device=tokens.device, dtype=tokens.dtype)
        relation_max = torch.full_like(relation_sum, -torch.inf)
        pair_count = torch.zeros((1, len(masks), 1), device=tokens.device, dtype=tokens.dtype)
        for i, j in combinations(range(modalities), 2):
            pi, pj = singleton_probabilities[:, i].clamp_min(1e-8), singleton_probabilities[:, j].clamp_min(1e-8)
            middle = .5 * (pi + pj)
            js = .5 * ((pi * (pi.log()-middle.log())).sum(1) + (pj * (pj.log()-middle.log())).sum(1))
            cosine = F.cosine_similarity(raw[:, i], raw[:, j], dim=-1)
            agreement = pi.argmax(1).eq(pj.argmax(1)).to(tokens.dtype)
            gap = (confidence[:, i]-confidence[:, j]).abs()
            entropy_gap = (entropy[:, i]-entropy[:, j]).abs()
            values = torch.stack([js, cosine, agreement, gap, entropy_gap], -1)[:, None]
            on = (masks[:, i] * masks[:, j]).bool()[None, :, None]
            relation_sum = relation_sum + values * on
            relation_max = torch.where(on, torch.maximum(relation_max, values), relation_max)
            pair_count = pair_count + on.to(tokens.dtype)
        relation_mean = relation_sum / pair_count.clamp_min(1)
        relation_max = torch.where(pair_count > 0, relation_max, torch.zeros_like(relation_max))
        relations = self.relation_projector(torch.cat([relation_mean, relation_max], -1))

        cp = coalition_probabilities.clamp_min(1e-8)
        cp_features = torch.cat([cp, cp.max(-1).values[..., None], self._entropy(cp)[..., None]], -1)
        query = (pooled + relations + self.mask_projector(masks)[None]
                 + self.size_embedding(masks.sum(1).long())[None]
                 + self.coalition_probability(cp_features))
        encoded = self.encoder(query)
        output = self.head(encoded)
        confidence_risk = -coalition_probabilities.clamp_min(1e-8).max(-1).values.log()
        mean = (confidence_risk + .5*torch.tanh(output[..., 0])).clamp_min(1e-4)
        sigma = F.softplus(output[..., 1]) + 1e-4
        safe_logit = output[..., 2]
        return mean, sigma, safe_logit


def focal_binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                               gamma: float = 2., alpha: float = .25) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    pt = targets * probability + (1-targets) * (1-probability)
    weight = targets * alpha + (1-targets) * (1-alpha)
    return (weight * (1-pt).pow(gamma) * ce).mean()


def risk_objective(mean: torch.Tensor, sigma: torch.Tensor, safe_logit: torch.Tensor,
                   target_mean: torch.Tensor, teacher_variance: torch.Tensor,
                   safe_target: torch.Tensor, edge_base: torch.Tensor, edge_added: torch.Tensor,
                   edge_sign: torch.Tensor, *, epsilon: float = .01, variance_floor: float = 1e-3,
                   rank_weight: float = .5, safe_weight: float = 1., sign_weight: float = .5
                   ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_variance = sigma.square() + teacher_variance + variance_floor ** 2
    risk = .5 * (((mean-target_mean).square()/total_variance) + total_variance.log()).mean()
    differences = target_mean[:, :, None] - target_mean[:, None, :]
    useful = differences.abs() > epsilon
    prediction_difference = mean[:, :, None] - mean[:, None, :]
    if useful.any():
        ranking = F.softplus(-torch.sign(differences[useful]) * prediction_difference[useful]).mean()
    else:
        ranking = mean.sum() * 0
    safe = focal_binary_cross_entropy(safe_logit, safe_target)
    predicted_edge = mean[:, edge_base] - mean[:, edge_added]
    ordinal_logits = torch.stack([
        -predicted_edge-epsilon,
        epsilon-predicted_edge.abs(),
        predicted_edge-epsilon,
    ], -1) / .1
    sign = F.cross_entropy(ordinal_logits.flatten(0, 1), edge_sign.flatten())
    total = risk + rank_weight*ranking + safe_weight*safe + sign_weight*sign
    return total, {"risk": risk, "ranking": ranking, "safe": safe, "sign": sign}


@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    quantile: float
    n_calibration: int
    delta: float = 1e-6
    mode: str = "risk"
    reference_index: int = -1


def fit_joint_conformal(mean: np.ndarray, sigma: np.ndarray, observed_loss: np.ndarray,
                        alpha: float = .1, delta: float = 1e-6, *,
                        mode: str = "risk", reference_index: int = -1) -> ConformalCalibration:
    """Fit a finite-sample split-conformal max-residual quantile."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0,1)")
    mean, sigma, observed_loss = map(np.asarray, (mean, sigma, observed_loss))
    if mean.shape != sigma.shape or mean.shape != observed_loss.shape or mean.ndim != 2:
        raise ValueError("mean, sigma and observed_loss must have equal [sample,coalition] shapes")
    if mode == "risk":
        score = np.max(np.abs(observed_loss-mean)/(sigma+delta), axis=1)
        resolved_reference = -1
    elif mode == "improvement":
        resolved_reference = mean.shape[1]-1 if reference_index < 0 else int(reference_index)
        predicted = mean[:, resolved_reference, None]-mean
        observed = observed_loss[:, resolved_reference, None]-observed_loss
        scale = np.sqrt(sigma[:, resolved_reference, None]**2+sigma**2)
        keep = np.arange(mean.shape[1]) != resolved_reference
        score = np.max(np.abs(observed[:, keep]-predicted[:, keep])/(scale[:, keep]+delta), axis=1)
    else:
        raise ValueError("mode must be 'risk' or 'improvement'")
    n = len(score)
    if n == 0:
        raise ValueError("conformal calibration requires at least one sample")
    if alpha == 0:
        quantile = math.inf
    else:
        rank = min(n, math.ceil((n+1)*(1-alpha)))
        quantile = float(np.partition(score, rank-1)[rank-1])
    return ConformalCalibration(alpha, quantile, n, delta, mode, resolved_reference)


def conformal_intervals(mean: np.ndarray, sigma: np.ndarray,
                        calibration: ConformalCalibration) -> tuple[np.ndarray, np.ndarray]:
    radius = calibration.quantile * np.asarray(sigma)
    return np.asarray(mean)-radius, np.asarray(mean)+radius


class ConformalSafeSelector:
    def __init__(self, calibration: ConformalCalibration, margin: float = .01):
        self.calibration = calibration
        self.margin = float(margin)

    def select(self, mean: np.ndarray, sigma: np.ndarray, full_index: int | None = None,
               valid_coalitions: np.ndarray | None = None
               ) -> dict[str, np.ndarray]:
        lower, upper = conformal_intervals(mean, sigma, self.calibration)
        full_index = mean.shape[1]-1 if full_index is None else int(full_index)
        if self.calibration.mode == "improvement":
            if full_index != self.calibration.reference_index:
                # A missing-modality condition changes the fallback coalition;
                # no calibrated pairwise guarantee is available for that base.
                selected = np.full(len(mean), full_index, dtype=np.int64)
                return {"selected": selected, "candidate": selected.copy(),
                        "switch": np.zeros(len(mean), dtype=bool), "lower": lower, "upper": upper,
                        "improvement_lower": np.full_like(mean, -np.inf),
                        "improvement_upper": np.full_like(mean, np.inf)}
            scale = np.sqrt(sigma[:, full_index, None]**2+sigma**2)
            predicted = mean[:, full_index, None]-mean
            improvement_lower = predicted-self.calibration.quantile*scale
            improvement_upper = predicted+self.calibration.quantile*scale
            candidates = improvement_lower.copy()
            if valid_coalitions is not None:
                valid_coalitions = np.asarray(valid_coalitions, dtype=bool)
                candidates[:, ~valid_coalitions] = -np.inf
            candidates[:, full_index] = -np.inf
            candidate = candidates.argmax(1)
            rows = np.arange(len(mean))
            switch = improvement_lower[rows, candidate] > self.margin
            selected = np.where(switch, candidate, full_index)
            return {"selected": selected, "candidate": candidate, "switch": switch,
                    "lower": lower, "upper": upper,
                    "improvement_lower": improvement_lower,
                    "improvement_upper": improvement_upper}
        candidate_upper = upper.copy()
        if valid_coalitions is not None:
            valid_coalitions = np.asarray(valid_coalitions, dtype=bool)
            if valid_coalitions.shape != (mean.shape[1],):
                raise ValueError("valid_coalitions must have one flag per coalition")
            if not valid_coalitions[full_index]:
                raise ValueError("fallback coalition must be valid")
            candidate_upper[:, ~valid_coalitions] = np.inf
        candidate_upper[:, full_index] = np.inf
        candidate = candidate_upper.argmin(1)
        rows = np.arange(len(mean))
        switch = upper[rows, candidate] + self.margin < lower[:, full_index]
        selected = np.where(switch, candidate, full_index)
        return {"selected": selected, "candidate": candidate, "switch": switch,
                "lower": lower, "upper": upper}
