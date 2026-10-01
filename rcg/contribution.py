from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.nn import functional as F

from . import MODALITIES
from .coalition import MASKS, coalition_name
from .io import write_json
from .models import Concat, Head
from .train import fit_temperature, seed_all, tensors, train_one


CANDIDATES = MASKS[1:]


def _predict_targets(model, xs, y, prior):
    """Observed probabilities/losses for all seven non-empty coalitions."""
    probabilities, losses = [], []
    with torch.no_grad():
        for mask in CANDIDATES:
            present = torch.tensor(mask, dtype=torch.float32, device=y.device)[None].repeat(len(y), 1)
            prob = model(xs, present).softmax(-1)
            probabilities.append(prob)
            losses.append(F.nll_loss(prob.clamp_min(1e-12).log(), y, reduction="none"))
    empty = -torch.log(torch.as_tensor(prior, dtype=torch.float32, device=y.device)[y].clamp_min(1e-12))
    return torch.stack(probabilities, 1), torch.stack(losses, 1), empty


def _unimodal_probabilities(models, temperatures, xs):
    with torch.no_grad():
        logits = torch.stack([models[i](xs[i]) for i in range(3)], 1)
        temp = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)[None, :, None]
        return (logits / temp).softmax(-1)


def crossfit_targets(split, cfg, seed, device, folds=5):
    """Create leakage-controlled OOF coalition losses and unimodal probabilities."""
    n = len(split["y"])
    dims = [split[m].shape[1] for m in MODALITIES]
    probs = np.empty((n, 3, 2), dtype=np.float32)
    coalition_probs = np.empty((n, len(CANDIDATES), 2), dtype=np.float32)
    losses = np.empty((n, len(CANDIDATES)), dtype=np.float32)
    empty = np.empty(n, dtype=np.float32)
    assignment = np.full(n, -1, dtype=np.int16)
    metadata = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit_index, hold_index) in enumerate(splitter.split(split["y"], groups=split["video"])):
        inner = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed * 100 + fold)
        inner_fit, inner_valid = next(inner.split(fit_index, groups=split["video"][fit_index]))
        train_index = fit_index[inner_fit]
        valid_index = fit_index[inner_valid]
        train = tensors({k: v[train_index] for k, v in split.items()}, device)
        valid = tensors({k: v[valid_index] for k, v in split.items()}, device)
        hold = tensors({k: v[hold_index] for k, v in split.items()}, device)
        args = {"hidden": cfg["hidden"], "dropout": cfg["dropout"]}
        coalition = Concat(dims, **args).to(device)
        seed_all(seed * 1000 + fold * 10)
        history = train_one(coalition, "concat_masked", train, valid, cfg)
        uni = []
        temperatures = []
        for mi, dim in enumerate(dims):
            head = Head(dim, **args).to(device)
            seed_all(seed * 1000 + fold * 10 + mi + 1)
            train_one(head, "unimodal", train, valid, cfg, mi)
            with torch.no_grad():
                valid_logits = head(valid[0][mi]).cpu().numpy()
            temperature, _ = fit_temperature(valid_logits, valid[1].cpu().numpy())
            uni.append(head)
            temperatures.append(temperature)
        fold_counts = np.bincount(split["y"][train_index], minlength=2).astype(np.float64)
        prior = fold_counts / fold_counts.sum()
        p, target, empty_target = _predict_targets(coalition, hold[0], hold[1], prior)
        mono = _unimodal_probabilities(uni, temperatures, hold[0])
        probs[hold_index] = mono.cpu().numpy()
        coalition_probs[hold_index] = p.cpu().numpy()
        losses[hold_index] = target.cpu().numpy()
        empty[hold_index] = empty_target.cpu().numpy()
        assignment[hold_index] = fold
        metadata.append({
            "fold": fold, "n_fit": len(train_index), "n_selection": len(valid_index),
            "n_holdout": len(hold_index), "epochs": len(history),
            "best_selection_nll": min(x["selection_nll"] for x in history),
            "temperatures": temperatures,
        })
    if np.any(assignment < 0) or not np.isfinite(losses).all() or not np.isfinite(probs).all():
        raise RuntimeError("Incomplete or non-finite cross-fitted targets")
    return {"probabilities": probs, "coalition_probabilities": coalition_probs,
            "losses": losses, "empty_loss": empty,
            "fold": assignment, "metadata": metadata}


class RelationalLossPredictor(nn.Module):
    """Set-style coalition loss predictor with explicit pair relationships."""
    def __init__(self, dims, hidden=64):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for dim in dims
        ])
        self.probability = nn.Linear(2, hidden)
        self.modality = nn.Parameter(torch.randn(3, hidden) * .02)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 16, hidden), nn.ReLU(), nn.Dropout(.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 2),
        )

    def forward(self, xs, probabilities, masks, coalition_probabilities=None):
        tokens = torch.stack([
            projector(x) + self.probability(probabilities[:, i]) + self.modality[i]
            for i, (projector, x) in enumerate(zip(self.projectors, xs))
        ], 1)
        masks = masks.to(tokens.device, tokens.dtype)
        expanded = tokens[:, None, :, :].expand(-1, len(masks), -1, -1)
        active = masks[None, :, :, None]
        count = active.sum(2).clamp_min(1)
        mean = (expanded * active).sum(2) / count
        maximum = expanded.masked_fill(~active.bool(), -torch.inf).amax(2)
        pair_features = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            pair_on = (masks[:, i] * masks[:, j])[None, :]
            pi = probabilities[:, i].clamp_min(1e-8)
            pj = probabilities[:, j].clamp_min(1e-8)
            midpoint = .5 * (pi + pj)
            js = .5 * ((pi * (pi.log()-midpoint.log())).sum(1) +
                       (pj * (pj.log()-midpoint.log())).sum(1))
            cosine = F.cosine_similarity(tokens[:, i], tokens[:, j], dim=1)
            agreement = probabilities[:, i].argmax(1).eq(probabilities[:, j].argmax(1)).float()
            pair_features.extend([js[:, None] * pair_on, cosine[:, None] * pair_on,
                                  agreement[:, None] * pair_on])
        pair = torch.stack(pair_features, -1)
        mask_features = masks[None].expand(len(tokens), -1, -1)
        if coalition_probabilities is None:
            coalition_probabilities = torch.full(
                (len(tokens), len(masks), 2), .5, dtype=tokens.dtype, device=tokens.device
            )
        cp = coalition_probabilities.clamp_min(1e-8)
        confidence = cp.max(-1).values[..., None]
        entropy = -(cp * cp.log()).sum(-1, keepdim=True)
        coalition_features = torch.cat([cp, confidence, entropy], -1)
        output = self.head(torch.cat([mean, maximum, pair, mask_features, coalition_features], -1))
        mean_loss = F.softplus(output[..., 0])
        log_variance = output[..., 1].clamp(-6, 6)
        return mean_loss, log_variance


def estimator_objective(mean, log_variance, target, masks, nll_weight=.2,
                        rank_weight=1., sign_weight=1., oracle_weight=2.):
    nll = .5 * (torch.exp(-log_variance) * (mean-target).square() + log_variance).mean()
    rank_terms = []
    for i in range(target.shape[1]):
        for j in range(i + 1, target.shape[1]):
            sign = torch.sign(target[:, j]-target[:, i])
            useful = sign != 0
            if useful.any():
                rank_terms.append(F.softplus(-sign[useful] * (mean[useful, j]-mean[useful, i])).mean())
    ranking = torch.stack(rank_terms).mean()
    full_index = CANDIDATES.index((1, 1, 1))
    sign_terms = []
    for modality in range(3):
        without = list((1, 1, 1))
        without[modality] = 0
        without_index = CANDIDATES.index(tuple(without))
        actual_positive = (target[:, without_index] > target[:, full_index]).float()
        predicted_utility = mean[:, without_index] - mean[:, full_index]
        sign_terms.append(F.binary_cross_entropy_with_logits(predicted_utility, actual_positive))
    sign_loss = torch.stack(sign_terms).mean()
    teacher = F.softmax(-target.detach(), 1)
    distillation = F.kl_div(F.log_softmax(-mean, 1), teacher, reduction="batchmean")
    total = nll_weight * nll + rank_weight * ranking + sign_weight * sign_loss + oracle_weight * distillation
    return total, {"nll": nll, "ranking": ranking, "sign": sign_loss, "oracle": distillation}


def _as_predictor_tensors(split, probabilities, coalition_probabilities, device):
    xs = [torch.as_tensor(split[m], dtype=torch.float32, device=device) for m in MODALITIES]
    probs = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    coalition = torch.as_tensor(coalition_probabilities, dtype=torch.float32, device=device)
    return xs, probs, coalition


def train_estimator(model, train_split, train_targets, valid_split, valid_targets, cfg, device,
                    epochs=100, patience=10, batch_size=128):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    masks = torch.tensor(CANDIDATES, dtype=torch.float32, device=device)
    train_x, train_p, train_cp = _as_predictor_tensors(
        train_split, train_targets["probabilities"], train_targets["coalition_probabilities"], device
    )
    valid_x, valid_p, valid_cp = _as_predictor_tensors(
        valid_split, valid_targets["probabilities"], valid_targets["coalition_probabilities"], device
    )
    train_y = torch.as_tensor(train_targets["losses"], dtype=torch.float32, device=device)
    valid_y = torch.as_tensor(valid_targets["losses"], dtype=torch.float32, device=device)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for index in torch.randperm(len(train_y), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            mean, logvar = model([x[index] for x in train_x], train_p[index], masks, train_cp[index])
            loss, parts = estimator_objective(mean, logvar, train_y[index], masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            mean, logvar = model(valid_x, valid_p, masks, valid_cp)
            validation, parts = estimator_objective(mean, logvar, valid_y, masks)
        value = float(validation)
        history.append({"epoch": epoch + 1, "train": float(np.mean(batch_losses)),
                        "validation": value, **{f"valid_{k}": float(v) for k, v in parts.items()}})
        if value < best - 1e-6:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return history


def final_targets(models, temperatures, split, prior, device):
    xs, y = tensors(split, device)
    probabilities = _unimodal_probabilities(
        [models[f"uni_{m}"] for m in MODALITIES], [temperatures[m] for m in MODALITIES], xs
    )
    coalition_probabilities, losses, empty = _predict_targets(models["concat_masked"], xs, y, prior)
    return {"probabilities": probabilities.cpu().numpy(), "losses": losses.cpu().numpy(),
            "empty_loss": empty.cpu().numpy(), "coalition_probabilities": coalition_probabilities.cpu().numpy()}


def sparsemax(logits, dim=-1):
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits, _ = torch.sort(shifted, descending=True, dim=dim)
    cumulative = sorted_logits.cumsum(dim) - 1
    ranks = torch.arange(1, logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.ndim
    shape[dim] = -1
    support = ranks.view(shape) * sorted_logits > cumulative
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cumulative.gather(dim, k.long()-1) / k
    return torch.clamp(shifted-tau, min=0)


@torch.no_grad()
def _selection(model, split, targets, device, risk_lambda, margin):
    masks = torch.tensor(CANDIDATES, dtype=torch.float32, device=device)
    xs, probabilities, coalition_probabilities = _as_predictor_tensors(
        split, targets["probabilities"], targets["coalition_probabilities"], device
    )
    mean, logvar = model(xs, probabilities, masks, coalition_probabilities)
    standard_deviation = (.5 * logvar).exp()
    score = -mean-risk_lambda*standard_deviation
    weights = sparsemax(score, 1)
    proposed = score.argmax(1)
    full_index = CANDIDATES.index((1, 1, 1))
    proposed_score = score.gather(1, proposed[:, None]).squeeze(1)
    switch = (proposed != full_index) & (proposed_score-score[:, full_index] > margin)
    selected = torch.where(switch, proposed, torch.full_like(proposed, full_index))
    weights[~switch] = 0
    weights[~switch, full_index] = 1
    target = torch.as_tensor(targets["losses"], dtype=torch.float32, device=device)
    coalition_prob = torch.as_tensor(targets["coalition_probabilities"], dtype=torch.float32, device=device)
    mixture = (coalition_prob * weights[:, :, None]).sum(1)
    y = torch.as_tensor(split["y"], dtype=torch.long, device=device)
    mixture_loss = -mixture[torch.arange(len(y), device=device), y].clamp_min(1e-12).log()
    oracle_loss, oracle = target.min(1)
    selected_loss = target[torch.arange(len(y), device=device), selected]
    return mean, standard_deviation, weights, selected, target, mixture, mixture_loss, oracle_loss, oracle, selected_loss, switch


@torch.no_grad()
def tune_selector(model, split, targets, device):
    candidates = []
    for risk_lambda in (0., .1, .25, .5, 1.):
        for margin in (0., .01, .02, .05, .1, .2):
            result = _selection(model, split, targets, device, risk_lambda, margin)
            selected_loss, switch = result[9], result[10]
            candidates.append({"risk_lambda": risk_lambda, "margin": margin,
                               "selection_nll": float(selected_loss.mean()),
                               "switch_rate": float(switch.float().mean())})
    # Prefer the safer (larger margin) policy when validation NLL is numerically tied.
    best = min(candidates, key=lambda x: (round(x["selection_nll"], 8), -x["margin"], x["risk_lambda"]))
    return best, candidates


@torch.no_grad()
def evaluate_estimator(model, split, targets, seed, device, risk_lambda=.25, margin=0.):
    (mean, standard_deviation, weights, selected, target, mixture, mixture_loss,
     oracle_loss, oracle, selected_loss, switch) = _selection(
        model, split, targets, device, risk_lambda, margin
    )
    y = torch.as_tensor(split["y"], dtype=torch.long, device=device)
    full_index = CANDIDATES.index((1, 1, 1))
    frame = pd.DataFrame({
        "sample_id": split["id"], "group_or_video_id": split["video"], "label": split["y"],
        "train_seed": seed, "predicted_coalition_loss_mae": (mean-target).abs().mean(1).cpu().numpy(),
        "observed_full_loss": target[:, full_index].cpu().numpy(),
        "oracle_loss": oracle_loss.cpu().numpy(), "selected_loss": selected_loss.cpu().numpy(),
        "mixture_loss": mixture_loss.cpu().numpy(),
        "full_fusion_regret": (target[:, full_index]-oracle_loss).cpu().numpy(),
        "selection_regret": (selected_loss-oracle_loss).cpu().numpy(),
        "switched_from_full": switch.cpu().numpy(),
        "risk_lambda": risk_lambda, "selection_margin": margin,
        "selected_coalition": [coalition_name(CANDIDATES[i]) for i in selected.cpu().numpy()],
        "oracle_coalition": [coalition_name(CANDIDATES[i]) for i in oracle.cpu().numpy()],
        "final_p0": mixture[:, 0].cpu().numpy(), "final_p1": mixture[:, 1].cpu().numpy(),
    })
    for ci, mask in enumerate(CANDIDATES):
        name = coalition_name(mask)
        frame[f"predicted_loss_{name}"] = mean[:, ci].cpu().numpy()
        frame[f"predicted_std_{name}"] = standard_deviation[:, ci].cpu().numpy()
        frame[f"weight_{name}"] = weights[:, ci].cpu().numpy()
        frame[f"observed_loss_{name}"] = target[:, ci].cpu().numpy()
    for modality in range(3):
        without = list((1, 1, 1))
        without[modality] = 0
        wi = CANDIDATES.index(tuple(without))
        name = MODALITIES[modality]
        frame[f"observed_contribution_{name}"] = (target[:, wi]-target[:, full_index]).cpu().numpy()
        frame[f"predicted_contribution_{name}"] = (mean[:, wi]-mean[:, full_index]).cpu().numpy()
    return frame


def summarize_estimator(frame):
    observed = np.column_stack([frame[f"observed_loss_{coalition_name(m)}"] for m in CANDIDATES])
    predicted = np.column_stack([frame[f"predicted_loss_{coalition_name(m)}"] for m in CANDIDATES])
    metrics = {
        "coalition_loss_mae": float(np.abs(observed-predicted).mean()),
        "coalition_loss_spearman": float(spearmanr(observed.ravel(), predicted.ravel()).statistic),
        "best_subset_top1": float((frame.selected_coalition == frame.oracle_coalition).mean()),
        "mean_full_fusion_regret": float(frame.full_fusion_regret.mean()),
        "mean_selection_regret": float(frame.selection_regret.mean()),
        "regret_reduction_fraction": float(1-frame.selection_regret.mean()/frame.full_fusion_regret.mean()),
        "mixture_nll": float(frame.mixture_loss.mean()),
        "selection_nll": float(frame.selected_loss.mean()),
        "switch_rate": float(frame.switched_from_full.mean()),
    }
    for modality in MODALITIES:
        actual = frame[f"observed_contribution_{modality}"].to_numpy()
        estimate = frame[f"predicted_contribution_{modality}"].to_numpy()
        harmful = actual < -.01
        metrics[f"contribution_spearman_{modality}"] = float(spearmanr(actual, estimate).statistic)
        if len(np.unique(harmful)) == 2:
            metrics[f"harm_auroc_{modality}"] = float(roc_auc_score(harmful, -estimate))
            metrics[f"harm_auprc_{modality}"] = float(average_precision_score(harmful, -estimate))
        else:
            metrics[f"harm_auroc_{modality}"] = np.nan
            metrics[f"harm_auprc_{modality}"] = np.nan
    return metrics
