"""End-to-end training entry point for RCG-Fusion.

Example (a one-seed smoke run):

    python -m rcg.rcg_fusion_pipeline --dataset mosi --seeds 11 \
        --teacher-seeds 11 --folds 2 --epochs 2 --risk-epochs 2 \
        --output runs/rcg-fusion-smoke

Formal runs should use the defaults: five OOF folds, five teacher seeds and five
method seeds.  Every stage is cached in a new output directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold
from torch.nn import functional as F

from .io import load_json, write_json
from .rcg_fusion import (CoalitionAwareBackbone, CoalitionRiskPredictor,
                         ConformalSafeSelector, build_teacher_targets,
                         coalition_losses, draw_coalition_masks,
                         evaluate_all_coalitions, fit_joint_conformal,
                         nonempty_coalitions, risk_objective)
from .strong_observation import cremadsplits, mmsasplits


DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_avmnist(root: str | Path) -> tuple[list[dict], tuple[str, ...]]:
    source = Path(root) / "prepared"
    splits = {}
    for split_name in ("train", "selection", "calibration", "test"):
        with np.load(source / f"{split_name}.npz") as data:
            ids = data["id"].copy()
            splits[split_name] = {
                "x": [data["image"].copy(), data["audio"].copy()],
                "y": data["y"].copy(), "id": ids, "group": ids,
            }
    return [splits], ("image", "audio")


def load_dataset(dataset: str, data: str | Path):
    data = Path(data)
    if dataset in ("mosi", "mosei"):
        return mmsasplits(data, dataset)
    if dataset == "cremad":
        path = data / "cremad" / "pretrained" / "cremad_wav2vec2_r3d18.npz"
        if not path.exists() and data.suffix == ".npz":
            path = data
        return cremadsplits(path)
    if dataset == "avmnist":
        if (data / "avmnist" / "prepared").exists():
            root = data / "avmnist"
        elif (data / "avmnist_hf" / "prepared").exists():
            root = data / "avmnist_hf"
        else:
            root = data
        return load_avmnist(root)
    raise ValueError(dataset)


def as_tensors(split: dict, device: str):
    return ([torch.as_tensor(x, dtype=torch.float32, device=device) for x in split["x"]],
            torch.as_tensor(split["y"], dtype=torch.long, device=device))


def fit_backbone(model: CoalitionAwareBackbone, train_split: dict, selection_split: dict,
                 *, seed: int, epochs: int, batch_size: int, patience: int = 10,
                 learning_rate: float = 1e-3) -> list[dict]:
    seed_all(seed)
    device = str(next(model.parameters()).device)
    train_x, train_y = as_tensors(train_split, device)
    selection_x, selection_y = as_tensors(selection_split, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train(); values = []
        for index in torch.randperm(len(train_y), device=device).split(batch_size):
            mask = draw_coalition_masks(len(index), model.n_modalities, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model([x[index] for x in train_x], mask)["logits"]
            loss = F.cross_entropy(logits, train_y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite coalition-backbone loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            values.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            full = torch.ones((len(selection_y), model.n_modalities), device=device)
            validation = float(F.cross_entropy(model(selection_x, full)["logits"], selection_y))
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(values)),
                        "selection_full_nll": validation})
        if validation < best-1e-6:
            best, best_state, stale = validation, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("backbone training did not produce a checkpoint")
    model.load_state_dict(best_state); model.eval()
    return history


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64); labels = np.asarray(labels)
    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float((logsumexp(scaled, axis=1)-scaled[np.arange(len(labels)), labels]).mean())
    return float(np.exp(minimize_scalar(objective, bounds=(-4, 4), method="bounded").x))


@torch.inference_mode()
def predict_outputs(model: CoalitionAwareBackbone, features, device: str,
                    temperatures: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Label-free coalition inference used by the deployment path."""
    xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in features]
    logits, probabilities = evaluate_all_coalitions(model, xs, temperatures)
    return {"logits": logits.cpu().numpy(), "probabilities": probabilities.cpu().numpy()}


def attach_observed_losses(outputs: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, np.ndarray]:
    """Offline evaluation only; do not call before the selector has produced its decision."""
    probability = torch.as_tensor(outputs["probabilities"])
    label = torch.as_tensor(labels, dtype=torch.long)
    return {**outputs, "losses": coalition_losses(probability, label).numpy()}


def predict_bundle(model: CoalitionAwareBackbone, split: dict, device: str,
                   temperatures: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Training/calibration helper where observed labels are intentionally available."""
    return attach_observed_losses(predict_outputs(model, split["x"], device, temperatures), split["y"])


def calibrate_temperatures(model: CoalitionAwareBackbone, split: dict, device: str) -> np.ndarray:
    bundle = predict_bundle(model, split, device)
    return np.asarray([fit_temperature(bundle["logits"][:, i], split["y"])
                       for i in range(bundle["logits"].shape[1])], dtype=np.float32)


def _inner_split(indices: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed)
    fit_local, selection_local = next(splitter.split(indices, groups=groups[indices]))
    return indices[fit_local], indices[selection_local]


def _take(split: dict, indices: np.ndarray) -> dict:
    return {"x": [x[indices] for x in split["x"]], "y": split["y"][indices],
            "id": split["id"][indices], "group": split["group"][indices]}


def generate_oof_teachers(split: dict, dims: list[int], classes: int, device: str, *,
                          folds: int, teacher_seeds: tuple[int, ...], epochs: int,
                          batch_size: int, output: Path) -> dict[str, np.ndarray]:
    cache = output / "oof_targets.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return {key: saved[key] for key in saved.files}
    n, n_coalitions = len(split["y"]), len(nonempty_coalitions(len(dims)))
    probabilities = np.full((len(teacher_seeds), n, n_coalitions, classes), np.nan, dtype=np.float32)
    losses = np.full((len(teacher_seeds), n, n_coalitions), np.nan, dtype=np.float32)
    assignment = np.full(n, -1, dtype=np.int16)
    metadata = []
    # AV-MNIST has one unique group per row; stratification is more useful there.
    if len(np.unique(split["group"])) == n:
        iterator = StratifiedKFold(folds, shuffle=True, random_state=20260927).split(np.arange(n), split["y"])
    else:
        iterator = GroupKFold(folds).split(np.arange(n), split["y"], split["group"])
    for fold, (fit_indices, holdout_indices) in enumerate(iterator):
        assignment[holdout_indices] = fold
        train_indices, selection_indices = _inner_split(np.asarray(fit_indices), split["group"], 20260927+fold)
        train_split, selection_split, holdout_split = (_take(split, train_indices),
                                                        _take(split, selection_indices),
                                                        _take(split, holdout_indices))
        for teacher_index, teacher_seed in enumerate(teacher_seeds):
            actual_seed = teacher_seed + fold*1000
            model = CoalitionAwareBackbone(dims, classes).to(device)
            history = fit_backbone(model, train_split, selection_split, seed=actual_seed,
                                   epochs=epochs, batch_size=batch_size, patience=min(10, max(2, epochs)))
            temperatures = calibrate_temperatures(model, selection_split, device)
            prediction = predict_bundle(model, holdout_split, device, temperatures)
            probabilities[teacher_index, holdout_indices] = prediction["probabilities"]
            losses[teacher_index, holdout_indices] = prediction["losses"]
            metadata.append({"fold": fold, "teacher_seed": teacher_seed,
                             "n_train": len(train_indices), "n_selection": len(selection_indices),
                             "n_holdout": len(holdout_indices), "epochs": len(history),
                             "best_selection_nll": min(x["selection_full_nll"] for x in history)})
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    if (assignment < 0).any() or not np.isfinite(probabilities).all() or not np.isfinite(losses).all():
        raise RuntimeError("incomplete OOF teacher coverage")
    np.savez_compressed(cache, probabilities=probabilities, losses=losses,
                        fold=assignment, teacher_seeds=np.asarray(teacher_seeds))
    write_json(output / "oof_metadata.json", metadata)
    return {"probabilities": probabilities, "losses": losses, "fold": assignment,
            "teacher_seeds": np.asarray(teacher_seeds)}


def save_target_tables(output: Path, split: dict, oof: dict, targets: dict,
                       masks: tuple[tuple[int, ...], ...], names: tuple[str, ...]) -> None:
    coalition_rows = []
    for teacher_index, teacher_seed in enumerate(oof["teacher_seeds"]):
        for coalition_index, mask in enumerate(masks):
            coalition_rows.append(pd.DataFrame({
                "sample_id": split["id"], "group_id": split["group"],
                "teacher_fold": oof["fold"], "teacher_seed": int(teacher_seed),
                "coalition_mask": "".join(map(str, mask)),
                "coalition_probability": [json.dumps(x.tolist(), separators=(",", ":"))
                                            for x in oof["probabilities"][teacher_index, :, coalition_index]],
                "coalition_loss": oof["losses"][teacher_index, :, coalition_index],
            }))
    pd.concat(coalition_rows, ignore_index=True).to_parquet(output / "oof_teacher_outputs.parquet", index=False)
    aggregate = []
    for coalition_index, mask in enumerate(masks):
        aggregate.append(pd.DataFrame({
            "sample_id": split["id"], "group_id": split["group"],
            "coalition_mask": "".join(map(str, mask)),
            "coalition_loss_mean": targets["loss_mean"][:, coalition_index],
            "coalition_loss_variance": targets["loss_variance"][:, coalition_index],
            "improvement_over_full": targets["improvement_over_full"][:, coalition_index],
            "safe_improvement": targets["safe_improvement"][:, coalition_index],
            "oracle_coalition": ["+".join(n for n, keep in zip(names, masks[i]) if keep)
                                 for i in targets["oracle_coalition_index"]],
            "full_fusion_regret": targets["full_fusion_regret"],
        }))
    pd.concat(aggregate, ignore_index=True).to_parquet(output / "aggregated_targets.parquet", index=False)
    edges = []
    for edge_index, (base, added) in enumerate(zip(targets["edge_base_index"], targets["edge_added_index"])):
        edges.append(pd.DataFrame({
            "sample_id": split["id"], "group_id": split["group"],
            "base_coalition_mask": "".join(map(str, masks[int(base)])),
            "added_coalition_mask": "".join(map(str, masks[int(added)])),
            "edge_contribution": targets["edge_contribution"][:, edge_index],
            "edge_sign_class": targets["edge_sign_class"][:, edge_index],
        }))
    pd.concat(edges, ignore_index=True).to_parquet(output / "edge_targets.parquet", index=False)
    pd.DataFrame({"sample_id": np.repeat(split["id"], len(names)),
                  "group_id": np.repeat(split["group"], len(names)),
                  "modality": np.tile(names, len(split["y"])),
                  "shapley_contribution": targets["shapley_contribution"].reshape(-1)}).to_parquet(
                      output / "shapley_targets.parquet", index=False)


def singleton_probabilities(probabilities: np.ndarray, masks: tuple[tuple[int, ...], ...]) -> np.ndarray:
    indices = [masks.index(tuple(int(j == i) for j in range(len(masks[0]))))
               for i in range(len(masks[0]))]
    return probabilities[:, indices]


def _risk_inputs(split: dict, probabilities: np.ndarray, masks, device):
    xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in split["x"]]
    cp = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    sp = torch.as_tensor(singleton_probabilities(probabilities, masks), dtype=torch.float32, device=device)
    mt = torch.as_tensor(masks, dtype=torch.float32, device=device)
    return xs, sp, cp, mt


def train_risk_predictor(model: CoalitionRiskPredictor, train_split: dict, train_targets: dict,
                         selection_split: dict, selection_targets: dict,
                         train_probabilities: np.ndarray, selection_probabilities: np.ndarray,
                         masks, device: str, *, seed: int, epochs: int, batch_size: int,
                         rank_weight: float = .5, safe_weight: float = 1., sign_weight: float = .5):
    seed_all(seed)
    tx, tsp, tcp, mt = _risk_inputs(train_split, train_probabilities, masks, device)
    vx, vsp, vcp, _ = _risk_inputs(selection_split, selection_probabilities, masks, device)
    tensors = {key: torch.as_tensor(value, device=device) for key, value in train_targets.items()
               if key in ("loss_mean", "loss_variance", "safe_improvement", "edge_sign_class")}
    valid = {key: torch.as_tensor(value, device=device) for key, value in selection_targets.items()
             if key in ("loss_mean", "loss_variance", "safe_improvement", "edge_sign_class")}
    edge_base = torch.as_tensor(train_targets["edge_base_index"], dtype=torch.long, device=device)
    edge_added = torch.as_tensor(train_targets["edge_added_index"], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train(); values = []
        for index in torch.randperm(len(train_split["y"]), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            mean, sigma, safe = model([x[index] for x in tx], tsp[index], tcp[index], mt)
            loss, _ = risk_objective(mean, sigma, safe, tensors["loss_mean"][index],
                                     tensors["loss_variance"][index], tensors["safe_improvement"][index],
                                     edge_base, edge_added, tensors["edge_sign_class"][index],
                                     rank_weight=rank_weight, safe_weight=safe_weight, sign_weight=sign_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            values.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            mean, sigma, safe = model(vx, vsp, vcp, mt)
            score, parts = risk_objective(mean, sigma, safe, valid["loss_mean"], valid["loss_variance"],
                                          valid["safe_improvement"], edge_base, edge_added,
                                          valid["edge_sign_class"], rank_weight=rank_weight,
                                          safe_weight=safe_weight, sign_weight=sign_weight)
        value = float(score)
        history.append({"epoch": epoch+1, "train": float(np.mean(values)), "selection": value,
                        **{f"selection_{key}": float(item) for key, item in parts.items()}})
        if value < best-1e-6:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(state); model.eval()
    return history


@torch.inference_mode()
def predict_risk(model, split, probabilities, masks, device, availability=None):
    xs, singleton, coalition, mask_tensor = _risk_inputs(split, probabilities, masks, device)
    available_tensor = None if availability is None else torch.as_tensor(
        availability, dtype=torch.float32, device=device)
    mean, sigma, safe = model(xs, singleton, coalition, mask_tensor,
                              availability=available_tensor)
    return mean.cpu().numpy(), sigma.cpu().numpy(), safe.sigmoid().cpu().numpy()


def targets_from_bundle(bundle: dict, split: dict, prior: np.ndarray, masks, epsilon=.01):
    return build_teacher_targets(bundle["losses"][None], bundle["probabilities"][None],
                                 split["y"], prior, masks, epsilon)


def audit_contribution_targets(oof_targets: dict, in_sample_bundle: dict, masks, names) -> dict:
    """Quantify training optimism and agreement of the two contribution definitions."""
    oof = oof_targets["loss_mean"]
    inside = in_sample_bundle["losses"]
    full = len(masks)-1; lookup = {tuple(mask): i for i, mask in enumerate(masks)}
    result = {
        "mean_oof_loss": float(oof.mean()), "mean_in_sample_loss": float(inside.mean()),
        "oof_minus_in_sample_loss": float((oof-inside).mean()),
        "oof_in_sample_loss_spearman": float(spearmanr(oof.ravel(), inside.ravel()).statistic),
        "mean_teacher_target_variance": float(oof_targets["loss_variance"].mean()),
    }
    for modality, name in enumerate(names):
        without = list((1,)*len(names)); without[modality] = 0
        deletion = oof[:, lookup[tuple(without)]]-oof[:, full]
        inside_deletion = inside[:, lookup[tuple(without)]]-inside[:, full]
        shapley = oof_targets["shapley_contribution"][:, modality]
        result[f"oof_in_sample_sign_agreement_{name}"] = float(
            (np.sign(deletion) == np.sign(inside_deletion)).mean())
        result[f"deletion_shapley_spearman_{name}"] = float(
            spearmanr(deletion, shapley).statistic)
        result[f"deletion_shapley_sign_agreement_{name}"] = float(
            (np.sign(deletion) == np.sign(shapley)).mean())
    return result


def tune_policy(mean: np.ndarray, sigma: np.ndarray, observed: np.ndarray,
                alphas=(.05, .1, .2), margins=(0., .01, .05)):
    rows = []
    full = observed.shape[1]-1; oracle = observed.min(1)
    for alpha in alphas:
        calibration = fit_joint_conformal(mean, sigma, observed, alpha, mode="improvement")
        for margin in margins:
            result = ConformalSafeSelector(calibration, margin).select(mean, sigma, full)
            chosen = observed[np.arange(len(observed)), result["selected"]]
            harmful = (chosen > observed[:, full]+.01)
            rows.append({"alpha": alpha, "margin": margin, "harmful_switch_rate": float(harmful.mean()),
                         "selection_regret": float((chosen-oracle).mean()),
                         "switch_rate": float(result["switch"].mean())})
    best = min(rows, key=lambda row: (row["harmful_switch_rate"], row["selection_regret"], row["switch_rate"]))
    return best, rows


def evaluate_method(split: dict, bundle: dict, mean: np.ndarray, sigma: np.ndarray,
                    calibration, margin: float, masks, names, seed: int,
                    available_mask=None, condition: str = "clean"):
    selector = ConformalSafeSelector(calibration, margin)
    if available_mask is None:
        available_mask = (1,) * len(names)
    available_mask = tuple(int(x) for x in available_mask)
    lookup = {tuple(mask): i for i, mask in enumerate(masks)}
    fallback = lookup[available_mask]
    valid = np.asarray([all(not keep or available_mask[i] for i, keep in enumerate(mask))
                        for mask in masks])
    result = selector.select(mean, sigma, fallback, valid)
    losses, probabilities, labels = bundle["losses"], bundle["probabilities"], split["y"]
    rows = np.arange(len(labels)); full = fallback
    selected = result["selected"]
    valid_indices = np.flatnonzero(valid)
    oracle = valid_indices[losses[:, valid_indices].argmin(1)]
    selected_loss = losses[rows, selected]; full_loss = losses[:, full]; oracle_loss = losses[rows, oracle]
    selected_probability = probabilities[rows, selected]
    full_probability = probabilities[:, full]
    if calibration.mode == "improvement" and full == calibration.reference_index:
        observed_improvement = full_loss[:, None]-losses
        keep = np.arange(losses.shape[1]) != full
        simultaneous_coverage = ((observed_improvement[:, keep] >= result["improvement_lower"][:, keep]) &
                                 (observed_improvement[:, keep] <= result["improvement_upper"][:, keep])).all(1)
        interval_width = result["improvement_upper"][:, keep]-result["improvement_lower"][:, keep]
    elif calibration.mode == "improvement":
        simultaneous_coverage = np.full(len(losses), np.nan)
        interval_width = np.full((len(losses), 1), np.nan)
    else:
        simultaneous_coverage = ((losses >= result["lower"]) & (losses <= result["upper"])).all(1)
        interval_width = result["upper"]-result["lower"]
    safe_truth = selected_loss + margin < full_loss
    switched = result["switch"]
    full_regret = full_loss-oracle_loss; selection_regret = selected_loss-oracle_loss
    width_value = np.nan if np.isnan(interval_width).all() else float(np.nanmean(interval_width))
    metrics = {
        "coalition_loss_mae": float(np.abs(mean-losses).mean()),
        "coalition_loss_spearman": float(spearmanr(mean.ravel(), losses.ravel()).statistic),
        "best_subset_top1": float((mean.argmin(1) == oracle).mean()),
        "best_subset_top2": float(np.mean([oracle[i] in np.argsort(mean[i])[:2] for i in rows])),
        "mean_full_fusion_regret": float(full_regret.mean()),
        "mean_selection_regret": float(selection_regret.mean()),
        "regret_reduction_fraction": float(1-selection_regret.mean()/max(full_regret.mean(), 1e-12)),
        "harmful_switch_rate": float(((selected_loss > full_loss+.01) & switched).mean()),
        "safe_switch_precision": float(safe_truth[switched].mean()) if switched.any() else np.nan,
        "switch_rate": float(switched.mean()), "full_fallback_rate": float((~switched).mean()),
        "oracle_recovery_rate": float((selected == oracle).mean()),
        "simultaneous_interval_coverage": float(simultaneous_coverage.mean()),
        "mean_interval_width": width_value,
        "accuracy": float(accuracy_score(labels, selected_probability.argmax(1))),
        "macro_f1": float(f1_score(labels, selected_probability.argmax(1), average="macro")),
        "nll": float(selected_loss.mean()),
        "full_accuracy": float(accuracy_score(labels, full_probability.argmax(1))),
        "full_macro_f1": float(f1_score(labels, full_probability.argmax(1), average="macro")),
        "full_nll": float(full_loss.mean()),
    }
    for modality, name in enumerate(names):
        if not available_mask[modality]:
            continue
        without = list(available_mask); without[modality] = 0
        contribution = losses[:, lookup[tuple(without)]]-full_loss
        predicted = mean[:, lookup[tuple(without)]]-mean[:, full]
        harmful = contribution < -.01
        metrics[f"contribution_spearman_{name}"] = float(spearmanr(contribution, predicted).statistic)
        if len(np.unique(harmful)) == 2:
            metrics[f"harm_auroc_{name}"] = float(roc_auc_score(harmful, -predicted))
            metrics[f"harm_auprc_{name}"] = float(average_precision_score(harmful, -predicted))
    coalition_names = ["+".join(n for n, keep in zip(names, mask) if keep) for mask in masks]
    frame = pd.DataFrame({
        "sample_id": split["id"], "group_or_video_id": split["group"], "label": labels,
        "condition": condition,
        "available_modalities": "+".join(name for name, keep in zip(names, available_mask) if keep),
        "train_seed": seed, "selected_coalition": [coalition_names[x] for x in selected],
        "oracle_coalition": [coalition_names[x] for x in oracle], "switched_from_full": switched,
        "selected_loss": selected_loss, "full_loss": full_loss, "oracle_loss": oracle_loss,
        "full_fusion_regret": full_regret, "selection_regret": selection_regret,
        "interval_jointly_covers": simultaneous_coverage,
    })
    for coalition_index, coalition_name in enumerate(coalition_names):
        frame[f"predicted_loss_{coalition_name}"] = mean[:, coalition_index]
        frame[f"predicted_std_{coalition_name}"] = sigma[:, coalition_index]
        frame[f"observed_loss_{coalition_name}"] = losses[:, coalition_index]
        frame[f"lcb_{coalition_name}"] = result["lower"][:, coalition_index]
        frame[f"ucb_{coalition_name}"] = result["upper"][:, coalition_index]
        if "improvement_lower" in result:
            frame[f"improvement_lcb_{coalition_name}"] = result["improvement_lower"][:, coalition_index]
            frame[f"improvement_ucb_{coalition_name}"] = result["improvement_upper"][:, coalition_index]
    for class_index in range(selected_probability.shape[1]):
        frame[f"final_p{class_index}"] = selected_probability[:, class_index]
    return frame, metrics


def write_go_no_go(root: Path, metrics_frame: pd.DataFrame) -> None:
    means = metrics_frame.select_dtypes(include=[np.number]).mean()
    checks = {
        "联盟风险 Spearman ≥ 0.70": means.get("coalition_loss_spearman", np.nan) >= .70,
        "平均选择后悔至少下降 20%": means.get("regret_reduction_fraction", np.nan) >= .20,
        "干净 Accuracy 下降不超过 0.5 个百分点":
            means.get("accuracy", -np.inf) >= means.get("full_accuracy", np.inf)-.005,
        "有害切换率不高于 5%": means.get("harmful_switch_rate", np.inf) <= .05,
        "联合覆盖率至少 85%": means.get("simultaneous_interval_coverage", -np.inf) >= .85,
    }
    passed = all(checks.values())
    lines = ["# RCG-Fusion 单数据集 Go/No-Go", "",
             f"结论：**{'通过，可以进入跨数据集正式验证' if passed else '未通过，不应宣称方法有效'}**。", "",
             "| 预注册检查 | 结果 | 实际均值 |", "|---|---|---:|",
             f"| 联盟风险 Spearman ≥ 0.70 | {'通过' if checks['联盟风险 Spearman ≥ 0.70'] else '未通过'} | {means.get('coalition_loss_spearman', np.nan):.4f} |",
             f"| 平均选择后悔至少下降 20% | {'通过' if checks['平均选择后悔至少下降 20%'] else '未通过'} | {means.get('regret_reduction_fraction', np.nan):.4f} |",
             f"| 干净 Accuracy 下降不超过 0.5 个百分点 | {'通过' if checks['干净 Accuracy 下降不超过 0.5 个百分点'] else '未通过'} | {(means.get('accuracy', np.nan)-means.get('full_accuracy', np.nan)):.4f} |",
             f"| 有害切换率不高于 5% | {'通过' if checks['有害切换率不高于 5%'] else '未通过'} | {means.get('harmful_switch_rate', np.nan):.4f} |",
             f"| 联合覆盖率至少 85% | {'通过' if checks['联合覆盖率至少 85%'] else '未通过'} | {means.get('simultaneous_interval_coverage', np.nan):.4f} |",
             "", "该报告只判断当前数据集，不能替代三个数据集改善和强基线比较的最终门槛。"]
    (root / "GO_NO_GO.md").write_text("\n".join(lines), encoding="utf-8")


def stress_conditions(split: dict, names: tuple[str, ...]):
    """Yield clean, feature-noise, feature-mask and whole-modality conditions."""
    yield "clean", split, (1,) * len(names)
    seeds = (101, 202, 303)
    for modality, name in enumerate(names):
        for corruption_seed in seeds:
            rng = np.random.default_rng(corruption_seed + modality*10000)
            gaussian = rng.standard_normal(split["x"][modality].shape, dtype=np.float32)
            uniform = rng.random(split["x"][modality].shape, dtype=np.float32)
            for level in (.25, .5, 1., 2.):
                changed = {**split, "x": [x.copy() for x in split["x"]]}
                changed["x"][modality] += level * gaussian
                yield f"gaussian:{name}:{level}:seed{corruption_seed}", changed, (1,) * len(names)
            for level in (.25, .5, .75):
                changed = {**split, "x": [x.copy() for x in split["x"]]}
                changed["x"][modality][uniform < level] = 0
                yield f"mask:{name}:{level}:seed{corruption_seed}", changed, (1,) * len(names)
        changed = {**split, "x": [x.copy() for x in split["x"]]}
        changed["x"][modality][:] = 0
        available = [1] * len(names); available[modality] = 0
        yield f"missing:{name}", changed, tuple(available)


def run(args) -> None:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    folds, names = load_dataset(args.dataset, args.data)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": args.dataset, "data": str(Path(args.data).resolve()),
                "modalities": names, "seeds": args.seeds, "teacher_seeds": args.teacher_seeds,
                "folds": args.folds, "epochs": args.epochs, "risk_epochs": args.risk_epochs,
                "alpha": args.alpha, "margin": args.margin, "tune_policy": args.tune_policy,
                "stress_tests": args.stress_tests,
                "loss_weights": {"rank": args.rank_weight, "safe": args.safe_weight,
                                 "sign": args.sign_weight},
                "epsilon": .01, "device": device, "python": sys.version,
                "torch": torch.__version__, "platform": platform.platform(),
                "protocol": "multi-teacher grouped OOF supervision -> relational coalition risk -> split-conformal safe fallback"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise ValueError("run manifest differs; use a fresh output directory")
    write_json(manifest_path, manifest)
    all_metrics = []
    start = time.perf_counter()
    for outer_fold, splits in enumerate(folds):
        dims = [x.shape[1] for x in splits["train"]["x"]]
        classes = int(max(np.max(part["y"]) for part in splits.values())+1)
        masks = nonempty_coalitions(len(dims))
        counts = np.bincount(splits["train"]["y"], minlength=classes).astype(float); prior = counts/counts.sum()
        fold_output = root / f"fold_{outer_fold}"; fold_output.mkdir(exist_ok=True)
        oof = generate_oof_teachers(splits["train"], dims, classes, device, folds=args.folds,
                                    teacher_seeds=tuple(args.teacher_seeds), epochs=args.epochs,
                                    batch_size=args.batch_size, output=fold_output)
        targets = build_teacher_targets(oof["losses"], oof["probabilities"], splits["train"]["y"],
                                        prior, masks, .01)
        if not (fold_output / "aggregated_targets.parquet").exists():
            save_target_tables(fold_output, splits["train"], oof, targets, masks, names)
        for seed in args.seeds:
            output = fold_output / f"seed_{seed}"; output.mkdir(exist_ok=True)
            if (output / "complete.json").exists():
                metrics = load_json(output / "metrics.json"); all_metrics.append(metrics); continue
            seed_all(seed + outer_fold*10000)
            if seed in oof["teacher_seeds"]:
                teacher_index = int(np.flatnonzero(oof["teacher_seeds"] == seed)[0])
                model_targets = build_teacher_targets(
                    oof["losses"][teacher_index:teacher_index+1],
                    oof["probabilities"][teacher_index:teacher_index+1],
                    splits["train"]["y"], prior, masks, .01)
                # Preserve ensemble disagreement as aleatoric target noise while
                # keeping the mean contribution tied to this task-model seed.
                model_targets["loss_variance"] = targets["loss_variance"]
                model_probability = oof["probabilities"][teacher_index]
                target_source = f"matching_oof_teacher_seed_{seed}"
            else:
                model_targets = targets
                model_probability = targets["probability_mean"]
                target_source = "multi_teacher_mean_fallback"
            backbone = CoalitionAwareBackbone(dims, classes).to(device)
            backbone_history = fit_backbone(backbone, splits["train"], splits["selection"],
                                             seed=seed+outer_fold*10000, epochs=args.epochs,
                                             batch_size=args.batch_size)
            # The predictor and temperature map are fixed before conformal calibration.
            temperatures = calibrate_temperatures(backbone, splits["selection"], device)
            in_sample_bundle = predict_bundle(backbone, splits["train"], device, temperatures)
            target_audit = audit_contribution_targets(model_targets, in_sample_bundle, masks, names)
            target_audit["risk_target_source"] = target_source
            selection_bundle = predict_bundle(backbone, splits["selection"], device, temperatures)
            calibration_bundle = predict_bundle(backbone, splits["calibration"], device, temperatures)
            selection_targets = targets_from_bundle(selection_bundle, splits["selection"], prior, masks)
            risk = CoalitionRiskPredictor(dims, classes).to(device)
            history = train_risk_predictor(
                risk, splits["train"], model_targets, splits["selection"], selection_targets,
                model_probability, selection_bundle["probabilities"], masks, device,
                seed=seed*100+outer_fold, epochs=args.risk_epochs, batch_size=args.batch_size,
                rank_weight=args.rank_weight, safe_weight=args.safe_weight, sign_weight=args.sign_weight,
            )
            selection_mean, selection_sigma, _ = predict_risk(
                risk, splits["selection"], selection_bundle["probabilities"], masks, device)
            tuned_policy, grid = tune_policy(selection_mean, selection_sigma, selection_bundle["losses"])
            # The paper's primary configuration is predeclared.  Tuning is an
            # explicit ablation and uses selection, never conformal/test labels.
            policy = tuned_policy if args.tune_policy else {
                "alpha": args.alpha, "margin": args.margin,
                "source": "predeclared_primary_configuration",
            }
            calibration_mean, calibration_sigma, _ = predict_risk(
                risk, splits["calibration"], calibration_bundle["probabilities"], masks, device)
            conformal = fit_joint_conformal(calibration_mean, calibration_sigma,
                                            calibration_bundle["losses"], policy["alpha"],
                                            mode="improvement")
            condition_frames, condition_metrics = [], []
            conditions = stress_conditions(splits["test"], names) if args.stress_tests else (
                ("clean", splits["test"], (1,) * len(names)),
            )
            for condition, condition_split, available_mask in conditions:
                # Deployment path: probabilities -> risk -> selection uses no label.
                test_outputs = predict_outputs(backbone, condition_split["x"], device, temperatures)
                availability = np.repeat(np.asarray(available_mask, dtype=np.float32)[None],
                                         len(condition_split["y"]), axis=0)
                test_mean, test_sigma, _ = predict_risk(
                    risk, condition_split, test_outputs["probabilities"], masks, device, availability)
                # Labels are attached only after all label-free predictions exist.
                test_bundle = attach_observed_losses(test_outputs, condition_split["y"])
                condition_frame, condition_result = evaluate_method(
                    condition_split, test_bundle, test_mean, test_sigma, conformal,
                    policy["margin"], masks, names, seed, available_mask, condition)
                condition_result["condition"] = condition
                condition_frames.append(condition_frame); condition_metrics.append(condition_result)
            frame = condition_frames[0]
            metrics = condition_metrics[0]
            metrics.update({"dataset": args.dataset, "fold": outer_fold, "train_seed": seed,
                            "alpha": policy["alpha"], "margin": policy["margin"],
                            "conformal_quantile": conformal.quantile,
                            "n_conformal": conformal.n_calibration})
            frame.to_parquet(output / "predictions.parquet", index=False)
            pd.concat(condition_frames, ignore_index=True).to_parquet(
                output / "stress_predictions.parquet", index=False)
            pd.DataFrame(condition_metrics).to_csv(output / "stress_metrics.csv", index=False)
            torch.save({"state_dict": backbone.state_dict(), "dims": dims, "classes": classes,
                        "temperatures": temperatures}, output / "backbone.pt")
            torch.save({"state_dict": risk.state_dict(), "dims": dims, "classes": classes}, output / "risk_predictor.pt")
            write_json(output / "training.json", {"backbone": backbone_history, "risk": history,
                                                    "policy": policy, "policy_grid": grid})
            write_json(output / "target_audit.json", target_audit)
            write_json(output / "conformal.json", conformal.__dict__)
            write_json(output / "metrics.json", metrics)
            write_json(output / "complete.json", {"n_test": len(frame)})
            all_metrics.append(metrics)
            print(f"{args.dataset} fold={outer_fold} seed={seed}: regret reduction "
                  f"{metrics['regret_reduction_fraction']:.3f}, switch {metrics['switch_rate']:.3f}", flush=True)
    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(root / "metrics_by_seed.csv", index=False)
    numeric = metrics_frame.select_dtypes(include=[np.number]).columns.difference(["fold", "train_seed"])
    pd.DataFrame({"metric": numeric, "mean": [metrics_frame[c].mean() for c in numeric],
                  "std": [metrics_frame[c].std(ddof=1) for c in numeric]}).to_csv(root / "metrics_summary.csv", index=False)
    write_go_no_go(root, metrics_frame)
    write_json(root / "runtime.json", {"wall_seconds": time.perf_counter()-start})


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate RCG-Fusion")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist"), required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--teacher-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--risk-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rank-weight", type=float, choices=(.25, .5, 1.), default=.5)
    parser.add_argument("--safe-weight", type=float, choices=(.5, 1., 2.), default=1.)
    parser.add_argument("--sign-weight", type=float, choices=(.25, .5, 1.), default=.5)
    parser.add_argument("--alpha", type=float, default=.10)
    parser.add_argument("--margin", type=float, default=.01)
    parser.add_argument("--tune-policy", action="store_true",
                        help="selection-set ablation; primary runs keep alpha=.10 and margin=.01")
    parser.add_argument("--stress-tests", action=argparse.BooleanOptionalAction, default=True,
                        help="evaluate Gaussian noise, feature masking and whole-modality missingness")
    return parser


def main():
    args = build_parser().parse_args()
    if args.folds < 2 or args.epochs < 1 or args.risk_epochs < 1:
        raise ValueError("folds >= 2 and positive epochs are required")
    if not 0 < args.alpha < 1 or args.margin < 0:
        raise ValueError("alpha must be in (0,1) and margin must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
