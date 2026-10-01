"""Train and evaluate posterior-structured benefit fusion over a frozen RCG run."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.nn import functional as F

from .benefit_fusion import (PosteriorStructuredBenefitEstimator, apply_crc,
                             benefit_objective, crc_risk_curve, ensemble_candidate,
                             fit_crc)
from .io import load_json, write_json
from .rcg_fusion import CoalitionAwareBackbone, nonempty_coalitions
from .rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                  predict_outputs, seed_all)


DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def _tensors(split, device):
    return ([torch.as_tensor(x, dtype=torch.float32, device=device) for x in split["x"]],
            torch.as_tensor(split["y"], dtype=torch.long, device=device))


def _model_forward(model, xs, probabilities, masks, no_posterior=False):
    override = probabilities[:, -1] if no_posterior else None
    return model(xs, probabilities, masks, posterior_override=override)


def train_estimator(model, train_split, oof_probabilities, oof_losses,
                    selection_split, selection_probabilities, selection_losses,
                    masks, device, *, seed, epochs=100, batch_size=128,
                    patience=10, no_posterior=False, teacher_mode="all"):
    seed_all(seed)
    train_x, train_y = _tensors(train_split, device)
    selection_x, selection_y = _tensors(selection_split, device)
    oof_p = torch.as_tensor(oof_probabilities, dtype=torch.float32, device=device)
    oof_l = torch.as_tensor(oof_losses, dtype=torch.float32, device=device)
    selection_p = torch.as_tensor(selection_probabilities, dtype=torch.float32, device=device)
    selection_l = torch.as_tensor(selection_losses, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(masks, dtype=torch.float32, device=device)
    teacher_count = 1 if teacher_mode == "single" else len(oof_p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train(); train_values = []
        for sample in torch.randperm(len(train_y), device=device).split(batch_size):
            # All teacher contexts for one original sample are averaged in a
            # single optimizer step.  This enforces total per-sample weight 1/T
            # without silently multiplying the effective epoch length by T.
            teacher = torch.arange(teacher_count, device=device)[:, None].expand(-1, len(sample)).reshape(-1)
            repeated_sample = sample[None].expand(teacher_count, -1).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            output = _model_forward(model, [x[repeated_sample] for x in train_x],
                                    oof_p[teacher, repeated_sample],
                                    mask_tensor, no_posterior)
            loss, _ = benefit_objective(output, train_y[repeated_sample],
                                        oof_l[teacher, repeated_sample],
                                        posterior_weight=0. if no_posterior else 1.)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            train_values.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            output = _model_forward(model, selection_x, selection_p, mask_tensor, no_posterior)
            value, parts = benefit_objective(output, selection_y, selection_l,
                                             posterior_weight=0. if no_posterior else 1.)
        record = {"epoch": epoch+1, "train": float(np.mean(train_values)), "selection": float(value),
                  **{f"selection_{key}": float(item) for key, item in parts.items()}}
        history.append(record)
        if float(value) < best-1e-6:
            best, state, stale = float(value), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state); model.eval()
    return history


@torch.inference_mode()
def predict_estimator(model, split, probabilities, masks, device, no_posterior=False):
    xs, _ = _tensors(split, device)
    probability = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(masks, dtype=torch.float32, device=device)
    output = _model_forward(model, xs, probability, mask_tensor, no_posterior)
    return {key: value.cpu().numpy() for key, value in output.items()}


def tune_kappa(outputs, losses, masks):
    oracle = losses.min(1); rows = np.arange(len(losses)); choices = []
    for kappa in (0., .5, 1.):
        candidate = ensemble_candidate(outputs, masks, kappa)
        selected_loss = losses[rows, candidate["candidate"]]
        observed_benefit = losses[:, -1]-selected_loss
        value_offset = float((observed_benefit-candidate["candidate_value"]).mean())
        choices.append({"kappa": kappa, "candidate_regret": float((selected_loss-oracle).mean()),
                        "candidate_vs_full": float((selected_loss-losses[:, -1]).mean()),
                        "value_offset": value_offset})
    return min(choices, key=lambda x: (x["candidate_regret"], x["kappa"])), choices


def apply_value_offset(candidate, offset):
    candidate = {key: value.copy() if isinstance(value, np.ndarray) else value
                 for key, value in candidate.items()}
    candidate["candidate_value"] += float(offset)
    candidate["mean_net"] += float(offset)
    return candidate


def candidate_metrics(outputs, candidate, losses, labels, posterior, tau=.01):
    rows = np.arange(len(losses)); full = losses.shape[1]-1
    observed_all = losses[:, full, None]-losses
    expected = np.mean([x["expected_benefit"] for x in outputs], 0)
    safe_probability = np.mean([x["safe_probability"] for x in outputs], 0)
    keep = np.arange(losses.shape[1]) != full
    safe_all = observed_all[:, keep] > tau
    result = {
        "benefit_spearman": float(spearmanr(expected[:, keep].ravel(), observed_all[:, keep].ravel()).statistic),
        "posterior_accuracy": float(accuracy_score(labels, posterior.argmax(1))),
        "all_pair_safe_auroc": float(roc_auc_score(safe_all.ravel(), safe_probability[:, keep].ravel())),
        "all_pair_safe_auprc": float(average_precision_score(safe_all.ravel(), safe_probability[:, keep].ravel())),
    }
    chosen = candidate["candidate"]
    benefit = observed_all[rows, chosen]
    score = candidate["candidate_safe_probability"]
    safe = benefit > tau
    result["candidate_safe_auroc"] = float(roc_auc_score(safe, score)) if len(np.unique(safe)) == 2 else np.nan
    result["candidate_safe_auprc"] = float(average_precision_score(safe, score)) if len(np.unique(safe)) == 2 else np.nan
    top = score >= np.quantile(score, .8)
    result["top20_safe_precision"] = float(safe[top].mean())
    candidate_loss = losses[rows, chosen]; oracle = losses.min(1); full_regret = losses[:, full]-oracle
    candidate_regret = candidate_loss-oracle
    result["ungated_candidate_regret"] = float(candidate_regret.mean())
    result["full_regret"] = float(full_regret.mean())
    result["ungated_regret_reduction"] = float(1-candidate_regret.mean()/max(full_regret.mean(), 1e-12))
    result["ungated_candidate_vs_full"] = float((candidate_loss-losses[:, full]).mean())
    return result, benefit


def evaluate_policy(split, probabilities, losses, outputs, candidate, calibration, masks):
    rows = np.arange(len(losses)); full = len(masks)-1
    decision = apply_crc(candidate, calibration, full)
    selected = decision["selected"]; switch = decision["switch"]
    selected_loss = losses[rows, selected]; full_loss = losses[:, full]; oracle_loss = losses.min(1)
    selected_probability = probabilities[rows, selected]; full_probability = probabilities[:, full]
    observed_benefit = full_loss-losses[rows, candidate["candidate"]]
    harmful = switch & (observed_benefit <= 0)
    clipped_harm = switch*np.minimum(np.maximum(-observed_benefit, 0), 1)
    full_prediction, selected_prediction = full_probability.argmax(1), selected_probability.argmax(1)
    label = split["y"]
    metrics = {
        "switch_rate": float(switch.mean()),
        "overall_harmful_switch_risk": float(harmful.mean()),
        "conditional_harmful_switch_rate": float(harmful.sum()/switch.sum()) if switch.any() else np.nan,
        "overall_clipped_harm": float(clipped_harm.mean()),
        "safe_switch_precision": float((observed_benefit[switch] > .01).mean()) if switch.any() else np.nan,
        "mean_full_regret": float((full_loss-oracle_loss).mean()),
        "mean_selection_regret": float((selected_loss-oracle_loss).mean()),
        "regret_reduction": float(1-(selected_loss-oracle_loss).mean()/max((full_loss-oracle_loss).mean(), 1e-12)),
        "accuracy": float(accuracy_score(label, selected_prediction)),
        "macro_f1": float(f1_score(label, selected_prediction, average="macro")),
        "nll": float(selected_loss.mean()),
        "full_accuracy": float(accuracy_score(label, full_prediction)),
        "full_macro_f1": float(f1_score(label, full_prediction, average="macro")),
        "full_nll": float(full_loss.mean()),
        "negative_flip_rate": float(((full_prediction == label) & (selected_prediction != label)).mean()),
        "correction_rate": float(((full_prediction != label) & (selected_prediction == label)).mean()),
    }
    frame = pd.DataFrame({
        "sample_id": split["id"], "group_or_video_id": split["group"], "label": label,
        "candidate": candidate["candidate"], "selected": selected, "switch": switch,
        "candidate_value": candidate["candidate_value"],
        "candidate_safe_probability": candidate["candidate_safe_probability"],
        "observed_candidate_benefit": observed_benefit,
        "full_loss": full_loss, "selected_loss": selected_loss, "oracle_loss": oracle_loss,
    })
    for k in range(selected_probability.shape[1]):
        frame[f"final_p{k}"] = selected_probability[:, k]
    return frame, metrics


def _load_backbone(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CoalitionAwareBackbone(checkpoint["dims"], checkpoint["classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model, np.asarray(checkpoint["temperatures"]), checkpoint["dims"], checkpoint["classes"]


def write_report(root, metrics):
    mean = metrics.select_dtypes(include=[np.number]).mean()
    ungated_positive = int((metrics.ungated_regret_reduction > 0).sum())
    checks = {
        "candidate_safe_auroc": mean.candidate_safe_auroc >= .70,
        "top20_safe_precision": mean.top20_safe_precision >= .80,
        "benefit_spearman": mean.benefit_spearman >= .50,
        "ungated_positive_seeds": ungated_positive >= 3,
        "switch_rate": mean.switch_rate >= .10,
        "event_risk": mean.overall_harmful_switch_risk <= .05,
        "harm_risk": mean.overall_clipped_harm <= .02,
        "regret": mean.regret_reduction >= .20,
        "accuracy": mean.accuracy >= mean.full_accuracy-.005,
    }
    lines = ["# Posterior-Structured Benefit + CRC Go/No-Go", "",
             f"结论：**{'通过' if all(checks.values()) else '未通过'}**。", "",
             "| 检查 | 实际 | 通过 |", "|---|---:|---|",
             f"| 候选安全 AUROC ≥ 0.70 | {mean.candidate_safe_auroc:.4f} | {checks['candidate_safe_auroc']} |",
             f"| Top-20% 安全精确率 ≥ 0.80 | {mean.top20_safe_precision:.4f} | {checks['top20_safe_precision']} |",
             f"| 收益 Spearman ≥ 0.50 | {mean.benefit_spearman:.4f} | {checks['benefit_spearman']} |",
             f"| 未门控候选至少3种子降后悔 | {ungated_positive}/5 | {checks['ungated_positive_seeds']} |",
             f"| 切换覆盖率 ≥ 0.10 | {mean.switch_rate:.4f} | {checks['switch_rate']} |",
             f"| 总体有害切换 ≤ 0.05 | {mean.overall_harmful_switch_risk:.4f} | {checks['event_risk']} |",
             f"| 总体截断损害 ≤ 0.02 | {mean.overall_clipped_harm:.4f} | {checks['harm_risk']} |",
             f"| 后悔下降 ≥ 0.20 | {mean.regret_reduction:.4f} | {checks['regret']} |",
             f"| Accuracy下降≤0.5pp | {(mean.accuracy-mean.full_accuracy):.4f} | {checks['accuracy']} |"]
    (root/"GO_NO_GO.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset(args.dataset, args.data)
    base = Path(args.base_run); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": args.dataset, "base_run": str(base.resolve()), "seeds": args.seeds,
                "benefit_seeds": args.benefit_seeds, "teacher_mode": args.teacher_mode,
                "no_posterior": args.no_posterior, "no_relations": args.no_relations,
                "no_structured": args.no_structured, "alpha_event": .05, "alpha_harm": .02}
    if (root/"manifest.json").exists() and load_json(root/"manifest.json") != manifest:
        raise ValueError("manifest differs; use a new output directory")
    write_json(root/"manifest.json", manifest)
    all_metrics = []; start = time.perf_counter()
    for fold_index, splits in enumerate(folds):
        fold_base = base/f"fold_{fold_index}"
        with np.load(fold_base/"oof_targets.npz") as saved:
            oof = {key: saved[key] for key in saved.files}
        masks = nonempty_coalitions(len(names)); fold_out = root/f"fold_{fold_index}"; fold_out.mkdir(exist_ok=True)
        oof_benefit = (oof["losses"][:, :, -1, None]-oof["losses"][:, :, :-1]).clip(-1, 1)
        output_priors = ((oof_benefit > .01).mean(), np.maximum(oof_benefit, 0).mean(),
                         np.maximum(-oof_benefit, 0).mean())
        for task_seed in args.seeds:
            out = fold_out/f"seed_{task_seed}"; out.mkdir(exist_ok=True)
            if (out/"complete.json").exists():
                all_metrics.append(load_json(out/"metrics.json")); continue
            backbone, temperatures, dims, classes = _load_backbone(
                fold_base/f"seed_{task_seed}"/"backbone.pt", device)
            bundles = {}
            for split_name in ("selection", "calibration", "test"):
                outputs = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
                bundles[split_name] = attach_observed_losses(outputs, splits[split_name]["y"])
            estimators, histories = [], []
            for estimator_seed in args.benefit_seeds:
                seed_all(task_seed*1000+estimator_seed)
                model = PosteriorStructuredBenefitEstimator(
                    dims, classes, len(masks), use_relations=not args.no_relations,
                    use_structured=not args.no_structured, output_priors=output_priors).to(device)
                history = train_estimator(
                    model, splits["train"], oof["probabilities"], oof["losses"],
                    splits["selection"], bundles["selection"]["probabilities"],
                    bundles["selection"]["losses"], masks, device,
                    seed=task_seed*1000+estimator_seed, epochs=args.epochs,
                    batch_size=args.batch_size, no_posterior=args.no_posterior,
                    teacher_mode=args.teacher_mode)
                estimators.append(model); histories.append(history)
            predictions = {split_name: [predict_estimator(
                model, splits[split_name], bundles[split_name]["probabilities"], masks, device,
                args.no_posterior) for model in estimators] for split_name in bundles}
            policy, kappa_grid = tune_kappa(predictions["selection"], bundles["selection"]["losses"], masks)
            kappa = policy["kappa"]
            candidates = {name: apply_value_offset(ensemble_candidate(value, masks, kappa),
                                                    policy["value_offset"])
                          for name, value in predictions.items()}
            calibration_rows = np.arange(len(splits["calibration"]["y"])); full = len(masks)-1
            cal_candidate = candidates["calibration"]["candidate"]
            cal_benefit = (bundles["calibration"]["losses"][:, full]
                           - bundles["calibration"]["losses"][calibration_rows, cal_candidate])
            crc, calibration_curve = fit_crc(
                candidates["calibration"]["candidate_safe_probability"],
                candidates["calibration"]["candidate_value"] > 0, cal_benefit)
            posterior = np.mean([x["posterior"] for x in predictions["test"]], 0)
            innovation2, _ = candidate_metrics(
                predictions["test"], candidates["test"], bundles["test"]["losses"],
                splits["test"]["y"], posterior)
            frame, policy_metrics = evaluate_policy(
                splits["test"], bundles["test"]["probabilities"], bundles["test"]["losses"],
                predictions["test"], candidates["test"], crc, masks)
            metrics = {"dataset": args.dataset, "fold": fold_index, "train_seed": task_seed,
                       "kappa": kappa, "crc_threshold": crc.threshold,
                       "candidate_value_offset": policy["value_offset"],
                       "crc_calibration_event_bound": crc.event_bound,
                       "crc_calibration_harm_bound": crc.harm_bound,
                       **innovation2, **policy_metrics}
            test_rows = np.arange(len(frame)); test_candidate = candidates["test"]["candidate"]
            test_benefit = (bundles["test"]["losses"][:, full]
                            - bundles["test"]["losses"][test_rows, test_candidate])
            pd.DataFrame(crc_risk_curve(
                candidates["test"]["candidate_safe_probability"],
                candidates["test"]["candidate_value"] > 0, test_benefit)).to_csv(
                    out/"test_risk_coverage_curve.csv", index=False)
            pd.DataFrame(calibration_curve).to_csv(out/"calibration_risk_curve.csv", index=False)
            frame.to_parquet(out/"predictions.parquet", index=False)
            write_json(out/"metrics.json", metrics)
            write_json(out/"crc.json", crc.__dict__)
            write_json(out/"training.json", {"histories": histories, "kappa_grid": kappa_grid})
            for estimator_seed, model in zip(args.benefit_seeds, estimators):
                torch.save({"state_dict": model.state_dict(), "dims": dims, "classes": classes,
                            "n_coalitions": len(masks), "output_priors": output_priors},
                           out/f"benefit_{estimator_seed}.pt")
            write_json(out/"complete.json", {"n_test": len(frame)})
            all_metrics.append(metrics)
            print(f"{args.dataset} seed={task_seed}: benefit_auc={metrics['candidate_safe_auroc']:.3f}, "
                  f"switch={metrics['switch_rate']:.3f}, regret_reduction={metrics['regret_reduction']:.3f}", flush=True)
    table = pd.DataFrame(all_metrics); table.to_csv(root/"metrics_by_seed.csv", index=False)
    numeric = table.select_dtypes(include=[np.number]).columns.difference(["fold", "train_seed"])
    pd.DataFrame({"metric": numeric, "mean": [table[c].mean() for c in numeric],
                  "std": [table[c].std(ddof=1) for c in numeric]}).to_csv(root/"metrics_summary.csv", index=False)
    write_report(root, table); write_json(root/"runtime.json", {"wall_seconds": time.perf_counter()-start})


def main():
    parser = argparse.ArgumentParser(description="Posterior-structured benefit estimation with CRC fusion")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist"), default="mosi")
    parser.add_argument("--data", default="data")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--benefit-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-mode", choices=("all", "single"), default="all")
    parser.add_argument("--no-posterior", action="store_true")
    parser.add_argument("--no-relations", action="store_true")
    parser.add_argument("--no-structured", action="store_true")
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
