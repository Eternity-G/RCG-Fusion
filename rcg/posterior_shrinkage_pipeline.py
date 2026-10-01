"""Evaluate bounded posterior-residual fusion over frozen RCG models."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .io import load_json, write_json
from .posterior_analytic_pipeline import _load_backbone
from .posterior_shrinkage import (contribution_adaptive_shrinkage,
                                  posterior_risk_certificate,
                                  select_alpha_by_nll,
                                  worst_case_loss_increase)
from .projected_fusion_pipeline import load_posteriors, observed_loss
from .rcg_fusion import nonempty_coalitions
from .rcg_fusion_pipeline import attach_observed_losses, load_dataset, predict_outputs


DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def probability_metrics(probability, labels):
    labels = np.asarray(labels); rows = np.arange(len(labels))
    prediction = probability.argmax(1); confidence = probability.max(1)
    correct = prediction == labels
    brier = ((probability-np.eye(probability.shape[1])[labels])**2).sum(1).mean()
    ece = 0.
    for lower in np.linspace(0, .9, 10):
        selected = (confidence >= lower) & (confidence < lower+.1 if lower < .9 else confidence <= 1)
        if selected.any():
            ece += selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    return {"accuracy": float(correct.mean()),
            "macro_f1": float(f1_score(labels, prediction, average="macro")),
            "nll": float(-np.log(np.clip(probability[rows, labels], 1e-12, 1.)).mean()),
            "brier": float(brier), "ece": float(ece)}, prediction


def cluster_bootstrap(records, repetitions=10000, seed=20260929):
    frame = pd.concat(records, ignore_index=True); groups = frame.group_or_video_id.unique()
    grouped = {group: frame.loc[frame.group_or_video_id == group,
                                ["nll_difference", "correct_difference"]].to_numpy()
               for group in groups}
    rng = np.random.default_rng(seed); nll = np.empty(repetitions); accuracy = np.empty(repetitions)
    for index in range(repetitions):
        sampled = rng.choice(groups, len(groups), replace=True)
        values = np.concatenate([grouped[group] for group in sampled])
        nll[index], accuracy[index] = values.mean(0)
    return {"nll_difference": float(frame.nll_difference.mean()),
            "nll_ci_low": float(np.quantile(nll, .025)),
            "nll_ci_high": float(np.quantile(nll, .975)),
            "accuracy_difference": float(frame.correct_difference.mean()),
            "accuracy_ci_low": float(np.quantile(accuracy, .025)),
            "accuracy_ci_high": float(np.quantile(accuracy, .975)),
            "groups": int(len(groups)), "repetitions": repetitions}


def write_report(root, table, paired, alpha_policy, alpha_bound, dataset):
    mean = table.select_dtypes(include=[np.number]).mean()
    nll_seeds = int((table.shrink_nll < table.full_nll).sum())
    accuracy_gain = mean.shrink_accuracy-mean.full_accuracy
    if dataset == "mosi":
        checks = {"nll_seeds": nll_seeds == len(table),
                  "accuracy": (accuracy_gain >= .005
                               and paired["accuracy_ci_low"] > 0),
                  "harm": mean.clipped_harm <= .02,
                  "flips": mean.correction_rate > mean.negative_flip_rate}
        context = f"MOSI is the development dataset; {alpha_policy}."
        conclusion = "通过，可进入无偏确认" if all(checks.values()) else "未完全通过"
    else:
        # Confirmation focuses on proper-score improvement and bounded harm.  The
        # alpha and these checks are fixed before inspecting confirmation results.
        checks = {"nll_direction": nll_seeds >= max(1, len(table)-1),
                  "nll_cluster_ci": paired["nll_ci_high"] < 0,
                  "harm": mean.clipped_harm <= .02,
                  "flips": mean.correction_rate >= mean.negative_flip_rate}
        context = (f"{dataset.upper()} cross-dataset evaluation; {alpha_policy}. "
                   "Whether this is an unbiased confirmation depends on the recorded protocol history.")
        conclusion = "跨数据集数值门槛通过" if all(checks.values()) else "跨数据集数值门槛未完全通过"
    lines = ["# Bounded Posterior Shrinkage Go/No-Go", "",
             context, "", f"结论：**{conclusion}**。", "",
             "| 检查 | 实际 | 通过 |", "|---|---:|---|"]
    if dataset == "mosi":
        lines.extend([
            f"| NLL改善种子 | {nll_seeds}/{len(table)} | {checks['nll_seeds']} |",
            f"| Accuracy提升≥0.5pp且簇CI>0 | {accuracy_gain:.4f} [{paired['accuracy_ci_low']:.4f}, {paired['accuracy_ci_high']:.4f}] | {checks['accuracy']} |"])
    else:
        lines.extend([
            f"| NLL改善种子≥{max(1, len(table)-1)}/{len(table)} | {nll_seeds}/{len(table)} | {checks['nll_direction']} |",
            f"| NLL配对簇CI上界<0 | [{paired['nll_ci_low']:.6f}, {paired['nll_ci_high']:.6f}] | {checks['nll_cluster_ci']} |",
            f"| Accuracy差（描述性） | {accuracy_gain:.4f} [{paired['accuracy_ci_low']:.4f}, {paired['accuracy_ci_high']:.4f}] | — |"])
    lines.extend([
        f"| 平均截断损害≤2% | {mean.clipped_harm:.4f} | {checks['harm']} |",
        f"| 纠错率≥负向翻转率 | {mean.correction_rate:.4f}/{mean.negative_flip_rate:.4f} | {checks['flips']} |",
        "", f"NLL差为 {paired['nll_difference']:.6f}，簇CI为 [{paired['nll_ci_low']:.6f}, {paired['nll_ci_high']:.6f}]。",
        f"最大启用 alpha={alpha_bound:.2f}；单样本CE增量上界为 "
        f"{-np.log1p(-alpha_bound):.4f} nats。"])
    (root/"GO_NO_GO.md").write_text("\n".join(lines), encoding="utf-8")
    return checks


def run(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset(args.dataset, args.data)
    masks = nonempty_coalitions(len(names))
    base, posterior_run, root = Path(args.base_run), Path(args.posterior_run), Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    alpha_grid = sorted(set(float(value) for value in args.alpha_grid))
    if args.select_alpha and args.adaptive_gamma is not None:
        raise ValueError("--select-alpha and --adaptive-gamma are mutually exclusive")
    alpha_bound = max(alpha_grid) if args.select_alpha else args.alpha
    manifest = {"dataset": args.dataset, "base_run": str(base.resolve()),
                "posterior_run": str(posterior_run.resolve()), "seeds": args.seeds,
                "alpha": args.alpha, "select_alpha": args.select_alpha,
                "adaptive_gamma": args.adaptive_gamma,
                "alpha_grid": alpha_grid,
                "worst_case_ce_increase": worst_case_loss_increase(alpha_bound),
                "protocol": ("contribution-adaptive bounded posterior shrinkage"
                             if args.adaptive_gamma is not None else
                             "selection-tuned bounded posterior shrinkage"
                             if args.select_alpha else
                             "fixed bounded posterior shrinkage"),
                "outer_folds": len(folds),
                "test_labels_role": "evaluation only"}
    if (root/"manifest.json").exists() and load_json(root/"manifest.json") != manifest:
        raise ValueError("manifest differs; use a new output directory")
    write_json(root/"manifest.json", manifest)
    metrics, records, start = [], [], time.perf_counter()
    for fold_index, splits in enumerate(folds):
      fold_out = root if len(folds) == 1 else root/f"fold_{fold_index}"
      fold_out.mkdir(exist_ok=True)
      for task_seed in args.seeds:
        out = fold_out/f"seed_{task_seed}"; out.mkdir(exist_ok=True)
        backbone, temperatures, dims, classes = _load_backbone(
            base/f"fold_{fold_index}"/f"seed_{task_seed}"/"backbone.pt", device)
        selected_alpha, alpha_scores = args.alpha, None
        if args.select_alpha:
            selection_raw = predict_outputs(
                backbone, splits["selection"]["x"], device, temperatures)
            selection_bundle = attach_observed_losses(
                selection_raw, splits["selection"]["y"])
            _, selection_posterior = load_posteriors(
                posterior_run, task_seed, dims, classes, masks,
                splits["selection"], selection_bundle["probabilities"], device,
                fold_index=fold_index)
            selected_alpha, alpha_scores = select_alpha_by_nll(
                selection_bundle["probabilities"][:, -1], selection_posterior,
                splits["selection"]["y"], alpha_grid)
            write_json(out/"alpha_selection.json",
                       {"selected_alpha": selected_alpha,
                        "selection_nll": {str(k): v for k, v in alpha_scores.items()},
                        "split": "selection"})
        raw = predict_outputs(backbone, splits["test"]["x"], device, temperatures)
        bundle = attach_observed_losses(raw, splits["test"]["y"])
        _, posterior = load_posteriors(
            posterior_run, task_seed, dims, classes, masks, splits["test"],
            bundle["probabilities"], device, fold_index=fold_index)
        full = bundle["probabilities"][:, -1]
        full_tensor = torch.as_tensor(full, dtype=torch.float64)
        posterior_tensor = torch.as_tensor(posterior, dtype=torch.float64)
        if args.adaptive_gamma is None:
            certificate = posterior_risk_certificate(
                full_tensor, posterior_tensor, selected_alpha)
            sample_alpha = np.full(len(full), selected_alpha)
            safe_probability = np.full(len(full), np.nan)
        else:
            certificate = contribution_adaptive_shrinkage(
                full_tensor, posterior_tensor, args.alpha,
                args.adaptive_gamma, args.benefit_tolerance)
            sample_alpha = certificate["alpha"].numpy()
            safe_probability = certificate["safe_probability"].numpy()
        shrink = certificate["probability"].numpy(); labels = splits["test"]["y"]
        full_metrics, full_prediction = probability_metrics(full, labels)
        shrink_metrics, shrink_prediction = probability_metrics(shrink, labels)
        full_loss = bundle["losses"][:, -1]; shrink_loss = observed_loss(shrink, labels)
        benefit = full_loss-shrink_loss; oracle_loss = bundle["losses"].min(1)
        negative = (full_prediction == labels) & (shrink_prediction != labels)
        correction = (full_prediction != labels) & (shrink_prediction == labels)
        full_regret = (full_loss-oracle_loss).mean()
        item = {"dataset": args.dataset, "fold": fold_index,
                "train_seed": task_seed, "n_test": len(labels),
                "alpha": float(sample_alpha.mean()),
                "max_alpha": float(sample_alpha.max()),
                **{f"full_{key}": value for key, value in full_metrics.items()},
                **{f"shrink_{key}": value for key, value in shrink_metrics.items()},
                "mean_benefit": float(benefit.mean()),
                "harmful_event_rate": float((benefit <= 0).mean()),
                "clipped_harm": float(np.minimum(np.maximum(-benefit, 0), 1).mean()),
                "negative_flip_rate": float(negative.mean()),
                "correction_rate": float(correction.mean()),
                "mean_full_regret": float(full_regret),
                "mean_shrink_regret": float((shrink_loss-oracle_loss).mean()),
                "regret_reduction": float(
                    1-(shrink_loss-oracle_loss).mean()/max(full_regret, 1e-12)),
                "posterior_expected_gain": float(
                    certificate["posterior_expected_gain"].mean()),
                "convexity_lower_bound": float(
                    certificate["convexity_lower_bound"].mean())}
        metrics.append(item); write_json(out/"metrics.json", item)
        frame = pd.DataFrame({"sample_id": splits["test"]["id"],
                              "group_or_video_id": splits["test"]["group"],
                              "fold": fold_index, "train_seed": task_seed,
                              "label": labels, "full_loss": full_loss,
                              "shrink_loss": shrink_loss, "benefit": benefit,
                              "nll_difference": shrink_loss-full_loss,
                              "correct_difference": correction.astype(int)-negative.astype(int)})
        frame["alpha"] = sample_alpha
        frame["analytic_safe_probability"] = safe_probability
        for k in range(classes):
            frame[f"full_p{k}"] = full[:, k]; frame[f"posterior_p{k}"] = posterior[:, k]
            frame[f"shrink_p{k}"] = shrink[:, k]
        frame.to_parquet(out/"predictions.parquet", index=False)
        records.append(frame[["group_or_video_id", "nll_difference", "correct_difference"]])
        print(f"{args.dataset} seed={task_seed}: acc_delta="
              f"{item['shrink_accuracy']-item['full_accuracy']:.3f}, "
              f"nll_delta={item['shrink_nll']-item['full_nll']:.4f}, "
              f"harm={item['clipped_harm']:.3f}", flush=True)
    fold_table = pd.DataFrame(metrics)
    fold_table.to_csv(root/"metrics_by_fold_seed.csv", index=False)
    numeric = fold_table.select_dtypes(include=[np.number]).columns.difference(
        ["fold", "train_seed", "n_test"])
    # Aggregate outer folds within each training seed.  This preserves five
    # independent training replicates rather than treating 5 folds x 5 seeds as
    # 25 independent seeds.
    seed_rows = []
    for task_seed, values in fold_table.groupby("train_seed", sort=True):
        weights = values["n_test"].to_numpy(dtype=float)
        row = {"dataset": args.dataset, "train_seed": int(task_seed),
               "n_test": int(weights.sum())}
        row.update({column: float(np.average(values[column], weights=weights))
                    for column in numeric})
        seed_rows.append(row)
    table = pd.DataFrame(seed_rows)
    table.to_csv(root/"metrics_by_seed.csv", index=False)
    summary_numeric = table.select_dtypes(include=[np.number]).columns.difference(
        ["train_seed", "n_test"])
    pd.DataFrame({"metric": summary_numeric,
                  "mean": [table[c].mean() for c in summary_numeric],
                  "std": [table[c].std(ddof=1) for c in summary_numeric]}).to_csv(
                      root/"metrics_summary.csv", index=False)
    paired = cluster_bootstrap(records); write_json(root/"paired_bootstrap.json", paired)
    if args.adaptive_gamma is not None:
        alpha_policy = (f"alpha(x)={args.alpha:.2f} pi(x)^{args.adaptive_gamma:g} "
                        "is computed without labels")
    elif args.select_alpha:
        alpha_policy = f"alpha is selected per task seed on the selection split from {alpha_grid}"
    else:
        alpha_policy = f"alpha={args.alpha:.2f} is fixed"
    checks = write_report(root, table, paired, alpha_policy,
                          float(alpha_bound), args.dataset)
    write_json(root/"complete.json", {"checks": checks, "passed": bool(all(checks.values())),
                                      "wall_seconds": time.perf_counter()-start})


def main():
    parser = argparse.ArgumentParser(description="Bounded posterior-residual fusion")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist"), default="mosi")
    parser.add_argument("--data", default="data")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--posterior-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=.25)
    parser.add_argument("--select-alpha", action="store_true",
                        help="select alpha by selection NLL for each task seed")
    parser.add_argument("--alpha-grid", type=float, nargs="+",
                        default=[0., .25, .5, .75])
    parser.add_argument("--adaptive-gamma", type=float,
                        help="use alpha(x)=alpha*pi(x)^gamma instead of a global alpha")
    parser.add_argument("--benefit-tolerance", type=float, default=.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
