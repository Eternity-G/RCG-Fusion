"""Stage-A evaluation of multi-coalition posterior projection with CRC."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .benefit_fusion import crc_risk_curve, fit_crc
from .io import load_json, write_json
from .posterior_analytic import (RelationalPosteriorEstimator, analytic_candidate)
from .posterior_analytic_pipeline import (_load_backbone, calibrate_members,
                                          predict_posterior)
from .projected_fusion import (analytic_action, build_projection, select_alpha,
                               shrink_probability)
from .rcg_fusion import nonempty_coalitions
from .rcg_fusion_pipeline import attach_observed_losses, load_dataset, predict_outputs


DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def load_posteriors(run, task_seed, dims, classes, masks, split, probabilities, device,
                    fold_index=0):
    """Load the posterior ensemble for one outer evaluation fold.

    ``fold_index`` used to be implicitly fixed to zero.  Keeping it explicit is
    required for grouped outer-fold datasets such as CREMA-D, where every actor
    is evaluated in exactly one held-out fold.
    """
    seed_root = Path(run)/f"fold_{fold_index}"/f"seed_{task_seed}"
    members = []
    for path in sorted(seed_root.glob("posterior_*.pt")):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = RelationalPosteriorEstimator(
            dims, classes, len(masks), use_relations=checkpoint["use_relations"]).to(device)
        model.load_state_dict(checkpoint["state_dict"]); model.eval()
        members.append(predict_posterior(model, split, probabilities, masks, device)["posterior"])
    if not members:
        raise FileNotFoundError(f"no posterior checkpoints in {seed_root}")
    temperature = load_json(seed_root/"metrics.json")["posterior_temperature"]
    calibrated = calibrate_members(members, temperature)
    return calibrated, np.mean(calibrated, axis=0)


def observed_loss(probability, labels):
    rows = np.arange(len(labels))
    return -np.log(np.clip(probability[rows, labels], 1e-12, 1.))


def evaluate_probabilities(probability, labels):
    loss = observed_loss(probability, labels); prediction = probability.argmax(1)
    return {"accuracy": float(accuracy_score(labels, prediction)),
            "macro_f1": float(f1_score(labels, prediction, average="macro")),
            "nll": float(loss.mean())}, loss


def apply_action(full_probability, action, calibration):
    activate = ((action["action_value"] > 0)
                & (action["safe_probability"] >= calibration.threshold))
    final = np.where(activate[:, None], action["action_probability"], full_probability)
    return final, activate


def hard_policy(candidate, probabilities, losses, calibration):
    rows = np.arange(len(losses)); full = losses.shape[1]-1
    activate = ((candidate["candidate_value"] > 0)
                & (candidate["candidate_safe_probability"] >= calibration.threshold))
    selected = np.where(activate, candidate["candidate"], full)
    return probabilities[rows, selected], activate


def projection_for_split(posterior, probabilities, coalition, alpha, top_k):
    return build_projection(
        posterior, probabilities, coalition["gain"], coalition["harm"],
        beta=2., top_k=top_k, alpha=alpha, iterations=50, learning_rate=.1)


def cluster_bootstrap_difference(records, repetitions=10000, seed=20260929):
    frame = pd.concat(records, ignore_index=True)
    groups = frame.group_or_video_id.unique(); rng = np.random.default_rng(seed)
    by_group = {group: frame.loc[frame.group_or_video_id == group, "loss_difference"].to_numpy()
                for group in groups}
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        estimates[index] = np.concatenate([by_group[group] for group in sampled]).mean()
    return {"mean_projected_minus_hard_nll": float(frame.loss_difference.mean()),
            "ci_low": float(np.quantile(estimates, .025)),
            "ci_high": float(np.quantile(estimates, .975)),
            "bootstrap_repetitions": repetitions,
            "groups": int(len(groups))}


def write_report(root, table, paired):
    mean = table.select_dtypes(include=[np.number]).mean()
    nonworse = int((table.final_nll <= table.full_nll+1e-12).sum())
    checks = {
        "coverage": mean.activation_rate >= .10,
        "event": mean.overall_harmful_action_risk <= .05,
        "harm": mean.overall_clipped_harm <= .02,
        "nll": nonworse >= 4,
        "task": (mean.final_accuracy >= mean.full_accuracy-.005
                 or mean.final_macro_f1 >= mean.full_macro_f1-.005),
        "regret": mean.regret_reduction >= .10,
        "hard": paired["ci_high"] < 0,
    }
    lines = ["# Projected Fusion Stage-A Go/No-Go", "",
             "MOSI is a development dataset for this method revision.", "",
             f"结论：**{'通过，可进入无偏确认' if all(checks.values()) else '未通过，停止跨数据集运行'}**。", "",
             "| 检查 | 实际 | 通过 |", "|---|---:|---|",
             f"| 激活覆盖率 ≥ 10% | {mean.activation_rate:.4f} | {checks['coverage']} |",
             f"| 总体有害动作风险 ≤ 5% | {mean.overall_harmful_action_risk:.4f} | {checks['event']} |",
             f"| 总体截断损害 ≤ 2% | {mean.overall_clipped_harm:.4f} | {checks['harm']} |",
             f"| NLL至少4/5种子不劣 | {nonworse}/5 | {checks['nll']} |",
             f"| Accuracy或Macro-F1下降≤0.5pp | {(mean.final_accuracy-mean.full_accuracy):.4f}/{(mean.final_macro_f1-mean.full_macro_f1):.4f} | {checks['task']} |",
             f"| 后悔下降 ≥ 10% | {mean.regret_reduction:.4f} | {checks['regret']} |",
             f"| 相对硬切换NLL显著改善 | {paired['mean_projected_minus_hard_nll']:.6f} [{paired['ci_low']:.6f}, {paired['ci_high']:.6f}] | {checks['hard']} |"]
    (root/"GO_NO_GO.md").write_text("\n".join(lines), encoding="utf-8")
    return checks


def run(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset(args.dataset, args.data)
    if len(folds) != 1:
        raise ValueError("Stage-A pipeline currently expects one prepared fold")
    splits = folds[0]; masks = nonempty_coalitions(len(names)); top_k = min(3, len(masks)-1)
    base, posterior_run, root = Path(args.base_run), Path(args.posterior_run), Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": args.dataset, "base_run": str(base.resolve()),
                "posterior_run": str(posterior_run.resolve()), "seeds": args.seeds,
                "top_k": top_k, "beta": 2., "iterations": 50, "learning_rate": .1,
                "alpha_candidates": [.25, .5, .75, 1.], "alpha_event": .05,
                "alpha_harm": .02, "protocol": "selection alpha -> calibration CRC -> test"}
    if (root/"manifest.json").exists() and load_json(root/"manifest.json") != manifest:
        raise ValueError("manifest differs; use a new output directory")
    write_json(root/"manifest.json", manifest)
    metrics, paired_records, start = [], [], time.perf_counter()
    for task_seed in args.seeds:
        out = root/f"seed_{task_seed}"; out.mkdir(exist_ok=True)
        backbone, temperatures, dims, classes = _load_backbone(
            base/"fold_0"/f"seed_{task_seed}"/"backbone.pt", device)
        bundles, posteriors, coalitions = {}, {}, {}
        for split_name in ("selection", "calibration", "test"):
            raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
            bundles[split_name] = attach_observed_losses(raw, splits[split_name]["y"])
            members, posterior = load_posteriors(
                posterior_run, task_seed, dims, classes, masks, splits[split_name],
                bundles[split_name]["probabilities"], device)
            posteriors[split_name] = posterior
            coalitions[split_name] = analytic_candidate(
                members, bundles[split_name]["probabilities"], masks, beta=2.)

        selection_projection = projection_for_split(
            posteriors["selection"], bundles["selection"]["probabilities"],
            coalitions["selection"], 1., top_k)
        alpha, alpha_table = select_alpha(
            splits["selection"]["y"], bundles["selection"]["probabilities"][:, -1],
            selection_projection["projected_probability"])
        actions = {name: projection_for_split(
            posteriors[name], bundles[name]["probabilities"], coalitions[name], alpha, top_k)
            for name in ("calibration", "test")}

        cal_labels = splits["calibration"]["y"]
        cal_full_loss = bundles["calibration"]["losses"][:, -1]
        cal_action_loss = observed_loss(actions["calibration"]["action_probability"], cal_labels)
        cal_benefit = cal_full_loss-cal_action_loss
        crc, curve = fit_crc(
            actions["calibration"]["safe_probability"],
            actions["calibration"]["action_value"] > 0, cal_benefit)

        test = actions["test"]; test_labels = splits["test"]["y"]
        full_probability = bundles["test"]["probabilities"][:, -1]
        final_probability, activate = apply_action(full_probability, test, crc)
        final_task, final_loss = evaluate_probabilities(final_probability, test_labels)
        full_task, full_loss = evaluate_probabilities(full_probability, test_labels)
        direct_task, direct_loss = evaluate_probabilities(
            test["projected_probability"], test_labels)
        shrink_task, shrink_loss = evaluate_probabilities(
            test["action_probability"], test_labels)
        oracle_loss = bundles["test"]["losses"].min(1)
        observed_benefit = full_loss-observed_loss(test["action_probability"], test_labels)
        harmful = activate & (observed_benefit <= 0)

        cal_hard = coalitions["calibration"]
        cal_rows = np.arange(len(cal_labels))
        hard_cal_benefit = (cal_full_loss
                            - bundles["calibration"]["losses"][cal_rows, cal_hard["candidate"]])
        hard_crc, _ = fit_crc(cal_hard["candidate_safe_probability"],
                              cal_hard["candidate_value"] > 0, hard_cal_benefit)
        hard_probability, hard_activate = hard_policy(
            coalitions["test"], bundles["test"]["probabilities"],
            bundles["test"]["losses"], hard_crc)
        hard_task, hard_loss = evaluate_probabilities(hard_probability, test_labels)

        full_regret = (full_loss-oracle_loss).mean()
        record = {"dataset": args.dataset, "train_seed": task_seed, "alpha": alpha,
                  "crc_threshold": crc.threshold, "activation_rate": float(activate.mean()),
                  "overall_harmful_action_risk": float(harmful.mean()),
                  "overall_clipped_harm": float((activate*np.minimum(
                      np.maximum(-observed_benefit, 0), 1)).mean()),
                  "safe_action_precision": float((observed_benefit[activate] > .01).mean())
                      if activate.any() else np.nan,
                  "mean_full_regret": float(full_regret),
                  "mean_final_regret": float((final_loss-oracle_loss).mean()),
                  "regret_reduction": float(1-(final_loss-oracle_loss).mean()/max(full_regret, 1e-12)),
                  **{f"full_{key}": value for key, value in full_task.items()},
                  **{f"final_{key}": value for key, value in final_task.items()},
                  **{f"direct_projection_{key}": value for key, value in direct_task.items()},
                  **{f"shrink_no_crc_{key}": value for key, value in shrink_task.items()},
                  **{f"hard_crc_{key}": value for key, value in hard_task.items()},
                  "hard_crc_activation_rate": float(hard_activate.mean()),
                  "hard_crc_regret_reduction": float(
                      1-(hard_loss-oracle_loss).mean()/max(full_regret, 1e-12))}
        metrics.append(record); write_json(out/"metrics.json", record)
        write_json(out/"crc.json", crc.__dict__)
        write_json(out/"alpha_selection.json", {"chosen": alpha, "grid": alpha_table})
        pd.DataFrame(curve).to_csv(out/"calibration_risk_curve.csv", index=False)
        pd.DataFrame(crc_risk_curve(
            test["safe_probability"], test["action_value"] > 0,
            observed_benefit)).to_csv(out/"test_risk_coverage_curve.csv", index=False)
        frame = pd.DataFrame({
            "sample_id": splits["test"]["id"],
            "group_or_video_id": splits["test"]["group"], "label": test_labels,
            "activate": activate, "hard_activate": hard_activate,
            "safe_probability": test["safe_probability"],
            "action_value": test["action_value"], "observed_benefit": observed_benefit,
            "full_loss": full_loss, "final_loss": final_loss, "hard_loss": hard_loss,
            "oracle_coalition_loss": oracle_loss,
            "loss_difference": final_loss-hard_loss,
        })
        for index in range(test["pool_indices"].shape[1]):
            frame[f"pool_{index}"] = test["pool_indices"][:, index]
            frame[f"weight_{index}"] = test["weights"][:, index]
        for k in range(classes): frame[f"final_p{k}"] = final_probability[:, k]
        frame.to_parquet(out/"predictions.parquet", index=False)
        np.savez_compressed(out/"projection_outputs.npz", **test)
        paired_records.append(frame[["group_or_video_id", "loss_difference"]])
        print(f"{args.dataset} seed={task_seed}: alpha={alpha:.2f}, "
              f"coverage={record['activation_rate']:.3f}, "
              f"risk={record['overall_harmful_action_risk']:.3f}, "
              f"regret_reduction={record['regret_reduction']:.3f}", flush=True)

    table = pd.DataFrame(metrics); table.to_csv(root/"metrics_by_seed.csv", index=False)
    numeric = table.select_dtypes(include=[np.number]).columns.difference(["train_seed"])
    pd.DataFrame({"metric": numeric, "mean": [table[c].mean() for c in numeric],
                  "std": [table[c].std(ddof=1) for c in numeric]}).to_csv(
                      root/"metrics_summary.csv", index=False)
    paired = cluster_bootstrap_difference(paired_records)
    write_json(root/"paired_projected_vs_hard.json", paired)
    checks = write_report(root, table, paired)
    write_json(root/"complete.json", {"checks": checks,
                                      "passed": bool(all(checks.values())),
                                      "wall_seconds": time.perf_counter()-start})


def main():
    parser = argparse.ArgumentParser(description="Risk-controlled posterior projection fusion")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist"), default="mosi")
    parser.add_argument("--data", default="data")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--posterior-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
