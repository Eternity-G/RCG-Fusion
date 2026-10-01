"""Train and evaluate posterior-analytic contribution estimation (v2)."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score

from .benefit_fusion import fit_crc
from .io import load_json, write_json
from .posterior_analytic import (RelationalPosteriorEstimator, analytic_candidate,
                                 posterior_objective)
from .rcg_fusion import CoalitionAwareBackbone, nonempty_coalitions
from .rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                  predict_outputs, seed_all)


DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def _tensors(split, device):
    return ([torch.as_tensor(x, dtype=torch.float32, device=device) for x in split["x"]],
            torch.as_tensor(split["y"], dtype=torch.long, device=device))


def _load_backbone(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CoalitionAwareBackbone(checkpoint["dims"], checkpoint["classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model, np.asarray(checkpoint["temperatures"]), checkpoint["dims"], checkpoint["classes"]


def train_posterior(model, train_split, oof_probabilities, selection_split,
                    selection_probabilities, masks, device, *, seed, epochs=100,
                    batch_size=128, patience=10, teacher_mode="all"):
    seed_all(seed)
    train_x, train_y = _tensors(train_split, device)
    selection_x, selection_y = _tensors(selection_split, device)
    oof_p = torch.as_tensor(oof_probabilities, dtype=torch.float32, device=device)
    selection_p = torch.as_tensor(selection_probabilities, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(masks, dtype=torch.float32, device=device)
    teacher_count = 1 if teacher_mode == "single" else len(oof_p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train(); train_values = []
        for sample in torch.randperm(len(train_y), device=device).split(batch_size):
            teacher = torch.arange(teacher_count, device=device)[:, None].expand(
                -1, len(sample)).reshape(-1)
            repeated = sample[None].expand(teacher_count, -1).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            output = model([x[repeated] for x in train_x], oof_p[teacher, repeated], mask_tensor)
            loss, _ = posterior_objective(output["posterior_logits"], train_y[repeated])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); train_values.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            output = model(selection_x, selection_p, mask_tensor)
            _, parts = posterior_objective(output["posterior_logits"], selection_y)
        selection_nll = float(parts["nll"])
        history.append({"epoch": epoch+1, "train": float(np.mean(train_values)),
                        "selection_nll": selection_nll,
                        "selection_brier": float(parts["brier"])})
        if selection_nll < best-1e-6:
            best, state, stale = selection_nll, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if state is None:
        raise RuntimeError("posterior training did not produce a checkpoint")
    model.load_state_dict(state); model.eval()
    return history


@torch.inference_mode()
def predict_posterior(model, split, probabilities, masks, device):
    xs, _ = _tensors(split, device)
    output = model(xs, torch.as_tensor(probabilities, dtype=torch.float32, device=device),
                   torch.as_tensor(masks, dtype=torch.float32, device=device))
    return {key: value.cpu().numpy() for key, value in output.items()}


def fit_ensemble_temperature(posteriors, labels):
    classes = posteriors[0].shape[1]
    objective = lambda log_t: log_loss(
        labels, np.mean(calibrate_members(posteriors, np.exp(log_t)), axis=0),
        labels=np.arange(classes))
    result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def calibrate_members(posteriors, temperature):
    return [softmax(np.log(value.clip(1e-8, 1.))/float(temperature), axis=1)
            for value in posteriors]


def candidate_metrics(candidate, losses, tau=.01):
    rows = np.arange(len(losses)); full = losses.shape[1]-1
    observed_all = losses[:, full, None]-losses
    keep = np.arange(losses.shape[1]) != full
    safe_all = observed_all[:, keep] > tau
    result = {
        "benefit_spearman": float(spearmanr(
            candidate["expected_benefit"][:, keep].ravel(),
            observed_all[:, keep].ravel()).statistic),
        "all_pair_safe_auroc": float(roc_auc_score(
            safe_all.ravel(), candidate["safe_probability"][:, keep].ravel())),
        "all_pair_safe_auprc": float(average_precision_score(
            safe_all.ravel(), candidate["safe_probability"][:, keep].ravel())),
    }
    chosen = candidate["candidate"]
    benefit = observed_all[rows, chosen]; safe = benefit > tau
    score = candidate["candidate_safe_probability"]
    result["candidate_safe_auroc"] = (float(roc_auc_score(safe, score))
                                      if len(np.unique(safe)) == 2 else np.nan)
    result["candidate_safe_auprc"] = (float(average_precision_score(safe, score))
                                      if len(np.unique(safe)) == 2 else np.nan)
    top = score >= np.quantile(score, .8)
    result["top20_safe_precision"] = float(safe[top].mean())
    candidate_loss = losses[rows, chosen]; oracle = losses.min(1)
    full_regret = losses[:, full]-oracle; candidate_regret = candidate_loss-oracle
    result.update({
        "safe_prevalence": float(safe.mean()),
        "ungated_candidate_regret": float(candidate_regret.mean()),
        "full_regret": float(full_regret.mean()),
        "ungated_regret_reduction": float(
            1-candidate_regret.mean()/max(full_regret.mean(), 1e-12)),
        "ungated_candidate_vs_full": float((candidate_loss-losses[:, full]).mean()),
    })
    return result


def posterior_metrics(posterior, labels):
    target = np.eye(posterior.shape[1])[labels]
    return {
        "posterior_accuracy": float(accuracy_score(labels, posterior.argmax(1))),
        "posterior_nll": float(log_loss(labels, posterior,
                                        labels=np.arange(posterior.shape[1]))),
        "posterior_brier": float(np.mean(np.sum((posterior-target)**2, axis=1))),
    }


def policy_metrics(split, probabilities, losses, candidate, calibration):
    rows = np.arange(len(losses)); full = losses.shape[1]-1
    switch = ((candidate["candidate_value"] > 0)
              & (candidate["candidate_safe_probability"] >= calibration.threshold))
    selected = np.where(switch, candidate["candidate"], full)
    benefit = losses[:, full]-losses[rows, candidate["candidate"]]
    selected_loss = losses[rows, selected]; oracle = losses.min(1)
    full_regret = losses[:, full]-oracle; selected_regret = selected_loss-oracle
    harmful = switch & (benefit <= 0)
    selected_prediction = probabilities[rows, selected].argmax(1)
    full_prediction = probabilities[:, full].argmax(1)
    return {
        "switch_rate": float(switch.mean()),
        "overall_harmful_switch_risk": float(harmful.mean()),
        "overall_clipped_harm": float((switch*np.minimum(np.maximum(-benefit, 0), 1)).mean()),
        "safe_switch_precision": float((benefit[switch] > .01).mean()) if switch.any() else np.nan,
        "mean_selection_regret": float(selected_regret.mean()),
        "regret_reduction": float(1-selected_regret.mean()/max(full_regret.mean(), 1e-12)),
        "accuracy": float(accuracy_score(split["y"], selected_prediction)),
        "full_accuracy": float(accuracy_score(split["y"], full_prediction)),
    }, selected, switch, benefit


def innovation2_pass(table):
    mean = table.select_dtypes(include=[np.number]).mean()
    return {
        "candidate_safe_auroc": mean.candidate_safe_auroc >= .70,
        "top20_safe_precision": mean.top20_safe_precision >= .80,
        "benefit_spearman": mean.benefit_spearman >= .50,
        "positive_seeds": int((table.ungated_regret_reduction > 0).sum()) >= 3,
    }


def write_report(root, table, baseline_path=None):
    mean = table.select_dtypes(include=[np.number]).mean()
    checks = innovation2_pass(table)
    lines = ["# Posterior-Analytic Innovation 2 Go/No-Go", "",
             "MOSI test has been used for method diagnosis; these are development results.", "",
             f"结论：**{'通过数值门槛' if all(checks.values()) else '未通过'}**。", "",
             "| 检查 | 实际 | 通过 |", "|---|---:|---|",
             f"| 候选安全 AUROC ≥ 0.70 | {mean.candidate_safe_auroc:.4f} | {checks['candidate_safe_auroc']} |",
             f"| Top-20% 安全精确率 ≥ 0.80 | {mean.top20_safe_precision:.4f} | {checks['top20_safe_precision']} |",
             f"| 收益 Spearman ≥ 0.50 | {mean.benefit_spearman:.4f} | {checks['benefit_spearman']} |",
             f"| 未门控候选至少3种子降后悔 | {int((table.ungated_regret_reduction > 0).sum())}/5 | {checks['positive_seeds']} |"]
    if baseline_path and Path(baseline_path).exists():
        baseline = pd.read_csv(baseline_path).select_dtypes(include=[np.number]).mean()
        lines += ["", "## 相对独立收益头 v1", "",
                  "| 指标 | v1 | v2 |", "|---|---:|---:|",
                  f"| 候选安全 AUROC | {baseline.candidate_safe_auroc:.4f} | {mean.candidate_safe_auroc:.4f} |",
                  f"| Top-20% 安全精确率 | {baseline.top20_safe_precision:.4f} | {mean.top20_safe_precision:.4f} |",
                  f"| 收益 Spearman | {baseline.benefit_spearman:.4f} | {mean.benefit_spearman:.4f} |"]
    (root/"GO_NO_GO.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset(args.dataset, args.data)
    base, root = Path(args.base_run), Path(args.output); root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": args.dataset, "base_run": str(base.resolve()),
                "task_seeds": args.seeds, "posterior_seeds": args.posterior_seeds,
                "teacher_mode": args.teacher_mode, "use_relations": not args.no_relations,
                "beta": args.beta, "tau": .01, "posterior_loss": "CE + 0.1 Brier",
                "selection_role": "posterior early stopping and ensemble temperature",
                "calibration_role": "CRC only", "run_crc": args.run_crc,
                "mosi_status": "development"}
    if (root/"manifest.json").exists() and load_json(root/"manifest.json") != manifest:
        raise ValueError("manifest differs; use a new output directory")
    write_json(root/"manifest.json", manifest)
    all_metrics, cached, start = [], [], time.perf_counter()
    for fold_index, splits in enumerate(folds):
        fold_base = base/f"fold_{fold_index}"
        with np.load(fold_base/"oof_targets.npz") as saved:
            oof = {key: saved[key] for key in saved.files}
        masks = nonempty_coalitions(len(names)); fold_out = root/f"fold_{fold_index}"
        fold_out.mkdir(exist_ok=True)
        for task_seed in args.seeds:
            out = fold_out/f"seed_{task_seed}"; out.mkdir(exist_ok=True)
            if (out/"complete.json").exists() and not args.run_crc:
                cached_metrics = load_json(out/"metrics.json")
                cached_metrics.setdefault("n_test", len(splits["test"]["y"]))
                all_metrics.append(cached_metrics); continue
            backbone, temperatures, dims, classes = _load_backbone(
                fold_base/f"seed_{task_seed}"/"backbone.pt", device)
            bundles = {}
            for split_name in ("selection", "calibration", "test"):
                outputs = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
                bundles[split_name] = attach_observed_losses(outputs, splits[split_name]["y"])
            models, histories = [], []
            for posterior_seed in args.posterior_seeds:
                seed_all(task_seed*1000+posterior_seed)
                model = RelationalPosteriorEstimator(
                    dims, classes, len(masks), use_relations=not args.no_relations).to(device)
                history = train_posterior(
                    model, splits["train"], oof["probabilities"], splits["selection"],
                    bundles["selection"]["probabilities"], masks, device,
                    seed=task_seed*1000+posterior_seed, epochs=args.epochs,
                    batch_size=args.batch_size, teacher_mode=args.teacher_mode)
                models.append(model); histories.append(history)
            predictions = {name: [predict_posterior(
                model, splits[name], bundle["probabilities"], masks, device)
                for model in models] for name, bundle in bundles.items()}
            selection_posteriors = [x["posterior"] for x in predictions["selection"]]
            temperature = fit_ensemble_temperature(selection_posteriors, splits["selection"]["y"])
            candidates, ensemble_posteriors = {}, {}
            for name, values in predictions.items():
                members = calibrate_members([x["posterior"] for x in values], temperature)
                posterior = np.mean(members, axis=0)
                ensemble_posteriors[name] = posterior
                candidates[name] = analytic_candidate(
                    members, bundles[name]["probabilities"], masks, beta=args.beta)
            metrics = {"dataset": args.dataset, "fold": fold_index, "train_seed": task_seed,
                       "n_test": len(splits["test"]["y"]),
                       "posterior_temperature": temperature, "beta": args.beta,
                       **posterior_metrics(ensemble_posteriors["test"], splits["test"]["y"]),
                       **candidate_metrics(candidates["test"], bundles["test"]["losses"])}
            rows = np.arange(len(splits["test"]["y"])); candidate = candidates["test"]
            frame = pd.DataFrame({
                "sample_id": splits["test"]["id"], "group_or_video_id": splits["test"]["group"],
                "label": splits["test"]["y"], "candidate": candidate["candidate"],
                "candidate_value": candidate["candidate_value"],
                "candidate_safe_probability": candidate["candidate_safe_probability"],
                "candidate_expected_benefit": candidate["candidate_expected_benefit"],
                "observed_candidate_benefit": (bundles["test"]["losses"][:, -1]
                    - bundles["test"]["losses"][rows, candidate["candidate"]]),
            })
            for k in range(classes): frame[f"posterior_p{k}"] = ensemble_posteriors["test"][:, k]
            frame.to_parquet(out/"predictions.parquet", index=False)
            np.savez_compressed(out/"analytic_outputs.npz",
                               posterior=ensemble_posteriors["test"],
                               label_benefit=candidate["label_benefit"],
                               expected_benefit=candidate["expected_benefit"],
                               safe_probability=candidate["safe_probability"],
                               gain=candidate["gain"], harm=candidate["harm"])
            write_json(out/"metrics.json", metrics)
            write_json(out/"training.json", {"histories": histories})
            for posterior_seed, model in zip(args.posterior_seeds, models):
                torch.save({"state_dict": model.state_dict(), "dims": dims,
                            "classes": classes, "n_coalitions": len(masks),
                            "use_relations": not args.no_relations},
                           out/f"posterior_{posterior_seed}.pt")
            write_json(out/"complete.json", {"n_test": len(frame)})
            all_metrics.append(metrics)
            cached.append((out, splits, bundles, candidates))
            print(f"{args.dataset} seed={task_seed}: auc={metrics['candidate_safe_auroc']:.3f}, "
                  f"top20={metrics['top20_safe_precision']:.3f}, "
                  f"regret_reduction={metrics['ungated_regret_reduction']:.3f}", flush=True)
    fold_table = pd.DataFrame(all_metrics)
    fold_table.to_csv(root/"metrics_by_fold_seed.csv", index=False)
    numeric = fold_table.select_dtypes(include=[np.number]).columns.difference(
        ["fold", "train_seed", "n_test"])
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
    baseline = Path("runs/rcg-benefit-mosi-v1/metrics_by_seed.csv") if args.dataset == "mosi" else None
    write_report(root, table, baseline)
    if args.run_crc:
        if not all(innovation2_pass(table).values()):
            raise RuntimeError("Innovation 2 gate failed; CRC was not run")
        for out, splits, bundles, candidates in cached:
            cal = candidates["calibration"]; cal_rows = np.arange(len(splits["calibration"]["y"]))
            cal_benefit = (bundles["calibration"]["losses"][:, -1]
                           - bundles["calibration"]["losses"][cal_rows, cal["candidate"]])
            crc, curve = fit_crc(cal["candidate_safe_probability"], cal["candidate_value"] > 0,
                                 cal_benefit)
            metrics, selected, switch, benefit = policy_metrics(
                splits["test"], bundles["test"]["probabilities"], bundles["test"]["losses"],
                candidates["test"], crc)
            write_json(out/"crc.json", crc.__dict__); write_json(out/"policy_metrics.json", metrics)
            pd.DataFrame(curve).to_csv(out/"calibration_risk_curve.csv", index=False)
    write_json(root/"runtime.json", {"wall_seconds": time.perf_counter()-start})


def main():
    parser = argparse.ArgumentParser(description="Posterior-analytic contribution estimation v2")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist"), default="mosi")
    parser.add_argument("--data", default="data")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--posterior-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-mode", choices=("all", "single"), default="all")
    parser.add_argument("--no-relations", action="store_true")
    parser.add_argument("--beta", type=float, default=2.)
    parser.add_argument("--run-crc", action="store_true")
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
