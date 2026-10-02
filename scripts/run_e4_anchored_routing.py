"""E4: compare analytic, listwise, pairwise, and anchor-preserving routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.listwise_router import (AnalyticResidualListwiseRouter,
                                 anchor_preserving_candidates,
                                 build_soft_coalition_targets, ranking_metrics)
from rcg.listwise_router_pipeline import (apply_residual_blend, predict_router,
                                          select_residual_blend, train_router)
from rcg.posterior_analytic import analytic_contribution
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                     predict_outputs, seed_all)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
CONFIGS = {
    "mosi": {"base": ROOT/"runs/rcg-fusion-mosi-v4",
             "posterior": ROOT/"runs/rcg-posterior-analytic-mosi-v2",
             "oof": ROOT/"runs/rcg-fusion-mosi-v4/fold_0/oof_targets.npz",
             "batch_size": 64},
    "mosei": {"base": ROOT/"runs/rcg-fusion-mosei-shrinkage-v1",
              "posterior": ROOT/"runs/rcg-posterior-analytic-mosei-shrinkage-v1",
              "oof": ROOT/"runs/formal-e2-mosei-oof/oof_targets.npz",
              "batch_size": 128},
    "cremad": {"base": ROOT/"runs/rcg-fusion-cremad-v1",
               "posterior": ROOT/"runs/rcg-posterior-analytic-cremad-v1",
               "oof_template": str(ROOT/"runs/formal-e2-cremad-oof/fold_{fold}/oof_targets.npz"),
               "batch_size": 64},
    "avmnist": {"base": ROOT/"runs/rcg-fusion-avmnist-v1",
                "posterior": ROOT/"runs/rcg-posterior-analytic-avmnist-v1",
                "oof": ROOT/"runs/formal-e2-avmnist-oof/oof_targets.npz",
                "batch_size": 128},
}

TRAIN_VARIANTS = {
    "hard_oracle_classifier": {"hard": True, "list_weight": 1., "pair_weight": 0.},
    "soft_listwise": {"hard": False, "list_weight": 1., "pair_weight": 0.},
    "pairwise_only": {"hard": False, "list_weight": 0., "pair_weight": 1.},
    "listwise_pairwise": {"hard": False, "list_weight": 1., "pair_weight": .5},
}


def make_target(losses: np.ndarray, *, hard: bool = False) -> dict[str, np.ndarray]:
    target = build_soft_coalition_targets(
        losses, oracle_temperature=.1, epsilon=.01, stable_fraction=.8)
    if hard:
        oracle = losses.mean(0).argmin(1)
        target["soft_oracle"] = np.eye(losses.shape[2], dtype=np.float32)[oracle]
    target["agreement_weight"] = np.ones(losses.shape[1], dtype=np.float32)
    return target


def analytic_scores(posterior: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        result = analytic_contribution(
            torch.as_tensor(posterior, dtype=torch.float64),
            torch.as_tensor(probabilities, dtype=torch.float64))
    return result["expected_benefit"].cpu().numpy()


def route_metrics(score: np.ndarray, analytic: np.ndarray, losses: np.ndarray,
                  *, anchored: bool, top_k: int) -> dict[str, float]:
    rows = np.arange(len(losses)); full = losses.shape[1]-1
    rank = ranking_metrics(score, losses)
    if anchored:
        candidates = anchor_preserving_candidates(analytic, score, top_k=top_k)
    else:
        candidates = np.argsort(-score, axis=1)[:, :top_k]
    oracle = losses.argmin(1)
    coverage = np.mean([oracle[i] in candidates[i] for i in rows])
    candidate_loss = np.take_along_axis(losses, candidates, axis=1).min(1)
    # The deployed action set always retains full fusion as a fallback.
    action_loss = np.minimum(candidate_loss, losses[:, full])
    full_loss = losses[:, full]
    oracle_loss = losses.min(1)
    denominator = float(np.mean(full_loss-oracle_loss))
    return {
        "ndcg": rank["ndcg"], "pairwise_accuracy": rank["pairwise_accuracy"],
        "top1_recovery": rank["top1"], "top2_recovery": rank["top2"],
        "candidate_k": top_k, "candidate_oracle_recovery": float(coverage),
        "candidate_oracle_nll": float(candidate_loss.mean()),
        "action_oracle_nll": float(action_loss.mean()),
        "candidate_mean_benefit": float(np.mean(full_loss-action_loss)),
        "candidate_headroom_fraction": float(np.mean(full_loss-action_loss)/max(denominator, 1e-12)),
        "selected_nll": float(losses[rows, score.argmax(1)].mean()),
        "full_nll": float(full_loss.mean()), "global_oracle_nll": float(oracle_loss.mean()),
        "anchor_preserved": float(np.mean(candidates[:, 0] == analytic.argmax(1))) if anchored else np.nan,
    }


def run_dataset(dataset: str, device: str) -> None:
    cfg = CONFIGS[dataset]
    folds, names = load_dataset(dataset, ROOT/"data")
    masks = nonempty_coalitions(len(names))
    output_name = (f"formal-e4-{dataset}-top2"
                   if len(names) == 2 else f"formal-e4-{dataset}")
    output = ROOT/"runs"/output_name
    output.mkdir(parents=True, exist_ok=True)
    partial = output/"metrics_by_fold_seed.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    completed = {(str(x["variant"]), int(x["fold"]), int(x["train_seed"])) for x in rows}

    for fold_index, splits in enumerate(folds):
        oof_path = (Path(cfg["oof_template"].format(fold=fold_index))
                    if "oof_template" in cfg else cfg["oof"])
        with np.load(oof_path) as saved:
            oof = {key: saved[key] for key in saved.files}
        for task_seed in SEEDS:
            needed = {"analytic_only", "hard_oracle_classifier", "soft_listwise",
                      "pairwise_only", "listwise_pairwise_unanchored",
                      "listwise_pairwise_anchored"}
            if all((name, fold_index, task_seed) in completed for name in needed):
                continue
            backbone, temperatures, dims, classes = _load_backbone(
                cfg["base"]/f"fold_{fold_index}/seed_{task_seed}/backbone.pt", device)
            bundles = {}
            for split_name in ("selection", "test"):
                raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
                bundles[split_name] = attach_observed_losses(raw, splits[split_name]["y"])
            posterior = {"train": []}
            for teacher_probability in oof["probabilities"]:
                _, value = load_posteriors(
                    cfg["posterior"], task_seed, dims, classes, masks, splits["train"],
                    teacher_probability, device, fold_index=fold_index)
                posterior["train"].append(value)
            posterior["train"] = np.stack(posterior["train"])
            for split_name in ("selection", "test"):
                _, posterior[split_name] = load_posteriors(
                    cfg["posterior"], task_seed, dims, classes, masks, splits[split_name],
                    bundles[split_name]["probabilities"], device, fold_index=fold_index)

            # Top-3 is informative for seven three-modal coalitions, whereas
            # it would contain every coalition in a two-modal task.
            top_k = 3 if len(names) >= 3 else 2
            analytic = analytic_scores(posterior["test"], bundles["test"]["probabilities"])
            if ("analytic_only", fold_index, task_seed) not in completed:
                metric = route_metrics(analytic, analytic, bundles["test"]["losses"],
                                       anchored=False, top_k=top_k)
                rows.append({"dataset": dataset, "variant": "analytic_only", "fold": fold_index,
                             "train_seed": task_seed, "n_test": len(splits["test"]["y"]),
                             "epochs": 0, "residual_blend": 0., **metric})
                pd.DataFrame(rows).to_csv(partial, index=False)

            train_target_base = make_target(oof["losses"])
            train_target_base["posterior_override"] = posterior["train"]
            selection_target_base = make_target(bundles["selection"]["losses"][None])
            selection_target_base["posterior_override"] = posterior["selection"]
            for variant, params in TRAIN_VARIANTS.items():
                produced = ([variant] if variant != "listwise_pairwise" else
                            ["listwise_pairwise_unanchored", "listwise_pairwise_anchored"])
                if all((name, fold_index, task_seed) in completed for name in produced):
                    continue
                train_target = make_target(oof["losses"], hard=params["hard"])
                train_target["posterior_override"] = posterior["train"]
                selection_target = make_target(
                    bundles["selection"]["losses"][None], hard=params["hard"])
                selection_target["posterior_override"] = posterior["selection"]
                seed_all(task_seed+fold_index*10000)
                model = AnalyticResidualListwiseRouter(
                    dims, classes, len(masks), residual_scale=.5).to(device)
                history = train_router(
                    model, splits["train"], oof["probabilities"], train_target,
                    splits["selection"], bundles["selection"]["probabilities"],
                    selection_target, masks, device, seed=task_seed+fold_index*10000,
                    epochs=100, batch_size=cfg["batch_size"], patience=10,
                    objective_kwargs={"list_weight": params["list_weight"],
                                      "pair_weight": params["pair_weight"],
                                      "stable_weight": 0.})
                selection_out = predict_router(
                    model, splits["selection"], bundles["selection"]["probabilities"],
                    masks, device, posterior["selection"])
                blend, _ = select_residual_blend(
                    selection_out, bundles["selection"]["losses"])
                test_out = apply_residual_blend(predict_router(
                    model, splits["test"], bundles["test"]["probabilities"],
                    masks, device, posterior["test"]), blend)
                for name in produced:
                    anchored = name.endswith("anchored") and not name.endswith("unanchored")
                    metric = route_metrics(test_out["score"], test_out["analytic_score"],
                                           bundles["test"]["losses"],
                                           anchored=anchored, top_k=top_k)
                    rows.append({"dataset": dataset, "variant": name, "fold": fold_index,
                                 "train_seed": task_seed, "n_test": len(splits["test"]["y"]),
                                 "epochs": len(history), "residual_blend": blend, **metric})
                pd.DataFrame(rows).to_csv(partial, index=False)
                print(f"{dataset} fold={fold_index} seed={task_seed} {variant} "
                      f"blend={blend:.2f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output/"metrics_by_fold_seed.csv", index=False)
    id_columns = {"dataset", "variant", "fold", "train_seed", "n_test", "epochs", "candidate_k"}
    metrics = [c for c in frame.columns if c not in id_columns]
    seed_rows = []
    for (variant, seed), values in frame.groupby(["variant", "train_seed"]):
        weights = values.n_test.to_numpy(float)
        item = {"dataset": dataset, "variant": variant, "train_seed": int(seed),
                "n_test": int(weights.sum())}
        item.update({m: float(np.average(values[m], weights=weights)) for m in metrics})
        seed_rows.append(item)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(output/"metrics_by_seed.csv", index=False)
    summary = seed_frame.groupby("variant")[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(output/"metrics_summary.csv", index=False)
    (output/"manifest.json").write_text(json.dumps({
        "experiment": "E4 anchored listwise routing", "dataset": dataset,
        "seeds": list(SEEDS), "variants": TRAIN_VARIANTS,
        "selection_role": "early stopping and residual blend only",
        "test_labels_role": "evaluation only", "top_k": top_k,
    }, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    datasets = CONFIGS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run_dataset(dataset, device)


if __name__ == "__main__":
    main()
