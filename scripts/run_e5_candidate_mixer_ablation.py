"""E5: isolate candidate-action mixing before adaptive shrinkage or ensembling."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer import AnchoredCandidateMixer, anchored_mixer_objective
from rcg.anchored_mixer_pipeline import predict_mixer
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import (apply_residual_blend, predict_router,
                                          select_residual_blend, train_router)
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.posterior_shrinkage_pipeline import probability_metrics
from rcg.projected_fusion_pipeline import load_posteriors, observed_loss
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                     predict_outputs, seed_all)

from run_e4_anchored_routing import CONFIGS, SEEDS, make_target


ROOT = Path(__file__).resolve().parents[1]
LEARNED = {
    "softmax_mixer": {"anchor_full": False, "harm_weight": 0., "oracle_weight": 0.},
    "sparsemax_mixer": {"anchor_full": False, "harm_weight": 0., "oracle_weight": 0.,
                        "normalizer": "sparsemax"},
    "anchored_mixer": {"anchor_full": True, "harm_weight": 0., "oracle_weight": 0.},
    "anchored_harm": {"anchor_full": True, "harm_weight": 2., "oracle_weight": 0.},
    "anchored_harm_oracle": {"anchor_full": True, "harm_weight": 2., "oracle_weight": .05},
}
BASELINES = ("full_coalition", "posterior_only", "analytic_hard", "listwise_hard",
             "equal_action_average", "candidate_oracle")


def alternative_candidates(analytic: np.ndarray, learned: np.ndarray,
                           top_k: int) -> np.ndarray:
    """Preserve the best analytic alternative; full fusion is action zero."""
    alternatives = analytic.shape[1]-1
    if not 1 <= top_k <= alternatives:
        raise ValueError("top_k exceeds the number of non-full coalitions")
    anchor = analytic[:, :-1].argmax(1)
    order = np.argsort(-learned[:, :-1], axis=1)
    result = np.empty((len(anchor), top_k), dtype=np.int64)
    result[:, 0] = anchor
    for row in range(len(anchor)):
        support = [index for index in order[row] if index != anchor[row]]
        result[row, 1:] = support[:top_k-1]
    return result


def make_actions(probabilities: np.ndarray, posterior: np.ndarray,
                 route_output: dict[str, np.ndarray], top_k: int):
    candidates = alternative_candidates(
        route_output["analytic_score"], route_output["score"], top_k)
    rows = np.arange(len(probabilities))[:, None]
    selected = probabilities[rows, candidates]
    actions = np.concatenate((probabilities[:, -1:, :], posterior[:, None, :], selected), axis=1)
    return actions.astype(np.float32), candidates


def train_mixer(model, train_actions, train_posterior, train_labels,
                selection_actions, selection_posterior, selection_labels,
                device, *, seed, harm_weight, oracle_weight,
                batch_size, epochs=100, patience=10):
    seed_all(seed)
    ta = torch.as_tensor(train_actions, dtype=torch.float32, device=device)
    tq = torch.as_tensor(train_posterior, dtype=torch.float32, device=device)
    ty = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    va = torch.as_tensor(selection_actions, dtype=torch.float32, device=device)
    vq = torch.as_tensor(selection_posterior, dtype=torch.float32, device=device)
    vy = torch.as_tensor(selection_labels, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train(); train_values = []
        for index in torch.randperm(len(ty), device=device).split(batch_size):
            output = model(ta[index], tq[index])
            loss, parts = anchored_mixer_objective(
                output, ta[index], ty[index], harm_weight=harm_weight,
                oracle_weight=oracle_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            train_values.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            output = model(va, vq)
            value = float(-output["probability"].clamp_min(1e-8).log()[
                torch.arange(len(vy), device=device), vy].mean())
        history.append({"epoch": epoch+1, "train_objective": float(np.mean(train_values)),
                        "selection_nll": value})
        if value < best-1e-6:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if state is None:
        raise RuntimeError("mixer produced no checkpoint")
    model.load_state_dict(state); model.eval()
    return history


def evaluate_probability(probability, full, labels, coalition_losses, actions,
                         *, weight=None):
    metrics, prediction = probability_metrics(probability, labels)
    full_metrics, full_prediction = probability_metrics(full, labels)
    loss, full_loss = observed_loss(probability, labels), observed_loss(full, labels)
    global_oracle = coalition_losses.min(1)
    action_loss = -np.log(np.clip(actions[np.arange(len(labels)), :, labels], 1e-12, 1))
    action_oracle = action_loss.min(1)
    full_regret = float(np.mean(full_loss-global_oracle))
    negative = (full_prediction == labels) & (prediction != labels)
    correction = (full_prediction != labels) & (prediction == labels)
    result = {
        **metrics, "accuracy_gain": metrics["accuracy"]-full_metrics["accuracy"],
        "macro_f1_gain": metrics["macro_f1"]-full_metrics["macro_f1"],
        "nll_gain": full_metrics["nll"]-metrics["nll"],
        "relative_nll_reduction": (full_metrics["nll"]-metrics["nll"])/full_metrics["nll"],
        "negative_flip_rate": float(negative.mean()),
        "correction_rate": float(correction.mean()),
        "clipped_harm": float(np.minimum(np.maximum(loss-full_loss, 0), 1).mean()),
        "regret_reduction": float(1-np.mean(loss-global_oracle)/max(full_regret, 1e-12)),
        "action_oracle_nll": float(action_oracle.mean()),
        "action_oracle_gap": float(np.mean(loss-action_oracle)),
        "full_nll": full_metrics["nll"], "full_accuracy": full_metrics["accuracy"],
    }
    if weight is None:
        result.update({"mean_full_weight": np.nan, "mean_posterior_weight": np.nan,
                       "mean_candidate_weight": np.nan, "candidate_usage_rate": np.nan})
    else:
        result.update({"mean_full_weight": float(weight[:, 0].mean()),
                       "mean_posterior_weight": float(weight[:, 1].mean()),
                       "mean_candidate_weight": float(weight[:, 2:].sum(1).mean()),
                       "candidate_usage_rate": float((weight[:, 2:].sum(1) > .1).mean())})
    return result, loss, prediction


def run_dataset(dataset: str, device: str, *, output_root: Path | None = None) -> None:
    cfg = CONFIGS[dataset]
    folds, names = load_dataset(dataset, ROOT/"data")
    masks = nonempty_coalitions(len(names)); top_k = 3 if len(names) >= 3 else 2
    root = (Path(output_root) if output_root is not None
            else ROOT/"runs"/f"formal-e5-{dataset}")
    root.mkdir(parents=True, exist_ok=True)
    partial = root/"metrics_by_fold_seed.partial.csv"
    saved_frames = []
    if (root/"metrics_by_fold_seed.csv").exists():
        saved_frames.append(pd.read_csv(root/"metrics_by_fold_seed.csv"))
    if partial.exists():
        saved_frames.append(pd.read_csv(partial))
    if saved_frames:
        saved = pd.concat(saved_frames, ignore_index=True).drop_duplicates(
            ["variant", "fold", "train_seed"], keep="last")
        metrics_rows = saved.to_dict("records")
    else:
        metrics_rows = []
    completed = {(str(x["variant"]), int(x["fold"]), int(x["train_seed"]))
                 for x in metrics_rows}
    variants = {*BASELINES, *LEARNED}

    for fold_index, splits in enumerate(folds):
        oof_path = (Path(cfg["oof_template"].format(fold=fold_index))
                    if "oof_template" in cfg else cfg["oof"])
        with np.load(oof_path) as saved:
            oof = {key: saved[key] for key in saved.files}
        for task_seed in SEEDS:
            if all((name, fold_index, task_seed) in completed for name in variants):
                continue
            out = root/f"fold_{fold_index}"/f"seed_{task_seed}"; out.mkdir(parents=True, exist_ok=True)
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

            router_path = out/"router.pt"
            if router_path.exists():
                saved = torch.load(router_path, map_location=device, weights_only=True)
                router = AnalyticResidualListwiseRouter(
                    dims, classes, len(masks), residual_scale=.5).to(device)
                router.load_state_dict(saved["state_dict"]); blend = float(saved["blend"])
            else:
                train_target = make_target(oof["losses"])
                train_target["posterior_override"] = posterior["train"]
                selection_target = make_target(bundles["selection"]["losses"][None])
                selection_target["posterior_override"] = posterior["selection"]
                seed_all(task_seed+fold_index*10000)
                router = AnalyticResidualListwiseRouter(
                    dims, classes, len(masks), residual_scale=.5).to(device)
                train_router(
                    router, splits["train"], oof["probabilities"], train_target,
                    splits["selection"], bundles["selection"]["probabilities"],
                    selection_target, masks, device, seed=task_seed+fold_index*10000,
                    epochs=100, batch_size=cfg["batch_size"], patience=10,
                    objective_kwargs={"list_weight": 1., "pair_weight": .5,
                                      "stable_weight": 0.})
                selection_route = predict_router(
                    router, splits["selection"], bundles["selection"]["probabilities"],
                    masks, device, posterior["selection"])
                blend, _ = select_residual_blend(
                    selection_route, bundles["selection"]["losses"])
                torch.save({"state_dict": router.state_dict(), "blend": blend}, router_path)

            route = {}; actions = {}; candidates = {}
            for split_name in ("selection", "test"):
                value = predict_router(
                    router, splits[split_name], bundles[split_name]["probabilities"],
                    masks, device, posterior[split_name])
                route[split_name] = apply_residual_blend(value, blend)
                actions[split_name], candidates[split_name] = make_actions(
                    bundles[split_name]["probabilities"], posterior[split_name],
                    route[split_name], top_k)
            test_labels = splits["test"]["y"]; test_full = actions["test"][:, 0]
            output_probabilities, output_weights = {}, {}
            output_probabilities["full_coalition"] = test_full
            output_probabilities["posterior_only"] = posterior["test"]
            analytic_index = route["test"]["analytic_score"].argmax(1)
            learned_index = route["test"]["score"].argmax(1)
            rows = np.arange(len(test_labels))
            output_probabilities["analytic_hard"] = bundles["test"]["probabilities"][rows, analytic_index]
            output_probabilities["listwise_hard"] = bundles["test"]["probabilities"][rows, learned_index]
            output_probabilities["equal_action_average"] = actions["test"].mean(1)
            action_loss = -np.log(np.clip(actions["test"][rows, :, test_labels], 1e-12, 1))
            action_oracle_index = action_loss.argmin(1)
            output_probabilities["candidate_oracle"] = actions["test"][rows, action_oracle_index]

            missing_learned = [variant for variant in LEARNED
                               if (variant, fold_index, task_seed) not in completed]
            if missing_learned:
                train_actions, train_posteriors, train_labels = [], [], []
                for teacher_index, teacher_probability in enumerate(oof["probabilities"]):
                    value = predict_router(router, splits["train"], teacher_probability,
                                           masks, device, posterior["train"][teacher_index])
                    value = apply_residual_blend(value, blend)
                    current, _ = make_actions(
                        teacher_probability, posterior["train"][teacher_index], value, top_k)
                    train_actions.append(current)
                    train_posteriors.append(posterior["train"][teacher_index])
                    train_labels.append(splits["train"]["y"])
                train_actions = np.concatenate(train_actions)
                train_posteriors = np.concatenate(train_posteriors)
                train_labels = np.concatenate(train_labels)

            for variant, config in LEARNED.items():
                if variant not in missing_learned:
                    continue
                seed_all(task_seed+fold_index*10000)
                mixer = AnchoredCandidateMixer(
                    classes, anchor_full=config["anchor_full"],
                    normalizer=config.get("normalizer", "softmax")).to(device)
                history = train_mixer(
                    mixer, train_actions, train_posteriors, train_labels,
                    actions["selection"], posterior["selection"], splits["selection"]["y"],
                    device, seed=task_seed+fold_index*10000,
                    harm_weight=config["harm_weight"], oracle_weight=config["oracle_weight"],
                    batch_size=cfg["batch_size"])
                prediction = predict_mixer(mixer, actions["test"], posterior["test"], device)
                output_probabilities[variant] = prediction["probability"]
                output_weights[variant] = prediction["weight"]
                torch.save({"state_dict": mixer.state_dict(), "classes": classes,
                            "config": config, "epochs": len(history)}, out/f"{variant}.pt")

            prediction_frames = []
            for variant in BASELINES + tuple(LEARNED):
                if (variant, fold_index, task_seed) in completed:
                    continue
                metric, losses, predictions = evaluate_probability(
                    output_probabilities[variant], test_full, test_labels,
                    bundles["test"]["losses"], actions["test"],
                    weight=output_weights.get(variant))
                metrics_rows.append({"dataset": dataset, "variant": variant,
                                     "fold": fold_index, "train_seed": task_seed,
                                     "n_test": len(test_labels), "router_blend": blend, **metric})
                frame = pd.DataFrame({"sample_id": splits["test"]["id"],
                                      "group_or_video_id": splits["test"]["group"],
                                      "label": test_labels, "variant": variant,
                                      "loss": losses, "prediction": predictions})
                prediction_frames.append(frame)
            if prediction_frames:
                prediction_path = out/"predictions.parquet"
                if prediction_path.exists():
                    old = pd.read_parquet(prediction_path)
                    new_names = {frame.variant.iloc[0] for frame in prediction_frames}
                    old = old[~old.variant.isin(new_names)]
                    prediction_frames.insert(0, old)
                pd.concat(prediction_frames, ignore_index=True).to_parquet(
                    prediction_path, index=False)
            pd.DataFrame(metrics_rows).to_csv(partial, index=False)
            print(f"{dataset} fold={fold_index} seed={task_seed}: "
                  f"added={len(prediction_frames)}", flush=True)

    frame = pd.DataFrame(metrics_rows); frame.to_csv(root/"metrics_by_fold_seed.csv", index=False)
    id_columns = {"dataset", "variant", "fold", "train_seed", "n_test"}
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in id_columns]
    seed_rows = []
    for (variant, seed), values in frame.groupby(["variant", "train_seed"]):
        weights = values.n_test.to_numpy(float)
        row = {"dataset": dataset, "variant": variant, "train_seed": int(seed),
               "n_test": int(weights.sum())}
        row.update({metric: float(np.average(values[metric], weights=weights))
                    for metric in numeric})
        seed_rows.append(row)
    by_seed = pd.DataFrame(seed_rows); by_seed.to_csv(root/"metrics_by_seed.csv", index=False)
    summary = by_seed.groupby("variant")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(root/"metrics_summary.csv", index=False)
    (root/"manifest.json").write_text(json.dumps({
        "experiment": "E5 candidate-action mixer ablation", "dataset": dataset,
        "seeds": list(SEEDS), "top_k_nonfull": top_k, "variants": LEARNED,
        "adaptive_shrinkage": False, "model_ensemble": False,
        "selection_role": "router blend and mixer early stopping only",
        "test_labels_role": "evaluation and candidate-oracle upper bound only",
    }, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)


def main():
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
