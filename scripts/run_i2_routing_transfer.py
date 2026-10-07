"""I2-2: propagate routing objectives through a fixed mixer and risk controller."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import predict_mixer
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import (apply_residual_blend, predict_router,
                                          select_residual_blend, train_router)
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                     predict_outputs, seed_all)

from run_e4_anchored_routing import (CONFIGS, SEEDS, analytic_scores,
                                     make_target, route_metrics)
from run_e5_candidate_mixer_ablation import train_mixer
from run_e6_adaptive_shrinkage import (choose_strength, evaluate, gate_values, shrink)


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "analytic_only": {"trained": False, "hard": False, "list_weight": 0.,
                      "pair_weight": 0., "anchored": False},
    "hard_oracle_classifier": {"trained": True, "hard": True, "list_weight": 1.,
                               "pair_weight": 0., "anchored": False},
    "soft_listwise": {"trained": True, "hard": False, "list_weight": 1.,
                      "pair_weight": 0., "anchored": False},
    "pairwise_only": {"trained": True, "hard": False, "list_weight": 0.,
                      "pair_weight": 1., "anchored": False},
    "listwise_pairwise_unanchored": {"trained": True, "hard": False,
                                     "list_weight": 1., "pair_weight": .5,
                                     "anchored": False},
    "listwise_pairwise_anchored": {"trained": True, "hard": False,
                                   "list_weight": 1., "pair_weight": .5,
                                   "anchored": True},
}
MIXER_CONFIG = {"harm_weight": 2., "oracle_weight": .05}
CONTROLLER = "analytic_pi_g2"


def make_actions(probabilities, posterior, route, top_k, anchored):
    score = route["score"][:, :-1]
    analytic = route["analytic_score"][:, :-1]
    if anchored:
        anchor = analytic.argmax(1); order = np.argsort(-score, axis=1)
        candidates = np.empty((len(score), top_k), dtype=np.int64)
        candidates[:, 0] = anchor
        for row in range(len(score)):
            support = [index for index in order[row] if index != anchor[row]]
            candidates[row, 1:] = support[:top_k-1]
    else:
        candidates = np.argsort(-score, axis=1)[:, :top_k]
    rows = np.arange(len(probabilities))[:, None]
    selected = probabilities[rows, candidates]
    actions = np.concatenate((probabilities[:, -1:, :], posterior[:, None, :], selected), axis=1)
    return actions.astype(np.float32), candidates


def weighted_seed_metrics(frame):
    ids = {"dataset", "variant", "fold", "train_seed", "n_test"}
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns
               if column not in ids]
    rows = []
    for (variant, seed), values in frame.groupby(["variant", "train_seed"]):
        weight = values.n_test.to_numpy(float)
        row = {"dataset": values.dataset.iloc[0], "variant": variant,
               "train_seed": int(seed), "n_test": int(weight.sum())}
        row.update({column: float(np.average(values[column], weights=weight))
                    for column in numeric})
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset(dataset, device):
    cfg = CONFIGS[dataset]
    folds, modality_names = load_dataset(dataset, ROOT/"data")
    masks = nonempty_coalitions(len(modality_names)); top_k = 3 if len(modality_names) >= 3 else 2
    output_root = ROOT/"runs"/f"formal-i2-2-{dataset}"; output_root.mkdir(parents=True, exist_ok=True)
    partial = output_root/"metrics_by_fold_seed.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    completed = {(str(row["variant"]), int(row["fold"]), int(row["train_seed"])) for row in rows}

    for fold_index, splits in enumerate(folds):
        oof_path = (Path(cfg["oof_template"].format(fold=fold_index))
                    if "oof_template" in cfg else cfg["oof"])
        with np.load(oof_path) as saved:
            oof = {key: saved[key] for key in saved.files}
        for task_seed in SEEDS:
            missing = [name for name in VARIANTS if (name, fold_index, task_seed) not in completed]
            if not missing:
                continue
            backbone, temperatures, dims, classes = _load_backbone(
                cfg["base"]/f"fold_{fold_index}/seed_{task_seed}/backbone.pt", device)
            bundles = {}
            for split_name in ("selection", "test"):
                raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
                bundles[split_name] = attach_observed_losses(raw, splits[split_name]["y"])
            posterior = {"train_oof": []}
            for teacher_probability in oof["probabilities"]:
                _, value = load_posteriors(
                    cfg["posterior"], task_seed, dims, classes, masks, splits["train"],
                    teacher_probability, device, fold_index=fold_index)
                posterior["train_oof"].append(value)
            posterior["train_oof"] = np.stack(posterior["train_oof"])
            for split_name in ("selection", "test"):
                _, posterior[split_name] = load_posteriors(
                    cfg["posterior"], task_seed, dims, classes, masks, splits[split_name],
                    bundles[split_name]["probabilities"], device, fold_index=fold_index)

            router_cache, decision_cache = {}, {}
            seed = task_seed+fold_index*10000
            for variant in missing:
                config = VARIANTS[variant]
                training_key = (config["trained"], config["hard"],
                                config["list_weight"], config["pair_weight"])
                if training_key not in router_cache:
                    if not config["trained"]:
                        router_cache[training_key] = (None, 0., 0)
                    else:
                        target = make_target(oof["losses"], hard=config["hard"])
                        target["posterior_override"] = posterior["train_oof"]
                        selection_target = make_target(
                            bundles["selection"]["losses"][None], hard=config["hard"])
                        selection_target["posterior_override"] = posterior["selection"]
                        seed_all(seed)
                        router = AnalyticResidualListwiseRouter(
                            dims, classes, len(masks), residual_scale=.5).to(device)
                        history = train_router(
                            router, splits["train"], oof["probabilities"], target,
                            splits["selection"], bundles["selection"]["probabilities"],
                            selection_target, masks, device, seed=seed, epochs=100,
                            batch_size=cfg["batch_size"], patience=10,
                            objective_kwargs={"list_weight": config["list_weight"],
                                              "pair_weight": config["pair_weight"],
                                              "stable_weight": 0.})
                        selection_route = predict_router(
                            router, splits["selection"], bundles["selection"]["probabilities"],
                            masks, device, posterior["selection"])
                        blend, _ = select_residual_blend(
                            selection_route, bundles["selection"]["losses"])
                        router_cache[training_key] = (router, blend, len(history))
                router, blend, router_epochs = router_cache[training_key]

                route, actions = {}, {}
                for split_name in ("selection", "test"):
                    if router is None:
                        analytic = analytic_scores(
                            posterior[split_name], bundles[split_name]["probabilities"])
                        value = {"analytic_score": analytic, "score": analytic}
                    else:
                        value = predict_router(
                            router, splits[split_name], bundles[split_name]["probabilities"],
                            masks, device, posterior[split_name])
                        value = apply_residual_blend(value, blend)
                    route[split_name] = value
                    actions[split_name], _ = make_actions(
                        bundles[split_name]["probabilities"], posterior[split_name], value,
                        top_k, config["anchored"])

                train_actions, train_posteriors, train_labels = [], [], []
                for teacher_index, teacher_probability in enumerate(oof["probabilities"]):
                    if router is None:
                        analytic = analytic_scores(
                            posterior["train_oof"][teacher_index], teacher_probability)
                        value = {"analytic_score": analytic, "score": analytic}
                    else:
                        value = predict_router(
                            router, splits["train"], teacher_probability, masks, device,
                            posterior["train_oof"][teacher_index])
                        value = apply_residual_blend(value, blend)
                    current, _ = make_actions(
                        teacher_probability, posterior["train_oof"][teacher_index], value,
                        top_k, config["anchored"])
                    train_actions.append(current); train_posteriors.append(
                        posterior["train_oof"][teacher_index]); train_labels.append(splits["train"]["y"])
                train_actions = np.concatenate(train_actions)
                train_posteriors = np.concatenate(train_posteriors)
                train_labels = np.concatenate(train_labels)

                digest = hashlib.sha256()
                for value in (train_actions, actions["selection"], actions["test"]):
                    digest.update(np.ascontiguousarray(value).view(np.uint8))
                action_key = digest.hexdigest()
                if action_key in decision_cache:
                    (mixer_epochs, mixer_state, selected, raw_metric, adaptive_metric,
                     losses, prediction, alpha) = decision_cache[action_key]
                else:
                    seed_all(seed)
                    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
                    mixer_history = train_mixer(
                        mixer, train_actions, train_posteriors, train_labels,
                        actions["selection"], posterior["selection"], splits["selection"]["y"],
                        device, seed=seed, batch_size=cfg["batch_size"], **MIXER_CONFIG)
                    mixture = {split_name: predict_mixer(
                        mixer, actions[split_name], posterior[split_name], device)
                        for split_name in ("selection", "test")}
                    selected, _ = choose_strength(
                        actions["selection"][:, 0], mixture["selection"]["probability"],
                        posterior["selection"], splits["selection"]["y"], CONTROLLER)
                    gate = gate_values(CONTROLLER, actions["test"][:, 0],
                                       mixture["test"]["probability"], posterior["test"])
                    adaptive, alpha = shrink(actions["test"][:, 0],
                                             mixture["test"]["probability"], gate,
                                             selected["strength"])
                    raw_metric, _, _ = evaluate(
                        mixture["test"]["probability"], actions["test"][:, 0],
                        splits["test"]["y"], bundles["test"]["losses"], np.ones(len(gate)))
                    adaptive_metric, losses, prediction = evaluate(
                        adaptive, actions["test"][:, 0], splits["test"]["y"],
                        bundles["test"]["losses"], alpha)
                    mixer_epochs = len(mixer_history)
                    mixer_state = {key: value.detach().cpu().clone()
                                   for key, value in mixer.state_dict().items()}
                    decision_cache[action_key] = (mixer_epochs, mixer_state, selected,
                                                  raw_metric, adaptive_metric, losses,
                                                  prediction, alpha)

                routing = route_metrics(route["test"]["score"], route["test"]["analytic_score"],
                                        bundles["test"]["losses"],
                                        anchored=config["anchored"], top_k=top_k)
                rows.append({"dataset": dataset, "variant": variant, "fold": fold_index,
                             "train_seed": task_seed, "n_test": len(splits["test"]["y"]),
                             "router_epochs": router_epochs, "mixer_epochs": mixer_epochs,
                             "residual_blend": blend, "shrinkage_strength": selected["strength"],
                             **{f"route_{key}": value for key, value in routing.items()},
                             **{f"raw_{key}": value for key, value in raw_metric.items()},
                             **{f"adaptive_{key}": value for key, value in adaptive_metric.items()}})
                variant_root = output_root/f"fold_{fold_index}"/f"seed_{task_seed}"/variant
                variant_root.mkdir(parents=True, exist_ok=True)
                if router is not None:
                    torch.save({"state_dict": router.state_dict(), "blend": blend},
                               variant_root/"router.pt")
                torch.save({"state_dict": mixer_state, "classes": classes}, variant_root/"mixer.pt")
                pd.DataFrame({"sample_id": splits["test"]["id"],
                              "group_or_video_id": splits["test"]["group"],
                              "label": splits["test"]["y"], "loss": losses,
                              "prediction": prediction, "alpha": alpha}).to_parquet(
                                  variant_root/"predictions.parquet", index=False)
                pd.DataFrame(rows).to_csv(partial, index=False)
                print(f"{dataset} fold={fold_index} seed={task_seed} {variant}: "
                      f"coverage={routing['candidate_oracle_recovery']:.3f} "
                      f"nll={adaptive_metric['nll_gain']:.5f}", flush=True)

    frame = pd.DataFrame(rows); frame.to_csv(output_root/"metrics_by_fold_seed.csv", index=False)
    by_seed = weighted_seed_metrics(frame); by_seed.to_csv(output_root/"metrics_by_seed.csv", index=False)
    (output_root/"manifest.json").write_text(json.dumps({
        "experiment": "I2-2 routing-to-decision transfer", "dataset": dataset,
        "variants": VARIANTS, "seeds": list(SEEDS), "top_k": top_k,
        "fixed_mixer": MIXER_CONFIG, "fixed_controller": CONTROLLER,
        "selection_role": "early stopping, residual blend, and shrinkage strength only",
        "test_labels_role": "routing metrics and final evaluation only"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="mosi")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    datasets = tuple(CONFIGS) if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run_dataset(dataset, device)


if __name__ == "__main__":
    main()
