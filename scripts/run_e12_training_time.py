"""E12 representative MOSI training-time benchmark for the current RCG stack."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import predict_mixer, train_mixer
from rcg.listwise_router import (AnalyticResidualListwiseRouter,
                                 anchor_preserving_candidates,
                                 build_soft_coalition_targets)
from rcg.listwise_router_pipeline import (apply_residual_blend, predict_router,
                                          select_residual_blend, train_router)
from rcg.posterior_analytic import RelationalPosteriorEstimator
from rcg.posterior_analytic_pipeline import (calibrate_members,
                                              fit_ensemble_temperature,
                                              predict_posterior, train_posterior)
from rcg.rcg_fusion import CoalitionAwareBackbone, nonempty_coalitions
from rcg.rcg_fusion_pipeline import (attach_observed_losses, calibrate_temperatures,
                                     fit_backbone, generate_oof_teachers, load_dataset,
                                     predict_outputs, seed_all)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)


def timed(device, function):
    torch.cuda.synchronize(device); start = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    return value, time.perf_counter()-start


def actions(probabilities, posterior, route, top_k=3):
    analytic, score = route["analytic_score"][:, :-1], route["score"][:, :-1]
    candidates = anchor_preserving_candidates(analytic, score, min(top_k, analytic.shape[1]))
    rows = np.arange(len(probabilities))[:, None]
    selected = probabilities[rows, candidates]
    return np.concatenate((probabilities[:, -1:, :], posterior[:, None, :], selected), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT/"runs/formal-e12-training-benchmark"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal E12 timing requires CUDA")
    device = torch.device("cuda")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    timing_file = root/"training_times.csv"
    if timing_file.exists():
        print(pd.read_csv(timing_file).to_string(index=False)); return
    folds, names = load_dataset("mosi", ROOT/"data")
    splits = folds[0]; dims = [x.shape[1] for x in splits["train"]["x"]]
    classes = int(max(split["y"].max() for split in splits.values())+1)
    masks = nonempty_coalitions(len(names)); rows = []

    (root/"oof").mkdir(parents=True, exist_ok=True)
    oof, seconds = timed(device, lambda: generate_oof_teachers(
        splits["train"], dims, classes, str(device), folds=5, teacher_seeds=SEEDS,
        epochs=100, batch_size=64, output=root/"oof"))
    rows.append({"stage": "shared 5-fold x 5-seed OOF teacher bank",
                 "scope": "one-time per five-member group", "seconds": seconds})

    seed_all(11)
    backbone = CoalitionAwareBackbone(dims, classes).to(device)
    def train_task():
        history = fit_backbone(backbone, splits["train"], splits["selection"], seed=11,
                               epochs=100, batch_size=64, patience=10)
        temperature = calibrate_temperatures(backbone, splits["calibration"], str(device))
        return history, temperature
    (history, temperatures), seconds = timed(device, train_task)
    rows.append({"stage": "one coalition-aware task backbone + calibration",
                 "scope": "per deployed member", "seconds": seconds})
    bundles = {}
    for name in ("selection", "test"):
        raw = predict_outputs(backbone, splits[name]["x"], str(device), temperatures)
        bundles[name] = attach_observed_losses(raw, splits[name]["y"])

    posterior_models = []
    def train_posteriors():
        for posterior_seed in SEEDS:
            seed_all(11000+posterior_seed)
            model = RelationalPosteriorEstimator(dims, classes, len(masks),
                                                  use_relations=True).to(device)
            train_posterior(
                model, splits["train"], oof["probabilities"], splits["selection"],
                bundles["selection"]["probabilities"], masks, str(device),
                seed=11000+posterior_seed, epochs=100, batch_size=64,
                patience=10, teacher_mode="all")
            posterior_models.append(model)
        selection_members = [predict_posterior(
            model, splits["selection"], bundles["selection"]["probabilities"],
            masks, str(device))["posterior"] for model in posterior_models]
        temperature = fit_ensemble_temperature(selection_members, splits["selection"]["y"])
        selection_q = np.mean(calibrate_members(selection_members, temperature), 0)
        teacher_q = []
        for teacher_probability in oof["probabilities"]:
            members = [predict_posterior(model, splits["train"], teacher_probability,
                                         masks, str(device))["posterior"]
                       for model in posterior_models]
            teacher_q.append(np.mean(calibrate_members(members, temperature), 0))
        return selection_q, np.stack(teacher_q)
    (selection_q, teacher_q), seconds = timed(device, train_posteriors)
    rows.append({"stage": "five-model posterior ensemble + calibration",
                 "scope": "per deployed member", "seconds": seconds})

    train_target = build_soft_coalition_targets(oof["losses"])
    train_target["posterior_override"] = teacher_q
    selection_target = build_soft_coalition_targets(bundles["selection"]["losses"][None])
    selection_target["posterior_override"] = selection_q
    router = AnalyticResidualListwiseRouter(dims, classes, len(masks)).to(device)
    def train_route():
        train_router(router, splits["train"], oof["probabilities"], train_target,
                     splits["selection"], bundles["selection"]["probabilities"],
                     selection_target, masks, str(device), seed=12011, epochs=100,
                     batch_size=64, patience=10)
        selection = predict_router(router, splits["selection"],
                                   bundles["selection"]["probabilities"], masks,
                                   str(device), selection_q)
        blend, _ = select_residual_blend(selection, bundles["selection"]["losses"])
        return blend
    blend, seconds = timed(device, train_route)
    rows.append({"stage": "analytic-residual listwise router",
                 "scope": "per deployed member", "seconds": seconds})

    train_actions, train_q, train_y = [], [], []
    for teacher_probability, q in zip(oof["probabilities"], teacher_q):
        route = predict_router(router, splits["train"], teacher_probability,
                               masks, str(device), q)
        route = apply_residual_blend(route, blend)
        train_actions.append(actions(teacher_probability, q, route))
        train_q.append(q); train_y.append(splits["train"]["y"])
    train_actions, train_q, train_y = map(np.concatenate,
                                          (train_actions, train_q, train_y))
    selection_route = predict_router(router, splits["selection"],
                                     bundles["selection"]["probabilities"], masks,
                                     str(device), selection_q)
    selection_route = apply_residual_blend(selection_route, blend)
    selection_actions = actions(bundles["selection"]["probabilities"],
                                selection_q, selection_route)
    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
    _, seconds = timed(device, lambda: train_mixer(
        mixer, train_actions, train_q, train_y, selection_actions, selection_q,
        splits["selection"]["y"], str(device), seed=13011, epochs=100,
        batch_size=64, patience=10))
    rows.append({"stage": "anchored candidate mixer",
                 "scope": "per deployed member", "seconds": seconds})

    frame = pd.DataFrame(rows)
    marginal = frame.loc[frame.scope == "per deployed member", "seconds"].sum()
    shared = frame.loc[frame.scope.str.startswith("one-time"), "seconds"].sum()
    summary = pd.DataFrame([
        {"stage": "one deployed member (excluding shared OOF bank)",
         "scope": "derived total", "seconds": marginal},
        {"stage": "one standalone member (including one shared OOF bank)",
         "scope": "derived total", "seconds": shared+marginal},
        {"stage": "five-member complete training (including one shared OOF bank)",
         "scope": "derived total", "seconds": shared+5*marginal},
    ])
    frame = pd.concat((frame, summary), ignore_index=True)
    frame["gpu_hours"] = frame.seconds/3600
    frame.to_csv(timing_file, index=False)
    (root/"manifest.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(device), "dataset": "mosi",
        "features": "precomputed frozen features", "seeds": list(SEEDS),
        "oof": "5 grouped folds x 5 teacher seeds", "epochs": 100,
        "early_stopping": True, "feature_extraction_excluded": True,
        "derived_five_member_cost": "one shared OOF bank + 5 x per-member stages"
    }, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
