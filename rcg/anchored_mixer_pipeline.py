"""Train innovation three: full-anchored candidate-set fusion."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .anchored_mixer import (AnchoredCandidateMixer, anchored_mixer_objective,
                             gather_anchored_actions, shrink_to_full)
from .io import load_json, write_json
from .listwise_router import AnalyticResidualListwiseRouter, anchor_preserving_candidates
from .listwise_router_pipeline import apply_residual_blend, predict_router
from .posterior_analytic_pipeline import _load_backbone
from .posterior_shrinkage_pipeline import probability_metrics
from .projected_fusion_pipeline import load_posteriors, observed_loss
from .rcg_fusion import nonempty_coalitions
from .rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                  predict_outputs, seed_all)


DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def load_router(path: Path, device: str):
    saved = torch.load(path/"listwise_router.pt", map_location=device, weights_only=True)
    model = AnalyticResidualListwiseRouter(
        saved["dims"], saved["classes"], saved["n_coalitions"],
        residual_scale=saved["residual_scale"]).to(device)
    model.load_state_dict(saved["state_dict"]); model.eval()
    training = load_json(path/"training.json")
    return model, float(training["selected_residual_blend"])


def make_actions(probabilities, posterior, router_output, top_k=3):
    candidates = anchor_preserving_candidates(
        router_output["analytic_score"], router_output["score"],
        top_k=min(top_k, probabilities.shape[1]))
    with torch.no_grad():
        actions = gather_anchored_actions(
            torch.as_tensor(probabilities, dtype=torch.float32),
            torch.as_tensor(posterior, dtype=torch.float32),
            torch.as_tensor(candidates, dtype=torch.long)).numpy()
    return actions, candidates


def train_mixer(model, train_actions, train_posterior, train_labels,
                selection_actions, selection_posterior, selection_labels,
                device, *, seed, epochs=100, batch_size=128, patience=10):
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
            loss, _ = anchored_mixer_objective(
                output, ta[index], ty[index], harm_weight=2., oracle_weight=.05)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            train_values.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            output = model(va, vq)
            value = float(-output["probability"].clamp_min(1e-8).log()[
                torch.arange(len(vy), device=device), vy].mean())
        history.append({"epoch": epoch+1, "train": float(np.mean(train_values)),
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


@torch.inference_mode()
def predict_mixer(model, actions, posterior, device):
    output = model(torch.as_tensor(actions, dtype=torch.float32, device=device),
                   torch.as_tensor(posterior, dtype=torch.float32, device=device))
    return {key: value.cpu().numpy() for key, value in output.items()}


def analytic_action_probability(full, mixture, posterior, tolerance=.01):
    delta = np.log(np.clip(mixture, 1e-6, 1.)/np.clip(full, 1e-6, 1.))
    return (posterior*(delta > tolerance)).sum(1)


def apply_adaptive_policy(full, mixture, posterior, alpha_max, gamma):
    safe = analytic_action_probability(full, mixture, posterior)
    alpha = float(alpha_max)*safe**float(gamma)
    return (1-alpha[:, None])*full+alpha[:, None]*mixture, alpha, safe


def select_adaptive_policy(full, mixture, posterior, labels,
                           alpha_grid=(.5, .75, 1.), gamma_grid=(.5, 1., 2., 3., 4.)):
    rows = np.arange(len(labels)); records = []
    full_loss = -np.log(np.clip(full[rows, labels], 1e-12, 1))
    for alpha_max in alpha_grid:
      for gamma in gamma_grid:
        probability, alpha, _ = apply_adaptive_policy(
            full, mixture, posterior, alpha_max, gamma)
        loss = -np.log(np.clip(probability[rows, labels], 1e-12, 1))
        records.append({"alpha_max": float(alpha_max), "gamma": float(gamma),
                        "mean_alpha": float(alpha.mean()), "nll": float(loss.mean()),
                        "clipped_harm": float(np.minimum(np.maximum(loss-full_loss, 0), 1).mean())})
    feasible = [item for item in records if item["clipped_harm"] <= .02]
    best = min(feasible or records, key=lambda item: (item["nll"], item["mean_alpha"]))
    return best, records


def run(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset(args.dataset, args.data); masks = nonempty_coalitions(len(names))
    base, posterior_root = Path(args.base_run), Path(args.posterior_run)
    router_root, root = Path(args.router_run), Path(args.output); root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": args.dataset, "base_run": str(base.resolve()),
                "posterior_run": str(posterior_root.resolve()),
                "router_run": str(router_root.resolve()), "seeds": args.seeds,
                "top_k": args.top_k, "test_labels_role": "evaluation only",
                "method": "full anchor + posterior + analytic top1 + residual support"}
    if (root/"manifest.json").exists() and load_json(root/"manifest.json") != manifest:
        raise ValueError("manifest differs; use a new output directory")
    write_json(root/"manifest.json", manifest); metrics, start = [], time.perf_counter()
    for fold_index, splits in enumerate(folds):
        fold_base = base/f"fold_{fold_index}"
        with np.load(fold_base/"oof_targets.npz") as saved:
            oof = {key: saved[key] for key in saved.files}
        for task_seed in args.seeds:
            out = root/f"fold_{fold_index}"/f"seed_{task_seed}"; out.mkdir(parents=True, exist_ok=True)
            backbone, temperatures, dims, classes = _load_backbone(
                fold_base/f"seed_{task_seed}"/"backbone.pt", device)
            router, blend = load_router(
                router_root/f"fold_{fold_index}"/f"seed_{task_seed}", device)
            bundles, posteriors, router_outputs = {}, {}, {}
            for split_name in ("selection", "test"):
                raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
                bundles[split_name] = attach_observed_losses(raw, splits[split_name]["y"])
                _, posteriors[split_name] = load_posteriors(
                    posterior_root, task_seed, dims, classes, masks, splits[split_name],
                    bundles[split_name]["probabilities"], device, fold_index=fold_index)
                value = predict_router(router, splits[split_name],
                                       bundles[split_name]["probabilities"], masks,
                                       device, posteriors[split_name])
                router_outputs[split_name] = apply_residual_blend(value, blend)
            train_actions, train_posteriors, train_labels = [], [], []
            for teacher_probability in oof["probabilities"]:
                _, q = load_posteriors(posterior_root, task_seed, dims, classes, masks,
                                       splits["train"], teacher_probability, device,
                                       fold_index=fold_index)
                value = predict_router(router, splits["train"], teacher_probability,
                                       masks, device, q)
                value = apply_residual_blend(value, blend)
                actions, _ = make_actions(teacher_probability, q, value, args.top_k)
                train_actions.append(actions); train_posteriors.append(q)
                train_labels.append(splits["train"]["y"])
            train_actions = np.concatenate(train_actions)
            train_posteriors = np.concatenate(train_posteriors)
            train_labels = np.concatenate(train_labels)
            selection_actions, _ = make_actions(
                bundles["selection"]["probabilities"], posteriors["selection"],
                router_outputs["selection"], args.top_k)
            seed_all(task_seed+fold_index*10000)
            mixer = AnchoredCandidateMixer(classes).to(device)
            history = train_mixer(
                mixer, train_actions, train_posteriors, train_labels,
                selection_actions, posteriors["selection"], splits["selection"]["y"],
                device, seed=task_seed+fold_index*10000, epochs=args.epochs,
                batch_size=args.batch_size)
            selection_output = predict_mixer(mixer, selection_actions,
                                             posteriors["selection"], device)
            policy, policy_grid = select_adaptive_policy(
                selection_actions[:, 0], selection_output["probability"],
                posteriors["selection"], splits["selection"]["y"])
            test_actions, candidates = make_actions(
                bundles["test"]["probabilities"], posteriors["test"],
                router_outputs["test"], args.top_k)
            output = predict_mixer(mixer, test_actions, posteriors["test"], device)
            full = test_actions[:, 0]; mixture = output["probability"]
            final, sample_alpha, safe_probability = apply_adaptive_policy(
                full, mixture, posteriors["test"], policy["alpha_max"], policy["gamma"])
            labels = splits["test"]["y"]
            full_metrics, full_pred = probability_metrics(full, labels)
            final_metrics, final_pred = probability_metrics(final, labels)
            full_loss = observed_loss(full, labels); final_loss = observed_loss(final, labels)
            oracle_loss = bundles["test"]["losses"].min(1)
            negative = (full_pred == labels) & (final_pred != labels)
            correction = (full_pred != labels) & (final_pred == labels)
            full_regret = float((full_loss-oracle_loss).mean())
            item = {"dataset": args.dataset, "fold": fold_index, "train_seed": task_seed,
                    "n_test": len(labels), "alpha_max": policy["alpha_max"],
                    "gamma": policy["gamma"], "mean_alpha": float(sample_alpha.mean()),
                    **{f"full_{k}": v for k, v in full_metrics.items()},
                    **{f"final_{k}": v for k, v in final_metrics.items()},
                    "relative_nll_reduction": float((full_metrics["nll"]-final_metrics["nll"])/full_metrics["nll"]),
                    "accuracy_gain": float(final_metrics["accuracy"]-full_metrics["accuracy"]),
                    "clipped_harm": float(np.minimum(np.maximum(final_loss-full_loss, 0), 1).mean()),
                    "negative_flip_rate": float(negative.mean()),
                    "correction_rate": float(correction.mean()),
                    "regret_reduction": float(1-(final_loss-oracle_loss).mean()/max(full_regret, 1e-12)),
                    "mean_full_weight": float(output["weight"][:, 0].mean()),
                    "mean_posterior_weight": float(output["weight"][:, 1].mean())}
            metrics.append(item); write_json(out/"metrics.json", item)
            torch.save({"state_dict": mixer.state_dict(), "classes": classes}, out/"mixer.pt")
            write_json(out/"training.json", {"epochs": history,
                                               "adaptive_policy_grid": policy_grid,
                                               "selected_policy": policy})
            frame = pd.DataFrame({"sample_id": splits["test"]["id"],
                                  "group_or_video_id": splits["test"]["group"],
                                  "label": labels, "train_seed": task_seed,
                                  "full_loss": full_loss, "final_loss": final_loss})
            frame["adaptive_alpha"] = sample_alpha
            frame["analytic_safe_probability"] = safe_probability
            for rank in range(candidates.shape[1]):
                frame[f"candidate_{rank+1}"] = candidates[:, rank]
            for index in range(final.shape[1]):
                frame[f"full_p{index}"] = full[:, index]
                frame[f"final_p{index}"] = final[:, index]
            frame.to_parquet(out/"predictions.parquet", index=False)
            write_json(out/"complete.json", {"n_test": len(labels)})
            print(f"{args.dataset} fold={fold_index} seed={task_seed}: "
                  f"acc={item['accuracy_gain']:+.4f} rel_nll={item['relative_nll_reduction']:+.3f} "
                  f"regret={item['regret_reduction']:+.3f}", flush=True)
    table = pd.DataFrame(metrics); table.to_csv(root/"metrics_by_fold_seed.csv", index=False)
    weighted = []
    for seed, values in table.groupby("train_seed"):
        w = values.n_test.to_numpy(float); row = {"dataset": args.dataset,
                                                   "train_seed": int(seed), "n_test": int(w.sum())}
        for column in values.select_dtypes(include=[np.number]).columns.difference(
                ["fold", "train_seed", "n_test"]):
            row[column] = float(np.average(values[column], weights=w))
        weighted.append(row)
    by_seed = pd.DataFrame(weighted); by_seed.to_csv(root/"metrics_by_seed.csv", index=False)
    numeric = by_seed.select_dtypes(include=[np.number]).columns.difference(["train_seed", "n_test"])
    pd.DataFrame({"metric": numeric, "mean": [by_seed[c].mean() for c in numeric],
                  "std": [by_seed[c].std(ddof=1) for c in numeric]}).to_csv(
                      root/"metrics_summary.csv", index=False)
    mean = by_seed.mean(numeric_only=True)
    checks = {"top1_preserved_by_construction": True,
              "nll_noninferior_4_of_5": int((by_seed.final_nll <= by_seed.full_nll).sum()) >= 4,
              "regret_reduction_10pct": mean.regret_reduction >= .10,
              "clipped_harm_2pct": mean.clipped_harm <= .02,
              "corrections_exceed_flips": mean.correction_rate >= mean.negative_flip_rate}
    write_json(root/"complete.json", {"checks": checks, "passed": all(checks.values()),
                                      "wall_seconds": time.perf_counter()-start})


def main():
    parser = argparse.ArgumentParser(description="Full-anchored candidate-set mixer")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist"), required=True)
    parser.add_argument("--data", default="data"); parser.add_argument("--base-run", required=True)
    parser.add_argument("--posterior-run", required=True); parser.add_argument("--router-run", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--top-k", type=int, default=3); parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
