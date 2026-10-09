"""E11: transfer the same RCG correction across coalition-valid backbones.

This experiment deliberately reuses the frozen strong-observation checkpoints.
For each backbone and task seed it trains the same lightweight posterior,
analytic-residual router, and anchored candidate mixer.  Test labels are only
used after every selection decision has been frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import make_actions, predict_mixer, train_mixer
from rcg.listwise_router import (AnalyticResidualListwiseRouter,
                                 build_soft_coalition_targets)
from rcg.listwise_router_pipeline import (apply_residual_blend, predict_router,
                                          select_residual_blend, train_router)
from rcg.posterior_analytic import RelationalPosteriorEstimator
from rcg.posterior_analytic_pipeline import predict_posterior, train_posterior
from rcg.posterior_shrinkage_pipeline import probability_metrics
from rcg.projected_fusion_pipeline import observed_loss
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import attach_observed_losses, load_dataset, seed_all
from rcg.strong_models import build_model
from rcg.strong_observation import mask_name

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("concat", "tmc", "qmf", "pdf", "i2moe")
SEEDS = (11, 22, 33, 44, 55)
VARIANTS = ("base", "analytic", "anchored_router", "complete_rcg")
STRENGTHS = (0., .1, .25, .5, .75, 1.)


def gate_values(full, mixture, posterior):
    delta = np.log(np.clip(mixture, 1e-6, 1.)/np.clip(full, 1e-6, 1.))
    safe = (posterior*(delta > .01)).sum(1)
    return safe**2


def shrink(full, mixture, gate, strength):
    alpha = np.clip(float(strength)*gate, 0, 1)
    return (1-alpha[:, None])*full+alpha[:, None]*mixture, alpha


def choose_strength(full, mixture, posterior, labels):
    gate = gate_values(full, mixture, posterior)
    full_loss = observed_loss(full, labels)
    records = []
    for strength in STRENGTHS:
        probability, alpha = shrink(full, mixture, gate, strength)
        loss = observed_loss(probability, labels)
        records.append({"strength": strength, "selection_nll": float(loss.mean()),
                        "selection_clipped_harm": float(np.minimum(
                            np.maximum(loss-full_loss, 0), 1).mean()),
                        "selection_mean_alpha": float(alpha.mean())})
    feasible = [row for row in records if row["selection_clipped_harm"] <= .02]
    return min(feasible, key=lambda row: (row["selection_nll"],
                                          row["selection_mean_alpha"])), records


@torch.inference_mode()
def predict_strong(model, split, masks, temperatures, names, device, batch_size=512):
    """Return calibrated probabilities for every non-empty coalition."""
    model.eval()
    probabilities = []
    n = len(split["y"])
    for mask in masks:
        coalition = mask_name(tuple(int(x) for x in mask), names)
        temperature = float(temperatures[coalition])
        chunks = []
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            xs = [torch.as_tensor(x[start:stop], dtype=torch.float32, device=device)
                  for x in split["x"]]
            available = torch.as_tensor(mask, dtype=torch.float32, device=device)[None]
            available = available.expand(stop-start, -1)
            logits = model(xs, available)["logits"]
            chunks.append(torch.softmax(logits/temperature, -1).cpu().numpy())
        probabilities.append(np.concatenate(chunks))
    probability = np.stack(probabilities, axis=1).astype(np.float32)
    return attach_observed_losses({"probabilities": probability}, split["y"])


def load_strong(path, device):
    config = json.loads((path/"run.json").read_text(encoding="utf-8"))
    model = build_model(config["method"], config["dims"], config["classes"]).to(device)
    model.load_state_dict(torch.load(path/"model.pt", map_location=device,
                                     weights_only=True))
    model.eval()
    return model, config


def posterior_for(model, split, bundle, masks, device):
    return predict_posterior(model, split, bundle["probabilities"], masks, device)["posterior"]


def stage_probability(full, mixture, posterior, labels):
    selected, curve = choose_strength(full, mixture, posterior, labels)
    gate = gate_values(full, mixture, posterior)
    probability, alpha = shrink(full, mixture, gate, selected["strength"])
    return probability, alpha, selected, curve


def fit_adapter(splits, bundles, masks, dims, classes, seed, device, epochs, batch_size):
    # The transfer adapter is a normal label-posterior learner on a frozen task
    # model.  E11 does not reuse these in-sample targets as evidence for the OOF
    # contribution-supervision claim, which is evaluated separately in E2.
    seed_all(seed + 70000)
    posterior_model = RelationalPosteriorEstimator(
        dims, classes, len(masks), use_relations=True).to(device)
    posterior_history = train_posterior(
        posterior_model, splits["train"], bundles["train"]["probabilities"][None],
        splits["selection"], bundles["selection"]["probabilities"], masks, device,
        seed=seed+70000, epochs=epochs, batch_size=batch_size, patience=8,
        teacher_mode="single")
    posterior = {name: posterior_for(posterior_model, splits[name], bundles[name], masks, device)
                 for name in ("train", "selection", "test")}

    targets = {}
    for name in ("train", "selection"):
        targets[name] = build_soft_coalition_targets(bundles[name]["losses"][None])
        targets[name]["posterior_override"] = posterior[name]
    seed_all(seed + 71000)
    router = AnalyticResidualListwiseRouter(dims, classes, len(masks)).to(device)
    router_history = train_router(
        router, splits["train"], bundles["train"]["probabilities"], targets["train"],
        splits["selection"], bundles["selection"]["probabilities"], targets["selection"],
        masks, device, seed=seed+71000, epochs=epochs, batch_size=batch_size,
        patience=8)
    route = {name: predict_router(router, splits[name], bundles[name]["probabilities"],
                                  masks, device, posterior[name])
             for name in ("train", "selection", "test")}
    blend, blend_curve = select_residual_blend(route["selection"],
                                                bundles["selection"]["losses"])
    route = {name: apply_residual_blend(value, blend) for name, value in route.items()}
    top_k = min(3, len(masks))
    actions = {name: make_actions(bundles[name]["probabilities"], posterior[name],
                                  route[name], top_k)[0]
               for name in ("train", "selection", "test")}

    seed_all(seed + 72000)
    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
    mixer_history = train_mixer(
        mixer, actions["train"], posterior["train"], splits["train"]["y"],
        actions["selection"], posterior["selection"], splits["selection"]["y"],
        device, seed=seed+72000, epochs=epochs, batch_size=batch_size, patience=8)
    mixture = {name: predict_mixer(mixer, actions[name], posterior[name], device)["probability"]
               for name in ("selection", "test")}
    return posterior, actions, mixture, {
        "posterior_epochs": len(posterior_history), "router_epochs": len(router_history),
        "mixer_epochs": len(mixer_history), "residual_blend": float(blend),
        "blend_curve": blend_curve,
    }


def run(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset(args.dataset, args.data)
    masks = nonempty_coalitions(len(names))
    source = Path(args.source)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    metrics, prediction_rows, selection_rows = [], [], []
    for fold, splits in enumerate(folds):
      for method in args.methods:
        for seed in args.seeds:
            out = root/f"fold_{fold}"/method/f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            model, config = load_strong(source/args.dataset/f"fold_{fold}"/method/f"seed_{seed}", device)
            bundles = {name: predict_strong(model, splits[name], masks,
                                             config["temperatures"], names, device)
                       for name in ("train", "selection", "test")}
            posterior, actions, mixture, training = fit_adapter(
                splits, bundles, masks, config["dims"], config["classes"], seed,
                device, args.epochs, args.batch_size)

            selected = {}
            # Analytic stage: posterior correction only.
            selected["analytic"] = stage_probability(
                actions["selection"][:, 0], posterior["selection"], posterior["selection"],
                splits["selection"]["y"])
            # Anchored router stage: uniform convex use of the anchor-preserving action list.
            selected["anchored_router"] = stage_probability(
                actions["selection"][:, 0], actions["selection"].mean(1),
                posterior["selection"], splits["selection"]["y"])
            # Complete stage: learned anchored mixer followed by contribution-probability shrinkage.
            selected["complete_rcg"] = stage_probability(
                actions["selection"][:, 0], mixture["selection"], posterior["selection"],
                splits["selection"]["y"])

            full = actions["test"][:, 0]
            test_mix = {"base": full, "analytic": posterior["test"],
                        "anchored_router": actions["test"].mean(1),
                        "complete_rcg": mixture["test"]}
            labels = splits["test"]["y"]
            full_loss = observed_loss(full, labels)
            full_prediction = full.argmax(1)
            for variant in VARIANTS:
                if variant == "base":
                    probability, alpha, strength = full, np.zeros(len(full)), 0.
                else:
                    strength = float(selected[variant][2]["strength"])
                    gate = gate_values(full, test_mix[variant], posterior["test"])
                    probability, alpha = shrink(full, test_mix[variant], gate, strength)
                values, prediction = probability_metrics(probability, labels)
                loss = observed_loss(probability, labels)
                row = {"dataset": args.dataset, "fold": fold,
                       "backbone": method, "train_seed": seed,
                       "variant": variant, "n_test": len(labels), "strength": strength,
                       "mean_alpha": float(alpha.mean()), **values,
                       "nll_gain_vs_base": float(full_loss.mean()-loss.mean()),
                       "accuracy_gain_vs_base": float((prediction == labels).mean()-(full_prediction == labels).mean()),
                       "negative_flip_rate": float(((full_prediction == labels) & (prediction != labels)).mean()),
                       "correction_rate": float(((full_prediction != labels) & (prediction == labels)).mean()),
                       "clipped_harm": float(np.minimum(np.maximum(loss-full_loss, 0), 1).mean())}
                metrics.append(row)
                frame = pd.DataFrame({
                    "sample_id": splits["test"]["id"],
                    "group_or_video_id": splits["test"]["group"], "label": labels,
                    "dataset": args.dataset, "fold": fold,
                    "backbone": method, "train_seed": seed, "variant": variant,
                    "loss": loss, "correct": prediction == labels, "alpha": alpha})
                for k in range(probability.shape[1]):
                    frame[f"p{k}"] = probability[:, k]
                prediction_rows.append(frame)
                if variant == "base":
                    selection_probability = actions["selection"][:, 0]
                else:
                    selection_probability = selected[variant][0]
                selection_frame = pd.DataFrame({
                    "sample_id": splits["selection"]["id"],
                    "group_or_video_id": splits["selection"]["group"],
                    "label": splits["selection"]["y"], "dataset": args.dataset,
                    "fold": fold, "backbone": method, "train_seed": seed,
                    "variant": variant})
                for k in range(selection_probability.shape[1]):
                    selection_frame[f"p{k}"] = selection_probability[:, k]
                selection_rows.append(selection_frame)
            (out/"training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
            print(f"{method} seed={seed}: complete NLL gain="
                  f"{metrics[-1]['nll_gain_vs_base']:+.5f}, "
                  f"Acc gain={metrics[-1]['accuracy_gain_vs_base']:+.4f}", flush=True)

    table = pd.DataFrame(metrics)
    table.to_csv(root/"metrics_by_seed.csv", index=False)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_parquet(root/"predictions.parquet", index=False)
    pd.concat(selection_rows, ignore_index=True).to_parquet(
        root/"selection_predictions.parquet", index=False)
    summary = table.groupby(["backbone", "variant"]).agg(
        accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"), nll=("nll", "mean"), nll_std=("nll", "std"),
        brier=("brier", "mean"), ece=("ece", "mean"),
        nll_gain_vs_base=("nll_gain_vs_base", "mean"),
        accuracy_gain_vs_base=("accuracy_gain_vs_base", "mean"),
        negative_flip_rate=("negative_flip_rate", "mean"),
        correction_rate=("correction_rate", "mean"), clipped_harm=("clipped_harm", "mean"),
        mean_alpha=("mean_alpha", "mean")).reset_index()
    summary.to_csv(root/"metrics_summary.csv", index=False)
    manifest = {"experiment": "S4 backbone transfer", "dataset": args.dataset,
                "methods": list(args.methods), "seeds": list(args.seeds),
                "adapter": "shared posterior + analytic anchored listwise router + anchored mixer",
                "analytic_stage": "posterior action with analytic pi^2 shrinkage",
                "router_stage": "uniform anchored action mixture with analytic pi^2 shrinkage",
                "complete_stage": "learned anchored mixer with analytic pi^2 shrinkage",
                "selection_role": "early stopping, residual blend, and shrink strength",
                "test_labels_role": "evaluation only",
                "scope": "frozen-feature backbone portability; E2 separately validates OOF supervision"}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT/"data"))
    parser.add_argument("--source", default=str(ROOT/"runs"/"strong-observation"))
    parser.add_argument("--output", default=str(ROOT/"runs"/"formal-e11-backbone-transfer"))
    parser.add_argument("--dataset", choices=("mosi", "cremad"), default="mosi")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
