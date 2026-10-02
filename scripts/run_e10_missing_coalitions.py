"""E10 missing-modality and arbitrary available-coalition experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer_pipeline import analytic_action_probability, predict_mixer
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.posterior_shrinkage_pipeline import probability_metrics
from rcg.rcg_fusion import CoalitionAwareBackbone, nonempty_coalitions
from rcg.rcg_fusion_pipeline import (calibrate_temperatures, fit_backbone,
                                     load_dataset, predict_outputs)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402
from run_e6_adaptive_shrinkage import choose_strength, shrink  # noqa: E402
from run_e9_clean_stress import load_context, metrics, posterior  # noqa: E402


def mask_name(mask, names) -> str:
    return "+".join(name for name, keep in zip(names, mask) if keep)


def train_variant(dataset: str, fold: int, seed: int, splits: dict,
                  mode: str, device: str) -> Path:
    output = ROOT / f"runs/formal-e10-backbones/{dataset}/{mode}/fold_{fold}/seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "backbone.pt"
    if checkpoint.exists():
        return checkpoint
    dims = [part.shape[1] for part in splits["train"]["x"]]
    classes = int(max(np.max(part["y"]) for part in splits.values())+1)
    model = CoalitionAwareBackbone(dims, classes).to(device)
    history = fit_backbone(
        model, splits["train"], splits["selection"], seed=seed+fold*10000,
        epochs=100, batch_size=CONFIGS[dataset]["batch_size"], patience=10,
        coalition_sampling=mode,
    )
    temperatures = calibrate_temperatures(model, splits["selection"], device)
    torch.save({"state_dict": model.state_dict(), "dims": dims, "classes": classes,
                "temperatures": temperatures}, checkpoint)
    (output / "training.json").write_text(json.dumps({
        "dataset": dataset, "fold": fold, "seed": seed, "sampling": mode,
        "epochs_ran": len(history),
        "best_clean_selection_full_nll": min(row["selection_full_nll"] for row in history),
        "history": history,
    }, indent=2), encoding="utf-8")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return checkpoint


@torch.inference_mode()
def zeroing_probability(model, temperatures, split, availability, device):
    xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in split["x"]]
    xs = [value*float(keep) for value, keep in zip(xs, availability)]
    full = torch.ones((len(split["y"]), len(xs)), dtype=torch.float32, device=device)
    logits = model(xs, full)["logits"] / max(float(temperatures[-1]), 1e-6)
    return torch.softmax(logits, -1).cpu().numpy()


def analytic_candidates(q, probabilities, masks, availability, top_k):
    reference_index = masks.index(tuple(availability)); reference = probabilities[:, reference_index]
    delta = np.clip(np.log(np.clip(probabilities, 1e-6, 1))-
                    np.log(np.clip(reference[:, None], 1e-6, 1)), -1, 1)
    gain = (q[:, None]*np.maximum(delta, 0)).sum(-1)
    harm = (q[:, None]*np.maximum(-delta, 0)).sum(-1)
    score = gain-2*harm
    valid = np.asarray([all(not keep or availability[index]
                            for index, keep in enumerate(mask)) for mask in masks])
    score[:, ~valid] = -np.inf; score[:, reference_index] = -np.inf
    order = np.argsort(-score, axis=1)
    result = np.full((len(q), top_k), reference_index, dtype=np.int64)
    number = min(top_k, int(valid.sum())-1)
    if number > 0:
        result[:, :number] = order[:, :number]
    return result, reference_index, valid


def rcg_member(context, split, masks, availability, top_k, device, strength=None):
    raw = predict_outputs(context["backbone"], split["x"], device, context["temperatures"])
    available = np.repeat(np.asarray(availability, dtype=np.float32)[None], len(split["y"]), axis=0)
    q = posterior(context, split, raw["probabilities"], masks, device, availability=available)
    candidates, reference_index, valid = analytic_candidates(
        q, raw["probabilities"], masks, availability, top_k)
    reference = raw["probabilities"][:, reference_index]
    # With a singleton availability set there is no alternative coalition to
    # fuse. The only legal behavior is exact reference fallback.
    if int(valid.sum()) == 1:
        return {"probability": reference, "reference": reference, "posterior": q,
                "alpha": np.zeros(len(q)), "candidates": candidates, "strength": 0.,
                "reference_index": reference_index}
    rows = np.arange(len(q))[:, None]
    actions = np.concatenate((reference[:, None],
                              q[:, None], raw["probabilities"][rows, candidates]), axis=1)
    mixture = predict_mixer(context["mixer"], actions, q, device)["probability"]
    if strength is None:
        chosen, _ = choose_strength(actions[:, 0], mixture, q, split["y"], "analytic_pi_g2")
        strength = float(chosen["strength"])
    pi = analytic_action_probability(actions[:, 0], mixture, q, tolerance=.01)
    probability, alpha = shrink(actions[:, 0], mixture, pi**2, strength)
    if not all(valid[index] for index in np.unique(candidates)):
        raise RuntimeError("an unavailable coalition entered the candidate set")
    return {"probability": probability, "reference": actions[:, 0], "posterior": q,
            "alpha": alpha, "candidates": candidates, "strength": strength,
            "reference_index": reference_index}


def run_dataset(dataset: str, device: str) -> None:
    folds, names = load_dataset(dataset, ROOT / "data")
    masks = nonempty_coalitions(len(names)); top_k = 3 if len(names) >= 3 else 2
    output = ROOT / f"runs/formal-e10-{dataset}"; output.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_frames, oracle_rows, configurations = [], [], [], []
    for fold, splits in enumerate(folds):
        models = {"full_zeroing": [], "modality_dropout": [], "coalition_dropout": []}
        contexts = []
        for seed in SEEDS:
            full_path = train_variant(dataset, fold, seed, splits, "full_only", device)
            modality_path = train_variant(dataset, fold, seed, splits, "modality_dropout", device)
            models["full_zeroing"].append(_load_backbone(full_path, device)[:2])
            models["modality_dropout"].append(_load_backbone(modality_path, device)[:2])
            clean_path = CONFIGS[dataset]["base"] / f"fold_{fold}/seed_{seed}/backbone.pt"
            models["coalition_dropout"].append(_load_backbone(clean_path, device)[:2])
            contexts.append(load_context(dataset, fold, seed, masks, device))
        for availability in masks:
            availability = tuple(availability); alliance = mask_name(availability, names)
            alliance_index = masks.index(availability)
            selection_rcg, strengths = [], []
            for context in contexts:
                selected = rcg_member(context, splits["selection"], masks, availability,
                                      top_k, device, strength=None)
                selection_rcg.append(selected); strengths.append(selected["strength"])
            method_probabilities = {}
            for method, members in models.items():
                values = []
                for model, temperatures in members:
                    if method == "full_zeroing":
                        values.append(zeroing_probability(
                            model, temperatures, splits["test"], availability, device))
                    else:
                        values.append(predict_outputs(
                            model, splits["test"]["x"], device, temperatures
                        )["probabilities"][:, alliance_index])
                method_probabilities[method] = np.mean(values, axis=0)
            rcg_outputs = [rcg_member(context, splits["test"], masks, availability,
                                      top_k, device, strength=strength)
                           for context, strength in zip(contexts, strengths)]
            method_probabilities["availability_safe_rcg"] = np.mean(
                [value["probability"] for value in rcg_outputs], axis=0)
            labels = splits["test"]["y"]; reference = method_probabilities["coalition_dropout"]
            for method, probability in method_probabilities.items():
                row = {"dataset": dataset, "fold": fold, "available_mask": "".join(map(str, availability)),
                       "available_modalities": alliance, "available_count": sum(availability),
                       "method": method, "n_test": len(labels), **metrics(probability, labels, reference)}
                if method == "availability_safe_rcg":
                    row["mean_alpha"] = float(np.mean([value["alpha"].mean() for value in rcg_outputs]))
                    row["mean_strength"] = float(np.mean(strengths))
                metric_rows.append(row)
                frame = pd.DataFrame({
                    "dataset": dataset, "fold": fold, "method": method,
                    "available_mask": "".join(map(str, availability)),
                    "available_modalities": alliance, "sample_id": splits["test"]["id"],
                    "group_or_video_id": splits["test"]["group"], "label": labels,
                })
                for class_index in range(probability.shape[1]):
                    frame[f"p{class_index}"] = probability[:, class_index]
                prediction_frames.append(frame)

            # Oracle headroom uses the five-member coalition-dropout ensemble.
            member_all = [predict_outputs(model, splits["test"]["x"], device, temperature)["probabilities"]
                          for model, temperature in models["coalition_dropout"]]
            coalition_probability = np.mean(member_all, axis=0)
            valid_indices = [index for index, mask in enumerate(masks)
                             if all(not keep or availability[j] for j, keep in enumerate(mask))]
            rows = np.arange(len(labels))[:, None]
            all_indices = np.arange(len(masks))[None]
            all_loss = -np.log(np.clip(
                coalition_probability[rows, all_indices, labels[:, None]], 1e-12, 1))
            loss = all_loss[:, valid_indices]
            oracle_local = loss.argmin(1); oracle_index = np.asarray(valid_indices)[oracle_local]
            reference_loss = -np.log(np.clip(reference[np.arange(len(labels)), labels], 1e-12, 1))
            oracle_loss = loss.min(1)
            oracle_rows.append({
                "dataset": dataset, "fold": fold, "available_mask": "".join(map(str, availability)),
                "available_modalities": alliance, "available_count": sum(availability),
                "n_test": len(labels), "reference_nll": float(reference_loss.mean()),
                "valid_oracle_nll": float(oracle_loss.mean()),
                "reference_regret": float((reference_loss-oracle_loss).mean()),
                "reference_oracle_rate": float((oracle_index == alliance_index).mean()),
                "oracle_single_rate": float(np.mean([sum(masks[index]) == 1 for index in oracle_index])),
                "oracle_pair_rate": float(np.mean([sum(masks[index]) == 2 for index in oracle_index])),
                "oracle_full_available_rate": float(np.mean([
                    sum(masks[index]) == sum(availability) for index in oracle_index])),
            })
            configurations.extend({
                "dataset": dataset, "fold": fold, "train_seed": seed,
                "available_mask": "".join(map(str, availability)), "strength": strength,
                "selection_mean_alpha": float(value["alpha"].mean()),
            } for seed, strength, value in zip(SEEDS, strengths, selection_rcg))
            print(f"{dataset} fold={fold} available={alliance}: "
                  f"zero={metric_rows[-4]['nll']:.3f} modality={metric_rows[-3]['nll']:.3f} "
                  f"coalition={metric_rows[-2]['nll']:.3f} rcg={metric_rows[-1]['nll']:.3f}", flush=True)
        del models, contexts
        if device == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(metric_rows).to_csv(output / "metrics_by_fold_alliance.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    pd.DataFrame(oracle_rows).to_csv(output / "oracle_by_fold_alliance.csv", index=False)
    pd.DataFrame(configurations).to_csv(output / "configuration.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "experiment": "E10 missing modalities and arbitrary available coalitions",
        "dataset": dataset, "seeds": list(SEEDS), "modalities": list(names),
        "training_protocols": ["full_only_then_zero", "independent_modality_dropout",
                               "50_percent_full_uniform_incomplete_coalition_dropout"],
        "availability_safe_rcg": "masked posterior + valid analytic candidates + anchored mixer + selection shrink",
        "missing_modalities_in_posterior_or_candidates": False,
        "test_labels_role": "metrics and oracle analysis only",
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    for dataset in (CONFIGS if args.dataset == "all" else (args.dataset,)):
        run_dataset(dataset, device)


if __name__ == "__main__":
    main()
