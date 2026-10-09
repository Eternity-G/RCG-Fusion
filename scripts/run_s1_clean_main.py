"""S1: compute-matched clean-test comparison for the canonical P0-v2 system.

This script never trains or selects on test labels. It assembles already frozen
five-member predictions, evaluates every method on aligned samples, and uses
paired group bootstrap for the canonical RCG-vs-baseline comparisons.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcg.posterior_shrinkage_pipeline import probability_metrics  # noqa: E402
from rcg.stable_ensemble import fit_simplex_weights, mix_actions  # noqa: E402
from rcg.stats import holm  # noqa: E402
from rcg.strong_models import build_model  # noqa: E402
from rcg.strong_observation import (avmnistsplits, cremadsplits, mask_name,
                                    mmsasplits, split_tensor)  # noqa: E402

DATASETS = ("mosi", "mosei", "cremad", "avmnist")
METHODS = ("concat", "tmc", "qmf", "pdf", "i2moe")
SEEDS = (11, 22, 33, 44, 55)
DISPLAY = {
    "full_only": "Full-only concat",
    "modality_dropout": "Modality dropout",
    "coalition_dropout": "Coalition dropout",
    "best_unimodal": "Best unimodal",
    "equal_probability": "Equal probability",
    "confidence_weighted": "Confidence weighted",
    "entropy_weighted": "Entropy weighted",
    "static_late_fusion": "Static late fusion",
    "concat": "Coalition concat",
    "tmc": "TMC",
    "qmf": "QMF",
    "pdf": "PDF",
    "i2moe": "I²MoE",
    "rcg_fusion_a8_v2": "RCG-Fusion",
}


def probability_columns(frame: pd.DataFrame, prefix: str = "p") -> list[str]:
    columns = [column for column in frame if column.startswith(prefix)]
    if prefix == "p":
        columns = [column for column in columns if column[1:].isdigit()]
        columns = [column for column in columns if not frame[column].isna().all()]
        return sorted(columns, key=lambda value: int(value[1:]))
    columns = [column for column in columns if not frame[column].isna().all()]
    return sorted(columns, key=lambda value: int(value[len(prefix):]))


def canonical(dataset: str) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT/f"runs/formal-e8-{dataset}/final_predictions.parquet")
    columns = probability_columns(frame)
    keep = frame[["fold", "sample_id", "group_or_video_id", "label", *columns]].copy()
    keep["method"] = "rcg_fusion_a8_v2"
    return keep


def e10_baselines(dataset: str) -> list[pd.DataFrame]:
    source = ROOT/f"runs/formal-e10-{dataset}/predictions.parquet"
    frame = pd.read_parquet(source)
    full_mask = "1"*len(str(frame.available_mask.iloc[0]))
    frame.available_mask = frame.available_mask.astype(str).str.zfill(len(full_mask))
    frame = frame[(frame.available_mask == full_mask) & frame.method.isin(
        ("full_zeroing", "modality_dropout", "coalition_dropout"))].copy()
    frame.method = frame.method.replace({"full_zeroing": "full_only"})
    return [part.drop(columns=[column for column in ("available_mask", "available_modalities")
                               if column in part])
            for _, part in frame.groupby("method", sort=False)]


def strong_ensemble(dataset: str, method: str) -> pd.DataFrame:
    records = []
    root = ROOT/"runs/strong-observation"/dataset
    for fold_path in sorted(root.glob("fold_*"), key=lambda path: int(path.name.split("_")[-1])):
        fold = int(fold_path.name.split("_")[-1]); members = []
        for seed in SEEDS:
            path = fold_path/method/f"seed_{seed}"/"samples.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path)
            columns = sorted([column for column in frame
                              if column.startswith("full_p") and column[6:].isdigit()],
                             key=lambda value: int(value[6:]))
            member = frame[["sample_id", "group_or_video_id", "label", *columns]].copy()
            member = member.rename(columns={column: f"p{int(column[6:])}" for column in columns})
            members.append(member)
        keys = ["sample_id", "group_or_video_id", "label"]
        aligned = members[0][keys].copy()
        pcols = probability_columns(members[0])
        probability = np.mean([member[pcols].to_numpy(float) for member in members], axis=0)
        aligned[pcols] = probability; aligned["fold"] = fold; aligned["method"] = method
        records.append(aligned)
    return pd.concat(records, ignore_index=True)


def dataset_splits(dataset: str):
    if dataset == "cremad":
        return cremadsplits(ROOT/"data/cremad/pretrained/cremad_wav2vec2_r3d18.npz")
    if dataset == "avmnist":
        return avmnistsplits(ROOT/"data")
    return mmsasplits(ROOT/"data", dataset)


@torch.inference_mode()
def basic_probability_baselines(dataset: str) -> list[pd.DataFrame]:
    folds, names = dataset_splits(dataset); output = []
    for fold, splits in enumerate(folds):
        ensemble = {split: [[] for _ in names] for split in ("selection", "test")}
        for seed in SEEDS:
            root = ROOT/f"runs/strong-observation/{dataset}/fold_{fold}/concat/seed_{seed}"
            metadata = json.loads((root/"run.json").read_text(encoding="utf-8"))
            dims = [int(value) for value in metadata["dims"]]; classes = int(metadata["classes"])
            model = build_model("concat", dims, classes).to("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(root/"model.pt", map_location=next(model.parameters()).device,
                                             weights_only=True)); model.eval()
            for split_name in ("selection", "test"):
                xs, _ = split_tensor(splits[split_name], next(model.parameters()).device)
                for modality in range(len(names)):
                    mask = tuple(int(index == modality) for index in range(len(names)))
                    present = torch.tensor(mask, dtype=torch.float32, device=xs[0].device)[None].repeat(len(xs[0]), 1)
                    logits = model(xs, present)["logits"]
                    temperature = float(metadata["temperatures"][mask_name(mask, names)])
                    ensemble[split_name][modality].append(
                        torch.softmax(logits/max(temperature, 1e-6), -1).cpu().numpy())
            del model
        actions = {split: np.stack([np.mean(value, axis=0) for value in ensemble[split]])
                   for split in ensemble}
        selection_labels = splits["selection"]["y"]
        selection_losses = [-np.log(np.clip(action[np.arange(len(selection_labels)), selection_labels],
                                              1e-12, 1)).mean()
                            for action in actions["selection"]]
        best = int(np.argmin(selection_losses))
        static_weights = fit_simplex_weights(actions["selection"], selection_labels, l2=1e-3)
        test = actions["test"]
        confidence = test.max(-1); confidence /= np.clip(confidence.sum(0, keepdims=True), 1e-12, None)
        entropy = -(test*np.log(np.clip(test, 1e-12, 1))).sum(-1)
        quality = 1-entropy/np.log(test.shape[-1]); total = quality.sum(0, keepdims=True)
        quality = np.divide(quality, total, out=np.full_like(quality, 1/len(names)), where=total > 1e-12)
        values = {
            "best_unimodal": test[best],
            "equal_probability": test.mean(0),
            "confidence_weighted": np.einsum("mn,mnk->nk", confidence, test),
            "entropy_weighted": np.einsum("mn,mnk->nk", quality, test),
            "static_late_fusion": mix_actions(test, static_weights),
        }
        for method, probability in values.items():
            frame = pd.DataFrame({"dataset": dataset, "fold": fold, "method": method,
                                  "sample_id": splits["test"]["id"],
                                  "group_or_video_id": splits["test"]["group"],
                                  "label": splits["test"]["y"]})
            for class_index in range(probability.shape[1]):
                frame[f"p{class_index}"] = probability[:, class_index]
            output.append(frame)
    joined = pd.concat(output, ignore_index=True)
    return [part.copy() for _, part in joined.groupby("method", sort=False)]


def align_frames(dataset: str) -> pd.DataFrame:
    frames = [canonical(dataset), *e10_baselines(dataset), *basic_probability_baselines(dataset)]
    frames.extend(strong_ensemble(dataset, method) for method in METHODS)
    reference = frames[0][["fold", "sample_id", "group_or_video_id", "label"]].copy()
    reference.sample_id = reference.sample_id.astype(str)
    output = []
    for frame in frames:
        frame = frame.copy(); frame.sample_id = frame.sample_id.astype(str)
        if "fold" not in frame: frame["fold"] = 0
        pcols = probability_columns(frame)
        candidate = frame[["fold", "sample_id", *pcols]].copy()
        merged = reference.merge(candidate, on=["fold", "sample_id"], validate="one_to_one")
        if len(merged) != len(reference):
            raise AssertionError(f"unaligned {dataset}/{frame.method.iloc[0]}")
        merged["dataset"] = dataset; merged["method"] = frame.method.iloc[0]
        output.append(merged)
    return pd.concat(output, ignore_index=True)


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), frame in predictions.groupby(["dataset", "method"], sort=False):
        pcols = probability_columns(frame); probability = frame[pcols].to_numpy(float)
        metric, _ = probability_metrics(probability, frame.label.to_numpy(int))
        rows.append({"dataset": dataset, "method": method, "display": DISPLAY[method],
                     "n_test": len(frame), **metric})
    return pd.DataFrame(rows)


def paired_bootstrap(predictions: pd.DataFrame, repetitions=10_000, seed=20261009):
    rng = np.random.default_rng(seed); rows = []
    for dataset, data in predictions.groupby("dataset", sort=False):
        p0 = data[data.method == "rcg_fusion_a8_v2"].copy()
        pcols = probability_columns(p0); labels = p0.label.to_numpy(int)
        p0_prob = p0[pcols].to_numpy(float); indexes = np.arange(len(labels))
        p0_loss = -np.log(np.clip(p0_prob[indexes, labels], 1e-12, 1))
        p0_correct = (p0_prob.argmax(1) == labels).astype(float)
        groups = p0.group_or_video_id.astype(str).to_numpy()
        unique = np.unique(groups)
        group_indexes = {group: np.flatnonzero(groups == group) for group in unique}
        for method in sorted(set(data.method)-{"rcg_fusion_a8_v2"}):
            baseline = data[data.method == method]
            probability = baseline[pcols].to_numpy(float)
            loss = -np.log(np.clip(probability[indexes, labels], 1e-12, 1))
            correct = (probability.argmax(1) == labels).astype(float)
            nll_values = loss-p0_loss
            accuracy_values = p0_correct-correct
            nll_draws = np.empty(repetitions); acc_draws = np.empty(repetitions)
            for draw in range(repetitions):
                sampled = rng.choice(unique, len(unique), replace=True)
                take = np.concatenate([group_indexes[group] for group in sampled])
                nll_draws[draw] = nll_values[take].mean()
                acc_draws[draw] = accuracy_values[take].mean()
            rows.append({
                "dataset": dataset, "baseline": method,
                "nll_gain": float(nll_values.mean()),
                "nll_ci_low": float(np.quantile(nll_draws, .025)),
                "nll_ci_high": float(np.quantile(nll_draws, .975)),
                "nll_one_sided_p": float((np.count_nonzero(nll_draws <= 0)+1)/(repetitions+1)),
                "accuracy_gain": float(accuracy_values.mean()),
                "accuracy_ci_low": float(np.quantile(acc_draws, .025)),
                "accuracy_ci_high": float(np.quantile(acc_draws, .975)),
                "accuracy_one_sided_p": float((np.count_nonzero(acc_draws <= 0)+1)/(repetitions+1)),
            })
    result = pd.DataFrame(rows)
    result["nll_holm_p"] = holm(result.nll_one_sided_p.to_numpy())
    result["accuracy_holm_p"] = holm(result.accuracy_one_sided_p.to_numpy())
    return result


def main():
    result = ROOT/"results"; result.mkdir(exist_ok=True)
    predictions = pd.concat([align_frames(dataset) for dataset in DATASETS], ignore_index=True)
    metrics = metrics_table(predictions)
    paired = paired_bootstrap(predictions)
    predictions.to_parquet(result/"s1_clean_predictions.parquet", index=False)
    metrics.to_csv(result/"s1_clean_metrics.csv", index=False)
    paired.to_csv(result/"s1_clean_paired_bootstrap.csv", index=False)
    manifest = {"experiment": "S1 compute-matched clean main results",
                "method_version": "rcg-fusion-a8-v2", "datasets": list(DATASETS),
                "ensemble_members": list(SEEDS), "bootstrap_repetitions": 10000,
                "status": "single five-member ensemble per method; independent ensemble repeats pending"}
    (result/"s1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.drop(columns="display").to_string(index=False))


if __name__ == "__main__":
    main()
