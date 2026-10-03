"""P0: materialize and audit the one canonical RCG-Fusion final system."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"))
from rcg.final_system import METHOD_VERSION, method_manifest, prediction_hash  # noqa: E402
from run_e4_anchored_routing import CONFIGS  # noqa: E402
from run_e8_complete_ablation import run_dataset  # noqa: E402


def probability_columns(frame: pd.DataFrame, prefix: str = "p"):
    columns = [column for column in frame if column.startswith(prefix)]
    if prefix == "p":
        columns = [column for column in columns if column[1:].isdigit()]
        return sorted(columns, key=lambda value: int(value[1:]))
    return sorted(columns, key=lambda value: int(value[len(prefix):]))


def metrics(probability: np.ndarray, labels: np.ndarray):
    rows = np.arange(len(labels))
    return {
        "accuracy": float((probability.argmax(1) == labels).mean()),
        "nll": float(-np.log(np.clip(probability[rows, labels], 1e-12, 1)).mean()),
    }


def audit_dataset(dataset: str):
    root = ROOT/f"runs/formal-e8-{dataset}"
    frame = pd.read_parquet(root/"final_predictions.parquet")
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if frame.duplicated(["fold", "sample_id"]).any():
        raise AssertionError(f"duplicate final predictions for {dataset}")
    if not (frame.method_version == METHOD_VERSION).all():
        raise AssertionError(f"method version mismatch for {dataset}")
    pcols = probability_columns(frame)
    fcols = probability_columns(frame, "full_p")
    probability = frame[pcols].to_numpy(float)
    full = frame[fcols].to_numpy(float)
    if not np.allclose(probability.sum(1), 1, atol=1e-6):
        raise AssertionError(f"invalid final probability for {dataset}")
    digest = prediction_hash(frame.sample_id, frame.fold, probability)
    if digest != manifest.get("prediction_hash"):
        raise AssertionError(f"prediction hash mismatch for {dataset}")
    long = pd.read_parquet(root/"predictions.parquet")
    long = long[long.method == "A8_safe_fallback"].copy()
    long.sample_id = long.sample_id.astype(str)
    compare = frame[["fold", "sample_id", *pcols]].merge(
        long[["fold", "sample_id", *pcols]], on=["fold", "sample_id"],
        suffixes=("_canonical", "_long"), validate="one_to_one")
    for column in pcols:
        if not np.allclose(compare[f"{column}_canonical"], compare[f"{column}_long"], atol=1e-12):
            raise AssertionError(f"canonical and E8 long predictions differ: {dataset}/{column}")
    label = frame.label.to_numpy(int)
    base_metric, final_metric = metrics(full, label), metrics(probability, label)
    return {
        "dataset": dataset,
        "method_version": METHOD_VERSION,
        "n_samples": len(frame),
        "n_folds": int(frame.fold.nunique()),
        "n_classes": len(pcols),
        "prediction_hash": digest,
        "full_accuracy": base_metric["accuracy"],
        "final_accuracy": final_metric["accuracy"],
        "accuracy_gain": final_metric["accuracy"]-base_metric["accuracy"],
        "full_nll": base_metric["nll"],
        "final_nll": final_metric["nll"],
        "nll_gain": base_metric["nll"]-final_metric["nll"],
        "prediction_path": str((root/"final_predictions.parquet").relative_to(ROOT)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse", action="store_true",
                        help="Audit existing canonical predictions without rerunning inference")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    datasets = tuple(CONFIGS) if args.dataset == "all" else (args.dataset,)
    if not args.reuse:
        for dataset in datasets:
            run_dataset(dataset, device)
    rows = [audit_dataset(dataset) for dataset in datasets]
    result = ROOT/"results"; result.mkdir(exist_ok=True)
    registry_path = result/"p0_final_system_registry.csv"
    registry = pd.DataFrame(rows).sort_values("dataset")
    registry.to_csv(registry_path, index=False)
    combined = hashlib.sha256("".join(
        f"{row['dataset']}:{row['prediction_hash']}" for row in sorted(rows, key=lambda x: x["dataset"])
    ).encode("utf-8")).hexdigest()
    audit = {
        "E1": "historical full-plus-posterior aggregation; must be regenerated from canonical predictions",
        "E7": "historical aggregation decomposition; not the canonical full pipeline",
        "E8": "canonical clean final system source",
        "E9": "uses canonical A7 member actions and E8-fitted clean aggregation parameters",
        "E11": "adapter transfer protocol; must be regenerated under S4 for canonical system claims",
        "E13": "loads canonical E8 final predictions by sample id and fold",
    }
    payload = {**method_manifest(), "combined_prediction_hash": combined,
               "datasets": rows, "legacy_audit": audit}
    (result/"p0_final_system_registry.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(registry[["dataset", "n_samples", "accuracy_gain", "nll_gain",
                    "prediction_hash"]].to_string(index=False))


if __name__ == "__main__":
    main()
